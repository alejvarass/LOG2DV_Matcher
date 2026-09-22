"""
Lith_main.py – Analizador Litológico de Patrones
================================================
Segmenta imágenes de leyendas litológicas, ejecuta OCR sobre las etiquetas,
extrae embeddings densos con DINOv2 (Vision Transformer) y clasifica mediante
similitud de coseno contra el catálogo de patrones.

Soporte de exportación: CSV, MySQL y PostgreSQL.
Ejecución multihilo optimizada para CPU (QThread).
"""

import sys
import os
import csv
import cv2
import numpy as np
import easyocr
import torch
import torchvision.transforms as transforms
from PIL import Image
from difflib import get_close_matches, SequenceMatcher

from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QTextEdit, QSplitter, QTableWidget, QTableWidgetItem,
    QPushButton, QGroupBox, QStackedWidget, QProgressBar,
    QDialog, QFormLayout, QLineEdit, QMessageBox, QFrame, QGridLayout,
    QSpinBox, QHeaderView, QComboBox, QSizePolicy,
)

# ---------------------------------------------------------------------------
# Rutas relativas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "geo_lith.csv")
DEFAULT_PATTERNS_DIR = os.path.join(BASE_DIR, "Patterns")
PATTERN_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def list_pattern_files(patterns_dir):
    """Retorna una lista ordenada de archivos gráficos dentro del directorio."""
    if not os.path.isdir(patterns_dir):
        return []
    return sorted(f for f in os.listdir(patterns_dir) if f.lower().endswith(PATTERN_EXTS))


# ---------------------------------------------------------------------------
# Hoja de estilos (Dark Catppuccin Theme)
# ---------------------------------------------------------------------------
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Cantarell', sans-serif;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton#accentBtn {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
}
QPushButton#accentBtn:hover {
    background-color: #74c7ec;
}
QPushButton#dangerBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    background-color: #313244;
    color: #cdd6f4;
    height: 22px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 3px;
}
QTableWidget {
    background-color: #181825;
    alternate-background-color: #1e1e2e;
    color: #cdd6f4;
    gridline-color: #45475a;
    border: 1px solid #45475a;
    border-radius: 4px;
    selection-background-color: #45475a;
}
QHeaderView::section {
    background-color: #313244;
    color: #89b4fa;
    border: 1px solid #45475a;
    padding: 4px;
    font-weight: bold;
}
QTextEdit, QLineEdit, QSpinBox, QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
}
QLabel#phaseLabel {
    font-size: 13px;
    font-weight: bold;
    color: #a6e3a1;
    padding: 4px;
}
QLabel#statusLabel {
    font-size: 12px;
    color: #f9e2af;
    padding: 2px;
}
QSplitter::handle {
    background-color: #45475a;
}
"""


# ---------------------------------------------------------------------------
# Carga del catálogo maestro CSV
# ---------------------------------------------------------------------------
def load_geo_lith_csv(path):
    """Devuelve una lista estructurada con {'idLith': int, 'nombre': str}."""
    records = []
    if not os.path.isfile(path):
        return records
    with open(path, mode="r", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = ";" if ";" in sample else ","
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            return records
        col_id, col_name = None, None
        for idx, h in enumerate(header):
            hl = h.strip().strip('"').lower()
            if hl in ("idlith",):
                col_id = idx
            elif hl in ("nombre", "name", "litologia"):
                col_name = idx
            elif hl in ("id", "code", "codigo") and col_id is None:
                col_id = idx
        if col_id is None:
            col_id = 0
        if col_name is None:
            col_name = 1
        for row in reader:
            if len(row) <= max(col_id, col_name):
                continue
            id_str = row[col_id].strip().strip('"')
            name_str = row[col_name].strip().strip('"')
            if not id_str:
                continue
            try:
                id_val = int(id_str)
            except ValueError:
                continue
            records.append({"idLith": id_val, "nombre": name_str})
    return records


# ---------------------------------------------------------------------------
# Módulo de Inteligencia Artificial (DINOv2)
# ---------------------------------------------------------------------------
class DINOv2FeatureExtractor:
    """Extrae descriptores morfológicos usando ViT-Small/14 preentrenado."""
    def __init__(self):
        self.device = torch.device("cpu")
        # Modelo ViT-Small con parches de 14x14 (~21M parámetros, rápido en CPU)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def get_embedding(self, img_input):
        """Genera un vector unitario de 384 dimensiones desde ruta o matriz NumPy."""
        if isinstance(img_input, str):
            if not os.path.isfile(img_input):
                return None
            pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if img_input.size == 0:
                return None
            pil_img = Image.fromarray(img_input).convert("RGB")
        else:
            return None

        t = self.transform(pil_img).unsqueeze(0).to(self.device)
        feat = self.model(t).squeeze(0).numpy()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-7)

    @staticmethod
    def cosine_similarity_score(emb1, emb2):
        """Calcula el porcentaje de similitud coseno entre dos vectores normalizados."""
        if emb1 is None or emb2 is None:
            return 0.0
        sim = float(np.dot(emb1, emb2))
        return round(max(0.0, sim) * 100.0, 2)


# ---------------------------------------------------------------------------
# Fuzzy Match de Texto
# ---------------------------------------------------------------------------
def match_lith_name(ocr_text, geo_lith_records):
    """Enlaza texto detectado con el registro formal de geo_lith.csv."""
    if not geo_lith_records or not ocr_text.strip():
        return -1, ocr_text
    upper = ocr_text.upper().strip()
    names = [r["nombre"].upper().strip() for r in geo_lith_records]

    if upper in names:
        idx = names.index(upper)
        return geo_lith_records[idx]["idLith"], geo_lith_records[idx]["nombre"]

    matches = get_close_matches(upper, names, n=1, cutoff=0.45)
    if matches:
        idx = names.index(matches[0])
        return geo_lith_records[idx]["idLith"], geo_lith_records[idx]["nombre"]

    for i, n in enumerate(names):
        if n and (n in upper or upper in n):
            return geo_lith_records[i]["idLith"], geo_lith_records[i]["nombre"]

    best_score, best_idx = 0, -1
    for i, n in enumerate(names):
        if not n:
            continue
        s = SequenceMatcher(None, upper, n).ratio()
        if s > best_score:
            best_score = s
            best_idx = i
    if best_score >= 0.35 and best_idx >= 0:
        return geo_lith_records[best_idx]["idLith"], geo_lith_records[best_idx]["nombre"]

    return -1, ocr_text


# ---------------------------------------------------------------------------
# Worker Thread de Procesamiento Multitarea
# ---------------------------------------------------------------------------
class ProcessingWorker(QThread):
    progress = Signal(int, str)
    stage_changed = Signal(str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, image_path, geo_lith, patterns_dir, id_operadora):
        super().__init__()
        self.image_path = image_path
        self.geo_lith = geo_lith
        self.patterns_dir = patterns_dir
        self.id_operadora = id_operadora

    def run(self):
        try:
            self._process()
        except Exception as e:
            self.error.emit(str(e))

    def _process(self):
        # 1. Segmentación
        self.stage_changed.emit("Etapa 1: Segmentación de recuadros litológicos")
        self.progress.emit(5, "Cargando imagen...")

        img_pil = Image.open(self.image_path).convert("RGB")
        img_np = np.array(img_pil)
        img_h, img_w = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        self.progress.emit(10, "Extrayendo contornos...")
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            # Filtro geométrico estándar para casillas litológicas
            if 15 < w < 100 and 8 < h < 60 and 0.4 < aspect < 4.0:
                boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: (b[1], b[0]))
        self.progress.emit(20, f"{len(boxes)} parches detectados.")

        if not boxes:
            self.error.emit("No se detectaron recuadros de leyenda válidos en la imagen.")
            return

        xs = sorted(set(b[0] for b in boxes))
        col_starts = []
        for xv in xs:
            if not col_starts or xv - col_starts[-1] > 50:
                col_starts.append(xv)
        col_ends = col_starts[1:] + [img_w]

        # 2. OCR
        self.stage_changed.emit("Etapa 2: Reconocimiento Óptico de Caracteres")
        self.progress.emit(25, "Iniciando EasyOCR...")
        reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)

        segments = []
        total = len(boxes)
        for i, (bx, by, bw, bh) in enumerate(boxes):
            pct = 30 + int((i / total) * 25)
            self.progress.emit(pct, f"OCR en elemento {i+1}/{total}...")

            patch = img_np[by:by+bh, bx:bx+bw]

            # Remoción de borde perimetral para evitar contaminación del embedding
            ph, pw = patch.shape[:2]
            if ph > 8 and pw > 8:
                pure_patch = np.ascontiguousarray(patch[3:ph-3, 3:pw-3])
            else:
                pure_patch = patch

            end_x = img_w
            for ce in col_ends:
                if bx + bw + 5 < ce:
                    end_x = ce
                    break
            text_region = img_np[max(0, by-2):min(img_h, by+bh+2), (bx + bw + 1):end_x]

            ocr_text = ""
            if text_region.size > 0:
                results = reader.readtext(text_region, detail=0)
                ocr_text = " ".join(results).strip()
                for ch in ["?", "_", "|", "}", "{"]:
                    ocr_text = ocr_text.replace(ch, "")
                ocr_text = ocr_text.strip()

            segments.append({
                "patch": patch,
                "pure_patch": pure_patch,
                "ocr_text": ocr_text,
                "box": (bx, by, bw, bh),
            })

        # 3. Correlación Textual
        self.stage_changed.emit("Etapa 3: Identificación Litológica (geo_lith)")
        self.progress.emit(60, "Buscando coincidencias de texto...")
        for seg in segments:
            id_lith, matched_name = match_lith_name(seg["ocr_text"], self.geo_lith)
            seg["idLith"] = id_lith
            seg["matched_nombre"] = matched_name

        # 4. Inferencia DINOv2
        self.stage_changed.emit("Etapa 4: Clasificación Visual con DINOv2")
        self.progress.emit(65, "Instanciando DINOv2 (ViT-Small/14)...")
        extractor = DINOv2FeatureExtractor()

        self.progress.emit(75, "Vectorizando catálogo de patrones de referencia...")
        pattern_files = list_pattern_files(self.patterns_dir)
        catalog_embs = []
        for pf in pattern_files:
            fp = os.path.join(self.patterns_dir, pf)
            emb = extractor.get_embedding(fp)
            if emb is not None:
                catalog_embs.append((pf, emb))

        self.progress.emit(85, "Calculando similitudes coseno...")
        for i, seg in enumerate(segments):
            pct = 85 + int((i / total) * 14)
            self.progress.emit(pct, f"Evaluando vector {i+1}/{total}...")

            seg_emb = extractor.get_embedding(seg["pure_patch"])
            if seg_emb is None:
                seg_emb = extractor.get_embedding(seg["patch"])

            best_file = ""
            best_score = 0.0

            for pf, cat_emb in catalog_embs:
                sc = extractor.cosine_similarity_score(seg_emb, cat_emb)
                if sc > best_score:
                    best_score = sc
                    best_file = pf

            seg["patternImage"] = best_file
            seg["pattern_score"] = best_score

        self.stage_changed.emit("Completado")
        self.progress.emit(100, f"Procesamiento finalizado ({len(segments)} items).")

        results = []
        for idx, seg in enumerate(segments, start=1):
            results.append({
                "id": idx,
                "patternImage": seg["patternImage"],
                "idLith": seg["idLith"],
                "idOperadora": self.id_operadora,
                "ocr_text": seg["ocr_text"],
                "matched_nombre": seg["matched_nombre"],
                "pattern_score": seg["pattern_score"],
                "patch": seg["patch"],
                "box": seg["box"],
            })
        self.finished.emit(results)


# ---------------------------------------------------------------------------
# Módulo de Exportación Multibase de Datos
# ---------------------------------------------------------------------------
class ExportDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("Exportar Correlación Litológica")
        self.resize(460, 200)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.cmb_format = QComboBox()
        self.cmb_format.addItems([
            "CSV (.csv)",
            "SQL para MySQL (.sql)",
            "SQL para PostgreSQL (.sql)",
        ])
        form.addRow("Destino:", self.cmb_format)

        self.txt_table = QLineEdit("correlacion_litologica")
        form.addRow("Tabla SQL:", self.txt_table)
        layout.addLayout(form)

        btn = QPushButton("Generar Archivo")
        btn.setObjectName("accentBtn")
        btn.setFixedHeight(36)
        btn.clicked.connect(self.do_export)
        layout.addWidget(btn)

    def do_export(self):
        fmt = self.cmb_format.currentIndex()
        if fmt == 0:
            p, _ = QFileDialog.getSaveFileName(self, "Exportar CSV", "resultado.csv", "CSV (*.csv)")
            if p: self._export_csv(p)
        elif fmt == 1:
            p, _ = QFileDialog.getSaveFileName(self, "Exportar MySQL", "resultado_mysql.sql", "SQL (*.sql)")
            if p: self._export_mysql(p)
        else:
            p, _ = QFileDialog.getSaveFileName(self, "Exportar PostgreSQL", "resultado_postgres.sql", "SQL (*.sql)")
            if p: self._export_postgresql(p)

    def _export_csv(self, path):
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "patternImage", "idLith", "idOperadora"])
                for r in self.results:
                    w.writerow([r["id"], r["patternImage"], r["idLith"], r["idOperadora"]])
            QMessageBox.information(self, "Éxito", f"Archivo CSV guardado:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _export_mysql(self, path):
        try:
            tbl = self.txt_table.text().strip() or "correlacion_litologica"
            lines = [
                f"CREATE TABLE IF NOT EXISTS `{tbl}` (",
                "    `id` INT AUTO_INCREMENT PRIMARY KEY,",
                "    `patternImage` VARCHAR(255) NOT NULL,",
                "    `idLith` INT NOT NULL,",
                "    `idOperadora` INT NOT NULL",
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n"
            ]
            for r in self.results:
                pi = r["patternImage"].replace("'", "\\'")
                lines.append(
                    f"INSERT INTO `{tbl}` (`patternImage`, `idLith`, `idOperadora`) "
                    f"VALUES ('{pi}', {r['idLith']}, {r['idOperadora']});"
                )
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            QMessageBox.information(self, "Éxito", f"Script MySQL generado:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _export_postgresql(self, path):
        try:
            tbl = self.txt_table.text().strip() or "correlacion_litologica"
            lines = [
                f'CREATE TABLE IF NOT EXISTS "{tbl}" (',
                '    "id" BIGSERIAL PRIMARY KEY,',
                '    "patternImage" VARCHAR(255) NOT NULL,',
                '    "idLith" INTEGER NOT NULL,',
                '    "idOperadora" INTEGER NOT NULL',
                ');\n'
            ]
            for r in self.results:
                pi = r["patternImage"].replace("'", "''")
                lines.append(
                    f'INSERT INTO "{tbl}" ("patternImage", "idLith", "idOperadora") '
                    f"VALUES ('{pi}', {r['idLith']}, {r['idOperadora']});"
                )
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            QMessageBox.information(self, "Éxito", f"Script PostgreSQL generado:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ---------------------------------------------------------------------------
# Contenedor Visual de Arrastre
# ---------------------------------------------------------------------------
class DropImageZone(QLabel):
    imageSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("🖼️ Arrastrá el log/imagen aquí\no hacé clic para explorar")
        self.setAcceptDrops(True)
        self.setMinimumSize(280, 120)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #585b70;
                border-radius: 10px;
                background-color: #181825;
                color: #a6adc8;
                font-size: 13px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #89b4fa;
                background-color: #1e1e2e;
            }
        """)
        self._path = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            p, _ = QFileDialog.getOpenFileName(
                self, "Seleccionar Imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )
            if p: self._set_image(p)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith(PATTERN_EXTS):
                self._set_image(p)
                break

    def _set_image(self, path):
        self._path = path
        pix = QPixmap(path).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)
        self.imageSelected.emit(path)


# ---------------------------------------------------------------------------
# Ventana Principal (Interfaz PySide6)
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lith Analizador Litológico (DINOv2 ViT)")
        self.resize(1340, 800)

        self.image_path = None
        self.geo_lith = load_geo_lith_csv(DEFAULT_CSV_PATH)
        self.patterns_dir = DEFAULT_PATTERNS_DIR
        self.results = []
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)

        # Barra superior de control
        phase_bar = QHBoxLayout()
        self.phase_btns = []
        for i, txt in enumerate(["① Configuración", "② Procesamiento IA", "③ Validación"]):
            b = QPushButton(txt)
            b.setFixedHeight(36)
            b.setEnabled(i == 0)
            b.clicked.connect(lambda _, idx=i: self.stacked.setCurrentIndex(idx))
            phase_bar.addWidget(b)
            self.phase_btns.append(b)
        root.addLayout(phase_bar)

        self.lbl_status = QLabel("Listo para configurar parámetros.")
        self.lbl_status.setObjectName("statusLabel")
        root.addWidget(self.lbl_status)

        self.stacked = QStackedWidget()
        root.addWidget(self.stacked)
        self.stacked.addWidget(self._build_phase1())
        self.stacked.addWidget(self._build_phase2())
        self.stacked.addWidget(self._build_phase3())

    def _build_phase1(self):
        w = QWidget()
        lay = QHBoxLayout(w)

        # Sección izquierda: Dropzone
        grp_img = QGroupBox("Imagen Litológica de Entrada")
        gl = QVBoxLayout(grp_img)
        self.drop_zone = DropImageZone()
        self.drop_zone.imageSelected.connect(self._on_image_selected)
        gl.addWidget(self.drop_zone)
        lay.addWidget(grp_img, stretch=2)

        # Sección derecha: Parámetros de Operación
        right = QVBoxLayout()

        grp_op = QGroupBox("Metadatos de Perforación")
        ol = QFormLayout(grp_op)
        self.spn_operadora = QSpinBox()
        self.spn_operadora.setRange(1, 999999)
        self.spn_operadora.setValue(1)
        ol.addRow("idOperadora:", self.spn_operadora)
        right.addWidget(grp_op)

        grp_csv = QGroupBox("Catálogo geo_lith.csv")
        cl = QVBoxLayout(grp_csv)
        self.lbl_csv_status = QLabel(
            f"✅ {len(self.geo_lith)} litologías cargadas" if self.geo_lith else "❌ No encontrado"
        )
        cl.addWidget(self.lbl_csv_status)
        btn_csv = QPushButton("Cargar CSV alternativo")
        btn_csv.clicked.connect(self._load_csv)
        cl.addWidget(btn_csv)
        right.addWidget(grp_csv)

        grp_pat = QGroupBox("Patrones de Referencia")
        pl = QVBoxLayout(grp_pat)
        n_pats = len(list_pattern_files(self.patterns_dir))
        self.lbl_patterns = QLabel(f"📁 {self.patterns_dir}\n({n_pats} patrones indexados)")
        pl.addWidget(self.lbl_patterns)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(3)
        self._update_pattern_grid()
        pl.addWidget(self.grid_widget)

        btn_pat = QPushButton("Cambiar carpeta de patrones")
        btn_pat.clicked.connect(self._select_patterns_dir)
        pl.addWidget(btn_pat)
        right.addWidget(grp_pat)

        self.btn_process = QPushButton("▶  PROCESAR CON DINOv2")
        self.btn_process.setObjectName("accentBtn")
        self.btn_process.setFixedHeight(44)
        self.btn_process.setStyleSheet(
            "font-size: 14px; font-weight: bold; background-color: #a6e3a1; color: #1e1e2e; border:none; border-radius:6px;"
        )
        self.btn_process.clicked.connect(self._start_processing)
        right.addWidget(self.btn_process)

        right.addStretch()
        lay.addLayout(right, stretch=1)
        return w

    def _update_pattern_grid(self):
        while self.grid_layout.count():
            it = self.grid_layout.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        if not os.path.isdir(self.patterns_dir): return
        for i, fname in enumerate(list_pattern_files(self.patterns_dir)[:9]):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(60, 42)
            lbl.setStyleSheet("border:1px solid #45475a; background:#181825; border-radius:3px;")
            pix = QPixmap(os.path.join(self.patterns_dir, fname))
            lbl.setPixmap(pix.scaled(56, 38, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.grid_layout.addWidget(lbl, i // 3, i % 3)

    def _build_phase2(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Estado de Inferencia")
        gl = QVBoxLayout(grp)

        self.lbl_stage = QLabel("Esperando inicio...")
        self.lbl_stage.setObjectName("phaseLabel")
        gl.addWidget(self.lbl_stage)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        gl.addWidget(self.progress_bar)

        self.lbl_progress_detail = QLabel("")
        self.lbl_progress_detail.setObjectName("statusLabel")
        gl.addWidget(self.lbl_progress_detail)
        lay.addWidget(grp)

        grp_log = QGroupBox("Consola de Operaciones")
        ll = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        ll.addWidget(self.txt_log)
        lay.addWidget(grp_log)
        return w

    def _build_phase3(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        self.lbl_result_summary = QLabel("")
        self.lbl_result_summary.setObjectName("phaseLabel")
        top.addWidget(self.lbl_result_summary)
        top.addStretch()

        btn_exp = QPushButton("📄 Exportar (CSV / MySQL / PostgreSQL)")
        btn_exp.setObjectName("accentBtn")
        btn_exp.setFixedHeight(34)
        btn_exp.clicked.connect(self._open_export_dialog)
        top.addWidget(btn_exp)
        lay.addLayout(top)

        splitter = QSplitter(Qt.Vertical)

        # Panel de Inspección Visual
        grp_insp = QGroupBox("Inspector Morfológico")
        il = QHBoxLayout(grp_insp)

        f1 = QFrame()
        l1 = QVBoxLayout(f1)
        l1.addWidget(QLabel("<b>Parche Extraído</b>"), alignment=Qt.AlignCenter)
        self.lbl_insp_patch = QLabel("—")
        self.lbl_insp_patch.setAlignment(Qt.AlignCenter)
        self.lbl_insp_patch.setFixedSize(120, 80)
        self.lbl_insp_patch.setStyleSheet("border:1px solid #45475a; background:#181825;")
        l1.addWidget(self.lbl_insp_patch, alignment=Qt.AlignCenter)
        il.addWidget(f1)

        f2 = QFrame()
        l2 = QVBoxLayout(f2)
        l2.addWidget(QLabel("<b>Identificación OCR</b>"), alignment=Qt.AlignCenter)
        self.lbl_insp_info = QLabel("idLith: —\nNombre: —\nOCR: —")
        self.lbl_insp_info.setAlignment(Qt.AlignCenter)
        self.lbl_insp_info.setStyleSheet("font-size:12px; color:#a6e3a1;")
        l2.addWidget(self.lbl_insp_info)
        il.addWidget(f2)

        f3 = QFrame()
        l3 = QVBoxLayout(f3)
        l3.addWidget(QLabel("<b>Patrón Clasificado (DINOv2)</b>"), alignment=Qt.AlignCenter)
        self.lbl_insp_match = QLabel("—")
        self.lbl_insp_match.setAlignment(Qt.AlignCenter)
        self.lbl_insp_match.setFixedSize(120, 80)
        self.lbl_insp_match.setStyleSheet("border:1px solid #45475a; background:#181825;")
        l3.addWidget(self.lbl_insp_match, alignment=Qt.AlignCenter)
        self.lbl_match_detail = QLabel("Archivo: —\nSimilitud: —")
        self.lbl_match_detail.setAlignment(Qt.AlignCenter)
        l3.addWidget(self.lbl_match_detail)
        il.addWidget(f3)

        splitter.addWidget(grp_insp)

        # Tabla de asignaciones
        self.table_results = QTableWidget()
        self.table_results.setColumnCount(8)
        self.table_results.setHorizontalHeaderLabels([
            "Parche", "id", "patternImage", "idLith", "idOperadora",
            "Texto OCR", "Nombre Coincidente", "Ajuste Manual",
        ])
        self.table_results.setAlternatingRowColors(True)
        self.table_results.verticalHeader().setDefaultSectionSize(46)

        hdr = self.table_results.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeToContents)

        self.table_results.itemSelectionChanged.connect(self._on_result_row_changed)
        splitter.addWidget(self.table_results)

        splitter.setSizes([180, 420])
        lay.addWidget(splitter)
        return w

    # -- Callbacks y Gestión de Estado -------------------------------------
    def _on_image_selected(self, p):
        self.image_path = p
        self.lbl_status.setText(f"Imagen seleccionada: {os.path.basename(p)}")

    def _load_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "Cargar geo_lith.csv", "", "CSV (*.csv)")
        if p:
            self.geo_lith = load_geo_lith_csv(p)
            self.lbl_csv_status.setText(f"✅ {len(self.geo_lith)} registros desde {os.path.basename(p)}")

    def _select_patterns_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Directorio de patrones")
        if d:
            self.patterns_dir = d
            n = len(list_pattern_files(d))
            self.lbl_patterns.setText(f"📁 {d}\n({n} imágenes)")
            self._update_pattern_grid()

    def _start_processing(self):
        if not self.image_path:
            QMessageBox.warning(self, "Atención", "Cargá una imagen de registro primero.")
            return
        if not self.geo_lith:
            QMessageBox.warning(self, "Atención", "El catálogo geo_lith.csv no está presente.")
            return

        self.stacked.setCurrentIndex(1)
        self.phase_btns[1].setEnabled(True)
        self.progress_bar.setValue(0)
        self.txt_log.clear()

        self.worker = ProcessingWorker(
            self.image_path,
            self.geo_lith,
            self.patterns_dir,
            self.spn_operadora.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.stage_changed.connect(self._on_stage)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.lbl_progress_detail.setText(msg)
        self.txt_log.append(f"[{pct:3d}%] {msg}")

    def _on_stage(self, stage):
        self.lbl_stage.setText(stage)
        self.txt_log.append(f"\n>>> {stage}")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Falla en ejecución", msg)
        self.lbl_stage.setText("Error")

    def _on_finished(self, results):
        self.results = results
        self.phase_btns[2].setEnabled(True)
        self.stacked.setCurrentIndex(2)
        self.lbl_result_summary.setText(f"✅ {len(results)} estratos analizados")
        self._populate_results_table()
        if results:
            self.table_results.selectRow(0)

    def _populate_results_table(self):
        pat_files = list_pattern_files(self.patterns_dir)
        self.table_results.blockSignals(True)
        self.table_results.setRowCount(len(self.results))

        for i, r in enumerate(self.results):
            # Columna 0: Miniatura
            patch = r.get("patch")
            if patch is not None and getattr(patch, "size", 0) > 0:
                patch_c = np.ascontiguousarray(patch)
                h, w, c = patch_c.shape
                qimg = QImage(patch_c.data, w, h, c * w, QImage.Format_RGB888)
                lbl_thumb = QLabel()
                lbl_thumb.setAlignment(Qt.AlignCenter)
                lbl_thumb.setPixmap(QPixmap.fromImage(qimg).scaled(56, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.table_results.setCellWidget(i, 0, lbl_thumb)

            # Columnas informativas
            for col, key in enumerate(["id", "patternImage", "idLith", "idOperadora", "ocr_text", "matched_nombre"], start=1):
                item = QTableWidgetItem(str(r[key]))
                if col <= 4:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                self.table_results.setItem(i, col, item)

            # Columna 7: Selector de anulación manual
            combo = QComboBox()
            combo.setIconSize(QSize(56, 38))
            combo.addItem("(sin patrón)", "")
            for pf in pat_files:
                combo.addItem(QIcon(os.path.join(self.patterns_dir, pf)), pf, pf)
            idx = combo.findData(r["patternImage"])
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.setProperty("score_text", f"{r['pattern_score']:.1f}")
            combo.currentIndexChanged.connect(lambda _, row=i, c=combo: self._on_manual_pattern_selected(row, c))
            self.table_results.setCellWidget(i, 7, combo)

        self.table_results.blockSignals(False)

    def _on_manual_pattern_selected(self, row, combo):
        if row < 0 or row >= len(self.results): return
        selected = combo.currentData() or ""
        self.results[row]["patternImage"] = selected
        item = self.table_results.item(row, 2)
        if item: item.setText(selected)
        if row == self.table_results.currentRow():
            self._on_result_row_changed()

    def _on_result_row_changed(self):
        row = self.table_results.currentRow()
        if row < 0 or row >= len(self.results): return
        r = self.results[row]

        # Actualizar vista del parche
        patch = r.get("patch")
        if patch is not None and patch.size > 0:
            pc = np.ascontiguousarray(patch)
            h, w, c = pc.shape
            qimg = QImage(pc.data, w, h, c * w, QImage.Format_RGB888)
            self.lbl_insp_patch.setPixmap(
                QPixmap.fromImage(qimg).scaled(self.lbl_insp_patch.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        self.lbl_insp_info.setText(f"idLith: {r['idLith']}\nNombre: {r['matched_nombre']}\nOCR: {r['ocr_text']}")

        # Actualizar vista del patrón clasificado
        combo = self.table_results.cellWidget(row, 7)
        score = combo.property("score_text") if isinstance(combo, QComboBox) else f"{r['pattern_score']:.1f}"
        if r["patternImage"]:
            p = os.path.join(self.patterns_dir, r["patternImage"])
            if os.path.isfile(p):
                self.lbl_insp_match.setPixmap(
                    QPixmap(p).scaled(self.lbl_insp_match.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            self.lbl_match_detail.setText(f"Archivo: {r['patternImage']}\nSimilitud Coseno: {score}%")
        else:
            self.lbl_insp_match.setText("—")
            self.lbl_match_detail.setText("Sin coincidencia")

    def _open_export_dialog(self):
        if not self.results:
            QMessageBox.warning(self, "Atención", "No hay datos para exportar.")
            return
        ExportDialog(self.results, self).exec()


# ---------------------------------------------------------------------------
# Entrada de la Aplicación
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())