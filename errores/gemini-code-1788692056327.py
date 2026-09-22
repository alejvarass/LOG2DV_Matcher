"""
Lith_main.py – Analizador Litológico de Patrones (Deep Learning ResNet50)
========================================================================
Segmenta imágenes de leyendas litológicas, ejecuta OCR sobre las etiquetas,
extrae embeddings con ResNet50 (preentrenado en ImageNet) y compara mediante
similitud de coseno normalizada contra el catálogo de patrones.

Soporta exportación a CSV, MySQL y PostgreSQL.
Optimizado para CPU en segundo plano (QThread).
"""

import sys
import os
import csv
import cv2
import numpy as np
import easyocr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from difflib import get_close_matches, SequenceMatcher

from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPalette, QIcon
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
    """Lista ordenada de archivos de imagen de patrones en un directorio."""
    if not os.path.isdir(patterns_dir):
        return []
    return sorted(f for f in os.listdir(patterns_dir) if f.lower().endswith(PATTERN_EXTS))


# ---------------------------------------------------------------------------
# Estilos Dark Theme
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
# Carga de catálogo geo_lith.csv
# ---------------------------------------------------------------------------
def load_geo_lith_csv(path):
    """Devuelve una lista de diccionarios con {'idLith': int, 'nombre': str}."""
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
# Extractor de Características ResNet50 y Similitud Coseno
# ---------------------------------------------------------------------------
class ResNetFeatureExtractor:
    """Extrae embeddings L2 normalizados (dim=2048) usando ResNet50 en CPU."""
    def __init__(self):
        self.device = torch.device("cpu")
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def get_embedding(self, img_input):
        """Genera embedding normalizado L2 a partir de un arreglo NumPy RGB o ruta de archivo."""
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
        """Calcula el puntaje de similitud de coseno en escala 0.0 a 100.0%."""
        if emb1 is None or emb2 is None:
            return 0.0
        sim = float(np.dot(emb1, emb2))
        return round(max(0.0, sim) * 100.0, 2)


# ---------------------------------------------------------------------------
# Fuzzy match de texto OCR con geo_lith.csv
# ---------------------------------------------------------------------------
def match_lith_name(ocr_text, geo_lith_records):
    """Devuelve (idLith, matched_nombre) o (-1, ocr_text)."""
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
# Worker Thread: Segmentación, OCR y Matching con ResNet50
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
        # --- Etapa 1: Segmentación ---
        self.stage_changed.emit("Etapa 1: Cargando imagen y segmentando")
        self.progress.emit(5, "Cargando imagen...")

        img_pil = Image.open(self.image_path).convert("RGB")
        img_np = np.array(img_pil)
        img_h, img_w = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        self.progress.emit(10, "Detectando recuadros de patrones...")
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if 15 < w < 100 and 8 < h < 60 and 0.4 < aspect < 4.0:
                boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: (b[1], b[0]))
        self.progress.emit(20, f"Encontrados {len(boxes)} segmentos.")

        if not boxes:
            self.error.emit("No se detectaron segmentos de patrones en la imagen.")
            return

        xs = sorted(set(b[0] for b in boxes))
        col_starts = []
        for xv in xs:
            if not col_starts or xv - col_starts[-1] > 50:
                col_starts.append(xv)
        col_ends = col_starts[1:] + [img_w]

        # --- Etapa 2: OCR ---
        self.stage_changed.emit("Etapa 2: Reconocimiento de texto (OCR)")
        self.progress.emit(25, "Inicializando EasyOCR...")
        reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        self.progress.emit(35, "Motor OCR listo. Procesando textos...")

        segments = []
        total = len(boxes)
        for i, (bx, by, bw, bh) in enumerate(boxes):
            pct = 35 + int((i / total) * 20)
            self.progress.emit(pct, f"OCR segmento {i+1}/{total}...")

            patch = img_np[by:by+bh, bx:bx+bw]

            ph, pw = patch.shape[:2]
            if ph > 8 and pw > 8:
                pure_patch = np.ascontiguousarray(patch[2:ph-2, 2:pw-2])
            else:
                pure_patch = patch

            end_x = img_w
            for ce in col_ends:
                if bx + bw + 5 < ce:
                    end_x = ce
                    break
            text_x_start = bx + bw + 1
            text_region = img_np[max(0, by-2):min(img_h, by+bh+2), text_x_start:end_x]

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

        # --- Etapa 3: Identificación OCR con geo_lith ---
        self.stage_changed.emit("Etapa 3: Identificación litológica (geo_lith)")
        self.progress.emit(55, "Correlacionando textos con geo_lith.csv...")

        for i, seg in enumerate(segments):
            id_lith, matched_name = match_lith_name(seg["ocr_text"], self.geo_lith)
            seg["idLith"] = id_lith
            seg["matched_nombre"] = matched_name

        # --- Etapa 4: Comparación con Patrones vía ResNet50 ---
        self.stage_changed.emit("Etapa 4: Análisis Deep Learning con ResNet50")
        self.progress.emit(65, "Cargando pesos de ResNet50...")
        extractor = ResNetFeatureExtractor()

        self.progress.emit(75, "Vectorizando catálogo de patrones...")
        pattern_files = list_pattern_files(self.patterns_dir)
        catalog_embs = []
        for pf in pattern_files:
            fp = os.path.join(self.patterns_dir, pf)
            emb = extractor.get_embedding(fp)
            if emb is not None:
                catalog_embs.append((pf, emb))

        self.progress.emit(85, "Calculando similitudes de coseno...")
        for i, seg in enumerate(segments):
            pct = 85 + int((i / total) * 13)
            self.progress.emit(pct, f"Evaluando similitud {i+1}/{total}...")

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

        # --- Finalización ---
        self.stage_changed.emit("Completado")
        self.progress.emit(100, f"Procesamiento finalizado: {len(segments)} elementos.")

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
# Diálogo de Exportación (CSV, MySQL o PostgreSQL)
# ---------------------------------------------------------------------------
class ExportDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("Exportar Resultados")
        self.resize(480, 210)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.cmb_format = QComboBox()
        self.cmb_format.addItems([
            "CSV (.csv)",
            "SQL MySQL (.sql)",
            "SQL PostgreSQL (.sql)"
        ])
        form.addRow("Formato:", self.cmb_format)

        self.txt_table = QLineEdit("correlacion_litologica")
        form.addRow("Nombre de tabla (SQL):", self.txt_table)
        layout.addLayout(form)

        btn = QPushButton("Exportar")
        btn.setObjectName("accentBtn")
        btn.setFixedHeight(36)
        btn.clicked.connect(self.do_export)
        layout.addWidget(btn)

    def do_export(self):
        fmt = self.cmb_format.currentIndex()
        if fmt == 0:
            path, _ = QFileDialog.getSaveFileName(
                self, "Guardar CSV", "resultado.csv", "CSV (*.csv)"
            )
            if path:
                self._export_csv(path)
        elif fmt == 1:
            path, _ = QFileDialog.getSaveFileName(
                self, "Guardar SQL (MySQL)", "resultado_mysql.sql", "SQL (*.sql)"
            )
            if path:
                self._export_mysql(path)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Guardar SQL (PostgreSQL)", "resultado_postgres.sql", "SQL (*.sql)"
            )
            if path:
                self._export_postgresql(path)

    def _export_csv(self, path):
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "patternImage", "idLith", "idOperadora"])
                for r in self.results:
                    writer.writerow([r["id"], r["patternImage"], r["idLith"], r["idOperadora"]])
            QMessageBox.information(self, "Exportado", f"CSV guardado en:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _export_mysql(self, path):
        try:
            table = self.txt_table.text().strip() or "correlacion_litologica"
            lines = [
                f"CREATE TABLE IF NOT EXISTS `{table}` (",
                "    `id` INT AUTO_INCREMENT PRIMARY KEY,",
                "    `patternImage` VARCHAR(255) NOT NULL,",
                "    `idLith` INT NOT NULL,",
                "    `idOperadora` INT NOT NULL",
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                ""
            ]
            for r in self.results:
                pi = r["patternImage"].replace("'", "\\'")
                lines.append(
                    f"INSERT INTO `{table}` (`patternImage`, `idLith`, `idOperadora`) "
                    f"VALUES ('{pi}', {r['idLith']}, {r['idOperadora']});"
                )
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            QMessageBox.information(self, "Exportado", f"SQL (MySQL) guardado en:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _export_postgresql(self, path):
        try:
            table = self.txt_table.text().strip() or "correlacion_litologica"
            lines = [
                f'CREATE TABLE IF NOT EXISTS "{table}" (',
                '    "id" BIGSERIAL PRIMARY KEY,',
                '    "patternImage" VARCHAR(255) NOT NULL,',
                '    "idLith" INTEGER NOT NULL,',
                '    "idOperadora" INTEGER NOT NULL',
                ');',
                ''
            ]
            for r in self.results:
                # En SQL estándar y Postgres las comillas simples se escapan duplicándolas
                pi = r["patternImage"].replace("'", "''")
                lines.append(
                    f'INSERT INTO "{table}" ("patternImage", "idLith", "idOperadora") '
                    f"VALUES ('{pi}', {r['idLith']}, {r['idOperadora']});"
                )
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            QMessageBox.information(self, "Exportado", f"SQL (PostgreSQL) guardado en:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ---------------------------------------------------------------------------
# Zona de Drag and Drop de imagen
# ---------------------------------------------------------------------------
class DropImageZone(QLabel):
    imageSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("🖼️  Arrastrá una imagen aquí\no hacé clic para seleccionar")
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
            path, _ = QFileDialog.getOpenFileName(
                self, "Seleccionar Imagen", "",
                "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )
            if path:
                self._set_image(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            fp = url.toLocalFile()
            if fp.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                self._set_image(fp)
                break

    def _set_image(self, path):
        self._path = path
        pix = QPixmap(path)
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.imageSelected.emit(path)

    def get_path(self):
        return self._path


# ---------------------------------------------------------------------------
# Ventana Principal (UI)
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lith Analizador Litológico (ResNet50)")
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

        # Barra de fases
        phase_bar = QHBoxLayout()
        self.phase_btns = []
        labels = [
            "① Configuración",
            "② Procesamiento (DL)",
            "③ Resultados y Exportación",
        ]
        for i, txt in enumerate(labels):
            btn = QPushButton(txt)
            btn.setFixedHeight(36)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, idx=i: self.stacked.setCurrentIndex(idx))
            phase_bar.addWidget(btn)
            self.phase_btns.append(btn)
        self.phase_btns[0].setEnabled(True)
        root.addLayout(phase_bar)

        self.lbl_status = QLabel("Configurá los parámetros y presioná Procesar.")
        self.lbl_status.setObjectName("statusLabel")
        root.addWidget(self.lbl_status)

        self.stacked = QStackedWidget()
        root.addWidget(self.stacked)

        self.stacked.addWidget(self._build_phase1())
        self.stacked.addWidget(self._build_phase2())
        self.stacked.addWidget(self._build_phase3())

    # -- Fase 1: Configuración ---------------------------------------------
    def _build_phase1(self):
        w = QWidget()
        lay = QHBoxLayout(w)

        # Izquierda: Imagen
        left = QVBoxLayout()
        grp_img = QGroupBox("Imagen de Referencias Litológicas")
        gl = QVBoxLayout(grp_img)
        self.drop_zone = DropImageZone()
        self.drop_zone.imageSelected.connect(self._on_image_selected)
        gl.addWidget(self.drop_zone)
        left.addWidget(grp_img)
        lay.addLayout(left, stretch=2)

        # Derecha: Parámetros y Catálogos
        right = QVBoxLayout()

        # idOperadora
        grp_op = QGroupBox("Datos de Operadora")
        ol = QFormLayout(grp_op)
        self.spn_operadora = QSpinBox()
        self.spn_operadora.setRange(1, 999999)
        self.spn_operadora.setValue(1)
        ol.addRow("idOperadora:", self.spn_operadora)
        right.addWidget(grp_op)

        # CSV geo_lith
        grp_csv = QGroupBox("Archivo geo_lith.csv")
        cl = QVBoxLayout(grp_csv)
        csv_status = f"✅ {len(self.geo_lith)} registros cargados" if self.geo_lith else "❌ No encontrado"
        self.lbl_csv_status = QLabel(csv_status)
        cl.addWidget(self.lbl_csv_status)
        self.txt_csv_preview = QTextEdit()
        self.txt_csv_preview.setReadOnly(True)
        self.txt_csv_preview.setFixedHeight(120)
        self.txt_csv_preview.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        preview = "idLith | Nombre\n" + "─" * 35 + "\n"
        for r in self.geo_lith[:8]:
            preview += f"{r['idLith']:<6} | {r['nombre']}\n"
        if len(self.geo_lith) > 8:
            preview += f"... y {len(self.geo_lith)-8} más"
        self.txt_csv_preview.setPlainText(preview)
        cl.addWidget(self.txt_csv_preview)
        btn_csv = QPushButton("Cargar otro CSV")
        btn_csv.clicked.connect(self._load_csv)
        cl.addWidget(btn_csv)
        right.addWidget(grp_csv)

        # Carpeta de Patrones
        grp_pat = QGroupBox("Catálogo de Patrones")
        pl = QVBoxLayout(grp_pat)
        n_patterns = len(list_pattern_files(self.patterns_dir))
        self.lbl_patterns = QLabel(f"📁 Patrones: {self.patterns_dir}\n({n_patterns} imágenes)")
        pl.addWidget(self.lbl_patterns)

        # Grid preview
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(3)
        self._update_pattern_grid()
        pl.addWidget(self.grid_widget)

        btn_pat = QPushButton("Seleccionar otra carpeta de patrones")
        btn_pat.clicked.connect(self._select_patterns_dir)
        pl.addWidget(btn_pat)
        right.addWidget(grp_pat)

        # Botón de Procesar
        self.btn_process = QPushButton("▶  PROCESAR CON IA")
        self.btn_process.setObjectName("accentBtn")
        self.btn_process.setFixedHeight(44)
        self.btn_process.setStyleSheet(
            "font-size: 15px; font-weight: bold; background-color: #a6e3a1; color: #1e1e2e; border:none; border-radius:6px;"
        )
        self.btn_process.clicked.connect(self._start_processing)
        right.addWidget(self.btn_process)

        right.addStretch()
        lay.addLayout(right, stretch=1)
        return w

    def _update_pattern_grid(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not os.path.isdir(self.patterns_dir):
            return
        files = list_pattern_files(self.patterns_dir)[:9]
        for i, fname in enumerate(files):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(60, 42)
            lbl.setStyleSheet("border:1px solid #45475a; background:#181825; border-radius:3px;")
            pix = QPixmap(os.path.join(self.patterns_dir, fname))
            lbl.setPixmap(pix.scaled(56, 38, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lbl.setToolTip(fname)
            self.grid_layout.addWidget(lbl, i // 3, i % 3)

    # -- Fase 2: Progreso --------------------------------------------------
    def _build_phase2(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("Progreso del Procesamiento")
        gl = QVBoxLayout(grp)

        self.lbl_stage = QLabel("Esperando...")
        self.lbl_stage.setObjectName("phaseLabel")
        gl.addWidget(self.lbl_stage)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        gl.addWidget(self.progress_bar)

        self.lbl_progress_detail = QLabel("")
        self.lbl_progress_detail.setObjectName("statusLabel")
        gl.addWidget(self.lbl_progress_detail)

        lay.addWidget(grp)

        grp_log = QGroupBox("Log de Procesamiento")
        ll = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        ll.addWidget(self.txt_log)
        lay.addWidget(grp_log)

        return w

    # -- Fase 3: Resultados y Tabla Final -----------------------------------
    def _build_phase3(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        self.lbl_result_summary = QLabel("")
        self.lbl_result_summary.setObjectName("phaseLabel")
        top.addWidget(self.lbl_result_summary)
        top.addStretch()

        btn_csv_exp = QPushButton("📄 Exportar (CSV / MySQL / PostgreSQL)")
        btn_csv_exp.setObjectName("accentBtn")
        btn_csv_exp.setFixedHeight(34)
        btn_csv_exp.clicked.connect(self._open_export_dialog)
        top.addWidget(btn_csv_exp)
        lay.addLayout(top)

        splitter = QSplitter(Qt.Vertical)

        # Inspector visual
        grp_insp = QGroupBox("Inspector Visual")
        il = QHBoxLayout(grp_insp)

        # Parche extraído
        f1 = QFrame()
        f1.setFrameShape(QFrame.StyledPanel)
        l1 = QVBoxLayout(f1)
        l1.addWidget(QLabel("<b>Patrón Extraído</b>"), alignment=Qt.AlignCenter)
        self.lbl_insp_patch = QLabel("—")
        self.lbl_insp_patch.setAlignment(Qt.AlignCenter)
        self.lbl_insp_patch.setFixedSize(120, 80)
        self.lbl_insp_patch.setStyleSheet("border:1px solid #45475a; background:#181825;")
        l1.addWidget(self.lbl_insp_patch, alignment=Qt.AlignCenter)
        il.addWidget(f1)

        # Identificación
        f2 = QFrame()
        f2.setFrameShape(QFrame.StyledPanel)
        l2 = QVBoxLayout(f2)
        l2.addWidget(QLabel("<b>Identificación</b>"), alignment=Qt.AlignCenter)
        self.lbl_insp_info = QLabel("idLith: —\nNombre: —\nOCR: —")
        self.lbl_insp_info.setAlignment(Qt.AlignCenter)
        self.lbl_insp_info.setStyleSheet("font-size:12px; color:#a6e3a1;")
        l2.addWidget(self.lbl_insp_info)
        il.addWidget(f2)

        # Patrón coincidente
        f3 = QFrame()
        f3.setFrameShape(QFrame.StyledPanel)
        l3 = QVBoxLayout(f3)
        l3.addWidget(QLabel("<b>Patrón Coincidente</b>"), alignment=Qt.AlignCenter)
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

        # Tabla de resultados final
        self.table_results = QTableWidget()
        self.table_results.setColumnCount(8)
        self.table_results.setHorizontalHeaderLabels([
            "Parche", "id", "patternImage", "idLith", "idOperadora",
            "Texto OCR", "Nombre Coincidente", "Seleccionar Patrón",
        ])
        self.table_results.setAlternatingRowColors(True)
        self.table_results.verticalHeader().setDefaultSectionSize(46)

        header = self.table_results.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)

        self.table_results.itemSelectionChanged.connect(self._on_result_row_changed)
        splitter.addWidget(self.table_results)

        splitter.setSizes([180, 420])
        lay.addWidget(splitter)
        return w

    # -- Controladores y Callbacks -----------------------------------------
    def _on_image_selected(self, path):
        self.image_path = path
        self.lbl_status.setText(f"Imagen cargada: {os.path.basename(path)}")

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar CSV", "", "CSV (*.csv)")
        if path:
            self.geo_lith = load_geo_lith_csv(path)
            self.lbl_csv_status.setText(f"✅ {len(self.geo_lith)} registros cargados desde {os.path.basename(path)}")

    def _select_patterns_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de patrones")
        if d:
            self.patterns_dir = d
            n = len(list_pattern_files(d))
            self.lbl_patterns.setText(f"📁 Patrones: {d}\n({n} imágenes)")
            self._update_pattern_grid()

    def _start_processing(self):
        if not self.image_path:
            QMessageBox.warning(self, "Falta imagen", "Cargá una imagen de referencias litológicas primero.")
            return
        if not self.geo_lith:
            QMessageBox.warning(self, "Falta CSV", "No se encontró geo_lith.csv. Cargá un archivo CSV.")
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
        self.txt_log.append(f"\n{'='*50}\n  {stage}\n{'='*50}")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.lbl_stage.setText("Error")
        self.lbl_progress_detail.setText(msg)

    def _on_finished(self, results):
        self.results = results
        self.phase_btns[2].setEnabled(True)
        self.stacked.setCurrentIndex(2)

        self.lbl_result_summary.setText(f"✅ {len(results)} elementos procesados")
        self._populate_results_table()

        if results:
            self.table_results.selectRow(0)

    def _populate_results_table(self):
        results = self.results
        pattern_files = list_pattern_files(self.patterns_dir)

        self.table_results.blockSignals(True)
        self.table_results.setRowCount(len(results))

        for i, r in enumerate(results):
            # Columna 0: Parche Extraído
            patch = r.get("patch")
            if patch is not None and getattr(patch, "size", 0) > 0:
                patch_c = np.ascontiguousarray(patch)
                h, w, c = patch_c.shape
                qimg = QImage(patch_c.data, w, h, c * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(56, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl_thumb = QLabel()
                lbl_thumb.setAlignment(Qt.AlignCenter)
                lbl_thumb.setPixmap(pix)
                lbl_thumb.setStyleSheet("background: transparent; padding: 2px;")
                self.table_results.setCellWidget(i, 0, lbl_thumb)
            else:
                item_empty = QTableWidgetItem("—")
                item_empty.setTextAlignment(Qt.AlignCenter)
                self.table_results.setItem(i, 0, item_empty)

            # Columna 1: ID
            item_id = QTableWidgetItem(str(r["id"]))
            item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
            item_id.setTextAlignment(Qt.AlignCenter)
            self.table_results.setItem(i, 1, item_id)

            # Columna 2: patternImage
            item_pi = QTableWidgetItem(r["patternImage"])
            item_pi.setFlags(item_pi.flags() & ~Qt.ItemIsEditable)
            self.table_results.setItem(i, 2, item_pi)

            # Columna 3: idLith
            item_lith = QTableWidgetItem(str(r["idLith"]))
            item_lith.setFlags(item_lith.flags() & ~Qt.ItemIsEditable)
            item_lith.setTextAlignment(Qt.AlignCenter)
            self.table_results.setItem(i, 3, item_lith)

            # Columna 4: idOperadora
            item_op = QTableWidgetItem(str(r["idOperadora"]))
            item_op.setFlags(item_op.flags() & ~Qt.ItemIsEditable)
            item_op.setTextAlignment(Qt.AlignCenter)
            self.table_results.setItem(i, 4, item_op)

            # Columna 5: Texto OCR
            self.table_results.setItem(i, 5, QTableWidgetItem(r["ocr_text"]))

            # Columna 6: Nombre Coincidente
            self.table_results.setItem(i, 6, QTableWidgetItem(r["matched_nombre"]))

            # Columna 7: Selector de patrón
            combo = QComboBox()
            combo.setIconSize(QSize(56, 38))
            combo.addItem("(elegir patrón)", "")
            for pf in pattern_files:
                combo.addItem(QIcon(os.path.join(self.patterns_dir, pf)), pf, pf)
            idx = combo.findData(r["patternImage"])
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.setProperty("score_text", f"{r['pattern_score']:.1f}")
            combo.currentIndexChanged.connect(
                lambda _i, row=i, c=combo: self._on_pattern_choice(row, c)
            )
            self.table_results.setCellWidget(i, 7, combo)

        self.table_results.blockSignals(False)

    def _on_pattern_choice(self, row, combo):
        if row < 0 or row >= len(self.results):
            return
        selected_file = combo.currentData() or ""
        self.results[row]["patternImage"] = selected_file
        item = self.table_results.item(row, 2)
        if item is not None:
            item.setText(selected_file)
        if row == self.table_results.currentRow():
            self._on_result_row_changed()

    def _on_result_row_changed(self):
        row = self.table_results.currentRow()
        if row < 0 or row >= len(self.results):
            return
        r = self.results[row]

        # Inspector: Parche extraído
        patch = r.get("patch")
        if patch is not None and patch.size > 0:
            patch_c = np.ascontiguousarray(patch)
            h, w, c = patch_c.shape
            qimg = QImage(patch_c.data, w, h, c * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.lbl_insp_patch.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_insp_patch.setPixmap(pix)
        else:
            self.lbl_insp_patch.setText("—")

        # Inspector: Información
        self.lbl_insp_info.setText(
            f"idLith: {r['idLith']}\n"
            f"Nombre: {r['matched_nombre']}\n"
            f"OCR: {r['ocr_text']}"
        )

        # Inspector: Patrón coincidente
        score_text = f"{r['pattern_score']:.1f}"
        combo = self.table_results.cellWidget(row, 7)
        if isinstance(combo, QComboBox):
            score_text = combo.property("score_text") or score_text
        if r["patternImage"]:
            ppath = os.path.join(self.patterns_dir, r["patternImage"])
            if os.path.isfile(ppath):
                pix2 = QPixmap(ppath).scaled(
                    self.lbl_insp_match.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.lbl_insp_match.setPixmap(pix2)
            else:
                self.lbl_insp_match.setText("—")
            self.lbl_match_detail.setText(
                f"Archivo: {r['patternImage']}\nSimilitud: {score_text}%"
            )
        else:
            self.lbl_insp_match.setText("—")
            self.lbl_match_detail.setText("Sin coincidencia")

    def _open_export_dialog(self):
        if not self.results:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para exportar.")
            return
        dlg = ExportDialog(self.results, self)
        dlg.exec()


# ---------------------------------------------------------------------------
# Punto de Entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())