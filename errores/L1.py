"""
Lith_main.py – Lithological Pattern Analyzer
=============================================
Segments a lithological legend image, performs OCR on each text label,
matches extracted pattern patches against a library of reference patterns,
cross-references with geo_lith.csv for idLith, and exports the result
as CSV or PostgreSQL-compatible SQL.

Designed to run on CPU (no GPU required).
"""

import sys
import os
import io
import csv
import json
import base64
import hashlib
import cv2
import numpy as np
import easyocr
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
# Paths (auto-resolved relative to this script)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "geo_lith.csv")
DEFAULT_PATTERNS_DIR = os.path.join(BASE_DIR, "Patterns")
RESULTS_STORE_PATH = os.path.join(BASE_DIR, "lith_results.json")
PATTERN_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def list_pattern_files(patterns_dir):
    """Sorted list of pattern image file names inside a directory."""
    if not os.path.isdir(patterns_dir):
        return []
    return sorted(f for f in os.listdir(patterns_dir) if f.lower().endswith(PATTERN_EXTS))

# ---------------------------------------------------------------------------
# Dark palette / stylesheet
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
# Load reference CSV into list of dicts
# ---------------------------------------------------------------------------
def load_geo_lith_csv(path):
    """Returns list of {'idLith': int, 'nombre': str}."""
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
        # Detect column indices
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
# Persistent results store (JSON: rows + extracted patch images)
# ---------------------------------------------------------------------------
def _patch_to_data_url(patch_rgb):
    """Encode an RGB numpy patch as a base64 PNG data URL."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("No se pudo codificar la imagen del parche.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _data_url_to_patch(data_url):
    """Decode a base64 PNG data URL back to an RGB numpy patch."""
    b64 = data_url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen del parche.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_results_store(path):
    """Load saved results rows. Returns [] if the file does not exist or is invalid."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("rows", [])
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    except Exception:
        return []


def save_results_store(path, new_rows):
    """Append rows whose extracted patch is not already stored, then save the file.

    Rows are deduplicated by patch image content (fallback: patternImage+OCR text).
    Returns the full list of rows after merging.
    """
    rows = load_results_store(path)
    known = set()
    for r in rows:
        if r.get("extractedImage"):
            known.add("img:" + hashlib.sha1(r["extractedImage"].encode("utf-8")).hexdigest())
        else:
            known.add(f"txt:{r.get('patternImage', '')}|{r.get('ocr_text', '')}")
    for r in new_rows:
        if r.get("extractedImage"):
            key = "img:" + hashlib.sha1(r["extractedImage"].encode("utf-8")).hexdigest()
        else:
            key = f"txt:{r.get('patternImage', '')}|{r.get('ocr_text', '')}"
        if key not in known:
            known.add(key)
            rows.append(r)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return rows


def _row_from_saved(saved, row_id, id_operadora):
    """Rebuild a result dict from a saved store row."""
    patch = None
    try:
        if saved.get("extractedImage"):
            patch = _data_url_to_patch(saved["extractedImage"])
    except Exception:
        patch = None
    return {
        "id": row_id,
        "patternImage": saved.get("patternImage", ""),
        "idLith": saved.get("idLith", -1),
        "idOperadora": saved.get("idOperadora", id_operadora),
        "ocr_text": saved.get("ocr_text", ""),
        "matched_nombre": saved.get("matched_nombre", ""),
        "pattern_score": saved.get("pattern_score", 0.0),
        "patch": patch,
        "box": None,
    }


def _result_to_store_row(r):
    """Convert a live result dict to a serializable store row."""
    row = {
        "patternImage": r.get("patternImage", ""),
        "idLith": r.get("idLith", -1),
        "idOperadora": r.get("idOperadora", 0),
        "ocr_text": r.get("ocr_text", ""),
        "matched_nombre": r.get("matched_nombre", ""),
        "pattern_score": r.get("pattern_score", 0.0),
        "extractedImage": r.get("extractedImage", ""),
    }
    if not row["extractedImage"]:
        patch = r.get("patch")
        if patch is not None and getattr(patch, "size", 0) > 0:
            row["extractedImage"] = _patch_to_data_url(patch)
    return row


# ---------------------------------------------------------------------------
# Image comparison: structural + colour similarity
# ---------------------------------------------------------------------------
def compare_images(patch_rgb, target_path, target_size=None, _target_arr=None):
    """Return similarity score 0-100 between a patch (numpy RGB) and a file."""
    try:
        if _target_arr is not None:
            target = cv2.cvtColor(_target_arr, cv2.COLOR_RGB2BGR)
        else:
            target = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
            if target is None:
                return 0.0
        # Handle RGBA: set transparent pixels to white background
        if target.ndim == 3 and target.shape[2] == 4:
            alpha = target[:, :, 3]
            target_bgr = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
            mask = alpha < 128
            target_bgr[mask] = [255, 255, 255]
            target = target_bgr
        elif target.ndim == 2:
            target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

        patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

        # Resize both to common size
        sz = target_size or (64, 44)
        p = cv2.resize(patch_bgr, sz, interpolation=cv2.INTER_AREA)
        t = cv2.resize(target, sz, interpolation=cv2.INTER_AREA)

        # 1. Colour histogram comparison (HSV, finer bins)
        p_hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
        t_hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
        score_hist = 0.0
        for ch in range(3):
            h1 = cv2.calcHist([p_hsv], [ch], None, [64], [0, 256])
            h2 = cv2.calcHist([t_hsv], [ch], None, [64], [0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            score_hist += cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        score_hist = max(0.0, (score_hist / 3.0)) * 100.0

        # 2. Structural: normalized cross-correlation on grayscale
        g1 = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g1_n = (g1 - g1.mean()) / max(g1.std(), 1e-5)
        g2_n = (g2 - g2.mean()) / max(g2.std(), 1e-5)
        ncc = float(np.mean(g1_n * g2_n))
        score_struct = max(0.0, ncc) * 100.0

        # 3. Mean absolute difference in LAB colour space
        lab1 = cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(t, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta_e = np.mean(np.abs(lab1 - lab2))
        score_lab = max(0.0, 100.0 - delta_e * 1.5)

        return round(score_hist * 0.40 + score_struct * 0.35 + score_lab * 0.25, 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Fuzzy match OCR text to geo_lith nombre
# ---------------------------------------------------------------------------
def match_lith_name(ocr_text, geo_lith_records):
    """Returns (idLith, matched_nombre) or (-1, ocr_text)."""
    if not geo_lith_records or not ocr_text.strip():
        return -1, ocr_text
    upper = ocr_text.upper().strip()
    names = [r["nombre"].upper().strip() for r in geo_lith_records]

    # Exact
    if upper in names:
        idx = names.index(upper)
        return geo_lith_records[idx]["idLith"], geo_lith_records[idx]["nombre"]

    # Close match
    matches = get_close_matches(upper, names, n=1, cutoff=0.45)
    if matches:
        idx = names.index(matches[0])
        return geo_lith_records[idx]["idLith"], geo_lith_records[idx]["nombre"]

    # Substring
    for i, n in enumerate(names):
        if n and (n in upper or upper in n):
            return geo_lith_records[i]["idLith"], geo_lith_records[i]["nombre"]

    # Partial ratio
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
# Worker thread: segmentation + OCR + pattern matching
# ---------------------------------------------------------------------------
class ProcessingWorker(QThread):
    progress = Signal(int, str)       # (percentage, message)
    stage_changed = Signal(str)       # stage name
    finished = Signal(list)           # list of result dicts
    error = Signal(str)

    def __init__(self, image_path, geo_lith, patterns_dir, id_operadora, results_store=None):
        super().__init__()
        self.image_path = image_path
        self.geo_lith = geo_lith
        self.patterns_dir = patterns_dir
        self.id_operadora = id_operadora
        self.results_store = results_store or []

    def run(self):
        try:
            self._process()
        except Exception as e:
            self.error.emit(str(e))

    def _process(self):
        # --- Stage 1: Load & Segment ---
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

        # Determine column boundaries for text extraction
        xs = sorted(set(b[0] for b in boxes))
        col_starts = []
        for xv in xs:
            if not col_starts or xv - col_starts[-1] > 50:
                col_starts.append(xv)
        col_ends = col_starts[1:] + [img_w]

        # --- Stage 2: OCR ---
        self.stage_changed.emit("Etapa 2: Reconocimiento de texto (OCR)")
        self.progress.emit(25, "Inicializando EasyOCR (puede tardar)...")
        reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        self.progress.emit(35, "Motor OCR listo. Procesando textos...")

        segments = []  # list of (patch_rgb, ocr_text, box)
        total = len(boxes)
        for i, (bx, by, bw, bh) in enumerate(boxes):
            pct = 35 + int((i / total) * 25)
            self.progress.emit(pct, f"OCR segmento {i+1}/{total}...")

            # Extract pattern patch
            patch = img_np[by:by+bh, bx:bx+bw]

            # Determine text region end (use full width for last column)
            end_x = img_w
            for ce in col_ends:
                if bx + bw + 5 < ce:
                    end_x = ce
                    break
            # Start text extraction right after the box, with slight vertical padding
            text_x_start = bx + bw + 1
            text_region = img_np[max(0, by-2):min(img_h, by+bh+2), text_x_start:end_x]

            ocr_text = ""
            if text_region.size > 0:
                results = reader.readtext(text_region, detail=0)
                ocr_text = " ".join(results).strip()
                # Clean up common OCR artifacts
                for ch in ["?", "_", "|", "}", "{"]:
                    ocr_text = ocr_text.replace(ch, "")
                ocr_text = ocr_text.strip()

            segments.append({
                "patch": patch,
                "ocr_text": ocr_text,
                "box": (bx, by, bw, bh),
            })

        # --- Stage 3: Match OCR to geo_lith ---
        self.stage_changed.emit("Etapa 3: Identificación litológica (geo_lith)")
        self.progress.emit(60, "Correlacionando textos con geo_lith.csv...")

        for i, seg in enumerate(segments):
            pct = 60 + int((i / total) * 10)
            self.progress.emit(pct, f"Identificando {i+1}/{total}: '{seg['ocr_text']}'")
            id_lith, matched_name = match_lith_name(seg["ocr_text"], self.geo_lith)
            seg["idLith"] = id_lith
            seg["matched_nombre"] = matched_name

        # --- Stage 4: Pattern image matching ---
        self.stage_changed.emit("Etapa 4: Comparación con patrones de referencia")
        self.progress.emit(70, "Cargando patrones de referencia...")

        pattern_files = list_pattern_files(self.patterns_dir)

        # Known matches from previous runs (saved results store)
        known = []
        for row in self.results_store:
            if not row.get("patternImage") or not row.get("extractedImage"):
                continue
            try:
                known.append({
                    "patch": _data_url_to_patch(row["extractedImage"]),
                    "patternImage": row["patternImage"],
                    "idLith": row.get("idLith", -1),
                    "matched_nombre": row.get("matched_nombre", ""),
                })
            except Exception:
                continue

        for i, seg in enumerate(segments):
            pct = 70 + int((i / total) * 25)
            self.progress.emit(pct, f"Comparando patrón {i+1}/{total}...")

            patch = seg["patch"]
            best_file = ""
            best_score = 0.0
            known_hit = None
            if patch.size > 0:
                # 1. Consult the saved results store first (pre-resolved match)
                for k in known:
                    sc = compare_images(patch, None, _target_arr=k["patch"])
                    if sc > best_score:
                        best_score = sc
                        known_hit = k
                # 2. Fall back to the pattern library only when no known match
                if best_score < 95.0:
                    best_file = ""
                    best_score = 0.0
                    known_hit = None
                    for pf in pattern_files:
                        fp = os.path.join(self.patterns_dir, pf)
                        sc = compare_images(patch, fp)
                        if sc > best_score:
                            best_score = sc
                            best_file = pf
            if known_hit is not None:
                seg["patternImage"] = known_hit["patternImage"]
                seg["idLith"] = known_hit["idLith"]
                seg["matched_nombre"] = known_hit["matched_nombre"]
            else:
                seg["patternImage"] = best_file
            seg["pattern_score"] = best_score

        # --- Build final results ---
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
# Export dialog (CSV or SQL)
# ---------------------------------------------------------------------------
class ExportDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("Exportar Resultados")
        self.resize(460, 200)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["CSV (.csv)", "SQL MySQL (.sql)"])
        form.addRow("Formato:", self.cmb_format)

        self.txt_table = QLineEdit("correlacion_litologica")
        form.addRow("Nombre tabla (SQL):", self.txt_table)
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
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Guardar SQL", "resultado.sql", "SQL (*.sql)"
            )
            if path:
                self._export_sql(path)

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

    def _export_sql(self, path):
        try:
            table = self.txt_table.text().strip() or "correlacion_litologica"
            lines = []
            lines.append(f"CREATE TABLE IF NOT EXISTS `{table}` (")
            lines.append("    `id` INT AUTO_INCREMENT PRIMARY KEY,")
            lines.append("    `patternImage` VARCHAR(255) NOT NULL,")
            lines.append("    `idLith` INT NOT NULL,")
            lines.append("    `idOperadora` INT NOT NULL")
            lines.append(") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;")
            lines.append("")
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


# ---------------------------------------------------------------------------
# Drop zone widget
# ---------------------------------------------------------------------------
class DropImageZone(QLabel):
    imageSelected = Signal(str)  # emits file path

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
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lith → MySQL  ·  Analizador Litológico")
        self.resize(1280, 780)

        self.image_path = None
        self.geo_lith = load_geo_lith_csv(DEFAULT_CSV_PATH)
        self.patterns_dir = DEFAULT_PATTERNS_DIR
        self.results_store_path = RESULTS_STORE_PATH
        self.results_store = load_results_store(self.results_store_path)
        self.results = []
        self.worker = None

        self._build_ui()

    # -- UI setup ----------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)

        # Phase indicator bar
        phase_bar = QHBoxLayout()
        self.phase_btns = []
        labels = [
            "① Configuración",
            "② Procesamiento",
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

        # Status
        self.lbl_status = QLabel("Configurá los parámetros y presioná Procesar.")
        self.lbl_status.setObjectName("statusLabel")
        root.addWidget(self.lbl_status)

        # Stacked phases
        self.stacked = QStackedWidget()
        root.addWidget(self.stacked)

        self.stacked.addWidget(self._build_phase1())
        self.stacked.addWidget(self._build_phase2())
        self.stacked.addWidget(self._build_phase3())

    # -- Phase 1: Config ---------------------------------------------------
    def _build_phase1(self):
        w = QWidget()
        lay = QHBoxLayout(w)

        # Left: image
        left = QVBoxLayout()
        grp_img = QGroupBox("Imagen de Referencias Litológicas")
        gl = QVBoxLayout(grp_img)
        self.drop_zone = DropImageZone()
        self.drop_zone.imageSelected.connect(self._on_image_selected)
        gl.addWidget(self.drop_zone)
        left.addWidget(grp_img)
        lay.addLayout(left, stretch=2)

        # Right: settings
        right = QVBoxLayout()

        # idOperadora
        grp_op = QGroupBox("Datos de Operadora")
        ol = QFormLayout(grp_op)
        self.spn_operadora = QSpinBox()
        self.spn_operadora.setRange(1, 999999)
        self.spn_operadora.setValue(1)
        ol.addRow("idOperadora:", self.spn_operadora)
        right.addWidget(grp_op)

        # CSV info
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

        # Patterns folder
        grp_pat = QGroupBox("Carpeta de Patrones")
        pl = QVBoxLayout(grp_pat)
        n_patterns = len(list_pattern_files(self.patterns_dir))
        self.lbl_patterns = QLabel(f"📁 {self.patterns_dir}\n({n_patterns} imágenes)")
        pl.addWidget(self.lbl_patterns)

        # Persistent results store status
        self.lbl_known = QLabel(f"💾 Registros guardados previamente: {len(self.results_store)}")
        pl.addWidget(self.lbl_known)

        # Grid preview
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(3)
        self._update_pattern_grid()
        pl.addWidget(self.grid_widget)

        btn_pat = QPushButton("Seleccionar otra carpeta")
        btn_pat.clicked.connect(self._select_patterns_dir)
        pl.addWidget(btn_pat)
        right.addWidget(grp_pat)

        # Process button
        self.btn_process = QPushButton("▶  PROCESAR")
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
        # Clear
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

    # -- Phase 2: Processing -----------------------------------------------
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

        # Live log
        grp_log = QGroupBox("Log de Procesamiento")
        ll = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        ll.addWidget(self.txt_log)
        lay.addWidget(grp_log)

        return w

    # -- Phase 3: Results --------------------------------------------------
    def _build_phase3(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # Top bar
        top = QHBoxLayout()
        self.lbl_result_summary = QLabel("")
        self.lbl_result_summary.setObjectName("phaseLabel")
        top.addWidget(self.lbl_result_summary)
        top.addStretch()

        btn_csv_exp = QPushButton("📄 Exportar CSV / SQL (MySQL)")
        btn_csv_exp.setObjectName("accentBtn")
        btn_csv_exp.setFixedHeight(34)
        btn_csv_exp.clicked.connect(self._open_export_dialog)
        top.addWidget(btn_csv_exp)

        self.btn_end_tasks = QPushButton("✔ End tasks")
        self.btn_end_tasks.setObjectName("dangerBtn")
        self.btn_end_tasks.setFixedHeight(34)
        self.btn_end_tasks.setEnabled(False)
        self.btn_end_tasks.setToolTip(
            "Guarda la tabla de resultados junto con las imágenes extraídas\n"
            "y el patrón coincidente elegido, y finaliza la tarea."
        )
        self.btn_end_tasks.clicked.connect(self._end_tasks)
        top.addWidget(self.btn_end_tasks)
        lay.addLayout(top)

        splitter = QSplitter(Qt.Vertical)

        # Inspector
        grp_insp = QGroupBox("Inspector Visual")
        il = QHBoxLayout(grp_insp)

        # Extracted patch
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

        # Info
        f2 = QFrame()
        f2.setFrameShape(QFrame.StyledPanel)
        l2 = QVBoxLayout(f2)
        l2.addWidget(QLabel("<b>Identificación</b>"), alignment=Qt.AlignCenter)
        self.lbl_insp_info = QLabel("idLith: —\nNombre: —\nOCR: —")
        self.lbl_insp_info.setAlignment(Qt.AlignCenter)
        self.lbl_insp_info.setStyleSheet("font-size:12px; color:#a6e3a1;")
        l2.addWidget(self.lbl_insp_info)
        il.addWidget(f2)

        # Matched pattern
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

        # Table
        self.table_results = QTableWidget()
        self.table_results.setColumnCount(7)
        self.table_results.setHorizontalHeaderLabels([
            "id", "patternImage", "idLith", "idOperadora",
            "Texto OCR", "Nombre Coincidente", "Similitud %",
        ])
        self.table_results.setAlternatingRowColors(True)
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_results.itemSelectionChanged.connect(self._on_result_row_changed)
        splitter.addWidget(self.table_results)

        splitter.setSizes([200, 400])
        lay.addWidget(splitter)
        return w

    # -- Callbacks ---------------------------------------------------------
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
            self.lbl_patterns.setText(f"📁 {d}\n({n} imágenes)")
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
            self.results_store,
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
        self.btn_end_tasks.setEnabled(True)

        if results:
            self.table_results.selectRow(0)
        else:
            self.btn_end_tasks.setEnabled(False)

    def _populate_results_table(self):
        results = self.results
        pattern_files = list_pattern_files(self.patterns_dir)

        self.table_results.blockSignals(True)
        self.table_results.setRowCount(len(results))
        for i, r in enumerate(results):
            item_id = QTableWidgetItem(str(r["id"]))
            item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
            self.table_results.setItem(i, 0, item_id)
            item_pi = QTableWidgetItem(r["patternImage"])
            item_pi.setFlags(item_pi.flags() & ~Qt.ItemIsEditable)
            self.table_results.setItem(i, 1, item_pi)
            item_lith = QTableWidgetItem(str(r["idLith"]))
            item_lith.setFlags(item_lith.flags() & ~Qt.ItemIsEditable)
            self.table_results.setItem(i, 2, item_lith)
            item_op = QTableWidgetItem(str(r["idOperadora"]))
            item_op.setFlags(item_op.flags() & ~Qt.ItemIsEditable)
            self.table_results.setItem(i, 3, item_op)
            self.table_results.setItem(i, 4, QTableWidgetItem(r["ocr_text"]))
            self.table_results.setItem(i, 5, QTableWidgetItem(r["matched_nombre"]))

            # Free selection of the matching pattern from the full Patterns folder
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
            self.table_results.setCellWidget(i, 6, combo)
        self.table_results.blockSignals(False)

    def _on_pattern_choice(self, row, combo):
        if row < 0 or row >= len(self.results):
            return
        self.results[row]["patternImage"] = combo.currentData() or ""
        item = self.table_results.item(row, 1)
        if item is not None:
            item.setText(self.results[row]["patternImage"])
        if row == self.table_results.currentRow():
            self._on_result_row_changed()

    def _end_tasks(self):
        if not self.results:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para guardar.")
            return
        try:
            new_rows = [_result_to_store_row(r) for r in self.results]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudieron preparar los datos:\n{e}")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar archivo de resultados",
            self.results_store_path,
            "JSON (*.json)",
        )
        if not path:
            return
        prev_count = len(load_results_store(path))
        try:
            all_rows = save_results_store(path, new_rows)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo:\n{e}")
            return
        self.results_store_path = path
        self.results_store = all_rows
        self.lbl_known.setText(f"💾 Registros guardados previamente: {len(all_rows)}")
        added = len(all_rows) - prev_count
        QMessageBox.information(
            self,
            "Tarea finalizada",
            f"Archivo de resultados guardado en:\n{path}\n\n"
            f"Filas nuevas agregadas: {added}\n"
            f"Total de filas acumuladas: {len(all_rows)}",
        )
        self.btn_end_tasks.setEnabled(False)

    def _on_result_row_changed(self):
        row = self.table_results.currentRow()
        if row < 0 or row >= len(self.results):
            return
        r = self.results[row]

        # Show extracted patch
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

        self.lbl_insp_info.setText(
            f"idLith: {r['idLith']}\n"
            f"Nombre: {r['matched_nombre']}\n"
            f"OCR: {r['ocr_text']}"
        )

        # Show matched pattern
        score_text = f"{r['pattern_score']:.1f}"
        combo = self.table_results.cellWidget(row, 6)
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
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
