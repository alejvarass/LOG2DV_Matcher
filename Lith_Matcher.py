"""
Lith_main.py – Analizador Litológico de Patrones
================================================
Pipeline híbrido de análisis litológico:

  1. Segmentación de casillas (contornos + filtros geométricos).
  2. OCR multi-estrategia (EasyOCR) con fuzzy matching geológico contra geo_lith.csv.
  3. Clasificación visual multi-métrica:
       - Color LAB continuo
       - Histogramas BGR + HSV (con penalización de anti-correlación)
       - NCC estructural + gradientes Sobel
       - Bordes Laplaciano
       - Textura ILBP (LBP invariante a rotación) + estadísticas de Haralick (GLCM)
       - Keypoints ORB con ratio test de Lowe
       - Embeddings DINOv2 (IA) multi-crop con caché en disco
  4. Desempate litológico ponderado por la confianza del OCR.
  5. Exportación a CSV, MySQL y PostgreSQL.

El detalle completo del método está documentado en ANALISIS.md.
"""

import sys
import os
import csv
import hashlib
import unicodedata
import cv2
import numpy as np
import easyocr
from PIL import Image
from difflib import get_close_matches, SequenceMatcher

# --- Dependencias de IA (opcionales): si torch/DINOv2 no están disponibles, ---
# --- el sistema sigue funcionando solo con las métricas clásicas de OpenCV.  ---
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except Exception:  # torch no instalado o incompatible
    torch = None
    transforms = None
    TORCH_AVAILABLE = False

from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QTextEdit, QSplitter, QTableWidget, QTableWidgetItem,
    QPushButton, QGroupBox, QStackedWidget, QProgressBar,
    QDialog, QFormLayout, QLineEdit, QMessageBox, QFrame, QGridLayout,
    QSpinBox, QHeaderView, QComboBox, QSizePolicy,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "geo_lith.csv")
DEFAULT_PATTERNS_DIR = os.path.join(BASE_DIR, "Patterns")
PATTERN_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# ============================================================================
# CONFIGURACIÓN DEL MOTOR DE COMPARACIÓN
# ============================================================================

# Pesos de cada métrica clásica dentro del score visual (suman 1.0).
# La textura (LBP-ri + distancia + grosor + orientación + GLCM) es la más
# fiable para tramas geológicas, por eso tiene el mayor peso. La estructura
# NCC queda limitada porque es sensible a la correlación espuria entre
# tramas periódicas (una retícula correlaciona alta con líneas). Los pesos
# se renormalizan según las métricas disponibles en cada comparación.
W_COLOR, W_HIST, W_STRUCT, W_EDGES, W_TEXTURE, W_ORB = 0.10, 0.07, 0.06, 0.07, 0.48, 0.22

# Peso de la IA (DINOv2) frente a las métricas clásicas cuando está disponible.
DINO_WEIGHT = 0.30  # 30% IA + 70% métricas clásicas

# Intensidad máxima del ajuste litológico (desempate) en puntos de score.
# El ajuste real se escala por la confianza del OCR (0..1), así un OCR
# dudoso no fuerza desempates equivocados (esto causaba fallos en patrones básicos).
TIE_BREAK_STRENGTH = 3.5

# Tabla de afinidades litología -> patrón para el desempate fino.
# Claves: idLith del catálogo geo_lith. Valores: {basename_patron: ajuste}.
LITHOLOGY_AFFINITIES = {
    8:  {"12": +1.0, "4-8": -1.0},    # ARCILITA TOBACEA
    21: {"18": +1.0, "7-3": +1.0, "5-2": -1.0},  # DOLOMITA
    5:  {"5": +1.0, "5-4": -1.0},     # ARENISCA CALCAREA
    38: {"17": +1.0, "15": -1.0, "5-2": -1.0},   # CALIZA ARCILLOSA
}

# Stopwords geológicas: no aportan discriminación al fuzzy matching de nombres.
_GEO_STOPWORDS = {"DE", "LA", "EL", "Y", "CON", "DEL", "EN"}

# Correcciones fonéticas/ortográficas frecuentes del OCR en español geológico.
# Forma: (fragmento_erróneo, fragmento_correcto). Se aplican sobre tokens.
_TOKEN_FIXES = (
    ("LIM0", "LIMO"), ("ARC1", "ARCI"), ("CALC", "CALC"),
    ("4RE", "ARE"), ("AR3N", "AREN"), ("T0B", "TOB"),
    ("D0L", "DOL"), ("C0NGL", "CONGL"), ("GRAU", "GRAV"),
)

def _normalize_geo_text(text):
    """
    Normaliza texto geológico para comparación robusta:
      - Mayúsculas, sin acentos (NFKD), solo caracteres alfanuméricos.
      - Separa en tokens y elimina stopwords.
    """
    if not text:
        return []
    txt = unicodedata.normalize("NFKD", text.upper())
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = "".join(ch if (ch.isalnum() or ch == " ") else " " for ch in txt)
    return [t for t in txt.split() if t and t not in _GEO_STOPWORDS]

def _fix_token(tok):
    """Aplica correcciones fonéticas/números->letras típicas del OCR."""
    for wrong, right in _TOKEN_FIXES:
        if wrong in tok:
            tok = tok.replace(wrong, right)
    return tok

DARK_STYLE = """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; padding-top: 14px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 5px; padding: 6px 14px; font-weight: bold; }
QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
QPushButton#accentBtn { background-color: #89b4fa; color: #1e1e2e; border: none; }
QPushButton#accentBtn:hover { background-color: #74c7ec; }
QProgressBar { border: 1px solid #45475a; border-radius: 4px; text-align: center; background-color: #313244; color: #cdd6f4; height: 22px; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
QTableWidget { background-color: #181825; alternate-background-color: #1e1e2e; color: #cdd6f4; gridline-color: #45475a; border: 1px solid #45475a; border-radius: 4px; selection-background-color: #45475a; }
QHeaderView::section { background-color: #313244; color: #89b4fa; border: 1px solid #45475a; padding: 4px; font-weight: bold; }
QTextEdit, QLineEdit, QSpinBox, QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }
QLabel#phaseLabel { font-size: 13px; font-weight: bold; color: #a6e3a1; padding: 4px; }
QLabel#statusLabel { font-size: 12px; color: #f9e2af; padding: 2px; }
QSplitter::handle { background-color: #45475a; }
"""

def list_pattern_files(patterns_dir):
    if not os.path.isdir(patterns_dir):
        return []
    return sorted(f for f in os.listdir(patterns_dir) if f.lower().endswith(PATTERN_EXTS))

def load_geo_lith_csv(path):
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
        col_id, col_name = 0, 1
        for idx, h in enumerate(header):
            hl = h.strip().strip('"').lower()
            if hl in ("idlith", "id", "codigo"): col_id = idx
            elif hl in ("nombre", "name", "litologia"): col_name = idx
        for row in reader:
            if len(row) <= max(col_id, col_name): continue
            id_str = row[col_id].strip().strip('"')
            name_str = row[col_name].strip().strip('"')
            if not id_str: continue
            try:
                records.append({"idLith": int(id_str), "nombre": name_str})
            except ValueError:
                continue
    return records

def match_lith_name(ocr_text, geo_lith_records):
    """
    Hace coincidir el texto OCR con el catálogo geo_lith.

    Devuelve (idLith, nombre_catalogo, confianza) donde confianza ∈ [0, 1]:
      1.00 = coincidencia exacta (tras normalización)
      0.90 = todos los tokens del catálogo presentes en el OCR
      0.70+ = fuzzy matching por tokens (con correcciones fonéticas)
      0.40  = fuzzy matching de cadena completa
      0.00  = sin coincidencia (idLith = -1)

    La confianza se usa después para escalar el desempate litológico:
    un OCR poco fiable apenas influye en la decisión visual.
    """
    if not geo_lith_records or not ocr_text.strip():
        return -1, ocr_text, 0.0

    upper = ocr_text.upper().strip()
    names = [r["nombre"].upper().strip() for r in geo_lith_records]

    # --- 1) Coincidencia exacta de cadena completa ---
    if upper in names:
        idx = names.index(upper)
        return geo_lith_records[idx]["idLith"], geo_lith_records[idx]["nombre"], 1.0

    ocr_tokens = [_fix_token(t) for t in _normalize_geo_text(ocr_text)]
    cat_tokens_list = [[_fix_token(t) for t in _normalize_geo_text(n)] for n in names]

    # --- 2) Coincidencia exacta por conjunto de tokens.                  ---
    # Solo cuando el OCR tiene >=2 tokens: con un solo token, nombres del
    # catálogo más largos pero distintos ganarían por subconjunto
    # ("ARCILITA TOB." -> ARCILITA); esos casos los resuelve el paso 3.
    if ocr_tokens and len(ocr_tokens) >= 2:
        ocr_set = set(ocr_tokens)
        best2, best2_idx = -1, -1
        for i, toks in enumerate(cat_tokens_list):
            # Todos los tokens del nombre deben estar en el OCR y el nombre
            # debe ser AL MENOS tan específico como el OCR (si el OCR tiene
            # más tokens que el nombre, puede ser una abreviatura, p.ej.
            # "ARCILITA TOB." -> ARCILITA TOBACEA: lo resuelve el paso 3).
            if toks and set(toks).issubset(ocr_set) and len(toks) >= len(ocr_tokens) and len(toks) > best2:
                best2, best2_idx = len(toks), i   # gana el más específico
        if best2_idx >= 0:
            return geo_lith_records[best2_idx]["idLith"], geo_lith_records[best2_idx]["nombre"], 0.90

    # --- 3) Matching por tokens unificado ---------------------------------
    # Para cada nombre del catálogo se calcula:
    #   hits  = tokens del OCR explicados (exacto, fuzzy >=0.75 o
    #           abreviatura-prefijo como "TOB" -> "TOBACEA")
    #   cov_ocr = hits / tokens del OCR
    #   cov_cat = tokens distintos del nombre alcanzados / total del nombre
    # Solo se acepta un nombre si TODO el texto OCR queda explicado
    # (cov_ocr = 1) y además el nombre queda totalmente cubierto o hay >=2
    # coincidencias fuertes. Esto evita falsos positivos clásicos:
    #   "ARENA"        -> ARENI / TOBA ARENOSA   (cov insuficiente: -1)
    #   "ARCILITA TOB."-> ARCILITA TOBACEA       (cubre los 2 tokens)
    # El ranking prefiere: más hits, luego mayor cobertura del nombre,
    # luego mayor calidad media de coincidencia, luego nombre más largo.
    def _token_score(ot, t):
        r = SequenceMatcher(None, t, ot).ratio()
        if ot == t:
            return 1.0
        # Abreviatura: SOLO si el token OCR es el prefijo Y es claramente
        # más corto (>=3 letras) que el del catálogo ("TOB"->TOBACEA).
        # Si las longitudes son casi iguales ("ARENA" vs "AREN") no es una
        # abreviatura: son palabras distintas y caerá al fuzzy con guardia.
        if len(ot) >= 3 and len(t) >= len(ot) + 3 and t.startswith(ot):
            return max(r, 0.80)
        if len(ot) >= 3 and len(t) >= 3 and r >= 0.75:
            # Guardia de especificidad: un token corto (<=5 letras) es
            # peligroso, porque el fuzzy lo confunde con cualquier palabra
            # que lo contenga ("ARENA" vs "AREN"/"ARENI"). Solo pasa si el
            # ratio es casi exacto. Si difieren mucho en longitud, igual.
            if (min(len(ot), len(t)) <= 5 or abs(len(t) - len(ot)) >= 2) and r < 0.92:
                return r * 0.4           # queda < 0.75, no cuenta como hit
            return r                     # fuzzy fuerte
        return r

    if ocr_tokens:
        best_key, best_tok_idx = None, -1
        for i, toks in enumerate(cat_tokens_list):
            if not toks:
                continue
            matched_cat = set()
            per_ocr = []
            for ot in ocr_tokens:
                best_j, best_r = -1, 0.0
                for j, t in enumerate(toks):
                    r = _token_score(ot, t)
                    # La guardia de especificidad marca palabras distintas
                    # ("ARENA" vs "AREN") devolviendo < 0.75: nunca ganan.
                    if r > best_r:
                        best_r, best_j = r, j
                per_ocr.append(best_r)
                if best_j >= 0 and best_r >= 0.75:
                    matched_cat.add(best_j)
            hits = sum(1 for r in per_ocr if r >= 0.75)
            cov_ocr = hits / len(ocr_tokens)
            cov_cat = len(matched_cat) / len(toks)
            if cov_ocr >= 0.99 and hits >= 1 and (cov_cat >= 0.99 or hits >= 2):
                quality = sum(per_ocr) / len(per_ocr)
                key = (hits, cov_cat, quality, len(toks))
                if best_key is None or key > best_key:
                    best_key, best_tok_idx = key, i
        if best_tok_idx >= 0:
            quality = best_key[2]
            return (geo_lith_records[best_tok_idx]["idLith"],
                    geo_lith_records[best_tok_idx]["nombre"],
                    float(min(0.88, max(0.60, quality))))

    # --- 5) Subcadena: solo si el texto OCR es largo (>=6), la cadena ----
    # --- corta cubre casi toda la larga Y el match cae en límites de ----
    # --- palabra (evita que "ARENA" active "AREN" o "ARENOSA").       ---
    if len(upper) >= 6:
        for i, n in enumerate(names):
            if not n:
                continue
            if n in upper or upper in n:
                short, long_ = (upper, n) if len(upper) <= len(n) else (n, upper)
                if len(short) / len(long_) < 0.80:
                    continue
                # límites de palabra: la subcadena no puede cortar tokens
                pos = long_.find(short)
                end = pos + len(short)
                ok_start = pos == 0 or not long_[pos - 1].isalnum()
                ok_end = end >= len(long_) or not long_[end].isalnum()
                if ok_start and ok_end:
                    return geo_lith_records[i]["idLith"], geo_lith_records[i]["nombre"], 0.50

    # --- 6) Fuzzy clásico de cadena completa (difflib), umbral alto para ---
    # --- no asignar litologías a texto que claramente no es geológico.   ---
    # Se exige además que la longitud no difiera demasiado (evita que
    # "ARENA" (5) case con "AREN" (4) por ratio 0.889).
    matches = get_close_matches(upper, names, n=1, cutoff=0.65)
    if matches:
        idx = names.index(matches[0])
        n = names[idx]
        # Guardia: para nombres cortos (<6 letras) exige igualdad de
        # longitud, porque el fuzzy los confunde con facilidad ("ARENA"
        # casa con "AREN" por ratio 0.889 si solo se exige ±1).
        if min(len(upper), len(n)) >= 6 or len(upper) == len(n):
            return geo_lith_records[idx]["idLith"], geo_lith_records[idx]["nombre"], 0.45

    # --- 7) Último recurso: ratio de SequenceMatcher con umbral exigente ---
    best_score, best_idx = 0.0, -1
    for i, n in enumerate(names):
        if not n:
            continue
        s = SequenceMatcher(None, upper, n).ratio()
        if s > best_score:
            best_score, best_idx = s, i
    if best_score >= 0.55 and best_idx >= 0:
        n = names[best_idx]
        if min(len(upper), len(n)) >= 6 or len(upper) == len(n):
            return geo_lith_records[best_idx]["idLith"], geo_lith_records[best_idx]["nombre"], best_score * 0.6

    return -1, ocr_text, 0.0

def _geology_from_name(name_clean, id_val, geo_lith_records):
    """
    Devuelve una clave geológica normalizada para el desempate litológico,
    resolviendo cualquier combinación id / nombre / sinónimo en UN solo lugar:

      - Primero por idLith directo (si está en la tabla de afinidades).
      - Luego por tokens del nombre: busca el término geológico principal
        (DOLOMITA, TOBA, ARENISCA, CALIZA, ARCILITA...) con fuzzy matching,
        de modo que "ARCILITA TOBACEA" y "ARCILITA TOB." llegan al mismo
        grupo aunque el OCR haya elegido nombres distintos.
    """
    if id_val in LITHOLOGY_AFFINITIES:
        return id_val
    toks = _normalize_geo_text(name_clean)
    toks = [_fix_token(t) for t in toks]
    def _has(word):
        return any(t == word or (len(t) >= 3 and word.startswith(t)) or
                   (len(t) >= 4 and SequenceMatcher(None, t, word).ratio() >= 0.80)
                   for t in toks)
    if _has("DOLOMITA"):
        return 21
    if _has("ARCILITA") and _has("TOBACEA"):
        return 8
    if _has("ARENISCA") and _has("CALCAREA"):
        return 5
    if _has("CALIZA") and (_has("ARCILLOSA") or _has("ARCILLITA")):
        return 38
    return None

# ---------------------------------------------------------------------------
# Métricas individuales de comparación de imágenes
# ---------------------------------------------------------------------------

def _texture_features(gray_f32):
    """
    Extrae un vector de textura discriminativo:
      - LBP-ri: Local Binary Pattern crudo invariante a rotación
        (36 bins) — separa puntos, líneas, cruces y bordes sin depender
        de la orientación de la trama. NO se usa la variante "uniforme"
        porque colapsa tramas distintas al mismo perfil de densidad.
      - Estadísticas de la GLCM (contraste, energía, homogeneidad y
        correlación) calculadas sobre la imagen cuantizada a 8 niveles
        en 2 direcciones (0° y 90°). Son mucho más discriminativas que
        la co-ocurrencia del LBP.
    Todos los sub-vectores se normalizan a norma 1 antes de concatenar,
    para que ninguno domine la similitud de coseno final.
    """
    g = gray_f32
    if g.shape[0] < 8 or g.shape[1] < 8:
        return None
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    # Suavizado leve: el LBP usa diferencias de signo, muy sensibles al
    # ruido de escaneo en zonas planas y bordes anti-aliased.
    g8 = cv2.GaussianBlur(g8, (3, 3), 0)
    c = g8[1:-1, 1:-1].astype(np.int16)
    # 8 vecinos del LBP en sentido horario con umbral de 2 niveles de gris:
    # ignora micro-variaciones de contraste pero conserva los bordes reales.
    codes = np.zeros_like(c, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        nb = g8[1 + dy:g8.shape[0] - 1 + dy, 1 + dx:g8.shape[1] - 1 + dx].astype(np.int16)
        codes |= ((nb >= c + 2).astype(np.uint8) << bit)

    lut = np.zeros(256, dtype=np.uint8)
    uniq = {}
    for v in range(256):
        b = format(v, "08b")
        rot_min = min(int(b[i:] + b[:i], 2) for i in range(8))
        if rot_min not in uniq:
            uniq[rot_min] = len(uniq)
        lut[v] = uniq[rot_min]
    n_bins = len(uniq)  # 36
    lbp = lut[codes]
    hist = cv2.calcHist([lbp], [0], None, [n_bins], [0, n_bins]).flatten()
    hist = hist / max(hist.sum(), 1e-6)
    # Raíz cuadrada SUAVE: atenúa el bin dominante (fondo) sin destruir la
    # señal de los bins de borde (que separan líneas de puntos).
    hist = hist ** 0.75
    hist /= max(np.linalg.norm(hist), 1e-6)

    # --- Histograma de la transformada de distancia (grosor/espaciado) ---
    # Mide a qué distancia está cada píxel del fondo respecto a la tinta:
    # separa tramas densas de tramas con líneas aisladas (retícula vs
    # líneas). Bins de rango FIJO (0..12 px) para que el histograma sea
    # comparable entre parches de distinta densidad.
    _, ink = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(255 - ink, cv2.DIST_L2, 3)
    dhist = np.histogram(dist, bins=8, range=(0.0, 12.0))[0].astype(np.float64)
    dhist = dhist / max(dhist.sum(), 1e-6)
    dhist /= max(np.linalg.norm(dhist), 1e-6)

    # --- Perfil de grosor de tinta multi-escala ---------------------------
    # Ratios de supervivencia de la tinta tras erosiones progresivas.
    # Se usan RATIOS (no fracciones absolutas) porque las tramas geológicas
    # suelen saturar la erosión (casi todo sobrevive): los ratios conservan
    # la diferencia entre retículas (las intersecciones aguantan) y líneas
    # finas (desaparecen). Es lo que mejor separa "cross" de "hlines".
    ink_frac = max(float((ink > 0).mean()), 1e-6)
    eroded = ink
    thick = []
    for _ in range(3):
        eroded = cv2.erode(eroded, np.ones((3, 3), np.uint8))
        thick.append(float((eroded > 0).mean()) / ink_frac)
    thick = np.asarray(thick, dtype=np.float64)
    thick /= max(np.linalg.norm(thick), 1e-6)

    # --- Orientación local dominante (histograma de 4 bins, SIN sqrt) ----
    # Gradientes fuertes votan por su orientación (0°, 45°, 90°, 135°):
    # la retícula tiene energía repartida entre 2 orientaciones, las
    # líneas simples concentran casi todo en una. NO se aplica raíz:
    # atenuaría precisamente la señal que distingue 1 de 2 orientaciones.
    gx = cv2.Sobel(g8, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g8, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    thr = np.percentile(mag, 60) if mag.size else 0.0
    mask = mag >= thr
    oh = np.zeros(4, dtype=np.float64)
    if mask.any():
        bins = np.clip((ang[mask] / 45.0).astype(np.int32), 0, 3)
        np.add.at(oh, bins, mag[mask])
    oh = oh / max(oh.sum(), 1e-6)
    oh /= max(np.linalg.norm(oh), 1e-6)

    # --- Perfil de tinta por filas y columnas -----------------------------
    # Proyección de la tinta sobre el eje X e Y (8 bins cada uno). Es la
    # característica más directa para distinguir tramas periódicas:
    #   - líneas horizontales: perfil de filas con picos, columnas plano
    #   - líneas verticales:   al revés
    #   - retícula:            picos en AMBAS proyecciones
    ink_bin = (ink > 0).astype(np.float64)
    hh, ww = ink_bin.shape
    def _proj(v, n=8):
        parts = np.array_split(v, n)
        p = np.asarray([x.mean() for x in parts])
        return p / max(p.sum(), 1e-6)
    proj_rows = _proj(ink_bin.mean(axis=1))   # proyección vertical (filas)
    proj_cols = _proj(ink_bin.mean(axis=0))   # proyección horizontal (cols)
    proj = np.concatenate([proj_rows, proj_cols])
    proj /= max(np.linalg.norm(proj), 1e-6)

    # --- Simetría horizontal/vertical de la tinta -------------------------
    # Fracción de tinta en cada mitad: una retícula es simétrica en ambas
    # direcciones; líneas puras solo en una.
    sym_h = abs(ink_bin[: hh // 2].mean() - ink_bin[hh - hh // 2:].mean())
    sym_v = abs(ink_bin[:, : ww // 2].mean() - ink_bin[:, ww - ww // 2:].mean())
    sym = np.asarray([1.0 - min(sym_h * 4, 1.0), 1.0 - min(sym_v * 4, 1.0)])
    sym /= max(np.linalg.norm(sym), 1e-6)

    # --- GLCM sobre imagen cuantizada (8 niveles), distancia 1, 0° y 90° ---
    n_g = 8
    q = (g8.astype(np.int32) * n_g // 256).astype(np.int32)
    stats = []
    for dy, dx in ((0, 1), (1, 0)):
        if dy == 0:
            a, b = q[:, :-1].ravel(), q[:, 1:].ravel()
        else:
            a, b = q[:-1, :].ravel(), q[1:, :].ravel()
        gl = np.zeros((n_g, n_g), dtype=np.float64)
        np.add.at(gl, (a, b), 1)
        gl += gl.T                      # simétrica
        if gl.sum() > 0:
            gl /= gl.sum()
        i_idx = np.arange(n_g).reshape(n_g, 1)
        j_idx = np.arange(n_g).reshape(1, n_g)
        contrast = float((gl * (i_idx - j_idx) ** 2).sum())
        energy = float(np.sqrt((gl ** 2).sum()))       # raíz de la energía (ASM)
        homogeneity = float((gl / (1.0 + np.abs(i_idx - j_idx))).sum())
        mu_i = float((gl * i_idx).sum()); mu_j = float((gl * j_idx).sum())
        sd_i = np.sqrt(float((gl * (i_idx - mu_i) ** 2).sum()))
        sd_j = np.sqrt(float((gl * (j_idx - mu_j) ** 2).sum()))
        corr = float((gl * (i_idx - mu_i) * (j_idx - mu_j)).sum() / (sd_i * sd_j + 1e-9))
        stats += [contrast / n_g, energy, homogeneity, (corr + 1.0) / 2.0]
    stats = np.asarray(stats, dtype=np.float64)
    stats /= max(np.linalg.norm(stats), 1e-6)
    return np.concatenate([hist, dhist, thick, oh, proj, sym, stats]).astype(np.float32)

def _orb_match_score(patch_bgr, target_bgr, orb_cache):
    """
    Compara keypoints ORB con ratio test de Lowe (0.75).
    Las tramas litológicas tienen poco gradiente, así que se aplica CLAHE
    (ecualización adaptativa) y un FAST threshold bajo para que ORB
    encuentre puntos incluso en tramas suaves.
    Devuelve 0..100 según la fracción de descriptores del parche con
    correspondencia fiable en el patrón.
    """
    key = id(target_bgr)
    cached = orb_cache.get(key)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    g1 = clahe.apply(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY))
    g2 = clahe.apply(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY))
    if min(g1.shape[:2]) < 8 or min(g2.shape[:2]) < 8:
        return 0.0
    orb = cv2.ORB_create(nfeatures=150, fastThreshold=6, edgeThreshold=4, patchSize=16)
    kp1, des1 = orb.detectAndCompute(g1, None)
    if cached is None:
        kp2, des2 = orb.detectAndCompute(g2, None)
        orb_cache[key] = (kp2, des2)
    else:
        kp2, des2 = cached
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = 0
    try:
        for pair in bf.knnMatch(des1, des2, k=2):
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:  # ratio test de Lowe
                    good += 1
    except cv2.error:
        return 0.0
    return min(100.0, (good / max(len(des1), 1)) * 200.0)

def _combined_image_score(patch_bgr, target_bgr, orb_cache=None):
    """
    Score visual combinado (0..100 aprox.) entre un parche de casilla y un
    patrón del catálogo, integrando 6 métricas complementarias:

      1. Color LAB continuo ............ distancia euclidiana media por píxel
      2. Histogramas BGR+HSV ........... correlación (con penalización de
                                         anti-correlación, que antes se
                                         truncaba a 0 y perdía información)
      3. NCC estructural ............... correlación cruzada normalizada en
                                         grises + sobre gradientes Sobel
      4. Bordes Laplaciano ............. TM_CCOEFF_NORMED sobre Laplaciano
      5. Textura ILBP + Haralick ....... distancia de coseno entre vectores
      6. ORB (opcional) ................ keypoints con ratio test de Lowe

    Los pesos se renormalizan según las métricas realmente disponibles.
    """
    if patch_bgr is None or target_bgr is None:
        return 0.0
    h, w = patch_bgr.shape[:2]
    if h < 5 or w < 5:
        return 0.0

    target_resized = cv2.resize(target_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    # 1. Color LAB continuo
    patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff_lab = np.linalg.norm(patch_lab - target_lab, axis=2)
    mean_diff = float(np.mean(diff_lab))
    score_color = max(0.0, 100.0 - mean_diff * 1.25)

    # 2. Histogramas en 6 canales (BGR + HSV)
    #    Se conservan las correlaciones NEGATIVAS: indican patrones
    #    claramente distintos y mejoran la discriminación entre tramas
    #    con colores parecidos pero distribución diferente.
    hist_vals = []
    for src, dst in (
        (patch_bgr, target_resized),
        (cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV), cv2.cvtColor(target_resized, cv2.COLOR_BGR2HSV))
    ):
        for ch in range(3):
            h1 = cv2.calcHist([src], [ch], None, [32], [0, 256])
            h2 = cv2.calcHist([dst], [ch], None, [32], [0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            hist_vals.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    mean_corr = float(np.mean(hist_vals))
    # Mapea [-1, 1] -> [0, 100] con leve énfasis en el extremo positivo
    score_hist = max(0.0, (mean_corr + 0.15) / 1.15) * 100.0

    # 3. NCC estructural: se evalúa sobre la imagen original y sobre el
    #    mapa de gradientes Sobel. Además se prueba la inversión fotométrica
    #    del objetivo (tinta clara/oscura intercambiada), porque los logs
    #    escaneados a veces invierten el contraste de la trama.
    g1 = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(target_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1_n = (g1 - g1.mean()) / max(g1.std(), 1e-5)
    so1 = np.hypot(cv2.Sobel(g1, cv2.CV_32F, 1, 0), cv2.Sobel(g1, cv2.CV_32F, 0, 1))
    so1_n = (so1 - so1.mean()) / max(so1.std(), 1e-5)
    ncc = ncc_grad = -1.0
    for cand in (g2, 255.0 - g2):
        cand_n = (cand - cand.mean()) / max(cand.std(), 1e-5)
        ncc = max(ncc, float(np.mean(g1_n * cand_n)))
        so2 = np.hypot(cv2.Sobel(cand, cv2.CV_32F, 1, 0), cv2.Sobel(cand, cv2.CV_32F, 0, 1))
        so2_n = (so2 - so2.mean()) / max(so2.std(), 1e-5)
        ncc_grad = max(ncc_grad, float(np.mean(so1_n * so2_n)))
    score_struct = max(0.0, ncc * 0.6 + ncc_grad * 0.4) * 100.0

    # 4. Bordes con Laplaciano (también tolerante a inversión fotométrica)
    g1_blur = cv2.GaussianBlur(g1, (3, 3), 0)
    g2_blur = cv2.GaussianBlur(g2, (3, 3), 0)
    g2_inv_blur = cv2.GaussianBlur(255.0 - g2, (3, 3), 0)
    lap1 = cv2.Laplacian(g1_blur, cv2.CV_32F)
    max_edge_score = -1.0
    for cand in (g2_blur, g2_inv_blur):
        lap2 = cv2.Laplacian(cand, cv2.CV_32F)
        res_edge = cv2.matchTemplate(lap2, lap1, cv2.TM_CCOEFF_NORMED)
        _, ms, _, _ = cv2.minMaxLoc(res_edge)
        max_edge_score = max(max_edge_score, ms)
    score_edges = max(0.0, float(max_edge_score) * 100.0)

    # 5. Textura: ILBP + Haralick comparados por similitud de coseno
    f1 = _texture_features(g1)
    f2 = _texture_features(g2)
    if f1 is not None and f2 is not None:
        num = float(np.dot(f1, f2))
        den = float(np.linalg.norm(f1) * np.linalg.norm(f2)) + 1e-7
        score_texture = max(0.0, num / den) * 100.0
    else:
        score_texture = None

    # 6. ORB (solo si se proporciona caché, es la métrica más costosa)
    score_orb = _orb_match_score(patch_bgr, target_resized, orb_cache) if orb_cache is not None else None

    # --- Fusión ponderada con renormalización según métricas disponibles ---
    parts = [(score_color, W_COLOR), (score_hist, W_HIST), (score_struct, W_STRUCT), (score_edges, W_EDGES)]
    if score_texture is not None:
        parts.append((score_texture, W_TEXTURE))
    else:
        parts.append((score_color, W_TEXTURE))  # color sustituye a textura
    if score_orb is not None and score_orb > 0.0:
        parts.append((score_orb, W_ORB))
    elif score_orb is not None:
        # ORB se calculó pero no encontró puntos (trama suave/ruido):
        # no es una evidencia negativa, simplemente no aporta.
        parts.append((score_struct, W_ORB))
    else:
        parts.append((score_struct, W_ORB))     # estructura sustituye a ORB
    total_w = sum(wt for _, wt in parts)
    return sum(s * wt for s, wt in parts) / total_w

def evaluate_pattern_multiscale(patch_bgr, target_variants, norm_size=(64, 44), use_orb=True):
    """
    Evalúa un parche contra todas las variantes de un patrón y devuelve
    el mejor score, combinando:
      - Score en escala nativa (máxima resolución, incluye ORB).
      - Media de las escalas normalizada (64x44) y reducida (48x33):
        promediar escalas pequeñas reduce el ruido de cualquier escala
        individual, que era una fuente de falsos positivos.
    Se devuelve max(nativa, media_escalas) para no perder ni la precisión
    de la escala nativa ni la robustez de las reducidas.
    """
    best = 0.0
    orb_cache = {}
    for tv in target_variants:
        s_native = _combined_image_score(patch_bgr, tv, orb_cache if use_orb else None)
        p_norm = cv2.resize(patch_bgr, norm_size, interpolation=cv2.INTER_AREA)
        t_norm = cv2.resize(tv, norm_size, interpolation=cv2.INTER_AREA)
        s_norm = _combined_image_score(p_norm, t_norm)
        small_w, small_h = max(16, int(norm_size[0] * 0.75)), max(16, int(norm_size[1] * 0.75))
        p_small = cv2.resize(patch_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        t_small = cv2.resize(tv, (small_w, small_h), interpolation=cv2.INTER_AREA)
        s_small = _combined_image_score(p_small, t_small)
        s_scaled = (s_norm + s_small) / 2.0
        best = max(best, s_native, s_scaled)
    return best

class DINOv2Extractor:
    """
    Extractor de embeddings semánticos con DINOv2 (Visión por IA).

    Mejoras respecto a la versión básica:
      - Disponibilidad opcional: si torch o el modelo no pueden cargarse,
        `available` queda en False y el pipeline sigue con métricas clásicas.
      - Multi-crop: el embedding de una imagen es la media de 3 vistas
        (completa, zoom central 80% y zoom central 60%), más robusta a
        márgenes y encuadres distintos entre parche y patrón.
      - Caché en disco (.dino_cache/): los embeddings de los patrones del
        catálogo se guardan por hash del archivo, acelerando ejecuciones
        posteriores.
    """

    def __init__(self, cache_dir=None):
        self.available = False
        self.device = None
        self.model = None
        self.transform = None
        self.cache_dir = cache_dir or os.path.join(BASE_DIR, ".dino_cache")
        self._mem_cache = {}
        if not TORCH_AVAILABLE:
            return
        try:
            self.device = torch.device("cpu")
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.model.eval()
            self.model.to(self.device)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            os.makedirs(self.cache_dir, exist_ok=True)
            self.available = True
        except Exception:
            # Sin red o sin pesos: el sistema opera sin IA
            self.model = None
            self.available = False

    def _embed_single(self, pil_img):
        t = self.transform(pil_img).unsqueeze(0).to(self.device)
        feat = self.model(t).squeeze(0).numpy()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-7)

    def get_embedding(self, rgb_numpy):
        """Embedding multi-crop normalizado (o None si no disponible)."""
        if not self.available or rgb_numpy is None or rgb_numpy.size == 0:
            return None
        # Caché en memoria por contenido (evita recalcular el mismo parche)
        key = hashlib.md5(rgb_numpy.tobytes()).hexdigest()
        if key in self._mem_cache:
            return self._mem_cache[key]
        pil_img = Image.fromarray(rgb_numpy).convert("RGB")
        w, h = pil_img.size
        crops = [pil_img]
        if w >= 8 and h >= 8:
            # Zoom central 80% y 60%: robustez a bordes y encuadre
            for f in (0.8, 0.6):
                cw, ch = max(4, int(w * f)), max(4, int(h * f))
                x0, y0 = (w - cw) // 2, (h - ch) // 2
                crops.append(pil_img.crop((x0, y0, x0 + cw, y0 + ch)))
        with torch.no_grad():
            embs = [self._embed_single(c) for c in crops]
        feat = np.mean(np.stack(embs), axis=0)
        feat = feat / (np.linalg.norm(feat) + 1e-7)
        self._mem_cache[key] = feat
        return feat

    def get_embedding_cached_file(self, filepath, rgb_numpy):
        """
        Embedding de un archivo de patrón con caché persistente en disco.
        La clave incluye ruta + tamaño + mtime, así se invalida si el
        archivo cambia.
        """
        if not self.available:
            return None
        try:
            st = os.stat(filepath)
            sig = f"{os.path.abspath(filepath)}|{st.st_size}|{int(st.st_mtime)}"
            disk_key = os.path.join(self.cache_dir, hashlib.md5(sig.encode()).hexdigest() + ".npy")
            if os.path.isfile(disk_key):
                return np.load(disk_key)
            emb = self.get_embedding(rgb_numpy)
            if emb is not None:
                try:
                    np.save(disk_key, emb)
                except OSError:
                    pass  # caché no escribible: no es crítico
            return emb
        except OSError:
            return self.get_embedding(rgb_numpy)

    @staticmethod
    def cosine_similarity(emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        return max(0.0, float(np.dot(emb1, emb2))) * 100.0

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
        self.stage_changed.emit("Etapa 1: Segmentando casillas litológicas")
        self.progress.emit(5, "Cargando imagen...")

        img_pil = Image.open(self.image_path).convert("RGB")
        img_np = np.array(img_pil)
        img_h, img_w = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if 15 < w < 100 and 8 < h < 60 and 0.4 < aspect < 4.0:
                boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: (b[1], b[0]))
        if not boxes:
            self.error.emit("No se detectaron casillas en la imagen.")
            return

        xs = sorted(set(b[0] for b in boxes))
        col_starts = []
        for xv in xs:
            if not col_starts or xv - col_starts[-1] > 50: 
                col_starts.append(xv)
        col_ends = col_starts[1:] + [img_w]

        self.stage_changed.emit("Etapa 2: Reconocimiento OCR de etiquetas")
        self.progress.emit(25, "Iniciando EasyOCR...")
        reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)

        segments = []
        total = len(boxes)
        for i, (bx, by, bw, bh) in enumerate(boxes):
            pct = 25 + int((i / total) * 25)
            self.progress.emit(pct, f"OCR {i+1}/{total}...")

            patch = img_np[by:by+bh, bx:bx+bw]
            ph, pw = patch.shape[:2]
            pure_patch = np.ascontiguousarray(patch[2:ph-2, 2:pw-2]) if ph > 8 and pw > 8 else patch

            end_x = img_w
            for ce in col_ends:
                if bx + bw + 5 < ce:
                    end_x = ce
                    break
            text_region = img_np[max(0, by-2):min(img_h, by+bh+2), (bx + bw + 1):end_x]

            # --- OCR multi-estrategia -------------------------------------
            # El texto de los logs suele venir pequeño y con fondo irregular.
            # Se prueban 3 preprocesamientos y se conserva el resultado con
            # mayor confianza media del reconocedor:
            #   A) imagen original
            #   B) escala x2 en grises (mejora trazos finos)
            #   C) escala x2 + binarización de Otsu (fondo uniforme)
            ocr_text = ""
            if text_region.size > 0:
                variants = [text_region]
                try:
                    tr_gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
                    tr_big = cv2.resize(tr_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    variants.append(tr_big)
                    _, tr_otsu = cv2.threshold(tr_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    variants.append(tr_otsu)
                except cv2.error:
                    pass
                best_txt, best_conf = "", -1.0
                for variant in variants:
                    try:
                        res = reader.readtext(variant, detail=1, paragraph=False)
                    except Exception:
                        continue
                    if not res:
                        continue
                    txt = " ".join(r[1] for r in res).strip()
                    conf = float(np.mean([r[2] for r in res]))
                    # Se prefiere el texto más largo si la confianza empata
                    if conf > best_conf + 0.02 or (abs(conf - best_conf) <= 0.02 and len(txt) > len(best_txt)):
                        best_txt, best_conf = txt, conf
                ocr_text = best_txt
                for ch in ["?", "_", "|", "}", "{"]:
                    ocr_text = ocr_text.replace(ch, "")
                ocr_text = ocr_text.strip()

            segments.append({
                "patch": patch,
                "pure_patch": pure_patch,
                "ocr_text": ocr_text,
                "box": (bx, by, bw, bh),
            })

        self.stage_changed.emit("Etapa 3: Identificación Litológica")
        self.progress.emit(55, "Correlacionando con geo_lith.csv...")
        for seg in segments:
            id_lith, matched_name, ocr_conf = match_lith_name(seg["ocr_text"], self.geo_lith)
            seg["idLith"] = id_lith
            seg["matched_nombre"] = matched_name
            seg["ocr_conf"] = ocr_conf

        self.stage_changed.emit("Etapa 4: Clasificación Visual y Desempate Fino")
        self.progress.emit(65, "Indexando catálogo de patrones...")
        ai_extractor = DINOv2Extractor()
        pattern_files = list_pattern_files(self.patterns_dir)

        catalog = []
        for pf in pattern_files:
            fp = os.path.join(self.patterns_dir, pf)
            target = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if target is None: 
                continue
            variants = []
            if target.ndim == 3 and target.shape[2] == 4:
                alpha = target[:, :, 3]
                base = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
                for bg in ((255, 255, 255), (0, 0, 0)):
                    filled = base.copy()
                    filled[alpha < 128] = list(bg)
                    variants.append(filled)
                variants.append(base)
            elif target.ndim == 2:
                variants.append(cv2.cvtColor(target, cv2.COLOR_GRAY2BGR))
            else:
                variants.append(target)

            target_rgb = cv2.cvtColor(variants[0], cv2.COLOR_BGR2RGB)
            # Caché persistente: no recalcula embeddings de patrones ya vistos
            dino_emb = ai_extractor.get_embedding_cached_file(fp, target_rgb)

            catalog.append({
                "filename": pf,
                "basename": os.path.splitext(pf)[0],
                "variants": variants,
                "dino_emb": dino_emb,
            })

        self.progress.emit(75, "Evaluando patrones litológicos...")
        for i, seg in enumerate(segments):
            pct = 75 + int((i / total) * 23)
            self.progress.emit(pct, f"Evaluando casilla {i+1}/{total}...")

            pure_rgb = seg["pure_patch"]
            patch_rgb = seg["patch"]
            pure_bgr = cv2.cvtColor(pure_rgb, cv2.COLOR_RGB2BGR)
            patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
            seg_dino = ai_extractor.get_embedding(pure_rgb)
            name_clean = seg["matched_nombre"].upper()
            id_val = seg["idLith"]
            ocr_conf = seg.get("ocr_conf", 0.0)

            best_file = ""
            best_score = -1.0

            for cat_item in catalog:
                s_cv = max(
                    evaluate_pattern_multiscale(pure_bgr, cat_item["variants"]),
                    evaluate_pattern_multiscale(patch_bgr, cat_item["variants"])
                )

                # Fusión IA + métricas clásicas. Si DINOv2 no está disponible
                # para alguno de los lados, el score es 100% métricas clásicas.
                s_dino = ai_extractor.cosine_similarity(seg_dino, cat_item["dino_emb"])
                if seg_dino is not None and cat_item["dino_emb"] is not None:
                    score_final = (s_cv * (1.0 - DINO_WEIGHT)) + (s_dino * DINO_WEIGHT)
                else:
                    score_final = s_cv

                bname = cat_item["basename"]

                # --- Desempate litológico escalado por confianza OCR -------
                # La clave geológica se resuelve en _geology_from_name a
                # partir del idLith o del nombre (con fuzzy de tokens), así
                # "ARCILITA TOBACEA" y "ARCILITA TOB." comparten desempate
                # aunque el OCR las haya clasificado distinto.
                # El ajuste es proporcional a la confianza del OCR: con OCR
                # dudoso apenas influye y no fuerza errores en patrones
                # básicos (problema que tenía la versión anterior).
                geo_key = _geology_from_name(name_clean, id_val, self.geo_lith)
                if geo_key is not None and geo_key in LITHOLOGY_AFFINITIES:
                    adj = LITHOLOGY_AFFINITIES[geo_key].get(bname, 0.0)
                    score_final += adj * TIE_BREAK_STRENGTH * ocr_conf

                if score_final > best_score:
                    best_score = score_final
                    best_file = cat_item["filename"]

            seg["patternImage"] = best_file
            seg["pattern_score"] = round(max(0.0, best_score), 2)

        self.stage_changed.emit("Completado")
        self.progress.emit(100, f"Procesamiento finalizado ({len(segments)} elementos).")

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
                "ocr_conf": seg.get("ocr_conf", 0.0),
                "patch": seg["patch"],
                "box": seg["box"],
            })
        self.finished.emit(results)

class ExportDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("Exportar Resultados")
        self.resize(460, 200)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["CSV (.csv)", "SQL para MySQL (.sql)", "SQL para PostgreSQL (.sql)"])
        form.addRow("Formato:", self.cmb_format)
        self.txt_table = QLineEdit("correlacion_litologica")
        form.addRow("Tabla SQL:", self.txt_table)
        layout.addLayout(form)

        btn = QPushButton("Exportar")
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
            QMessageBox.information(self, "Exportado", f"CSV guardado:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _export_mysql(self, path):
        try:
            tbl = self.txt_table.text().strip() or "correlacion_litologica"
            lines = [
                f"CREATE TABLE IF NOT EXISTS `{tbl}` (",
                "  `id` INT AUTO_INCREMENT PRIMARY KEY,",
                "  `patternImage` VARCHAR(255) NOT NULL,",
                "  `idLith` INT NOT NULL,",
                "  `idOperadora` INT NOT NULL",
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n"
            ]
            for r in self.results:
                pi = r["patternImage"].replace("'", "\\'")
                lines.append(f"INSERT INTO `{tbl}` (`patternImage`, `idLith`, `idOperadora`) VALUES ('{pi}', {r['idLith']}, {r['idOperadora']});")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            QMessageBox.information(self, "Exportado", f"SQL MySQL guardado:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _export_postgresql(self, path):
        try:
            tbl = self.txt_table.text().strip() or "correlacion_litologica"
            lines = [
                f'CREATE TABLE IF NOT EXISTS "{tbl}" (',
                '  "id" BIGSERIAL PRIMARY KEY,',
                '  "patternImage" VARCHAR(255) NOT NULL,',
                '  "idLith" INTEGER NOT NULL,',
                '  "idOperadora" INTEGER NOT NULL',
                ');\n'
            ]
            for r in self.results:
                pi = r["patternImage"].replace("'", "''")
                lines.append(f'INSERT INTO "{tbl}" ("patternImage", "idLith", "idOperadora") VALUES (\'{pi}\', {r["idLith"]}, {r["idOperadora"]});')
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            QMessageBox.information(self, "Exportado", f"SQL PostgreSQL guardado:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

class DropImageZone(QLabel):
    imageSelected = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("🖼️ Arrastrá la imagen aquí\no hacé clic para buscar")
        self.setAcceptDrops(True)
        self.setMinimumSize(280, 120)
        self.setStyleSheet("QLabel { border: 2px dashed #585b70; border-radius: 10px; background-color: #181825; color: #a6adc8; font-size: 13px; padding: 20px; } QLabel:hover { border-color: #89b4fa; background-color: #1e1e2e; }")
        self._path = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            p, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff)")
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lith Analizador Litológico")
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

        phase_bar = QHBoxLayout()
        self.phase_btns = []
        for i, txt in enumerate(["① Configuración", "② Procesamiento", "③ Resultados"]):
            b = QPushButton(txt)
            b.setFixedHeight(36)
            b.setEnabled(i == 0)
            b.clicked.connect(lambda _, idx=i: self.stacked.setCurrentIndex(idx))
            phase_bar.addWidget(b)
            self.phase_btns.append(b)
        root.addLayout(phase_bar)

        self.lbl_status = QLabel("Configurá los parámetros y presioná Procesar.")
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

        left = QVBoxLayout()
        grp_img = QGroupBox("Imagen de Referencias Litológicas")
        gl = QVBoxLayout(grp_img)
        self.drop_zone = DropImageZone()
        self.drop_zone.imageSelected.connect(self._on_image_selected)
        gl.addWidget(self.drop_zone)
        left.addWidget(grp_img)
        lay.addLayout(left, stretch=2)

        right = QVBoxLayout()
        grp_op = QGroupBox("Metadatos de Pozo")
        ol = QFormLayout(grp_op)
        self.spn_operadora = QSpinBox()
        self.spn_operadora.setRange(1, 999999)
        self.spn_operadora.setValue(1)
        ol.addRow("idOperadora:", self.spn_operadora)
        right.addWidget(grp_op)

        grp_csv = QGroupBox("Catálogo geo_lith.csv")
        cl = QVBoxLayout(grp_csv)
        self.lbl_csv_status = QLabel(f"✅ {len(self.geo_lith)} registros cargados" if self.geo_lith else "❌ No encontrado")
        cl.addWidget(self.lbl_csv_status)
        btn_csv = QPushButton("Cargar otro CSV")
        btn_csv.clicked.connect(self._load_csv)
        cl.addWidget(btn_csv)
        right.addWidget(grp_csv)

        grp_pat = QGroupBox("Catálogo de Patrones")
        pl = QVBoxLayout(grp_pat)
        n_pats = len(list_pattern_files(self.patterns_dir))
        self.lbl_patterns = QLabel(f"📁 {self.patterns_dir}\n({n_pats} patrones indexados)")
        pl.addWidget(self.lbl_patterns)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(3)
        self._update_pattern_grid()
        pl.addWidget(self.grid_widget)

        btn_pat = QPushButton("Seleccionar carpeta de patrones")
        btn_pat.clicked.connect(self._select_patterns_dir)
        pl.addWidget(btn_pat)
        right.addWidget(grp_pat)

        self.btn_process = QPushButton("▶  PROCESAR")
        self.btn_process.setObjectName("accentBtn")
        self.btn_process.setFixedHeight(44)
        self.btn_process.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #a6e3a1; color: #1e1e2e; border:none; border-radius:6px;")
        self.btn_process.clicked.connect(self._start_processing)
        right.addWidget(self.btn_process)

        right.addStretch()
        lay.addLayout(right, stretch=1)
        return w

    def _update_pattern_grid(self):
        while self.grid_layout.count():
            it = self.grid_layout.takeAt(0)
            if it.widget(): 
                it.widget().deleteLater()
        if not os.path.isdir(self.patterns_dir): 
            return
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
        grp = QGroupBox("Progreso")
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

        grp_log = QGroupBox("Log")
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
        grp_insp = QGroupBox("Inspector Visual")
        il = QHBoxLayout(grp_insp)

        f1 = QFrame()
        l1 = QVBoxLayout(f1)
        l1.addWidget(QLabel("<b>Patrón Extraído</b>"), alignment=Qt.AlignCenter)
        self.lbl_insp_patch = QLabel("—")
        self.lbl_insp_patch.setAlignment(Qt.AlignCenter)
        self.lbl_insp_patch.setFixedSize(120, 80)
        self.lbl_insp_patch.setStyleSheet("border:1px solid #45475a; background:#181825;")
        l1.addWidget(self.lbl_insp_patch, alignment=Qt.AlignCenter)
        il.addWidget(f1)

        f2 = QFrame()
        l2 = QVBoxLayout(f2)
        l2.addWidget(QLabel("<b>Identificación</b>"), alignment=Qt.AlignCenter)
        self.lbl_insp_info = QLabel("idLith: —\nNombre: —\nOCR: —")
        self.lbl_insp_info.setAlignment(Qt.AlignCenter)
        self.lbl_insp_info.setStyleSheet("font-size:12px; color:#a6e3a1;")
        l2.addWidget(self.lbl_insp_info)
        il.addWidget(f2)

        f3 = QFrame()
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

    def _on_image_selected(self, p):
        self.image_path = p
        self.lbl_status.setText(f"Imagen seleccionada: {os.path.basename(p)}")

    def _load_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "Seleccionar geo_lith.csv", "", "CSV (*.csv)")
        if p:
            self.geo_lith = load_geo_lith_csv(p)
            self.lbl_csv_status.setText(f"✅ {len(self.geo_lith)} registros desde {os.path.basename(p)}")

    def _select_patterns_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de patrones")
        if d:
            self.patterns_dir = d
            n = len(list_pattern_files(d))
            self.lbl_patterns.setText(f"📁 {d}\n({n} imágenes)")
            self._update_pattern_grid()

    def _start_processing(self):
        if not self.image_path or not self.geo_lith:
            QMessageBox.warning(self, "Atención", "Cargá una imagen y el archivo geo_lith.csv primero.")
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

    def _on_finished(self, results):
        self.results = results
        self.phase_btns[2].setEnabled(True)
        self.stacked.setCurrentIndex(2)
        self.lbl_result_summary.setText(f"✅ {len(results)} elementos analizados")
        self._populate_results_table()
        if results: 
            self.table_results.selectRow(0)

    def _populate_results_table(self):
        pat_files = list_pattern_files(self.patterns_dir)
        self.table_results.blockSignals(True)
        self.table_results.setRowCount(len(self.results))
        for i, r in enumerate(self.results):
            patch = r.get("patch")
            if patch is not None and getattr(patch, "size", 0) > 0:
                patch_c = np.ascontiguousarray(patch)
                h, w, c = patch_c.shape
                qimg = QImage(patch_c.data, w, h, c * w, QImage.Format_RGB888)
                lbl_thumb = QLabel()
                lbl_thumb.setAlignment(Qt.AlignCenter)
                lbl_thumb.setPixmap(QPixmap.fromImage(qimg).scaled(56, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.table_results.setCellWidget(i, 0, lbl_thumb)

            for col, key in enumerate(["id", "patternImage", "idLith", "idOperadora", "ocr_text", "matched_nombre"], start=1):
                item = QTableWidgetItem(str(r[key]))
                if col <= 4:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignCenter)
                self.table_results.setItem(i, col, item)

            combo = QComboBox()
            combo.setIconSize(QSize(56, 38))
            combo.addItem("(elegir patrón)", "")
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
        patch = r.get("patch")
        if patch is not None and patch.size > 0:
            pc = np.ascontiguousarray(patch)
            h, w, c = pc.shape
            qimg = QImage(pc.data, w, h, c * w, QImage.Format_RGB888)
            self.lbl_insp_patch.setPixmap(QPixmap.fromImage(qimg).scaled(self.lbl_insp_patch.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        conf = r.get("ocr_conf", 0.0)
        self.lbl_insp_info.setText(f"idLith: {r['idLith']}\nNombre: {r['matched_nombre']}\nOCR: {r['ocr_text']} ({conf*100:.0f}%)")
        combo = self.table_results.cellWidget(row, 7)
        score = combo.property("score_text") if isinstance(combo, QComboBox) else f"{r['pattern_score']:.1f}"
        if r["patternImage"]:
            p = os.path.join(self.patterns_dir, r["patternImage"])
            if os.path.isfile(p):
                self.lbl_insp_match.setPixmap(QPixmap(p).scaled(self.lbl_insp_match.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_match_detail.setText(f"Archivo: {r['patternImage']}\nSimilitud: {score}%")
        else:
            self.lbl_insp_match.setText("—")
            self.lbl_match_detail.setText("Sin coincidencia")

    def _open_export_dialog(self):
        if not self.results:
            QMessageBox.warning(self, "Atención", "No hay datos para exportar.")
            return
        ExportDialog(self.results, self).exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())