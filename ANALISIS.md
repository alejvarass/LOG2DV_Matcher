# Documento de Análisis — Cómo compara y decide LOG2DV Matcher

Este documento explica en detalle el proceso que sigue la aplicación
(`Lith_Matcher.py`) para convertir una imagen de referencias litológicas
(log) en una tabla `id → patternImage / idLith / idOperadora` exportable a
CSV, MySQL o PostgreSQL.

---

## 1. Visión general del pipeline

El procesamiento tiene 4 etapas secuenciales:

```
Imagen del log
   │
   ├─ Etapa 1: Segmentación de casillas  (OpenCV, contornos)
   ├─ Etapa 2: OCR de la etiqueta de cada casilla  (EasyOCR multi-estrategia)
   ├─ Etapa 3: Identificación litológica  (fuzzy matching contra geo_lith.csv)
   └─ Etapa 4: Clasificación visual del patrón
                 (6 métricas clásicas + DINOv2/IA + desempate litológico)
```

Cada etapa alimenta a la siguiente: el OCR da el **nombre litológico** y una
**confianza (0–1)**; la clasificación visual da el **patrón gráfico**; y el
desempate litológico usa la confianza del OCR para ajustar la decisión visual
solo cuando el texto es fiable.

---

## 2. Etapa 1 — Segmentación de casillas

1. La imagen se pasa a escala de grises y se binariza con umbral 80
   (`THRESH_BINARY_INV`) para resaltar bordes y tinta.
2. `cv2.findContours` extrae los contornos externos.
3. Se filtran por geometría de casilla: `15 < ancho < 100`,
   `8 < alto < 60`, relación de aspecto `0.4 – 4.0`.
4. Las casillas se ordenan de arriba-abajo y de izquierda-derecha
   (`sort` por `(y, x)`), de modo que el `id` de salida sigue el orden
   visual de la columna del log.
5. Se detectan los límites de columna (saltos de x > 50 px) para saber
   hasta dónde llega la zona de texto de cada fila.

De cada casilla se guardan dos recortes:

- `patch`: la casilla completa (con borde).
- `pure_patch`: la casilla sin 2 px de borde, para que la línea del
  marco no contamine la comparación del relleno.

---

## 3. Etapa 2 — OCR multi-estrategia (EasyOCR)

El texto junto a cada casilla es pequeño y a veces viene con fondo irregular.
Para maximizar la lectura se prueban **tres preprocesamientos** de la región
de texto y se conserva el resultado con **mayor confianza media** del
reconocedor:

| Variante | Proceso | Cuándo ayuda |
|----------|---------|--------------|
| A | Imagen original | Texto limpio y grande |
| B | Grises + escalado ×2 (cúbico) | Trazos finos |
| C | Grises + ×2 + binarización Otsu | Fondo irregular |

Si dos variantes empatan en confianza (±0.02) se prefiere el texto más largo.
Después se eliminan caracteres espurios (`? _ | } {`).

EasyOCR trabaja en español + inglés (`["es", "en"]`), sin GPU.

---

## 4. Etapa 3 — Identificación litológica (fuzzy matching geológico)

`match_lith_name` compara el texto OCR con el catálogo `geo_lith.csv`
(`idLith, nombre`) y devuelve `(idLith, nombre, confianza)`.

### Normalización

Antes de comparar, el texto se normaliza: mayúsculas, sin acentos (NFKD),
solo alfanumérico, separado en *tokens* y sin stopwords geológicas
(`DE, LA, EL, Y, CON, DEL, EN`). Además se aplican correcciones típicas del
OCR (`LIM0→LIMO`, `AR3N→AREN`, `T0B→TOB`, …).

### Escalera de decisión (de mayor a menor confianza)

| Paso | Regla | Confianza |
|------|-------|-----------|
| 1 | Coincidencia exacta de la cadena | 1.00 |
| 2 | Todos los tokens del nombre están en el OCR (exige que el nombre sea al menos tan específico como el OCR; gana el más largo) | 0.90 |
| 3 | Matching por tokens: cada token del OCR se empareja con el mejor token del nombre (exacto, abreviatura-prefijo `"TOB"→TOBACEA`, o fuzzy ≥ 0.75 con **guardia de especificidad**) | 0.60–0.88 |
| 5 | Subcadena con límites de palabra y longitud comparable | 0.50 |
| 6 | `difflib.get_close_matches` (cutoff 0.65) con guardia de longitud | 0.45 |
| 7 | Mejor ratio `SequenceMatcher` ≥ 0.55 con guardia de longitud | ratio × 0.6 |
| — | Sin coincidencia fiable | 0.0 (`idLith = -1`) |

La **guardia de especificidad** evita falsos positivos clásicos:
palabras cortas (≤ 5 letras) o con gran diferencia de longitud solo cuentan
como coincidencia si el ratio es casi exacto (≥ 0.92). Así `"ARENA"` ya no
activa `AREN` ni `TOBA ARENOSA` (devuelve -1), mientras que
`"ARCILITA TOB."` sí llega a `ARCILITA TOBACEA` por abreviatura.

La **confianza** devuelta se usa en la Etapa 4 para escalar el desempate:
un OCR dudoso apenas influye en la decisión visual.

---

## 5. Etapa 4 — Clasificación visual del patrón

Cada casilla se compara contra todo el catálogo de patrones (`Patterns/*.png`)
con una fusión de **6 métricas clásicas + IA (DINOv2)**.

### 5.1 Preparación del catálogo

Cada archivo de patrón genera *variantes* para hacer la comparación robusta:

- PNG con transparencia → se compone sobre fondo **blanco** y sobre fondo
  **negro**, además de la versión base.
- Escala de grises → se convierte a BGR.

### 5.2 Métricas clásicas (`_combined_image_score`)

| # | Métrica | Qué mide | Peso |
|---|---------|----------|------|
| 1 | **Color LAB** | Distancia euclidiana media por píxel en espacio perceptual | 0.10 |
| 2 | **Histogramas BGR+HSV** | Correlación de histogramas en 6 canales; las correlaciones *negativas* ya no se truncan a 0, así patrones con colores parecidos pero distribución distinta se penalizan | 0.07 |
| 3 | **NCC estructural** | Correlación cruzada normalizada en grises **y** sobre gradientes Sobel, tolerante a **inversión fotométrica** (logs escaneados con tinta/fondo intercambiados) | 0.06 |
| 4 | **Bordes Laplaciano** | `matchTemplate` TM_CCOEFF_NORMED sobre el Laplaciano, también con inversión | 0.07 |
| 5 | **Textura** (ver 5.3) | LBP-ri + distancia + grosor + orientación + proyecciones + GLCM | 0.48 |
| 6 | **ORB** | Keypoints con ratio test de Lowe (0.75) sobre imagen ecualizada con CLAHE | 0.22 |

Los pesos se **renormalizan** automáticamente si alguna métrica no está
disponible (p.ej. ORB en parches muy pequeños), de modo que el score final
siempre está en escala 0–100 comparable.

### 5.3 Vector de textura (`_texture_features`)

Es la métrica más discriminativa para tramas geológicas. Se calcula sobre la
imagen en grises suavizada (Gauss 3×3) y concatena estos bloques
(normalizados a norma 1 para que ninguno domine el coseno):

1. **LBP-ri (36 bins)** — Local Binary Pattern crudo invariante a rotación
   con umbral de 2 niveles de gris (ignora micro-ruido de escaneo). Se
   atenúa con potencia 0.75 para que el bin dominante (fondo plano) no
   tape los bins de borde que separan puntos de líneas.
2. **Histograma de transformada de distancia (8 bins fijos 0–12 px)** —
   cuánto "fondo" hay alrededor de la tinta: separa tramas densas de
   líneas aisladas.
3. **Perfil de grosor de tinta (3 valores)** — ratio de supervivencia de la
   tinta tras erosiones de 1, 2 y 3 px: las retículas dejan intersecciones
   gruesas que aguantan; las líneas finas desaparecen.
4. **Orientación dominante (4 bins, 0°/45°/90°/135°)** — la retícula reparte
   energía en 2 orientaciones; las líneas simples concentran en 1.
5. **Proyecciones de tinta por filas y columnas (8+8 bins)** — líneas
   horizontales → picos en filas; verticales → picos en columnas;
   retícula → picos en ambas.
6. **Simetría de tinta (2 valores)** — retícula simétrica en ambos ejes.
7. **GLCM (8 valores)** — contraste, energía (ASM), homogeneidad y
   correlación a 0° y 90° sobre la imagen cuantizada a 8 niveles.

La similitud entre parche y patrón es el **coseno** entre sus vectores.

### 5.4 Evaluación multi-escala (`evaluate_pattern_multiscale`)

Cada parche se evalúa en 3 escalas: nativa, normalizada (64×44) y reducida
(48×33). El score del patrón es `max(nativa, media(norm, small))`:
la escala nativa conserva la máxima resolución (y es la única que usa ORB,
porque en escalas pequeñas ORB pierde los puntos), mientras que la media de
las escalas reducidas suaviza el ruido de una escala concreta.

### 5.5 IA — DINOv2 (`DINOv2Extractor`)

Cuando `torch` y el modelo `dinov2_vits14` están disponibles, se extrae un
embedding semántico de 384 dimensiones:

- **Multi-crop**: se promedian 3 vistas (completa, zoom central 80 % y
  60 %), lo que hace el embedding robusto a márgenes y encuadres distintos.
- **Caché en disco** (`.dino_cache/`): los embeddings de los patrones se
  guardan por hash de (ruta, tamaño, fecha), así las ejecuciones
  posteriores no recalculan el catálogo. Los parches usan caché en memoria
  por contenido.
- **Fallback**: si torch no está instalado, no hay red o el modelo no carga,
  `available = False` y el pipeline sigue funcionando solo con las métricas
  clásicas (la app no se rompe).

La similitud IA es el coseno entre embeddings × 100.

### 5.6 Fusión final y desempate litológico

```
score = (1 - DINO_WEIGHT) * score_clásico + DINO_WEIGHT * score_DINOv2
        + ajuste_litología * TIE_BREAK_STRENGTH * confianza_OCR
```

- `DINO_WEIGHT = 0.30` (70 % métricas clásicas + 30 % IA). Si la IA no está
  disponible, el score es 100 % clásico.
- El **desempate** usa la tabla `LITHOLOGY_AFFINITIES` (idLith →
  {patrón: ajuste ±1}). La clave geológica se resuelve en
  `_geology_from_name` por idLith **o** por tokens del nombre (con fuzzy),
  así `ARCILITA TOBACEA` y `ARCILITA TOB.` comparten el mismo desempate.
- El ajuste se **escala por la confianza del OCR**: si el OCR es dudoso,
  el desempate apenas mueve el score y no fuerza errores en patrones
  básicos (problema que tenía la versión anterior con ajustes fijos ±3.5).

El patrón con mayor `score` se asigna a la casilla y se muestra su
porcentaje en el inspector.

---

## 6. Resultados y exportación

Por cada casilla se genera:

| Campo | Origen |
|-------|--------|
| `id` | Orden visual (arriba→abajo, izquierda→derecha) |
| `patternImage` | Patrón del catálogo con mayor score |
| `idLith` | Catálogo geo_lith vía OCR (-1 si no hay match fiable) |
| `idOperadora` | Metadato introducido en la interfaz |
| `ocr_text` / `matched_nombre` | Texto leído y nombre del catálogo |
| `pattern_score` | Score final (0–100) |
| `ocr_conf` | Confianza del match litológico (0–1) |

Exportaciones: **CSV** (`id,patternImage,idLith,idOperadora`), **SQL MySQL**
(`CREATE TABLE` + `INSERT`) y **SQL PostgreSQL** (con `BIGSERIAL`). La tabla
de resultados permite **ajuste manual** del patrón mediante un combo con
miniaturas, sin alterar el layout original.

---

## 7. Resumen de mejoras respecto a la versión anterior

| Área | Antes | Ahora |
|------|-------|-------|
| OCR | 1 pasada, sin confianza | 3 preprocesamientos, elige por confianza |
| Match litológico | cadena fuzzy simple | tokens + abreviaturas + guardia de especificidad + confianza |
| Histogramas | correlación negativa truncada a 0 | se conserva (mejor discriminación) |
| Estructura | NCC solo en grises | NCC + Sobel, tolerante a inversión |
| Textura | — | LBP-ri + distancia + grosor + orientación + proyecciones + GLCM |
| Keypoints | — | ORB con CLAHE + ratio test de Lowe |
| IA (DINOv2) | 1 crop, sin caché, rompía sin torch | multi-crop, caché en disco, fallback seguro |
| Desempate | ±3.5 fijo por nombre parcial | ponderado por confianza OCR, clave geológica unificada |
