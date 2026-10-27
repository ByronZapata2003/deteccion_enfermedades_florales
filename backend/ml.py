"""Extracción de características, entrenamiento y predicción del detector foliar."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[1]
MODELOS = ROOT / "models"
RUTA_MODELO = MODELOS / "modelo.joblib"
RUTA_BASELINE = MODELOS / "baseline_color.joblib"
RUTA_METRICAS = MODELOS / "metricas.json"

CLASES = ("mancha_negra", "hoja_seca", "hoja_sana", "perforaciones")

CATALOGO = {
    "mancha_negra": {
        "etiqueta": "Mancha negra",
        "etiqueta_original": "Black Spot",
        "tipo": "Enfermedad fúngica foliar",
        "referencia": "Asociada con frecuencia a Diplocarpon rosae en rosal.",
        "recomendacion": (
            "Retira y no compostes las hojas con manchas. Mejora la ventilación, "
            "evita mojar el follaje al regar y valora un fungicida autorizado para "
            "mancha negra del rosal, siempre según la etiqueta local."
        ),
    },
    "hoja_seca": {
        "etiqueta": "Hoja seca",
        "etiqueta_original": "Dry Leaf",
        "tipo": "Síntoma foliar",
        "referencia": "Puede deberse a estrés hídrico, senescencia, calor o daño previo; no identifica un único patógeno.",
        "recomendacion": (
            "Revisa riego, drenaje, sales del sustrato y exposición al sol. "
            "Elimina tejido seco y compara con hojas nuevas antes de aplicar un producto."
        ),
    },
    "hoja_sana": {
        "etiqueta": "Hoja sana",
        "etiqueta_original": "Healthy Leaf",
        "tipo": "Control sano",
        "referencia": "Hoja sin síntomas evidentes en las clases de este dataset.",
        "recomendacion": (
            "Mantén un manejo preventivo: riego en la base, poda de aireación "
            "y vigilancia semanal de manchas o perforaciones nuevas."
        ),
    },
    "perforaciones": {
        "etiqueta": "Perforaciones foliares",
        "etiqueta_original": "Leaf Holes",
        "tipo": "Daño foliar",
        "referencia": "Suele relacionarse con insectos masticadores u otro daño mecánico, no siempre con una infección.",
        "recomendacion": (
            "Inspecciona el envés y los brotes en busca de larvas o escarabajos. "
            "Retira hojas muy dañadas y confirma la plaga antes de aplicar un insecticida."
        ),
    },
}


def extraer_color(imagen: Image.Image) -> np.ndarray:
    """Histograma HSV y estadísticas RGB. Baseline simple, sin forma de la hoja."""
    rgb = imagen.convert("RGB").resize((96, 96))
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
    hist_h, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 255))
    hist_s, _ = np.histogram(hsv[:, :, 1], bins=8, range=(0, 255))
    hist_v, _ = np.histogram(hsv[:, :, 2], bins=8, range=(0, 255))
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist /= hist.sum() + 1e-6
    arr = np.asarray(rgb.resize((32, 32)), dtype=np.float32) / 255.0
    medias = arr.mean(axis=(0, 1))
    desv = arr.std(axis=(0, 1))
    proporcion_verde = np.array([medias[1] / (float(medias.sum()) + 1e-6)], dtype=np.float32)
    return np.concatenate([hist, medias, desv, proporcion_verde]).astype(np.float32)


def extraer_caracteristicas(imagen: Image.Image) -> np.ndarray:
    """Color + HOG. El HOG aporta textura (manchas, bordes de agujeros, nervaduras)."""
    color = extraer_color(imagen)
    gris = np.asarray(imagen.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
    textura = hog(
        gris,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    ).astype(np.float32)
    return np.concatenate([color, textura])


def vectorizar(imagenes, extractor) -> np.ndarray:
    return np.vstack([extractor(imagen) for imagen in imagenes])


def cargar_modelo(ruta: Path = RUTA_MODELO):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta.name}. Ejecuta `python entrenar_modelo.py` desde la raíz del proyecto."
        )
    paquete = joblib.load(ruta)
    return paquete["modelo"], list(paquete["clases"])


def predecir(imagen: Image.Image, ruta: Path = RUTA_MODELO) -> dict:
    modelo, clases = cargar_modelo(ruta)
    vector = extraer_caracteristicas(imagen).reshape(1, -1)
    if hasattr(modelo, "predict_proba"):
        probabilidad = modelo.predict_proba(vector)[0]
        orden_modelo = list(modelo.classes_)
        por_clase = {str(clase): float(probabilidad[i]) for i, clase in enumerate(orden_modelo)}
    else:
        predicho = str(modelo.predict(vector)[0])
        por_clase = {clase: (1.0 if clase == predicho else 0.0) for clase in clases}
    clase = max(por_clase, key=por_clase.get)
    ficha = CATALOGO[clase]
    return {
        "clase": clase,
        "etiqueta": ficha["etiqueta"],
        "tipo": ficha["tipo"],
        "referencia": ficha["referencia"],
        "confianza": round(por_clase[clase], 4),
        "probabilidades": {CATALOGO[k]["etiqueta"]: round(por_clase[k], 4) for k in clases},
        "recomendacion": ficha["recomendacion"],
        "aviso": (
            "Resultado de un prototipo educativo entrenado con fotos de hojas de rosal. "
            "No es un diagnóstico agronómico ni sustituye la revisión de una persona especialista."
        ),
    }


def guardar_metricas(metricas: dict) -> None:
    MODELOS.mkdir(parents=True, exist_ok=True)
    RUTA_METRICAS.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")


def leer_metricas() -> dict:
    if not RUTA_METRICAS.exists():
        return {}
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
