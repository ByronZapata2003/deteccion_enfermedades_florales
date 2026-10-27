"""Entrena el baseline de color y el modelo de color + textura, y guarda métricas."""
from __future__ import annotations

import csv
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.ml import (
    CATALOGO,
    CLASES,
    MODELOS,
    RUTA_BASELINE,
    RUTA_MODELO,
    extraer_caracteristicas,
    extraer_color,
    guardar_metricas,
)

ROOT = Path(__file__).resolve().parent
META = ROOT / "data" / "processed" / "metadata.csv"


def cargar_imagenes():
    if not META.exists():
        raise SystemExit("No hay metadata.csv. Ejecuta primero: python descargar_dataset.py")
    imagenes = []
    etiquetas = []
    with META.open(encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            ruta = ROOT / "data" / "processed" / fila["archivo"]
            imagenes.append(Image.open(ruta).convert("RGB"))
            etiquetas.append(fila["clase"])
    return imagenes, np.array(etiquetas)


def _reporte(nombre, y_verdad, y_pred) -> dict:
    return {
        "nombre": nombre,
        "accuracy": round(float(accuracy_score(y_verdad, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_verdad, y_pred, average="macro")), 4),
        "matriz_confusion": confusion_matrix(y_verdad, y_pred, labels=list(CLASES)).tolist(),
        "reporte": classification_report(y_verdad, y_pred, labels=list(CLASES), output_dict=True, zero_division=0),
    }


def main() -> None:
    imagenes, y = cargar_imagenes()
    indices = np.arange(len(y))
    idx_temp, idx_test = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.1765, random_state=42, stratify=y[idx_temp])

    print(f"Imágenes: {len(y)} | train {len(idx_train)} | val {len(idx_val)} | test {len(idx_test)}")
    print("Extrayendo características...")
    color = np.vstack([extraer_color(imagenes[i]) for i in range(len(imagenes))])
    completo = np.vstack([extraer_caracteristicas(imagenes[i]) for i in range(len(imagenes))])

    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    resultados = []

    mayoria = DummyClassifier(strategy="most_frequent")
    mayoria.fit(color[idx_train], y_train)
    resultados.append(_reporte("mayoria", y_test, mayoria.predict(color[idx_test])))

    baseline = Pipeline(
        [
            ("escala", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, random_state=42)),
        ]
    )
    baseline.fit(color[idx_train], y_train)
    resultados.append(_reporte("regresion_logistica_color", y_test, baseline.predict(color[idx_test])))

    bosque = RandomForestClassifier(
        n_estimators=200,
        max_depth=24,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    bosque.fit(completo[idx_train], y_train)
    pred_val = bosque.predict(completo[idx_val])
    pred_test = bosque.predict(completo[idx_test])
    resultados.append(_reporte("bosque_color_textura", y_test, pred_test))
    validacion = _reporte("bosque_color_textura_validacion", y_val, pred_val)

    MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": baseline, "clases": list(CLASES), "tipo": "color"}, RUTA_BASELINE)
    joblib.dump({"modelo": bosque, "clases": list(CLASES), "tipo": "color+hog"}, RUTA_MODELO)

    import sklearn

    metricas = {
        "dataset": "RoseLeafSet (subconjunto augmented, CC BY 4.0)",
        "clases": list(CLASES),
        "etiquetas": {clave: CATALOGO[clave]["etiqueta"] for clave in CLASES},
        "n_imagenes": int(len(y)),
        "n_train": int(len(idx_train)),
        "n_validacion": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "semilla": 42,
        "validacion": validacion,
        "modelos": resultados,
        "modelo_elegido": "bosque_color_textura",
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "nota": (
            "Las imágenes augmented pueden ser transformaciones de la misma hoja. "
            "La partición aleatoria puede ser optimista frente a un cultivo nuevo. "
            "No usar como diagnóstico de campo."
        ),
    }
    guardar_metricas(metricas)
    for item in resultados:
        print(f"{item['nombre']}: accuracy={item['accuracy']:.3f} f1_macro={item['f1_macro']:.3f}")
    print(f"Validación del bosque: accuracy={validacion['accuracy']:.3f}")
    print(f"Modelo guardado en {RUTA_MODELO.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
