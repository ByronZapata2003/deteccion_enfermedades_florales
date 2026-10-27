"""Interfaz local para subir una hoja de rosal y ver el diagnóstico educativo."""
from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Enfermedades florales", page_icon="🌹", layout="wide")
st.title("Detección de enfermedades florales")
st.caption(
    "Clasifica una foto de hoja de rosal en mancha negra, hoja seca, hoja sana o perforaciones. "
    "Prototipo de laboratorio: no sustituye a una persona especialista."
)


def explicar(exc: Exception) -> None:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            st.error(exc.response.json().get("detail", str(exc)))
            return
        except Exception:
            pass
    st.error(f"No se pudo usar la API en {API}. Revisa que esté encendida. Detalle: {exc}")


def predecir_local(nombre: str, contenido: bytes, tipo: str) -> dict | None:
    import io
    import sys

    sys.path.insert(0, str(ROOT))
    from PIL import Image
    from backend.ml import predecir

    imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
    resultado = predecir(imagen)
    resultado.update(
        {
            "id": "local",
            "created_at": "",
            "archivo": nombre,
            "persistencia": "no guardado",
        }
    )
    return resultado


def mostrar(resultado: dict) -> None:
    st.success(f"{resultado['etiqueta']} · confianza {resultado['confianza']:.0%}")
    st.write(resultado["tipo"])
    st.info(resultado["recomendacion"])
    st.warning(resultado["aviso"])
    st.bar_chart(resultado["probabilidades"])
    with st.expander("Detalle"):
        st.json(resultado)


tab_nueva, tab_historial, tab_datos = st.tabs(["Diagnosticar", "Historial", "Dataset y métricas"])

with tab_nueva:
    archivo = st.file_uploader("Foto de la hoja (JPG, PNG o WEBP, máximo 10 MB)", type=["jpg", "jpeg", "png", "webp"])
    columna_img, columna_res = st.columns(2)
    if archivo is not None:
        columna_img.image(archivo, caption=archivo.name, use_container_width=True)
    if st.button("Analizar hoja", type="primary", disabled=archivo is None):
        try:
            with st.spinner("Clasificando..."):
                respuesta = requests.post(
                    API + "/diagnosticos",
                    files={"file": (archivo.name, archivo.getvalue(), archivo.type or "image/jpeg")},
                    timeout=60,
                )
                respuesta.raise_for_status()
                st.session_state["resultado"] = respuesta.json()
        except requests.RequestException as exc:
            st.warning("La API no respondió. Se calcula el resultado en local, sin guardarlo.")
            explicar(exc)
            try:
                st.session_state["resultado"] = predecir_local(archivo.name, archivo.getvalue(), archivo.type or "")
            except Exception as local_exc:
                st.error(f"Tampoco se pudo predecir en local: {local_exc}")
    if st.session_state.get("resultado"):
        with columna_res:
            mostrar(st.session_state["resultado"])

with tab_historial:
    st.caption("El historial vive en la API. Con MongoDB apagado se guarda en data/diagnosticos.json.")
    if st.button("Actualizar historial"):
        try:
            respuesta = requests.get(API + "/diagnosticos", timeout=10)
            respuesta.raise_for_status()
            st.session_state["historial"] = respuesta.json()
        except requests.RequestException as exc:
            explicar(exc)
    for item in st.session_state.get("historial", []):
        with st.expander(f"{item.get('created_at', '')} · {item.get('etiqueta', '')} · {item.get('archivo', '')}"):
            izquierda, derecha = st.columns([4, 1])
            try:
                detalle = requests.get(API + "/diagnosticos/" + item["id"], timeout=10)
                detalle.raise_for_status()
                izquierda.json(detalle.json())
            except requests.RequestException as exc:
                explicar(exc)
            if derecha.button("Eliminar", key="del_" + item["id"]):
                try:
                    borrado = requests.delete(API + "/diagnosticos/" + item["id"], timeout=10)
                    borrado.raise_for_status()
                    st.session_state["historial"] = [
                        fila for fila in st.session_state.get("historial", []) if fila["id"] != item["id"]
                    ]
                    st.rerun()
                except requests.RequestException as exc:
                    explicar(exc)

with tab_datos:
    st.subheader("Ejemplos del subconjunto RoseLeafSet")
    muestras = []
    for clase in ("mancha_negra", "hoja_seca", "hoja_sana", "perforaciones"):
        carpeta = ROOT / "data" / "processed" / clase
        imagenes = sorted(carpeta.glob("*.jpg"))
        if imagenes:
            muestras.append((clase, imagenes[0]))
    if muestras:
        columnas = st.columns(len(muestras))
        for columna, (clase, ruta) in zip(columnas, muestras):
            columna.image(str(ruta), caption=clase.replace("_", " "), use_container_width=True)
    st.subheader("Métricas guardadas")
    try:
        respuesta = requests.get(API + "/metricas", timeout=10)
        respuesta.raise_for_status()
        datos = respuesta.json()
    except requests.RequestException:
        import json

        ruta = ROOT / "models" / "metricas.json"
        datos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None
        if datos is None:
            st.info("Todavía no hay métricas. Entrena con python entrenar_modelo.py.")
    if datos:
        filas = [
            {"modelo": item["nombre"], "accuracy": item["accuracy"], "f1_macro": item["f1_macro"]}
            for item in datos.get("modelos", [])
        ]
        st.dataframe(filas, use_container_width=True)
        st.caption(datos.get("nota", ""))
