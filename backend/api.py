"""API del detector de condiciones foliares en rosal."""
from __future__ import annotations

import io
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from pymongo.errors import PyMongoError

from backend.almacen import backend_activo, eliminar, guardar, listar, mongo_disponible, obtener
from backend.ml import CATALOGO, CLASES, leer_metricas, predecir

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024
TIPOS = {"image/png", "image/jpeg", "image/webp"}
FORMATOS = {"PNG", "JPEG", "WEBP"}

app = FastAPI(
    title="Detección de enfermedades florales",
    version="1.0.0",
    description="Prototipo educativo para clasificar hojas de rosal. No es un diagnóstico agronómico.",
)


class DiagnosticoOut(BaseModel):
    id: str
    created_at: str
    archivo: str
    clase: str
    etiqueta: str
    tipo: str
    confianza: float
    probabilidades: dict[str, float]
    recomendacion: str
    aviso: str
    persistencia: str
    referencia: str = ""


def leer_imagen(data: bytes, content_type: str) -> Image.Image:
    if not data:
        raise HTTPException(422, "El archivo está vacío.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"La imagen supera el límite de {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    if content_type not in TIPOS:
        raise HTTPException(415, "Formato no admitido. Sube JPG, PNG o WEBP.")
    try:
        imagen = Image.open(io.BytesIO(data))
        imagen.verify()
        imagen = Image.open(io.BytesIO(data))
        if imagen.format not in FORMATOS:
            raise HTTPException(415, "El contenido no coincide con JPG, PNG o WEBP.")
        if imagen.width * imagen.height > 25_000_000:
            raise HTTPException(413, "La imagen supera el máximo de 25 megapíxeles.")
        return imagen.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "No se pudo leer la imagen. Comprueba que el archivo no esté dañado.")


@app.get("/health")
def health():
    try:
        destino = backend_activo()
        error = None
    except PyMongoError as exc:
        destino = "mongo"
        error = str(exc)
    return {
        "status": "ok" if error is None else "degraded",
        "persistencia": destino,
        "mongo_disponible": mongo_disponible(),
        "clases": list(CLASES),
        "detalle": error,
    }


@app.get("/clases")
def clases():
    return [{"id": clave, **CATALOGO[clave]} for clave in CLASES]


@app.get("/metricas")
def metricas():
    datos = leer_metricas()
    if not datos:
        raise HTTPException(404, "Aún no hay métricas. Ejecuta python entrenar_modelo.py.")
    return datos


@app.post("/diagnosticos", response_model=DiagnosticoOut)
async def crear_diagnostico(file: UploadFile = File(...)):
    if file.content_type not in TIPOS:
        raise HTTPException(415, "Formato no admitido. Sube JPG, PNG o WEBP.")
    data = await file.read(MAX_IMAGE_BYTES + 1)
    imagen = leer_imagen(data, file.content_type or "")
    try:
        resultado = predecir(imagen)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    try:
        guardado = guardar({"archivo": file.filename or "captura.jpg", **resultado})
    except PyMongoError:
        raise HTTPException(
            503,
            "MongoDB no está disponible y PERSISTENCE=mongo. Inicia MongoDB o usa PERSISTENCE=auto en .env.",
        )
    return guardado


@app.get("/diagnosticos")
def obtener_diagnosticos(limit: int = 50):
    if not 1 <= limit <= 200:
        raise HTTPException(422, "limit debe estar entre 1 y 200.")
    try:
        return listar(limit)
    except PyMongoError:
        raise HTTPException(503, "MongoDB no está disponible. Inícialo o cambia PERSISTENCE=json.")


@app.get("/diagnosticos/{identificador}")
def obtener_uno(identificador: str):
    if not identificador or len(identificador) > 64:
        raise HTTPException(422, "Identificador inválido.")
    try:
        doc = obtener(identificador)
    except PyMongoError:
        raise HTTPException(503, "MongoDB no está disponible.")
    if not doc:
        raise HTTPException(404, "No existe ese diagnóstico.")
    return doc


@app.delete("/diagnosticos/{identificador}")
def borrar(identificador: str):
    if not identificador or len(identificador) > 64 or not identificador.isalnum():
        raise HTTPException(422, "Identificador inválido.")
    try:
        ok = eliminar(identificador)
    except PyMongoError:
        raise HTTPException(503, "MongoDB no está disponible.")
    if not ok:
        raise HTTPException(404, "No existe ese diagnóstico.")
    return {"deleted": True, "id": identificador}
