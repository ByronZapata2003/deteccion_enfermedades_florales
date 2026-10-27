"""Persistencia de diagnósticos: MongoDB si está disponible y, si no, un JSON local."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pymongo import MongoClient
from pymongo.errors import PyMongoError

ROOT = Path(__file__).resolve().parents[1]


def ahora() -> datetime:
    return datetime.now(timezone.utc)


def ruta_json() -> Path:
    return Path(os.getenv("DIAGNOSTICOS_PATH", str(ROOT / "data" / "diagnosticos.json")))


def mongo_disponible() -> bool:
    try:
        cliente = MongoClient(
            os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"),
            serverSelectionTimeoutMS=800,
            connectTimeoutMS=800,
        )
        cliente.admin.command("ping")
        cliente.close()
        return True
    except Exception:
        return False


def backend_activo() -> str:
    modo = os.getenv("PERSISTENCE", "auto").strip().lower()
    if modo == "json":
        return "json"
    if mongo_disponible():
        return "mongo"
    if modo == "mongo":
        raise PyMongoError("MongoDB no responde")
    return "json"


def _coleccion():
    cliente = MongoClient(
        os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"),
        serverSelectionTimeoutMS=1500,
        connectTimeoutMS=1500,
    )
    cliente.admin.command("ping")
    base = os.getenv("MONGO_DB", "enfermedades_florales")
    return cliente, cliente[base]["diagnosticos"]


def _leer_json() -> list[dict]:
    ruta = ruta_json()
    if not ruta.exists():
        return []
    return json.loads(ruta.read_text(encoding="utf-8"))


def _escribir_json(filas: list[dict]) -> None:
    ruta = ruta_json()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")


def guardar(documento: dict) -> dict:
    creado = ahora()
    cuerpo = {**documento, "created_at": creado}
    destino = backend_activo()
    if destino == "mongo":
        cliente, coleccion = _coleccion()
        try:
            insertado = coleccion.insert_one(cuerpo)
        finally:
            cliente.close()
        cuerpo["id"] = str(insertado.inserted_id)
    else:
        filas = _leer_json()
        cuerpo["id"] = uuid4().hex
        filas.append({**cuerpo, "created_at": creado.isoformat()})
        _escribir_json(filas)
    cuerpo["created_at"] = creado.isoformat()
    cuerpo["persistencia"] = destino
    return cuerpo


def listar(limite: int = 50) -> list[dict]:
    destino = backend_activo()
    if destino == "mongo":
        cliente, coleccion = _coleccion()
        try:
            cursor = coleccion.find({}, {"imagen": 0}).sort("created_at", -1).limit(limite)
            filas = []
            for doc in cursor:
                filas.append(
                    {
                        "id": str(doc.pop("_id")),
                        "created_at": doc["created_at"].isoformat(),
                        "archivo": doc.get("archivo", ""),
                        "etiqueta": doc.get("etiqueta", ""),
                        "confianza": doc.get("confianza", 0),
                        "persistencia": "mongo",
                    }
                )
            return filas
        finally:
            cliente.close()
    filas = list(reversed(_leer_json()))[:limite]
    return [
        {
            "id": fila["id"],
            "created_at": fila["created_at"],
            "archivo": fila.get("archivo", ""),
            "etiqueta": fila.get("etiqueta", ""),
            "confianza": fila.get("confianza", 0),
            "persistencia": "json",
        }
        for fila in filas
    ]


def obtener(identificador: str) -> dict | None:
    destino = backend_activo()
    if destino == "mongo":
        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            oid = ObjectId(identificador)
        except (InvalidId, TypeError):
            return None
        cliente, coleccion = _coleccion()
        try:
            doc = coleccion.find_one({"_id": oid})
        finally:
            cliente.close()
        if not doc:
            return None
        doc["id"] = str(doc.pop("_id"))
        doc["created_at"] = doc["created_at"].isoformat()
        doc["persistencia"] = "mongo"
        return doc
    for fila in _leer_json():
        if fila["id"] == identificador:
            return {**fila, "persistencia": "json"}
    return None


def eliminar(identificador: str) -> bool:
    destino = backend_activo()
    if destino == "mongo":
        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            oid = ObjectId(identificador)
        except (InvalidId, TypeError):
            return False
        cliente, coleccion = _coleccion()
        try:
            resultado = coleccion.delete_one({"_id": oid})
        finally:
            cliente.close()
        return resultado.deleted_count == 1
    filas = _leer_json()
    nuevas = [fila for fila in filas if fila["id"] != identificador]
    if len(nuevas) == len(filas):
        return False
    _escribir_json(nuevas)
    return True
