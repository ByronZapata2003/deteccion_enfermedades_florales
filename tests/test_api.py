import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import api

cliente = TestClient(api.app)


@pytest.fixture(autouse=True)
def almacen_json(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSISTENCE", "json")
    monkeypatch.setenv("DIAGNOSTICOS_PATH", str(tmp_path / "diagnosticos.json"))


def png_blanco() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), (30, 140, 40)).save(buffer, "PNG")
    return buffer.getvalue()


def test_rechaza_tipo_incorrecto():
    respuesta = cliente.post("/diagnosticos", files={"file": ("nota.txt", b"hola", "text/plain")})
    assert respuesta.status_code == 415


def test_rechaza_imagen_danada():
    respuesta = cliente.post("/diagnosticos", files={"file": ("hoja.png", b"no soy imagen", "image/png")})
    assert respuesta.status_code == 422


def test_diagnostico_se_guarda_y_se_consulta():
    respuesta = cliente.post("/diagnosticos", files={"file": ("hoja.png", png_blanco(), "image/png")})
    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["clase"] in {"mancha_negra", "hoja_seca", "hoja_sana", "perforaciones"}
    assert 0 <= cuerpo["confianza"] <= 1
    assert cuerpo["persistencia"] == "json"
    assert "especialista" in cuerpo["aviso"]
    listado = cliente.get("/diagnosticos")
    assert listado.status_code == 200
    assert listado.json()[0]["id"] == cuerpo["id"]
    detalle = cliente.get("/diagnosticos/" + cuerpo["id"])
    assert detalle.status_code == 200
    assert detalle.json()["etiqueta"] == cuerpo["etiqueta"]
    borrado = cliente.delete("/diagnosticos/" + cuerpo["id"])
    assert borrado.status_code == 200
    assert cliente.get("/diagnosticos/" + cuerpo["id"]).status_code == 404


def test_limite_invalido():
    assert cliente.get("/diagnosticos?limit=0").status_code == 422
    assert cliente.get("/diagnosticos?limit=201").status_code == 422


def test_id_invalido():
    assert cliente.delete("/diagnosticos/id con espacios").status_code == 422
    assert cliente.get("/diagnosticos/no-existe").status_code == 404


def test_health_en_modo_json():
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["persistencia"] == "json"
    assert len(respuesta.json()["clases"]) == 4
