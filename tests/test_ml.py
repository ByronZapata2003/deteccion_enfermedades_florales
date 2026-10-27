import io

from PIL import Image

from backend.ml import CLASES, extraer_caracteristicas, extraer_color, predecir


def _imagen() -> Image.Image:
    return Image.new("RGB", (80, 60), (20, 120, 40))


def test_dimensiones_de_caracteristicas():
    color = extraer_color(_imagen())
    completo = extraer_caracteristicas(_imagen())
    assert color.shape == (39,)
    assert completo.shape == (1607,)
    assert completo[:39].tolist() == color.tolist()


def test_prediccion_tiene_contrato_estable():
    resultado = predecir(_imagen())
    assert resultado["clase"] in CLASES
    assert set(resultado["probabilidades"]) == {
        "Mancha negra",
        "Hoja seca",
        "Hoja sana",
        "Perforaciones foliares",
    }
    assert abs(sum(resultado["probabilidades"].values()) - 1) < 0.02
    assert resultado["recomendacion"]


def test_imagen_real_del_dataset():
    from pathlib import Path

    ruta = Path(__file__).resolve().parents[1] / "data" / "processed" / "hoja_seca" / "hoja_seca_010.jpg"
    resultado = predecir(Image.open(ruta))
    assert resultado["clase"] == "hoja_seca"
    buffer = io.BytesIO()
    Image.open(ruta).save(buffer, format="PNG")
    assert buffer.getvalue()
