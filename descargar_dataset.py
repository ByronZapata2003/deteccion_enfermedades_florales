"""Descarga un subconjunto equilibrado de RoseLeafSet (hojas de rosal).

Fuente: Jinia et al. (2025), Mendeley Data, CC BY 4.0.
https://doi.org/10.17632/9G668BFHY5.3
Réplica usada: Project-AgML/roseleafset_disease_classification (config augmented).

El subconjunto ya puede venir incluido en data/processed. Este script permite regenerarlo.
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DESTINO = ROOT / "data" / "processed"
URL = (
    "https://huggingface.co/datasets/Project-AgML/"
    "roseleafset_disease_classification/resolve/main/augmented/train-00000-of-00001.parquet"
)
ETIQUETAS = {
    0: ("mancha_negra", "Black Spot"),
    1: ("hoja_seca", "Dry Leaf"),
    2: ("hoja_sana", "Healthy Leaf"),
    3: ("perforaciones", "Leaf Holes"),
}
POR_CLASE = 180
PASO = 12
LADO = 160


def _indices_deseados(por_clase: int, paso: int) -> dict[int, int]:
    deseados = {}
    for clase in range(4):
        inicio = clase * 2500
        for copia in range(por_clase):
            deseados[inicio + copia * paso] = clase
    return deseados


def _guardar(imagen_bytes: bytes, ruta: Path) -> None:
    imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    imagen.thumbnail((LADO, LADO))
    lienzo = Image.new("RGB", (LADO, LADO), (0, 0, 0))
    lienzo.paste(imagen, ((LADO - imagen.width) // 2, (LADO - imagen.height) // 2))
    lienzo.save(ruta, format="JPEG", quality=80, optimize=True)


def ya_descargado(por_clase: int) -> bool:
    meta = DESTINO / "metadata.csv"
    if not meta.exists():
        return False
    with meta.open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    conteo = {clave: 0 for clave in ("mancha_negra", "hoja_seca", "hoja_sana", "perforaciones")}
    for fila in filas:
        conteo[fila["clase"]] = conteo.get(fila["clase"], 0) + 1
        if not (DESTINO / fila["archivo"]).exists():
            return False
    return all(valor >= por_clase for valor in conteo.values())


def extraer(parquet: Path, por_clase: int, paso: int) -> int:
    import pyarrow.parquet as pq

    DESTINO.mkdir(parents=True, exist_ok=True)
    for carpeta in ETIQUETAS.values():
        (DESTINO / carpeta[0]).mkdir(parents=True, exist_ok=True)
    deseados = _indices_deseados(por_clase, paso)
    guardados = {clave: 0 for clave, _ in ETIQUETAS.values()}
    filas = []
    indice = 0
    archivo_parquet = pq.ParquetFile(parquet)
    for lote in archivo_parquet.iter_batches(batch_size=32, columns=["image", "label"]):
        imagenes = lote.column("image").to_pylist()
        for imagen in imagenes:
            clase_id = deseados.get(indice)
            indice += 1
            if clase_id is None:
                continue
            clase, original = ETIQUETAS[clase_id]
            nombre = f"{clase}/{clase}_{guardados[clase]:03d}.jpg"
            _guardar(imagen["bytes"], DESTINO / nombre)
            guardados[clase] += 1
            filas.append(
                {
                    "archivo": nombre.replace("\\", "/"),
                    "clase": clase,
                    "clase_id": clase_id,
                    "etiqueta_original": original,
                    "indice_fuente": indice - 1,
                }
            )
            if sum(guardados.values()) == len(deseados):
                break
        if sum(guardados.values()) == len(deseados):
            break
    with (DESTINO / "metadata.csv").open("w", encoding="utf-8", newline="") as salida:
        escritor = csv.DictWriter(
            salida,
            fieldnames=["archivo", "clase", "clase_id", "etiqueta_original", "indice_fuente"],
        )
        escritor.writeheader()
        escritor.writerows(filas)
    return len(filas)


def descargar_parquet(destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando RoseLeafSet augmented (~383 MB) a {destino} ...")
    urllib.request.urlretrieve(URL, destino)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el subconjunto local de hojas de rosal.")
    parser.add_argument("--parquet", type=Path, default=ROOT / "data" / "raw" / "roseleafset_augmented.parquet")
    parser.add_argument("--por-clase", type=int, default=POR_CLASE)
    parser.add_argument("--forzar", action="store_true")
    args = parser.parse_args()
    if ya_descargado(args.por_clase) and not args.forzar:
        print(f"El subconjunto ya está en {DESTINO}. Usa --forzar para regenerarlo.")
        return
    if not args.parquet.exists():
        descargar_parquet(args.parquet)
    total = extraer(args.parquet, args.por_clase, PASO)
    print(f"Imágenes listas: {total} en {DESTINO}")


if __name__ == "__main__":
    main()
