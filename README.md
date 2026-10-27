# Detección de enfermedades florales

Prototipo educativo para clasificar una foto de **hoja de rosal** en cuatro condiciones:

| Clase del modelo | Etiqueta | Qué representa |
| --- | --- | --- |
| `mancha_negra` | Mancha negra | Enfermedad fúngica foliar, asociada con frecuencia a *Diplocarpon rosae* |
| `hoja_seca` | Hoja seca | Síntoma (estrés, senescencia o daño). No identifica un solo patógeno |
| `hoja_sana` | Hoja sana | Hoja sin síntomas evidentes |
| `perforaciones` | Perforaciones foliares | Agujeros, a menudo por insectos masticadores u otro daño mecánico |

No es un diagnóstico agronómico ni sustituye la revisión de una persona especialista. El modelo incluido, entrenado en este subconjunto, quedó así en el test de 108 imágenes (semilla 42):

- baseline de mayoría: accuracy 0.250
- regresión logística solo con color: accuracy 0.824
- bosque aleatorio con color y textura HOG: accuracy 0.880 y F1 macro 0.874

La clase más difícil es la mancha negra: comparte necrosis oscura con las perforaciones y con la hoja seca.

## Arquitectura

- **Notebook** `Laboratorio_IA.ipynb`: exploración, entrenamiento, métricas e inferencia.
- **FastAPI** `backend/api.py`: valida la imagen, predice y guarda el diagnóstico.
- **Streamlit** `frontend/app.py`: carga la foto, muestra probabilidades, recomendación e historial.
- **Persistencia**: MongoDB si está encendido (`MONGO_URI`). Si no responde y `PERSISTENCE=auto`, se usa `data/diagnosticos.json`.
- **Modelo** `models/modelo.joblib`: scikit-learn, sin GPU.

## Entorno con Anaconda Navigator y cmd.exe (Python 3.11)

1. Abre **Anaconda Navigator**.
2. Entra en **Environments** y pulsa **Create**.
3. Nombre: `flores-ia`. Versión de Python: **3.11**. Crea el entorno.
4. En la fila `flores-ia`, abre el botón de reproducción y elige **Open Terminal**. Esa ventana es **cmd.exe** con el entorno ya activado.
5. Si en lugar de eso usas **Anaconda Prompt**, activa el entorno a mano:

```bat
conda activate flores-ia
```

Si `conda` no se reconoce dentro de cmd.exe, ciérralo, abre Anaconda Prompt una vez y ejecuta `conda init cmd.exe`. Cierra y vuelve a abrir cmd.exe.

6. Ve a la carpeta del proyecto (ajusta la ruta) e instala las librerías **con el entorno activo**:

```bat
cd /d C:\ruta\deteccion_enfermedades_florales
python -m pip install -r requirements.txt
copy .env.example .env
python -m ipykernel install --user --name flores-ia --display-name "Python 3.11 (flores-ia)"
```

Usa `python -m pip`, no el `pip` de otro entorno. El subconjunto de imágenes ya está en `data/processed`, así que no hace falta descargar las 10 000 fotos.

## Orden para trabajar

En el mismo cmd.exe, con `flores-ia` activo y situado en la carpeta del proyecto:

```bat
python entrenar_modelo.py
python -m jupyter notebook
```

En el navegador abre `Laboratorio_IA.ipynb` y elige el kernel **Python 3.11 (flores-ia)**. Ejecuta las celdas de arriba hacia abajo. No lances el Jupyter del entorno `base` de Navigator: ese kernel no tiene las librerías de este proyecto.

Para la API y la interfaz hacen falta otras dos ventanas de cmd.exe, cada una con `conda activate flores-ia` y `cd` a la misma carpeta.

**Ventana API**

```bat
python -m uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Documentación interactiva: http://127.0.0.1:8000/docs

**Ventana interfaz**

```bat
python -m streamlit run frontend/app.py
```

Abre la URL que imprime Streamlit (normalmente http://localhost:8501). Sube una foto de hoja. Si la API está apagada, la pantalla avisa y puede clasificar en local, pero ese resultado no queda en el historial.

## MongoDB en Windows (opcional)

Con `PERSISTENCE=auto` el laboratorio funciona sin MongoDB. Para usar la base que pide la arquitectura:

1. Instala [MongoDB Community Server](https://www.mongodb.com/try/download/community) para Windows.
2. Crea la carpeta de datos, por ejemplo `C:\data\db`.
3. En otra ventana:

```bat
mongod --dbpath C:\data\db
```

4. Compass, si lo instalas, se conecta a `mongodb://127.0.0.1:27017`. La base `enfermedades_florales` y la colección `diagnosticos` aparecen al guardar el primer resultado.
5. En `.env` deja `PERSISTENCE=auto` o pon `PERSISTENCE=mongo` si quieres que la API falle con un mensaje claro cuando MongoDB esté apagado.

## Dataset

RoseLeafSet: fotos reales de hojas de rosal tomadas en viveros de Savar, Dhaka (Jinia, Rahman, Mojumdar y Bijoy, 2025). Licencia **CC BY 4.0**. DOI: https://doi.org/10.17632/9G668BFHY5.3

Aquí se usa un subconjunto de la configuración *augmented* publicada en Hugging Face por Project-AgML: 180 imágenes por clase (720 en total), tomadas con un salto para no quedarse solo con copias consecutivas, y reducidas a 160 px. El detalle está en `dataset_fuente.json`.

Para regenerar el subconjunto (descarga unos 383 MB):

```bat
python descargar_dataset.py
python entrenar_modelo.py
```

Las imágenes augmented pueden ser transformaciones de la misma hoja. Una partición aleatoria puede inflar la accuracy respecto a un cultivo nuevo. Por eso el resultado se presenta como línea base de laboratorio.

## Pruebas

```bat
python -m pytest -q
```

Cubren formato inválido, imagen dañada, predicción, guardado JSON, consulta, borrado y el tamaño de los vectores de características. No necesitan MongoDB ni la API encendida.

## Límites

- Solo cuatro etiquetas de **rosal**, no todas las enfermedades florales.
- La mancha negra se confunde con necrosis de otros tipos.
- Fondo, luz y especie distintos a los del dataset bajan el rendimiento.
- La recomendación es orientativa y no indica dosis de ningún producto.
- No expongas la API a internet: no tiene autenticación.
