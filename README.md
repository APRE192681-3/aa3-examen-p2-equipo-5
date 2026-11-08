# aa3-examen-p2-equipo-5

Examen Parcial II · IN1042 Aprendizaje Automático III · Otoño 2026 · Grupo A7
Equipo 5.

Dos casos de empresas ficticias: **CardioRitmo** (tamizaje de latidos ECG) y **Compra Fácil Bajío** (detección de reseñas negativas). Cada caso tiene su cuaderno DEV, su modelo guardado y su folleto en PDF.

## Integrantes y quién hizo qué

| Integrante | Responsabilidad |
|---|---|
| Andrea | Problema 1 (ECG): `ecg/DEV_ecg.ipynb`, `resultados_ecg.csv`, `modelo/`, borrador de limitaciones; estructura del repo y README |
| Paula | Problema 2 (reseñas): `resenas/DEV_resenas.ipynb`, `resultados_resenas.csv`, `modelo/` |
| Diana | Folletos en PDF (ambos) y maquetas PMV (`pmv/`) |

## Modelo elegido

| Problema | Modelo | Params | Cumple specs |
|---|---|---|---|
| ECG (CardioRitmo) | Conv1D | 5,857 (límite 20,000) | Sí: sensibilidad 98.6 %, especificidad 99.7 %, AUC 0.999 |
| Reseñas (Compra Fácil Bajío) | TF-IDF (palabras y bigramas) + red densa | 1,280,129 | Sí: exhaustividad 92.0 %, precisión 91.1 %, exactitud 91.5 % |

Todas las métricas son en la prueba del cliente, con umbral 0.5, usada una sola vez al final.

- **ECG:** se comparó contra línea base, TCN causal y BiGRU; los tres modelos cumplen. Se eligió Conv1D (≈107 KB, muy por debajo del límite); la justificación completa está en el cuaderno. La prueba tiene 416 latidos anormales y 584 normales; el modelo falla en 6 anormales y 2 normales.
- **Reseñas:** se comparó contra línea base, Embedding + Bi-LSTM y Embedding + Conv1D; los tres modelos cumplen. Se eligió TF-IDF + red densa: mayor precisión y exactitud que los otros dos, con exhaustividad sobre 90 %; la justificación completa está en el cuaderno. Las reseñas de 3 estrellas se quitaron del entrenamiento (decisión por validación, justificada en el cuaderno).

Los resultados completos están en `ecg/resultados_ecg.csv` y `resenas/resultados_resenas.csv`.

## Estructura

```
README.md
requirements.txt
ecg/
  DEV_ecg.ipynb
  resultados_ecg.csv
  modelo/modelo_ecg.keras
  folleto_ecg.pdf
resenas/
  DEV_resenas.ipynb
  resultados_resenas.csv
  modelo/modelo_resenas.keras, preprocesador_resenas.joblib
  folleto_resenas.pdf
pmv/
  ecg.html
  resenas.html
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook
```

Abrir `ecg/DEV_ecg.ipynb` y `resenas/DEV_resenas.ipynb` y ejecutar cada uno de arriba abajo (Restart & Run All). Los cuadernos descargan los datos de las ligas indicadas en el examen. El cuaderno de reseñas fija `KERAS_BACKEND=torch` antes de importar Keras, por eso `requirements.txt` incluye `torch`.

Para usar el modelo de reseñas hay que cargar también `resenas/modelo/preprocesador_resenas.joblib` (el vectorizador TF-IDF con el que se entrenó).

## PMV (maquetas)

`pmv/ecg.html` y `pmv/resenas.html` son maquetas interactivas de la interfaz. Se abren con doble clic en el navegador, sin instalar nada.

**Importante:** las maquetas todavía usan un clasificador simulado ("Modo demostración") y no cargan los modelos guardados. No son una app web que use `modelo/`, así que no reclamamos el punto extra. El punto de conexión está marcado en el código con `CONEXIÓN: reemplazar por fetch('/predict'...)`.

## Modelos grandes

Ningún modelo pesa más de 100 MB (el mayor es `modelo_resenas.keras`, 15.4 MB), así que todos están en el repositorio. No aplica enlace externo.
