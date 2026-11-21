# TP2 — Perceptrón multicapa (MLP) sobre 20 Newsgroups (TNG)

Redes Neuronales -  Fernández San Juan

## Qué hace

Continúa el TP1 (regresión softmax) sobre el mismo dataset (~18.800 posts de foros, 20 categorías), reemplazando la capa lineal única por un **perceptrón multicapa**.

- Mismo pipeline: preprocesamiento configurable, TF-IDF, PyTorch, MLflow, TensorBoard y Optuna.
- Búsqueda con Optuna, con pruning, de:
  - preprocesamiento, learning rate y batch size;
  - cantidad de capas (1 a 3) y de neuronas;
  - función de activación;
  - optimizador (Adam, RMSprop, SGD) con sus momentos de orden 1 y 2.
- Comparación controlada de optimizadores y activaciones (una cosa por vez).
- Experimento de overfitting variando el tamaño de la red.
- Entrenamiento final con train / validación / test separados, early stopping sobre validación y 3 semillas.
- Comparación controlada con la regresión softmax: mismo presupuesto de búsqueda y mismo protocolo final.

## Archivos

| Archivo | Contenido |
|---|---|
| `TP2_MLP_FernandezSanJuan.ipynb` | Notebook del trabajo (versión corregida), ejecutada con todas las salidas |
| `requirements.txt` | Librerías necesarias |
| `README.md` | Este archivo |

## Cómo correrlo

**En Google Colab (recomendado):** abrir la notebook, elegir *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU* y luego *Ejecutar todo*. La primera celda instala lo que falte. En CPU (como la corrida entregada) tarda unos 30 minutos en total, sobre todo por la búsqueda del MLP (11,8 min) y el experimento de overfitting con la red 3×1024 (10,6 s por época).

**Localmente** (Python 3.10 o superior):

```
pip install -r requirements.txt
jupyter notebook TP2_MLP_FernandezSanJuan.ipynb
```

El dataset se descarga solo con `sklearn.datasets.fetch_20newsgroups`. Las semillas están fijas (incluida la del sampler de Optuna), aunque en GPU puede haber diferencias mínimas entre corridas. Para ver los experimentos: `mlflow ui`; para las curvas del experimento de overfitting: `tensorboard --logdir runs`.


