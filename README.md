# Evaluación 2 - Inteligencia de Negocios

## 🚀 Configuración del Entorno

Recomiendo usar un ambiente virtual e instalar el contenido de "requirements.txt":

```bash
pip install -r requirements.txt
```

**Versión de Python utilizada:** 3.10.11

### 1️⃣ Limpieza de Datos

Ejecutar Limpieza con el Script de "LimpiarDataset.py" lo que creará el dataset limpio a partir del archivo "mermas_actividad_unidad_2.xlsx"
Este nuevo archivo se almacena en el directorio de "/dataset" junto con un reporte de los cambios.

```
/dataset/
├── mermas_actividad_unidad_2.xlsx  # Dataset original
├── mermas_limpio.csv               # Dataset limpio para análisis
```

### 2️⃣ Entrenamiento de Modelos

Ejecutar el Script "script_modificado.py" este entrenará 3 modelos: Linear Regression, Random Forest y CatBoost
Al finalizar la ejecución del script los resultados generados se mostrarán por terminal al igual que en el directorio "/Resultados"

## 📂 Directorio Collab

Dentro de esta se encuentran scripts distintos donde el principal es "optimizacion_catboost_colab.py" el cual busca y entrega los mejores hiperparámetros para el modelo CatBoost usando GPU en Google Colab.


