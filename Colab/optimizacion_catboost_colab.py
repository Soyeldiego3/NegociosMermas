# OPTIMIZACIÓN DE HIPERPARÁMETROS CATBOOST PARA GOOGLE COLAB
# Script para encontrar los mejores hiperparámetros usando GPU en Google Colab
# Autor: Generado para análisis predictivo de mermas
# Fecha: 2025

# ===================================================================
# INSTALACIÓN DE DEPENDENCIAS (ejecutar en la primera celda de Colab)
# ===================================================================
"""
# Ejecutar esta celda PRIMERO en Google Colab:
!pip install catboost optuna pandas numpy scikit-learn matplotlib seaborn
!pip install --upgrade catboost

# Verificar GPU disponible
!nvidia-smi
"""

# ===================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ===================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings
warnings.filterwarnings('ignore')

print("✅ Librerías importadas correctamente")
print("🔥 Iniciando optimización de hiperparámetros CatBoost con GPU")

# ===================================================================
# CONFIGURACIÓN DEL DATASET
# ===================================================================
# CAMBIAR ESTA RUTA POR LA UBICACIÓN DE TU DATASET EN COLAB
DATASET_PATH = '/content/mermas_limpio.csv'  # 👈 CAMBIAR AQUÍ TU RUTA

print(f"📊 Cargando dataset desde: {DATASET_PATH}")

# Cargar y preparar datos
try:
    data = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset cargado exitosamente: {data.shape[0]} filas x {data.shape[1]} columnas")
except FileNotFoundError:
    print("❌ ERROR: No se encontró el archivo del dataset")
    print("🔧 SOLUCIÓN: Sube tu archivo 'mermas_limpio.csv' a Colab y actualiza la variable DATASET_PATH")
    exit()

# ===================================================================
# PREPARACIÓN DE DATOS
# ===================================================================
print("🔧 Preparando datos...")

# Convertir fecha a datetime y extraer características temporales
data['fecha'] = pd.to_datetime(data['fecha'], errors='coerce')
data['año'] = data['fecha'].dt.year
data['mes_num'] = data['fecha'].dt.month

# Definir características (mismas que en el script original)
features = [
    'negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 
    'comuna', 'tienda', 'motivo', 'mes_num', 'año'
]

# Verificar que todas las columnas existan
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"❌ ERROR: Las siguientes columnas no están en el dataset: {missing_features}")
    print(f"📋 Columnas disponibles: {list(data.columns)}")
    exit()

# Preparar X e y
X = data[features]

# ===================================================================
# ANÁLISIS DE VARIABLE OBJETIVO
# ===================================================================
if 'merma_unidad_p' in data.columns:
    y = data['merma_unidad_p']
    print(f"✅ Usando 'merma_unidad_p' como variable objetivo")
else:
    y = data['merma_unidad']
    print(f"✅ Usando 'merma_unidad' como variable objetivo")

# Identificar variables categóricas para CatBoost
categorical_features = [
    'negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 
    'comuna', 'tienda', 'motivo'
]

# Obtener índices de las características categóricas
cat_features_idx = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]

print(f"✅ Características categóricas identificadas: {len(cat_features_idx)} variables")
print(f"📊 Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"🔄 División completada:")
print(f"   📈 Entrenamiento: {X_train.shape[0]} muestras")
print(f"   🧪 Prueba: {X_test.shape[0]} muestras")

# ===================================================================
# FUNCIÓN OBJETIVO BASADA EN TUS CONFIGURACIONES (OPTIMIZA R²)
# ===================================================================
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 2200),  # Cerca de tu 1800
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02),  # Cerca de tu 0.01
        'depth': trial.suggest_int('depth', 3, 6),  # Cerca de tu 4
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5),
        'task_type': 'GPU',
        'verbose': False,
        'random_state': 42
    }
    
    try:
        model = CatBoostRegressor(**params)
        # Pasar cat_features en fit(), no en el constructor
        model.fit(X_train, y_train, cat_features=cat_features_idx)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Retornar -R² para que Optuna maximice R² (más cercano a 1 = mejor)
        return -r2
        
    except Exception as e:
        return float('inf')

# ===================================================================
# OPTIMIZACIÓN (MAXIMIZAR R²)
# ===================================================================
study = optuna.create_study(direction='minimize')  # Minimiza -R² = Maximiza R²
study.optimize(objective, n_trials=50, timeout=3600)

best_params = study.best_params
best_r2 = -study.best_value  # Convertir de vuelta a R² positivo

# ===================================================================
# MODELO FINAL Y RESULTADOS
# ===================================================================
print(f"\n🏆 MEJOR R²: {best_r2:.4f} ({best_r2*100:.1f}% de varianza explicada)")

print(f"\n⚙️ MEJORES HIPERPARÁMETROS:")
for param, value in best_params.items():
    if isinstance(value, float):
        print(f"   {param}: {value:.4f}")
    else:
        print(f"   {param}: {value}")

final_model = CatBoostRegressor(**best_params, verbose=False)
final_model.fit(X_train, y_train, cat_features=cat_features_idx)
y_pred = final_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 MÉTRICAS FINALES:")
print(f"   R²: {r2:.4f} ({r2*100:.1f}% varianza explicada)")
print(f"   RMSE: {rmse:.4f} unidades")
print(f"   MAE: {mae:.4f} unidades")

# ===================================================================
# IMPORTANCIA DE CARACTERÍSTICAS
# ===================================================================
importances = final_model.get_feature_importance()
importance_df = pd.DataFrame({
    'Característica': X.columns,
    'Porcentaje': (importances / importances.sum()) * 100
}).sort_values('Porcentaje', ascending=False)

print(f"\n📊 IMPORTANCIA DE CARACTERÍSTICAS:")
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{i:2d}. {row['Característica']:<15} {row['Porcentaje']:6.2f}%")

# Gráfico simple
plt.figure(figsize=(10, 6))
top_features = importance_df.head(8)
plt.barh(range(len(top_features)), top_features['Porcentaje'])
plt.yticks(range(len(top_features)), top_features['Característica'])
plt.xlabel('Importancia (%)')
plt.title('Características Más Importantes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Hiperparámetros para copiar
print(f"\n📋 COPIAR HIPERPARÁMETROS:")
print("best_params = {")
for param, value in best_params.items():
    if isinstance(value, str):
        print(f"    '{param}': '{value}',")
    else:
        print(f"    '{param}': {value},")
print("}")
