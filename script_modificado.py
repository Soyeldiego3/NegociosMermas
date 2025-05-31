# IMPLEMENTACIÓN DE ANÁLISIS PREDICTIVO COMPLETO
# Utilizamos train.csv disponible en https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting (Es necesario 
# registrarse en la página de kaggle). Luego incorporar el archivo en el directorio de trabajo con python.
# Las librerias necesarias están en el archivo requirements.txt

# PASO 1: IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime as dt

#modelo Catboost para comparar
from catboost import CatBoostRegressor

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS. CASO PREDICTIVO DE VENTAS*")

# PASO 2: CARGA Y PREPARACIÓN DE DATOS
data = pd.read_csv('dataset/mermas_limpio.csv')

# Convertir fecha a datetime
data['fecha'] = pd.to_datetime(data['fecha'], errors='coerce')
data['año'] = data['fecha'].dt.year
data['mes_num'] = data['fecha'].dt.month

# PASO 3: SELECCIÓN DE CARACTERÍSTICAS
# Features Removidas: zonal, region, mes_num, año
features = [
    'negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 'comuna', 'tienda', 'motivo'
]
X = data[features]
y = data['merma_unidad_p'] # puede ser merma_monto o merma_unidad, dependiendo del objetivo del análisis

# PASO 4: DIVISIÓN DE DATOS
# 80% entrenamiento, 20% prueba. Este porcentaje es el habitual en la literatura para este tipo de modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PASO 5: PREPROCESAMIENTO
categorical_features = [
    'negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 'comuna', 'tienda','motivo'
]
numeric_features = []

# Crear preprocesador para manejar ambos tipos de variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# PASO 6: IMPLEMENTACIÓN DE MODELOS
# Modelo 1: Regresión Lineal. Este modelo es el habitual para este tipo de problemas debido a su simplicidad y interpretabilidad.
# En caso de mermas, es posible utilizar este modelo pero pueden explorar otros modelos mas eficientes.
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Modelo 2: Random Forest
num_estimators = 130  # Número de árboles en el bosque
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=num_estimators, random_state=42))
])

# Modelo 3: CatBoost
cat_features_idx = [X.columns.get_loc(col) for col in categorical_features]
pipeline_cb = CatBoostRegressor(
    iterations=2191,
    learning_rate=0.019156269747930194,
    depth=5,
    l2_leaf_reg=1.4568506876320768,
    cat_features=cat_features_idx,
    verbose=1,
    random_state=42,
    task_type='GPU'  # Cambia a 'GPU' si tienes una GPU compatible y drivers instalados
)

# PASO 7: ENTRENAMIENTO DE MODELOS
# Entrenamos ambos modelos
print("Entrenando Regresión Lineal...")
pipeline_lr.fit(X_train, y_train)

print("Entrenando Random Forest...")
pipeline_rf.fit(X_train, y_train)

print("Entrenando CatBoost...")
pipeline_cb.fit(X_train, y_train, cat_features=cat_features_idx)
print("CatBoost entrenado correctamente")


print("Modelos entrenados correctamente")

# -------------------------------------------------
# EVALUACIÓN DE LOS MODELOS
# -------------------------------------------------

print("\n=== EVALUACIÓN DE MODELOS PREDICTIVOS ===")

# PASO 8: REALIZAR PREDICCIONES CON LOS MODELOS ENTRENADOS
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)

y_pred_cb = pipeline_cb.predict(X_test)

# PASO 9: CALCULAR MÚLTIPLES MÉTRICAS DE EVALUACIÓN
# Error Cuadrático Medio (MSE)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)

mse_cb = mean_squared_error(y_test, y_pred_cb)

# Raíz del Error Cuadrático Medio (RMSE)
rmse_lr = np.sqrt(mse_lr)
rmse_rf = np.sqrt(mse_rf)

rmse_cb = np.sqrt(mse_cb)

# Error Absoluto Medio (MAE)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

mae_cb = mean_absolute_error(y_test, y_pred_cb)

# Coeficiente de Determinación (R²)
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

r2_cb = r2_score(y_test, y_pred_cb)

# NUEVO PASO: GUARDAR RESULTADOS DE PREDICCIÓN EN ARCHIVOS MARKDOWN
# Crear un DataFrame con las predicciones y valores reales
results_df = pd.DataFrame({
    'Valor_Real': y_test,
    'Prediccion_LR': y_pred_lr,
    'Prediccion_RF': y_pred_rf,
    'Prediccion_CB': y_pred_cb,
    'Error_LR': y_test - y_pred_lr,
    'Error_RF': y_test - y_pred_rf,
    'Error_CB': y_test - y_pred_cb,
    'Error_Porcentual_LR': ((y_test - y_pred_lr) / y_test) * 100,
    'Error_Porcentual_RF': ((y_test - y_pred_rf) / y_test) * 100,
    'Error_Porcentual_CB': ((y_test - y_pred_cb) / y_test) * 100
})

# Reiniciar el índice para añadir información de las características
results_df = results_df.reset_index(drop=True)

# Añadir algunas columnas con información de las características para mayor contexto
X_test_reset = X_test.reset_index(drop=True)
for feature in X_test.columns:
    results_df[feature] = X_test_reset[feature]

# Ordenar por valor real para facilitar la comparación
results_df = results_df.sort_values('Valor_Real', ascending=False)

# Guardar resultado para Regresión Lineal
with open('Resultados/prediccion_lr.md', 'w') as f:
    f.write('# Resultados de Predicción: Regresión Lineal\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_lr:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_lr:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_lr:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Regresión Lineal explica aproximadamente el {r2_lr*100:.1f}% de la variabilidad en las ventas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_lr:.2f} unidades.\n\n')
      # Mostrar muestra de predicciones (top 10)
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Comuna |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_LR']:.2f} | {row['Error_LR']:.2f} | {row['Error_Porcentual_LR']:.1f}% | {row['categoria']} | {row['comuna']} |\n")
    
    # Estadísticas de error
    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_LR"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_LR"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_LR"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_LR"].std():.2f}\n\n')
    
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')

# Guardar resultado para Random Forest
with open('Resultados/prediccion_rf.md', 'w') as f:
    f.write('# Resultados de Predicción: Random Forest\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_rf:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_rf:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_rf:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Random Forest explica aproximadamente el {r2_rf*100:.1f}% de la variabilidad en las ventas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_rf:.2f} unidades.\n\n')
      # Mostrar muestra de predicciones (top 10)
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Comuna |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_RF']:.2f} | {row['Error_RF']:.2f} | {row['Error_Porcentual_RF']:.1f}% | {row['categoria']} | {row['comuna']} |\n")
    
    # Estadísticas de error
    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_RF"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_RF"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_RF"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_RF"].std():.2f}\n\n')
    
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')

# Guardar resultado para CatBoost
with open('Resultados/prediccion_cb.md', 'w') as f:
    f.write('# Resultados de Predicción: CatBoost\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_cb:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_cb:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_cb:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    # f.write('## Interpretación\n\n')
    # f.write(f'El modelo de CatBoost explica aproximadamente el {r2_cb*100:.1f}% de la variabilidad en las ventas. ')
    # f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_cb:.2f} unidades.\n\n')
    #   # Mostrar muestra de predicciones (top 10)
    # f.write('## Muestra de Predicciones (Top 10)\n\n')
    # f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Comuna |\n')
    # f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    # for i, row in results_df.head(10).iterrows():
    #     f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_CB']:.2f} | {row['Error_CB']:.2f} | {row['Error_Porcentual_CB']:.1f}% | {row['categoria']} | {row['comuna']} |\n")
    
    # Estadísticas de error
    # f.write('\n## Distribución del Error\n\n')
    # f.write(f'- **Error Mínimo**: {results_df["Error_CB"].min():.2f}\n')
    # f.write(f'- **Error Máximo**: {results_df["Error_CB"].max():.2f}\n')
    # f.write(f'- **Error Promedio**: {results_df["Error_CB"].mean():.2f}\n')
    # f.write(f'- **Desviación Estándar del Error**: {results_df["Error_CB"].std():.2f}\n\n')
    
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')

print("Archivos de predicción generados: Resultados/prediccion_lr.md, Resultados/prediccion_rf.md y Resultados/prediccion_cb.md")

# PASO 10: PRESENTAR RESULTADOS DE LAS MÉTRICAS EN FORMATO TABULAR
metrics_df = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Random Forest', 'CatBoost'],
    'MSE': [mse_lr, mse_rf, mse_cb],
    'RMSE': [rmse_lr, rmse_rf, rmse_cb],
    'MAE': [mae_lr, mae_rf, mae_cb],
    'R²': [r2_lr, r2_rf, r2_cb]
})
print("\nComparación de métricas entre modelos:")
print(metrics_df)

# PASO 11: VISUALIZACIÓN DE PREDICCIONES VS VALORES REALES
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Random Forest: Predicciones vs Valores Reales')
plt.savefig('Resultados/predicciones_vs_reales.png')
print("\nGráfico guardado: Resultados/predicciones_vs_reales.png")

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Regresión Lineal: Predicciones vs Valores Reales')
plt.savefig('Resultados/predicciones_vs_reales_lr.png')
print("Gráfico guardado: Resultados/predicciones_vs_reales_lr.png")

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_cb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('CatBoost: Predicciones vs Valores Reales')
plt.savefig('Resultados/predicciones_vs_reales_cb.png')
print("Gráfico guardado: Resultados/predicciones_vs_reales_cb.png")



# PASO 12: VISUALIZACIÓN DE RESIDUOS PARA EVALUAR CALIDAD DEL MODELO
residuals = y_test - y_pred_rf
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos - Random Forest')
plt.savefig('Resultados/analisis_residuos.png')
print("Gráfico guardado: Resultados/analisis_residuos.png")

residuals_lr = y_test - y_pred_lr
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_lr, residuals_lr, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos - Regresión Lineal')
plt.savefig('Resultados/analisis_residuos_lr.png')
print("Gráfico guardado: Resultados/analisis_residuos_lr.png")

residuals_cb = y_test - y_pred_cb
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_cb, residuals_cb, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos - CatBoost')
plt.savefig('Resultados/analisis_residuos_cb.png')
print("Gráfico guardado: Resultados/analisis_residuos_cb.png")



# PASO 13: DISTRIBUCIÓN DE ERRORES
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribución de Errores - Random Forest')
plt.xlabel('Error')
plt.savefig('Resultados/distribucion_errores.png')
print("Gráfico guardado: Resultados/distribucion_errores.png")

plt.figure(figsize=(10, 6))
sns.histplot(residuals_lr, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribución de Errores - Regresión Lineal')
plt.xlabel('Error')
plt.savefig('Resultados/distribucion_errores_lr.png')
print("Gráfico guardado: Resultados/distribucion_errores_lr.png")

plt.figure(figsize=(10, 6))
sns.histplot(residuals_cb, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribución de Errores - CatBoost')
plt.xlabel('Error')
plt.savefig('Resultados/distribucion_errores_cb.png')
print("Gráfico guardado: Resultados/distribucion_errores_cb.png")


# -------------------------------------------------
# DOCUMENTACIÓN DEL PROCESO
# -------------------------------------------------

print("\n=== DOCUMENTACIÓN DEL PROCESO ===")

# PASO 14: DOCUMENTAR LA EXPLORACIÓN INICIAL DE DATOS
print(f"Dimensiones del dataset: {data.shape[0]} filas x {data.shape[1]} columnas")
print(f"Período de tiempo analizado: de {data['fecha'].min().date()} a {data['fecha'].max().date()}")
# print(f"Tipos de datos en las columnas principales:")
# print(data[features + ['merma_monto']].dtypes)

# PASO 15: DOCUMENTAR EL PREPROCESAMIENTO
print("\n--- PREPROCESAMIENTO APLICADO ---")
print(f"Variables numéricas: {numeric_features}")
print(f"Variables categóricas: {categorical_features}")
print("Transformaciones aplicadas:")
print("- Variables numéricas: Estandarización")
print("- Variables categóricas: One-Hot Encoding")

# PASO 16: DOCUMENTAR LA DIVISIÓN DE DATOS
print("\n--- DIVISIÓN DE DATOS ---")
print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/data.shape[0]:.1%} del total)")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/data.shape[0]:.1%} del total)")
print(f"Método de división: Aleatoria con random_state=42")

# PASO 17: DOCUMENTAR LOS MODELOS EVALUADOS
print("\n--- MODELOS IMPLEMENTADOS ---")
print("1. Regresión Lineal:")
print("   - Ventajas: Simple, interpretable")
print("   - Limitaciones: Asume relación lineal entre variables")

print("\n2. Random Forest Regressor:")
print(f"   - Hiperparámetros: n_estimators={num_estimators}, random_state=42")
print("   - Ventajas: Maneja relaciones no lineales, menor riesgo de overfitting")
print("   - Limitaciones: Menos interpretable, mayor costo computacional")

# PASO 18: DOCUMENTAR LA VALIDACIÓN DEL MODELO
print("\n--- VALIDACIÓN DEL MODELO ---")
print("Método de validación: Evaluación en conjunto de prueba separado")
print("Métricas utilizadas: MSE, RMSE, MAE, R²")

# PASO 19: VISUALIZAR IMPORTANCIA DE CARACTERÍSTICAS
if hasattr(pipeline_cb, 'get_feature_importance'):
    print("\n--- IMPORTANCIA DE CARACTERÍSTICAS (CATBOOST) ---")
    # CatBoost puede trabajar directamente con variables categóricas sin one-hot encoding
    # Obtener importancias de CatBoost
    importances = pipeline_cb.get_feature_importance()
    feature_names = X.columns.tolist()  # Usar los nombres originales de las columnas
    
    # Crear un DataFrame para visualización
    if len(feature_names) == len(importances):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_percentage': (importances / importances.sum() * 100)
        }).sort_values('importance', ascending=False)
        
        # Mostrar las 10 características más importantes
        print("Top 10 características más importantes (CatBoost):")
        for i, (idx, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:<25} → {row['importance']:8.2f} ({row['importance_percentage']:5.2f}%)")
          # Visualizar
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Características Más Importantes - CatBoost')
        plt.xlabel('Importancia')
        plt.savefig('Resultados/importancia_caracteristicas.png')
        print("Gráfico guardado: Resultados/importancia_caracteristicas.png")
    else:
        print("No se pudo visualizar la importancia de características debido a diferencias en la dimensionalidad")
else:
    print("CatBoost no tiene el método get_feature_importance disponible")

# PASO 20: CONCLUSIÓN
print("\n=== CONCLUSIÓN ===")

# Determinar el mejor modelo según R²
modelos = {
    'Regresión Lineal': r2_lr,
    'Random Forest': r2_rf,
    'CatBoost': r2_cb
}
mejor_modelo = max(modelos, key=modelos.get)
mejor_r2 = modelos[mejor_modelo]
mejor_rmse = {'Regresión Lineal': rmse_lr, 'Random Forest': rmse_rf, 'CatBoost': rmse_cb}[mejor_modelo]

print(f"El mejor modelo según R² es: {mejor_modelo}")
print(f"R² del mejor modelo: {mejor_r2:.4f}")
print(f"RMSE del mejor modelo: {mejor_rmse:.2f}")

# Explicaciones adicionales para facilitar la interpretación
print("\n--- INTERPRETACIÓN DE RESULTADOS ---")
print(f"• R² (Coeficiente de determinación): Valor entre 0 y 1 que indica qué proporción de la variabilidad")
print(f"  en las mermas/ventas es explicada por el modelo. Un valor de {mejor_r2:.4f} significa que")
print(f"  aproximadamente el {mejor_r2*100:.1f}% de la variación puede ser explicada por las variables utilizadas.")

print(f"\n• RMSE (Error cuadrático medio): Representa el error promedio de predicción en las mismas unidades")
print(f"  que la variable objetivo. Un RMSE de {mejor_rmse:.2f} significa que, en promedio,")
print(f"  las predicciones difieren de los valores reales en ±{mejor_rmse:.2f} unidades.")

print(f"\n• {mejor_modelo} es el mejor modelo porque:")
if mejor_modelo == 'Random Forest':
    print("  - Captura mejor las relaciones no lineales entre las variables")
    print("  - Tiene mayor capacidad predictiva (R² más alto)")
    print("  - Menor error de predicción (RMSE más bajo)")
elif mejor_modelo == 'CatBoost':
    print("  - Maneja eficientemente variables categóricas")
    print("  - Excelente rendimiento en datos tabulares")
    print("  - Mayor capacidad predictiva (R² más alto) y menor error (RMSE más bajo)")
else:
    print("  - Ofrece un buen equilibrio entre simplicidad y capacidad predictiva")
    print("  - Es más interpretable que modelos complejos")
    print("  - Presenta un mejor ajuste a los datos en este caso específico")

print("\nEl análisis predictivo ha sido completado exitosamente.")
