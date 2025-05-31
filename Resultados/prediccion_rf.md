# Resultados de Predicción: Random Forest

## Resumen de Métricas

- **R²**: 0.1354 (Proporción de varianza explicada por el modelo)
- **RMSE**: 10.24 (Error cuadrático medio, en unidades de la variable objetivo)
- **MAE**: 6.47 (Error absoluto medio, en unidades de la variable objetivo)

## Interpretación

El modelo de Random Forest explica aproximadamente el 13.5% de la variabilidad en las ventas. En promedio, las predicciones difieren de los valores reales en ±10.24 unidades.

## Muestra de Predicciones (Top 10)

| # | Valor Real | Predicción | Error | Error % | Categoría | Comuna |
|---|------------|------------|-------|---------|-----------|--------|
| 567 | 60.00 | 13.91 | 46.09 | 76.8% | FIDEOS Y PASTAS | TEMUCO |
| 586 | 60.00 | 20.05 | 39.95 | 66.6% | YOGHURT | TEMUCO |
| 1023 | 60.00 | 12.88 | 47.12 | 78.5% | FIDEOS Y PASTAS | ANGOL |
| 104 | 60.00 | 14.40 | 45.60 | 76.0% | LECHES ESPECIALES | ANGOL |
| 194 | 60.00 | 20.05 | 39.95 | 66.6% | YOGHURT | TEMUCO |
| 817 | 60.00 | 13.91 | 46.09 | 76.8% | FIDEOS Y PASTAS | TEMUCO |
| 719 | 60.00 | 15.91 | 44.09 | 73.5% | YOGHURT | TEMUCO |
| 757 | 60.00 | 15.91 | 44.09 | 73.5% | YOGHURT | TEMUCO |
| 1022 | 60.00 | 15.91 | 44.09 | 73.5% | YOGHURT | TEMUCO |
| 862 | 60.00 | 17.36 | 42.64 | 71.1% | LECHES SABORES | TEMUCO |

## Distribución del Error

- **Error Mínimo**: -29.00
- **Error Máximo**: 47.12
- **Error Promedio**: 0.39
- **Desviación Estándar del Error**: 10.24

*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*
