import pandas as pd

# Cargar el archivo (ajusta el path si es necesario)
df = pd.read_excel('dataset/mermas_actividad_unidad_2.xlsx')

# 1. Limpiar nombres de columnas
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 2. Convertir campos numéricos (quita puntos de miles y cambia comas por puntos si es necesario)
for col in ['merma_unidad', 'merma_monto', 'merma_unidad_p', 'merma_monto_p']:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('.', '', regex=False)   # Quita puntos de miles si los hay
        .str.replace(',', '.', regex=False)  # Cambia coma decimal por punto
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Convertir fecha a datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y', errors='coerce')

# 4. Eliminar filas con valores nulos en campos clave
df = df.dropna(subset=['merma_unidad', 'merma_monto', 'fecha'])

# 5. Estandarizar texto en columnas categóricas (opcional)
cat_cols = ['negocio', 'seccion', 'linea', 'categoria', 'abastecimiento', 'comuna', 'region', 'tienda', 'zonal', 'mes', 'motivo', 'ubicación_motivo']
for col in cat_cols:
    df[col] = df[col].str.strip().str.upper()

# Eliminar columnas no deseadas
#df = df.drop(columns=['zonal', 'tienda', 'comuna', 'region'])

# 6. Eliminar outliers usando el método IQR para las columnas numéricas
num_cols = ['merma_unidad', 'merma_monto', 'merma_unidad_p', 'merma_monto_p']
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Guardar el dataset limpio
nombre_archivo = 'dataset/mermas_limpio.csv'
df.to_csv(nombre_archivo, index=False)
print(f"¡Dataset limpio y guardado como '{nombre_archivo}'!")