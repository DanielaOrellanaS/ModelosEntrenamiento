import pandas as pd
import os
import glob

# Variables globales para las rutas
DOWNLOADS_PATH = r"C:\Users\user\Downloads"
BASE_PATH = r"C:\Users\user\OneDrive\Documentos\Trading\ModelPrPth\apiPrPthDataset\ModelosEntrenamiento\DataFiles"
CONVERTED_PATH = os.path.join(BASE_PATH, "TrasnformerTxConvertidos")
os.makedirs(CONVERTED_PATH, exist_ok=True)

symbol = "EURUSD"
 
TRANSFORMER_FILE = glob.glob(os.path.join(DOWNLOADS_PATH, f"transformer_{symbol}_*.csv"))
TX_FILE = glob.glob(os.path.join(DOWNLOADS_PATH, f"TX_{symbol}_*.csv"))

FILE_DATA_PROFIT = f"Data_Entrenamiento_{symbol}.xlsx"

# Lista de archivos con rutas dinámicas
archivos = []

for f in TRANSFORMER_FILE:
    archivos.append((f, os.path.join(CONVERTED_PATH, os.path.splitext(os.path.basename(f))[0] + ".xlsx")))

for f in TX_FILE:
    archivos.append((f, os.path.join(CONVERTED_PATH, os.path.splitext(os.path.basename(f))[0] + ".xlsx")))

def convertir_csv_a_excel(csv_path, excel_path):
    df = pd.read_csv(csv_path, delimiter=";")
    
    # Reemplazar los puntos por comas en todos los valores numéricos
    df = df.map(lambda x: str(x).replace('.', ',') if isinstance(x, str) else x)
    
    # Eliminar espacios al inicio y final de las columnas
    df.columns = df.columns.str.strip()
    
    # Eliminar espacios en la columna 'precioopen5'
    if 'precioopen5' in df.columns:
        df['precioopen5'] = df['precioopen5'].astype(str).str.strip()
    
    # Ordenar por fecha si la columna existe
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], format="%Y,%m,%d %H:%M", errors="coerce")
        df = df.sort_values(by="fecha", ascending=True)
        df["fecha"] = df["fecha"].dt.strftime("%Y,%m,%d %H:%M")
    
    df.to_excel(excel_path, index=False)
    print(f"Archivo convertido con éxito y guardado en: {excel_path}")

# Convertir ambos archivos
enum = [(csv, out) for csv, out in archivos]
for csv_file, output_file in enum:
    convertir_csv_a_excel(csv_file, output_file)

# Leer archivos Excel
transformer_excels = glob.glob(os.path.join(CONVERTED_PATH, f"transformer_{symbol}_*.xlsx"))
tx_excels = glob.glob(os.path.join(CONVERTED_PATH, f"TX_{symbol}_*.xlsx"))

df_transformer = pd.concat([pd.read_excel(f) for f in transformer_excels], ignore_index=True)
df_tx = pd.concat([pd.read_excel(f) for f in tx_excels], ignore_index=True)

# Eliminar espacios extra en los nombres de las columnas
df_tx.columns = df_tx.columns.str.strip()

# Contar registros
total_transformer = len(df_transformer)
total_tx = len(df_tx)
print(f"Número de registros transformer: {total_transformer}")
print(f"Número de registros TX: {total_tx}")

# Seleccionar y renombrar columnas de TX
columnas_necesarias = ['simbolo', 'timeframe', 'fecha', 'pr1', 'pr2', 'dif', 'periodos', 'tipo', 'minimo']
df_tx_cleaned = df_tx[columnas_necesarias].copy()
df_tx_cleaned = df_tx_cleaned.rename(columns={"dif": "profit"})

df_transformer = df_transformer.drop_duplicates(subset=["simbolo", "timeframe", "fecha"])
df_tx_cleaned = df_tx_cleaned.drop_duplicates(subset=["simbolo", "timeframe", "fecha"])

# Realizar left join
df_merged = pd.merge(
    df_transformer, 
    df_tx_cleaned, 
    on=["simbolo", "timeframe", "fecha"], 
    how="left"
)

# Ordenar por fecha
if "fecha" in df_merged.columns:
    df_merged["fecha"] = pd.to_datetime(df_merged["fecha"], format="%Y,%m,%d %H:%M", errors="coerce")
    df_merged = df_merged.sort_values(by="fecha", ascending=True)
    df_merged["fecha"] = df_merged["fecha"].dt.strftime("%Y,%m,%d %H:%M")

# Contar registros finales
total_final = len(df_merged)
print(f"Número de registros después de merge: {total_final}")

# Guardar archivo final
data_final_path = os.path.join(BASE_PATH, FILE_DATA_PROFIT)
df_merged.to_excel(data_final_path, index=False)
print(f"Archivo consolidado guardado en: {data_final_path}")
