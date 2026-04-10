"""
check_columns.py — Muestra las columnas exactas del Excel de entrenamiento
Uso: python check_columns.py
"""
import pandas as pd
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "DataFiles")
file_path = os.path.join(DATA_DIR, "Data_Entrenamiento_NAS100.xlsx")

data = pd.read_excel(file_path, nrows=2)
data.columns = data.columns.str.strip()

print("Columnas disponibles en el Excel:")
for i, col in enumerate(data.columns):
    print(f"  {i:>3}. '{col}'")