from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar modelo y features con rutas absolutas
model = joblib.load(os.path.join(BASE_DIR, "modelo.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# Crear instancia de la API
app = FastAPI(title="API Predicción de Ventas")

# Definir el esquema de entrada
class DatosEntrada(BaseModel):
    publicidad: float
    ubicacion: str  # "rural", "urbana", "suburbana"

# Ruta GET de inicio
@app.get("/")
def home():
    return {"titulo": "Bienvenido a la API de Predicción de Ventas"}

# Ruta POST para predicciones
@app.post("/predecir")
def predecir(data: DatosEntrada):
    # Convertir entrada en DataFrame
    df = pd.DataFrame([{
        "publicidad": data.publicidad,
        "ubicacion": data.ubicacion
    }])

    # Codificación de ubicación (dummy variables)
    df = pd.get_dummies(df, columns=["ubicacion"], drop_first=True)

    # Asegurar que tenga las mismas columnas que el modelo
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    # Hacer predicción
    prediccion = model.predict(df)[0]
    return {"prediccion_ventas": round(float(prediccion), 2)}
