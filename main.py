from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar modelo y features
model = joblib.load("modelo.pkl")
features = joblib.load("features.pkl")

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
