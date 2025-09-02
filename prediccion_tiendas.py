import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Leer CSV
df = pd.read_csv("ventas_tiendas (4).csv")

# Eliminar columnas irrelevantes
df = df.drop(columns=["tienda_id", "empleados"])

# Rellenar valores nulos
df["publicidad"] = df["publicidad"].fillna(df["publicidad"].mean())
df["ventas"] = df["ventas"].fillna(df["ventas"].median())
df["ubicacion"] = df["ubicacion"].fillna(df["ubicacion"].mode()[0])

# Eliminar outliers en ventas (usar minúscula)
df = df[(df["ventas"] > 0) & (df["ventas"] < 500000)]

# Codificar variable categórica (minúscula)
df = pd.get_dummies(df, columns=["ubicacion"], drop_first=True)

# Separar variables dependientes e independientes
X = df.drop(columns=["ventas"])   # features
y = df["ventas"]                  # target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

#ecuacioon de la recta
print("Ecuación de la recta: y =", model.intercept_, "+", " + ".join(f"{coef}*{name}" for coef, name in zip(model.coef_, X.columns)))   


# Métricas de evaluación
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

#Guardar modelo
joblib.dump(model, "modelo.pkl")
joblib.dump(X_train.columns.tolist(), "features.pkl")  # guardar nombres de columnas