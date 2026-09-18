import os
import joblib
import pandas as pd

from src.contexts.api.models import PredictorRequest


class TrainModelController:
    def execute(self, request: PredictorRequest):
        print(request)

        tipo_correo = request.tipo_correo
        pais = request.pais
        ciudad = request.ciudad

        lr_model_path = os.getenv("MODELO_ENTRENADO")

        # Cargar el modelo entrenado
        modelo_cargado = joblib.load(lr_model_path)

        # Armar el dato nuevo con las MISMAS columnas del entrenamiento
        nuevo_dato = pd.DataFrame([{
            "tipo_correo": tipo_correo,
            "pais": pais,
            "ciudad": ciudad,
        }])

        # Predecir el genero
        result = modelo_cargado.predict(nuevo_dato)
        genero = str(result[0])
        print(f"Prediccion de genero: {genero}")

        return {"status": "OK", "genero": genero}
