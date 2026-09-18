import joblib
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


class TrainModel:

    def entrenarModelo():

        load_dotenv("/app/.env")
        USER = os.getenv("SUPABASE_USER")
        PASSWORD = os.getenv("SUPABASE_PASSWORD")
        HOST = os.getenv("SUPABASE_HOST")
        PORT = os.getenv("SUPABASE_PORT")
        DBNAME = os.getenv("SUPABASE_DBNAME")

        if PORT is None:
            print("no se lee el env")
            return
        else:
            print("si se lee en env")

        try:
            with psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        'SELECT tipo_correo, pais, ciudad, genero FROM vista_genero_cliente;'
                    )
                    rows = cursor.fetchall()
                    print(f"Filas recuperadas: {len(rows)}")

        except Exception as e:
            print(f"Error al conectar o recuperar datos: {e}")
            return

        if not rows:
            print("No se recuperaron filas de la base de datos. Abortando entrenamiento.")
            return
        else:
            print(rows[:2])

        # Pasar los datos a un DataFrame
        df = pd.DataFrame(rows, columns=["tipo_correo", "pais", "ciudad", "genero"])

        # X = lo que uso para predecir | y = lo que quiero predecir
        X = df[["tipo_correo", "pais", "ciudad"]]
        y = df["genero"]

        # Como son texto, hay que convertirlas a numeros con OneHotEncoder
        preprocesador = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"),
                 ["tipo_correo", "pais", "ciudad"])
            ]
        )

        modelo = Pipeline(steps=[
            ("preprocesador", preprocesador),
            ("clasificador", DecisionTreeClassifier(random_state=42)),
        ])

        # Con pocos datos, entrenamos con todo el set
        modelo.fit(X, y)

        joblib.dump(modelo, str(os.getenv("MODELO_ENTRENADO")))
        print("modelo entrenado")
