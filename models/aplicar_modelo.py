"""
Aplica el modelo entrenado (models/trained/riesgo_climatico.pkl) sobre los
datos historicos y genera un CSV con las predicciones ya calculadas.

Se ejecuta UNA VEZ dentro del entorno virtual (donde scikit-learn si esta
instalado). Webots despues solo lee el CSV resultante, sin depender de
ninguna libreria de Machine Learning en tiempo de simulacion.
"""

from __future__ import annotations

from pathlib import Path

import joblib

import train_model as tm

SALIDA_PATH = tm.DATA_DIR / "predicciones_ml.csv"


def main() -> None:
    paquete = joblib.load(tm.MODEL_PATH)
    modelo = paquete["modelo"]
    features = paquete["features"]
    horizonte = paquete["horizonte_pasos"]

    print(f"Modelo cargado. Horizonte de prediccion: {horizonte} lecturas hacia adelante.")

    df_crudo = tm.cargar_datos(tm.DATA_DIR)
    df = tm.construir_features(df_crudo)

    df["riesgo_predicho_ml"] = modelo.predict(df[features])
    # Proporcion de arboles del bosque que "votaron" por riesgo: es la forma
    # correcta de interpretar esto como confianza del modelo, no como una
    # probabilidad exacta en el sentido estadistico estricto.
    df["probabilidad_riesgo"] = modelo.predict_proba(df[features])[:, 1].round(3)

    # 'objetivo_riesgo_futuro' es el riesgo real que SI ocurrio (lo conocemos
    # porque los datos son simulados). Se conserva solo para poder mostrar,
    # a modo de demo, si la prediccion acerto o no. En un sistema real, este
    # dato no estaria disponible de antemano.
    columnas_salida = [
        "fecha",
        "region_nombre",
        "temperatura_c",
        "humedad_pct",
        "riesgo_bin",
        "riesgo_predicho_ml",
        "probabilidad_riesgo",
        "objetivo_riesgo_futuro",
    ]
    salida = df[columnas_salida].rename(
        columns={
            "riesgo_bin": "riesgo_actual",
            "objetivo_riesgo_futuro": "riesgo_real_futuro_demo",
        }
    )
    salida = salida.sort_values(["region_nombre", "fecha"])
    salida.to_csv(SALIDA_PATH, index=False)

    aciertos = (salida["riesgo_predicho_ml"] == salida["riesgo_real_futuro_demo"]).mean()
    alertas_predichas = int(salida["riesgo_predicho_ml"].sum())

    print(f"Filas escritas: {len(salida)}")
    print(f"Alertas predichas en total: {alertas_predichas}")
    print(f"Acierto de la prediccion vs el resultado real (solo verificable en la demo): {aciertos:.1%}")
    print(f"Regiones incluidas: {sorted(salida['region_nombre'].unique())}")
    print(f"\nCSV guardado en: {SALIDA_PATH}")


if __name__ == "__main__":
    main()