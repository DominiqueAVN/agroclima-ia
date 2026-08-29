"""
Entrenamiento del modelo de prediccion de riesgo climatico.

Diferencia clave frente a un clasificador ingenuo: este modelo predice el
riesgo TERMICO FUTURO (varias lecturas hacia adelante) a partir de patrones
historicos (temperatura, humedad, hora del dia, tendencia reciente), en vez
de re-etiquetar la misma lectura que ya genero la alerta. Sin esta
separacion temporal, el modelo solo memoriza la regla de umbrales que ya
existia (data leakage), y su "precision" no significa nada.

Combina datos de todas las regiones simuladas (Latinoamerica y Africa) para
tener suficientes ejemplos de eventos de riesgo, ya que son poco frecuentes.
"""

from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURACION
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]  # sube de models/ a agroclima-ia/

# Permite importar main.py (ubicado en la raiz del proyecto) aunque este
# script viva dentro de models/.
sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data" / "synthetic"
MODEL_DIR = ROOT_DIR / "models" / "trained"
MODEL_PATH = MODEL_DIR / "riesgo_climatico.pkl"
METRICS_PATH = MODEL_DIR / "metricas_modelo.json"

# Cuantas lecturas hacia adelante se predice el riesgo.
# Con datos cada 3 horas, HORIZONTE=4 equivale a predecir riesgo con
# 12 horas de anticipacion (alerta temprana real, no deteccion del presente).
HORIZONTE_PASOS = 4

# Numero de lecturas pasadas usadas como tendencia reciente.
VENTANA_LAG = 3


# ============================================================
# 1. CARGAR TODOS LOS DATASETS DISPONIBLES
# ============================================================
def cargar_datos(carpeta: Path) -> pd.DataFrame:
    """Carga y combina todos los CSV de regiones generados por main.py."""
    archivos = sorted(glob.glob(str(carpeta / "*_temperaturas.csv")))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron datasets en {carpeta}. "
            "Corre main.py primero, por ejemplo:\n"
            "  python main.py --region todas --dias 365 --semilla 42"
        )

    partes = [pd.read_csv(a, parse_dates=["fecha"]) for a in archivos]
    df = pd.concat(partes, ignore_index=True)
    df = df.sort_values(["region_id", "fecha"]).reset_index(drop=True)

    print(f"Datasets combinados: {len(archivos)} regiones, {len(df)} registros totales.")
    return df


# ============================================================
# 2. INGENIERIA DE VARIABLES (features)
# ============================================================
def construir_features(df: pd.DataFrame, cultivo: str = "germinado") -> pd.DataFrame:
    """
    Construye variables predictoras y el objetivo (riesgo futuro), evitando
    fuga de informacion: ninguna variable usa datos posteriores al momento
    de la prediccion.
    """
    from main import detectar_riesgo_termico  # reutiliza la logica ya definida

    resultado = []
    for region_id, grupo in df.groupby("region_id"):
        grupo = grupo.sort_values("fecha").reset_index(drop=True)
        grupo = detectar_riesgo_termico(grupo, cultivo=cultivo)
        grupo["riesgo_bin"] = (grupo["riesgo"] == "alerta").astype(int)

        # Variables ciclicas: la hora 23 y la hora 0 son "vecinas" en el
        # tiempo, un numero crudo (0-23) no captura eso.
        hora = grupo["fecha"].dt.hour + grupo["fecha"].dt.minute / 60
        grupo["hora_sin"] = np.sin(2 * np.pi * hora / 24)
        grupo["hora_cos"] = np.cos(2 * np.pi * hora / 24)

        mes = grupo["fecha"].dt.month
        grupo["mes_sin"] = np.sin(2 * np.pi * mes / 12)
        grupo["mes_cos"] = np.cos(2 * np.pi * mes / 12)

        # Tendencia reciente: lecturas anteriores, NUNCA futuras.
        for lag in range(1, VENTANA_LAG + 1):
            grupo[f"temp_lag{lag}"] = grupo["temperatura_c"].shift(lag)
            grupo[f"humedad_lag{lag}"] = grupo["humedad_pct"].shift(lag)

        grupo["temp_tendencia"] = (
            grupo["temperatura_c"] - grupo["temperatura_c"].shift(VENTANA_LAG)
        )

        # Objetivo: riesgo HORIZONTE_PASOS lecturas hacia el futuro.
        grupo["objetivo_riesgo_futuro"] = grupo["riesgo_bin"].shift(-HORIZONTE_PASOS)

        resultado.append(grupo)

    combinado = pd.concat(resultado, ignore_index=True)
    combinado["region_nombre"] = combinado["region_id"]
    combinado = pd.get_dummies(combinado, columns=["region_id"], prefix="region")

    # Elimina filas sin suficiente historia (inicio de cada serie) o sin
    # objetivo futuro conocido (final de cada serie).
    combinado = combinado.dropna().reset_index(drop=True)

    return combinado


def columnas_features(df: pd.DataFrame) -> list[str]:
    excluir = {
        "fecha",
        "temperatura_c",
        "humedad_pct",
        "riesgo",
        "severidad",
        "cultivo",
        "riesgo_bin",
        "objetivo_riesgo_futuro",
        "region_nombre",
    }
    return [c for c in df.columns if c not in excluir]


# ============================================================
# 3. DIVISION TEMPORAL (nunca aleatoria en series de tiempo)
# ============================================================
def dividir_train_test(df: pd.DataFrame, proporcion_train: float = 0.8):
    """
    Divide por fecha, no al azar: el modelo se entrena con el pasado y se
    evalua con el futuro, tal como funcionaria en produccion. Una division
    aleatoria mezclaria informacion futura dentro del entrenamiento.
    """
    corte = df["fecha"].quantile(proporcion_train)
    train = df[df["fecha"] <= corte]
    test = df[df["fecha"] > corte]
    return train, test


# ============================================================
# 4. ENTRENAR Y EVALUAR
# ============================================================
def entrenar_y_evaluar(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    X_train, y_train = train[features], train["objetivo_riesgo_futuro"]
    X_test, y_test = test[features], test["objetivo_riesgo_futuro"]

    print(f"\nEntrenamiento: {len(X_train)} registros | Prueba: {len(X_test)} registros")
    print(f"Casos de riesgo en entrenamiento: {int(y_train.sum())} ({y_train.mean():.1%})")
    print(f"Casos de riesgo en prueba: {int(y_test.sum())} ({y_test.mean():.1%})")

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",  # compensa que los eventos de riesgo son escasos
        random_state=42,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    # predict_proba da la PROPORCION de arboles del bosque que votaron por
    # "riesgo" para cada caso: es la forma correcta de interpretar esto como
    # una probabilidad/confianza, no como un valor exacto de certeza absoluta.
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print(f"RESULTADOS - prediccion a {HORIZONTE_PASOS} lecturas de anticipacion")
    print("=" * 60)
    matriz = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusion (filas=real, columnas=prediccion):")
    print(matriz)
    reporte_global = classification_report(
        y_test, y_pred, target_names=["Normal", "Alerta"], zero_division=0, output_dict=True
    )
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Alerta"], zero_division=0))

    # Comparacion honesta: un modelo de "persistencia" que asume que el
    # riesgo futuro sera igual al riesgo actual. Este es el punto de
    # comparacion correcto, no un umbral que ya conoce la respuesta.
    y_persistencia = test["riesgo_bin"]
    reporte_persistencia = classification_report(
        y_test, y_persistencia, target_names=["Normal", "Alerta"], zero_division=0, output_dict=True
    )
    print("Comparacion contra linea base de persistencia (riesgo futuro = riesgo actual):")
    print(classification_report(y_test, y_persistencia, target_names=["Normal", "Alerta"], zero_division=0))

    # Desempeno DESGLOSADO POR REGION: el modelo puede ser mas confiable en
    # unas regiones que en otras (depende de cuantos casos de riesgo tuvo
    # cada una para aprender). Ocultar esto seria enganoso.
    metricas_por_region = {}
    for region in sorted(test["region_nombre"].unique()):
        mascara = test["region_nombre"] == region
        if mascara.sum() < 20:  # muy pocos datos para una metrica confiable
            continue
        rep = classification_report(
            y_test[mascara], y_pred[mascara],
            labels=[0, 1], target_names=["Normal", "Alerta"], zero_division=0, output_dict=True,
        )
        metricas_por_region[region] = {
            "casos_evaluados": int(mascara.sum()),
            "casos_alerta_reales": int(y_test[mascara].sum()),
            "recall_alerta": round(rep["Alerta"]["recall"], 3),
            "precision_alerta": round(rep["Alerta"]["precision"], 3),
            "f1_alerta": round(rep["Alerta"]["f1-score"], 3),
        }

    # Importancia de variables: que esta usando el modelo para decidir.
    importancias = pd.Series(modelo.feature_importances_, index=features)
    print("\nVariables mas influyentes:")
    print(importancias.sort_values(ascending=False).head(8).to_string())

    metricas = {
        "fecha_entrenamiento": pd.Timestamp.now().isoformat(),
        "horizonte_pasos": HORIZONTE_PASOS,
        "casos_entrenamiento": len(X_train),
        "casos_prueba": len(X_test),
        "matriz_confusion": {
            "normal_predicho_normal": int(matriz[0][0]),
            "normal_predicho_alerta": int(matriz[0][1]),
            "alerta_predicho_normal": int(matriz[1][0]),
            "alerta_predicho_alerta": int(matriz[1][1]),
        },
        "global": {
            "accuracy": round(reporte_global["accuracy"], 4),
            "precision_alerta": round(reporte_global["Alerta"]["precision"], 4),
            "recall_alerta": round(reporte_global["Alerta"]["recall"], 4),
            "f1_alerta": round(reporte_global["Alerta"]["f1-score"], 4),
        },
        "baseline_persistencia": {
            "accuracy": round(reporte_persistencia["accuracy"], 4),
            "recall_alerta": round(reporte_persistencia["Alerta"]["recall"], 4),
            "f1_alerta": round(reporte_persistencia["Alerta"]["f1-score"], 4),
        },
        "por_region": metricas_por_region,
        "variables_mas_influyentes": importancias.sort_values(ascending=False).head(8).round(4).to_dict(),
    }

    return modelo, metricas


# ============================================================
# 5. GUARDAR MODELO CON METADATA
# ============================================================
def guardar_modelo(modelo, features: list[str], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    paquete = {
        "modelo": modelo,
        "features": features,
        "horizonte_pasos": HORIZONTE_PASOS,
        "ventana_lag": VENTANA_LAG,
    }
    joblib.dump(paquete, ruta)
    print(f"\nModelo guardado en: {ruta}")


def guardar_metricas(metricas: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2, ensure_ascii=False)
    print(f"Metricas guardadas en: {ruta}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    df_crudo = cargar_datos(DATA_DIR)
    df_features = construir_features(df_crudo)
    features = columnas_features(df_features)

    train, test = dividir_train_test(df_features)
    modelo, metricas = entrenar_y_evaluar(train, test, features)
    guardar_modelo(modelo, features, MODEL_PATH)
    guardar_metricas(metricas, METRICS_PATH)

    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()