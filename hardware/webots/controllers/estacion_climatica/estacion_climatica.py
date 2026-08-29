"""
Controlador de la estacion climatica simulada (Webots).

Lee data/synthetic/predicciones_ml.csv, generado por models/aplicar_modelo.py,
y reproduce la secuencia de una region especifica, encendiendo el LED cuando
el modelo predice riesgo termico varias lecturas hacia adelante.

No requiere scikit-learn ni joblib en este script: la prediccion ya viene
calculada, Webots solo la reproduce.
"""

from __future__ import annotations

import csv
from pathlib import Path

from controller import Robot  # type: ignore

# Cambia esto para simular otra region. Opciones disponibles: arequipa_pe,
# cochabamba_bo, oaxaca_mx, nairobi_ke, ouagadougou_bf, antananarivo_mg.
# Uagadugu (Sahel) tiene mayor variabilidad termica y es mejor para ver el
# LED encenderse en una demo corta.
REGION_ACTIVA = "ouagadougou_bf"

ROOT_DIR = Path(__file__).resolve().parents[4]
CSV_PATH = ROOT_DIR / "data" / "synthetic" / "predicciones_ml.csv"

TIME_STEP_MS = 64
SEGUNDOS_POR_REGISTRO = 0.5


def cargar_region(ruta: Path, region: str) -> list[dict[str, str]]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro {ruta}. Corre models/aplicar_modelo.py primero."
        )

    with ruta.open(newline="", encoding="utf-8") as archivo:
        registros = [r for r in csv.DictReader(archivo) if r["region_nombre"] == region]

    if not registros:
        raise ValueError(f"No hay registros para la region '{region}' en el CSV.")

    return registros


def main() -> None:
    robot = Robot()
    led = robot.getDevice("led_alerta")

    registros = cargar_region(CSV_PATH, REGION_ACTIVA)
    print(f"Region activa: {REGION_ACTIVA} ({len(registros)} registros)")

    pasos_por_registro = max(1, int((SEGUNDOS_POR_REGISTRO * 1000) / TIME_STEP_MS))

    indice_actual = 0
    contador_pasos = 0

    while robot.step(TIME_STEP_MS) != -1:
        if indice_actual >= len(registros):
            print("Simulacion completa.")
            break

        registro = registros[indice_actual]
        en_alerta = registro["riesgo_predicho_ml"] == "1"

        led.set(1 if en_alerta else 0)

        if contador_pasos == 0:
            estado = "ALERTA TEMPRANA" if en_alerta else "estable"
            print(
                f"[{registro['fecha']}] {registro['temperatura_c']} C - {estado} "
                f"(prediccion ML, {registro.get('riesgo_real_futuro_demo', '?')} = resultado real)"
            )

        contador_pasos += 1
        if contador_pasos >= pasos_por_registro:
            contador_pasos = 0
            indice_actual += 1


if __name__ == "__main__":
    main()