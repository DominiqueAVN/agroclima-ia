"""
AgroClima IA
============

Sistema de alerta temprana climática para agricultura de pequeña escala en
Latinoamérica y África.

Este modulo genera series climaticas sinteticas para un conjunto de regiones
de referencia y evalua el riesgo termico para cultivos sensibles en etapa de
germinacion. Sirve como base para el desarrollo posterior del modelo de
prediccion (fase 2) y la integracion con hardware de sensores (fase 3).
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agroclima")


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """Perfil climatico de referencia para una zona de monitoreo."""

    id: str
    nombre: str
    pais: str
    continente: str
    latitud: float
    longitud: float
    altitud_m: float
    temp_media_c: float
    amplitud_diaria_c: float
    humedad_base_pct: float


@dataclass(frozen=True)
class UmbralCultivo:
    """Rango termico tolerado por un cultivo en etapa sensible."""

    temp_min_c: float
    temp_max_c: float


# ---------------------------------------------------------------------------
# Datos de referencia
#
# Las regiones cubren America Latina y Africa de forma deliberada: el
# proyecto no esta acotado a un pais, sino a un tipo de contexto (agricultura
# de pequena escala con baja cobertura meteorologica). Los valores de
# temperatura/humedad son aproximaciones climatologicas de referencia, no
# mediciones reales; se reemplazan por datos de campo o fuentes abiertas
# (NASA Earthdata, Copernicus) en fases posteriores.
# ---------------------------------------------------------------------------

REGIONES: dict[str, Region] = {
    "arequipa_pe": Region(
        id="arequipa_pe",
        nombre="Arequipa",
        pais="Peru",
        continente="America del Sur",
        latitud=-16.4,
        longitud=-71.5,
        altitud_m=2335,
        temp_media_c=15.5,
        amplitud_diaria_c=8.0,
        humedad_base_pct=45.0,
    ),
    "cochabamba_bo": Region(
        id="cochabamba_bo",
        nombre="Cochabamba",
        pais="Bolivia",
        continente="America del Sur",
        latitud=-17.4,
        longitud=-66.2,
        altitud_m=2558,
        temp_media_c=17.8,
        amplitud_diaria_c=9.0,
        humedad_base_pct=50.0,
    ),
    "oaxaca_mx": Region(
        id="oaxaca_mx",
        nombre="Oaxaca",
        pais="Mexico",
        continente="America del Norte",
        latitud=17.1,
        longitud=-96.7,
        altitud_m=1555,
        temp_media_c=21.0,
        amplitud_diaria_c=7.0,
        humedad_base_pct=55.0,
    ),
    "nairobi_ke": Region(
        id="nairobi_ke",
        nombre="Nairobi",
        pais="Kenia",
        continente="Africa Oriental",
        latitud=-1.3,
        longitud=36.8,
        altitud_m=1795,
        temp_media_c=19.0,
        amplitud_diaria_c=6.5,
        humedad_base_pct=60.0,
    ),
    "ouagadougou_bf": Region(
        id="ouagadougou_bf",
        nombre="Uagadugu",
        pais="Burkina Faso",
        continente="Africa Occidental / Sahel",
        latitud=12.4,
        longitud=-1.5,
        altitud_m=305,
        temp_media_c=28.5,
        amplitud_diaria_c=11.0,
        humedad_base_pct=35.0,
    ),
    "antananarivo_mg": Region(
        id="antananarivo_mg",
        nombre="Antananarivo",
        pais="Madagascar",
        continente="Africa Oriental",
        latitud=-18.9,
        longitud=47.5,
        altitud_m=1280,
        temp_media_c=18.5,
        amplitud_diaria_c=7.5,
        humedad_base_pct=58.0,
    ),
}

UMBRALES_CULTIVO: dict[str, UmbralCultivo] = {
    "germinado": UmbralCultivo(temp_min_c=10.0, temp_max_c=28.0),
    "lechuga": UmbralCultivo(temp_min_c=7.0, temp_max_c=24.0),
    "tomate": UmbralCultivo(temp_min_c=15.0, temp_max_c=30.0),
    "quinua": UmbralCultivo(temp_min_c=5.0, temp_max_c=25.0),
    "sorgo": UmbralCultivo(temp_min_c=18.0, temp_max_c=35.0),
    "mijo": UmbralCultivo(temp_min_c=16.0, temp_max_c=34.0),
    "maiz": UmbralCultivo(temp_min_c=12.0, temp_max_c=32.0),
}


# ---------------------------------------------------------------------------
# Simulacion climatica
# ---------------------------------------------------------------------------

def simular_serie_climatica(
    region: Region,
    dias: int = 30,
    mediciones_por_dia: int = 8,
    semilla: int | None = None,
) -> pd.DataFrame:
    """
    Genera una serie sintetica de temperatura y humedad para una region.

    El modelo combina un ciclo diurno senoidal, ruido gaussiano acotado y una
    probabilidad baja de eventos anomalos (olas de calor o frio), como
    aproximacion a la variabilidad observada en registros meteorologicos
    reales. No sustituye datos de campo; es una base para probar la logica
    de deteccion de riesgo antes de integrar sensores fisicos.

    Args:
        region: Perfil climatico de referencia.
        dias: Numero de dias a simular.
        mediciones_por_dia: Cantidad de lecturas por dia.
        semilla: Semilla opcional para reproducibilidad.

    Returns:
        DataFrame con columnas: fecha, region_id, temperatura_c, humedad_pct.
    """
    if semilla is not None:
        random.seed(semilla)
        np.random.seed(semilla)

    intervalo_horas = 24 / mediciones_por_dia
    fecha_inicio = datetime.now() - timedelta(days=dias)
    fechas = pd.date_range(
        start=fecha_inicio,
        periods=dias * mediciones_por_dia,
        freq=f"{intervalo_horas}h",
    )

    registros = []
    for fecha in fechas:
        hora = fecha.hour + fecha.minute / 60

        if 6 <= hora < 18:
            temp = region.temp_media_c + (region.amplitud_diaria_c / 2) * np.sin(
                np.pi * (hora - 6) / 12
            )
        else:
            hora_ajustada = hora if hora >= 18 else hora + 24
            temp = region.temp_media_c - (region.amplitud_diaria_c / 2) * np.cos(
                np.pi * (hora_ajustada - 18) / 12
            )

        ruido = np.random.normal(loc=0.0, scale=0.8)

        anomalia = 0.0
        if random.random() < 0.05:
            anomalia = random.uniform(-3.5, 3.5)

        temp_final = round(float(temp + ruido + anomalia), 2)

        humedad = (
            region.humedad_base_pct
            - (temp_final - region.temp_media_c) * 2.5
            + random.uniform(-8, 8)
        )
        humedad = round(max(15.0, min(95.0, humedad)), 2)

        registros.append(
            {
                "fecha": fecha,
                "region_id": region.id,
                "temperatura_c": temp_final,
                "humedad_pct": humedad,
            }
        )

    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# Deteccion de riesgo
# ---------------------------------------------------------------------------

def detectar_riesgo_termico(df: pd.DataFrame, cultivo: str) -> pd.DataFrame:
    """
    Anota cada registro con nivel de riesgo y severidad segun el cultivo.

    Args:
        df: DataFrame con columna 'temperatura_c'.
        cultivo: Clave presente en UMBRALES_CULTIVO.

    Returns:
        Copia del DataFrame de entrada con columnas 'cultivo', 'severidad' y
        'riesgo' agregadas.

    Raises:
        KeyError: Si el cultivo no esta definido en UMBRALES_CULTIVO.
    """
    if cultivo not in UMBRALES_CULTIVO:
        raise KeyError(
            f"Cultivo '{cultivo}' no definido. Opciones: {list(UMBRALES_CULTIVO)}"
        )

    umbral = UMBRALES_CULTIVO[cultivo]
    resultado = df.copy()

    def severidad(temp: float) -> float:
        if temp < umbral.temp_min_c:
            valor = (umbral.temp_min_c - temp) * 2
        elif temp > umbral.temp_max_c:
            valor = (temp - umbral.temp_max_c) * 2
        else:
            valor = 0.0
        return round(min(10.0, valor), 1)

    resultado["severidad"] = resultado["temperatura_c"].apply(severidad)
    resultado["riesgo"] = resultado["severidad"].apply(
        lambda s: "alerta" if s > 0 else "normal"
    )
    resultado["cultivo"] = cultivo

    return resultado


# ---------------------------------------------------------------------------
# Persistencia y reportes
# ---------------------------------------------------------------------------

def guardar_dataset(
    df: pd.DataFrame, region_id: str, carpeta: str = "data/synthetic"
) -> Path:
    """Guarda el dataset anotado en formato CSV dentro de la carpeta indicada."""
    ruta_carpeta = Path(carpeta)
    ruta_carpeta.mkdir(parents=True, exist_ok=True)
    ruta_archivo = ruta_carpeta / f"{region_id}_temperaturas.csv"
    df.to_csv(ruta_archivo, index=False)
    return ruta_archivo


def generar_reporte(df: pd.DataFrame, region: Region, cultivo: str) -> str:
    """Construye un reporte de texto plano con las estadisticas del periodo."""
    alertas = df[df["riesgo"] == "alerta"]

    lineas = [
        "=" * 60,
        f"Reporte de monitoreo climatico - {region.nombre}, {region.pais}",
        "=" * 60,
        f"Continente: {region.continente}",
        f"Coordenadas: {region.latitud}, {region.longitud} | Altitud: {region.altitud_m} m",
        f"Cultivo evaluado: {cultivo}",
        f"Periodo: {df['fecha'].min():%Y-%m-%d} a {df['fecha'].max():%Y-%m-%d}",
        f"Mediciones totales: {len(df)}",
        "",
        "Temperatura (C):",
        f"  Promedio: {df['temperatura_c'].mean():.2f}",
        f"  Minima:   {df['temperatura_c'].min():.2f}",
        f"  Maxima:   {df['temperatura_c'].max():.2f}",
        "",
        "Humedad relativa (%):",
        f"  Promedio: {df['humedad_pct'].mean():.2f}",
        f"  Minima:   {df['humedad_pct'].min():.2f}",
        f"  Maxima:   {df['humedad_pct'].max():.2f}",
        "",
        f"Eventos de riesgo detectados: {len(alertas)}",
    ]

    if not alertas.empty:
        alta = len(alertas[alertas["severidad"] >= 7])
        media = len(alertas[(alertas["severidad"] >= 4) & (alertas["severidad"] < 7)])
        baja = len(alertas[alertas["severidad"] < 4])
        lineas += [
            f"  Severidad alta (7-10): {alta}",
            f"  Severidad media (4-6): {media}",
            f"  Severidad baja (1-3):  {baja}",
        ]

    lineas.append("=" * 60)
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------

def ejecutar_monitoreo(
    region_id: str, cultivo: str, dias: int, semilla: int | None
) -> None:
    """Ejecuta el flujo completo para una region: simular, evaluar, guardar."""
    if region_id not in REGIONES:
        raise KeyError(f"Region '{region_id}' no definida. Opciones: {list(REGIONES)}")

    region = REGIONES[region_id]
    logger.info("Simulando serie climatica para %s (%s)", region.nombre, region.pais)

    df = simular_serie_climatica(region, dias=dias, semilla=semilla)
    df = detectar_riesgo_termico(df, cultivo=cultivo)

    ruta_datos = guardar_dataset(df, region_id)
    logger.info("Dataset guardado en %s", ruta_datos)

    reporte = generar_reporte(df, region, cultivo)
    print(reporte)

    ruta_reporte = Path("docs") / f"reporte_{region_id}_{cultivo}.txt"
    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    ruta_reporte.write_text(reporte, encoding="utf-8")
    logger.info("Reporte guardado en %s", ruta_reporte)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgroClima IA: simulacion y deteccion de riesgo termico agricola."
    )
    parser.add_argument(
        "--region",
        default="arequipa_pe",
        choices=[*REGIONES.keys(), "todas"],
        help="Region a simular. Usa 'todas' para procesar todas las regiones definidas.",
    )
    parser.add_argument(
        "--cultivo",
        default="germinado",
        choices=list(UMBRALES_CULTIVO.keys()),
        help="Cultivo a evaluar para el calculo de umbrales termicos.",
    )
    parser.add_argument(
        "--dias",
        type=int,
        default=30,
        help="Numero de dias de historial sintetico a generar.",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=None,
        help="Semilla aleatoria para reproducibilidad de los datos simulados.",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()

    regiones_a_procesar = (
        list(REGIONES.keys()) if args.region == "todas" else [args.region]
    )

    for region_id in regiones_a_procesar:
        ejecutar_monitoreo(
            region_id=region_id,
            cultivo=args.cultivo,
            dias=args.dias,
            semilla=args.semilla,
        )


if __name__ == "__main__":
    main()