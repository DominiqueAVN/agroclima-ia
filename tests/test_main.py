"""
Pruebas automatizadas para AgroClima IA.

Se ejecutan con: pytest
Verifican que la simulacion y la deteccion de riesgo se comporten como se
espera, para detectar errores automaticamente antes de que lleguen a
produccion o a una demo.
"""

import pandas as pd
import pytest

from main import (
    REGIONES,
    UMBRALES_CULTIVO,
    detectar_riesgo_termico,
    simular_serie_climatica,
)


def test_regiones_no_vacio():
    """Debe haber al menos una region de Latinoamerica y una de Africa."""
    continentes = {r.continente for r in REGIONES.values()}
    assert len(REGIONES) > 0
    assert any("America" in c for c in continentes)
    assert any("Africa" in c for c in continentes)


def test_umbrales_cultivo_no_vacio():
    assert len(UMBRALES_CULTIVO) > 0


def test_simular_serie_climatica_columnas():
    region = REGIONES["arequipa_pe"]
    df = simular_serie_climatica(region, dias=5, semilla=1)

    assert list(df.columns) == ["fecha", "region_id", "temperatura_c", "humedad_pct"]
    assert len(df) == 5 * 8  # 5 dias x 8 mediciones por dia


def test_simular_serie_climatica_reproducible():
    """Con la misma semilla, dos simulaciones deben ser identicas."""
    region = REGIONES["nairobi_ke"]
    df1 = simular_serie_climatica(region, dias=3, semilla=7)
    df2 = simular_serie_climatica(region, dias=3, semilla=7)

    pd.testing.assert_series_equal(df1["temperatura_c"], df2["temperatura_c"])


def test_detectar_riesgo_termico_cultivo_invalido():
    region = REGIONES["arequipa_pe"]
    df = simular_serie_climatica(region, dias=1, semilla=1)

    with pytest.raises(KeyError):
        detectar_riesgo_termico(df, cultivo="cultivo_inexistente")


def test_detectar_riesgo_termico_marca_alerta_correctamente():
    """Una temperatura muy por fuera del umbral debe marcarse como alerta."""
    df = pd.DataFrame(
        {
            "fecha": pd.date_range("2026-01-01", periods=2, freq="D"),
            "region_id": ["arequipa_pe", "arequipa_pe"],
            "temperatura_c": [2.0, 18.0],  # 2.0 esta fuera del rango "germinado"
            "humedad_pct": [50.0, 50.0],
        }
    )

    resultado = detectar_riesgo_termico(df, cultivo="germinado")

    assert resultado.loc[0, "riesgo"] == "alerta"
    assert resultado.loc[1, "riesgo"] == "normal"
    assert resultado.loc[0, "severidad"] > 0
