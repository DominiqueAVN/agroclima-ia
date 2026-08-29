"""
Dashboard de AgroClima IA.

Panel visual con mapa interactivo para mostrar el estado climatico y las
predicciones de riesgo de las regiones monitoreadas, pensado para presentar
a cooperativas, funcionarios publicos o evaluadores.

Se ejecuta con: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from main import REGIONES, UMBRALES_CULTIVO  # noqa: E402

CSV_PATH = ROOT_DIR / "data" / "synthetic" / "predicciones_ml.csv"
METRICS_PATH = ROOT_DIR / "models" / "trained" / "metricas_modelo.json"

NOMBRES_REGION = {rid: f"{r.nombre}, {r.pais}" for rid, r in REGIONES.items()}

st.set_page_config(page_title="AgroClima IA - Panel de monitoreo", layout="wide")


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["fecha"])
    df["region_display"] = df["region_nombre"].map(NOMBRES_REGION).fillna(df["region_nombre"])
    return df


@st.cache_data
def cargar_metricas() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open(encoding="utf-8") as archivo:
        return json.load(archivo)


def calcular_riesgo_por_cultivo(temperatura: float) -> pd.DataFrame:
    """Evalua la temperatura actual contra el umbral de cada cultivo definido."""
    filas = []
    for cultivo, umbral in UMBRALES_CULTIVO.items():
        en_riesgo = temperatura < umbral.temp_min_c or temperatura > umbral.temp_max_c
        filas.append(
            {
                "Cultivo": cultivo,
                "Rango seguro (C)": f"{umbral.temp_min_c:.0f} - {umbral.temp_max_c:.0f}",
                "Estado actual": "En riesgo" if en_riesgo else "Normal",
            }
        )
    return pd.DataFrame(filas)


if not CSV_PATH.exists():
    st.error("No se encontro el archivo de predicciones. Corre primero: python models/aplicar_modelo.py")
    st.stop()

df = cargar_datos()

st.title("AgroClima IA")
st.caption(
    "Sistema de alerta temprana climatica para agricultura de pequena escala "
    "en Latinoamerica y Africa."
)

# ---------------------------------------------------------------------------
# Datos mas recientes por region (para el mapa y el panorama general)
# ---------------------------------------------------------------------------
ultimos = df.sort_values("fecha").groupby("region_nombre").tail(1).set_index("region_nombre")

filas_mapa = []
for region_id, region in REGIONES.items():
    if region_id not in ultimos.index:
        continue
    fila = ultimos.loc[region_id]
    filas_mapa.append(
        {
            "region_id": region_id,
            "nombre": f"{region.nombre}, {region.pais}",
            "continente": region.continente,
            "lat": region.latitud,
            "lon": region.longitud,
            "temperatura": fila["temperatura_c"],
            "humedad": fila["humedad_pct"],
            "alerta": bool(fila["riesgo_predicho_ml"]),
        }
    )
df_mapa = pd.DataFrame(filas_mapa)

# ---------------------------------------------------------------------------
# Estado de seleccion (se sincroniza entre el mapa y el selector lateral)
# ---------------------------------------------------------------------------
if "region_activa" not in st.session_state:
    st.session_state["region_activa"] = df_mapa.iloc[0]["region_id"]

st.subheader("Mapa de monitoreo - Latinoamerica y Africa")
st.caption("Haz click en un punto para ver el informe completo de esa region.")

colores = df_mapa["alerta"].map({True: "#e63946", False: "#2a9d8f"})

texto_hover = [
    f"{row.nombre}<br>Temp: {row.temperatura:.1f} C<br>Humedad: {row.humedad:.0f} %"
    for row in df_mapa.itertuples()
]

fig_mapa = go.Figure(
    go.Scattergeo(
        lat=df_mapa["lat"],
        lon=df_mapa["lon"],
        text=df_mapa["nombre"],
        hovertext=texto_hover,
        customdata=df_mapa["region_id"],
        mode="markers",
        marker=dict(size=16, color=colores, line=dict(width=1, color="white")),
        hovertemplate="%{hovertext}<extra></extra>",
    )
)
fig_mapa.update_geos(
    showcountries=True,
    countrycolor="rgba(255,255,255,0.3)",
    showland=True,
    landcolor="#1a1a2e",
    showocean=True,
    oceancolor="#0d0d1a",
    projection_type="natural earth",
    lataxis_range=[-40, 40],
    lonaxis_range=[-100, 60],
)
fig_mapa.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0))

seleccion = st.plotly_chart(
    fig_mapa,
    width='stretch',
    on_select="rerun",
    selection_mode="points",
    key="mapa_regiones",
)

puntos = seleccion.get("selection", {}).get("points", []) if seleccion else []
if puntos:
    indice_click = puntos[0]["point_index"]
    st.session_state["region_activa"] = df_mapa.iloc[indice_click]["region_id"]

# Selector lateral, sincronizado con el mapa
region_ids_ordenados = df_mapa["region_id"].tolist()
nombres_ordenados = df_mapa["nombre"].tolist()
indice_actual = region_ids_ordenados.index(st.session_state["region_activa"])

nombre_elegido = st.sidebar.selectbox("Region", nombres_ordenados, index=indice_actual)
region_id_elegida = region_ids_ordenados[nombres_ordenados.index(nombre_elegido)]
st.session_state["region_activa"] = region_id_elegida

# ---------------------------------------------------------------------------
# Informe completo de la region seleccionada
# ---------------------------------------------------------------------------
region_obj = REGIONES[region_id_elegida]
df_region = df[df["region_nombre"] == region_id_elegida].sort_values("fecha")
ultimo_registro = df_region.iloc[-1]

st.divider()
st.header(f"Informe completo - {region_obj.nombre}, {region_obj.pais}")
st.caption(
    f"{region_obj.continente} | Altitud {region_obj.altitud_m:.0f} m | "
    f"Coordenadas {region_obj.latitud}, {region_obj.longitud}"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura actual", f"{ultimo_registro['temperatura_c']:.1f} C")
col2.metric("Humedad actual", f"{ultimo_registro['humedad_pct']:.0f} %")
col3.metric("Estado actual", "En riesgo" if ultimo_registro["riesgo_actual"] == 1 else "Normal")
col4.metric(
    "Probabilidad de riesgo (proximas horas)",
    f"{ultimo_registro['probabilidad_riesgo']:.0%}",
)

if ultimo_registro["riesgo_predicho_ml"] == 1:
    st.warning(
        f"El modelo predice riesgo termico en {region_obj.nombre} en las "
        f"proximas horas (probabilidad: {ultimo_registro['probabilidad_riesgo']:.0%}). "
        f"Se recomienda notificar a los productores de la zona."
    )

metricas = cargar_metricas()
if metricas and region_id_elegida in metricas.get("por_region", {}):
    m_region = metricas["por_region"][region_id_elegida]
    if m_region["casos_alerta_reales"] < 20:
        st.info(
            f"Confiabilidad del modelo en {region_obj.nombre}: en el periodo de "
            f"prueba hubo muy pocos casos de riesgo reales ({m_region['casos_alerta_reales']}) "
            f"para validar el modelo con solidez estadistica en esta region especifica. "
            f"Las predicciones aqui deben tomarse con mas cautela que en regiones con "
            f"mas casos historicos."
        )
    else:
        st.caption(
            f"Confiabilidad estimada para {region_obj.nombre}: de cada 100 eventos de "
            f"riesgo reales ocurridos en el periodo de prueba, el modelo detecto "
            f"{m_region['recall_alerta']:.0%} (recall) con una precision de "
            f"{m_region['precision_alerta']:.0%} en sus predicciones de alerta."
        )

st.subheader("Riesgo por tipo de cultivo (a la temperatura actual)")
st.caption(
    "Comparacion de la temperatura actual contra el rango seguro de cada "
    "cultivo. No requiere modelo entrenado por separado para cada uno."
)
tabla_cultivos = calcular_riesgo_por_cultivo(ultimo_registro["temperatura_c"])
st.dataframe(tabla_cultivos, width='stretch', hide_index=True)

st.subheader("Temperatura reciente")
ventana_reciente = df_region.tail(240)
fig_linea = px.line(
    ventana_reciente, x="fecha", y="temperatura_c",
    labels={"fecha": "Fecha", "temperatura_c": "Temperatura (C)"},
)
alertas = ventana_reciente[ventana_reciente["riesgo_predicho_ml"] == 1]
if not alertas.empty:
    fig_linea.add_scatter(
        x=alertas["fecha"], y=alertas["temperatura_c"],
        mode="markers", marker=dict(color="red", size=8), name="Riesgo predicho",
    )
st.plotly_chart(fig_linea, width='stretch')

st.subheader("Probabilidad de riesgo en el tiempo")
st.caption(
    "Proporcion de arboles del modelo (de 200 en total) que votaron por "
    "'riesgo' en cada punto. Es la forma correcta de leer esto como nivel de "
    "confianza, no como una certeza absoluta."
)
fig_prob = px.area(
    ventana_reciente, x="fecha", y="probabilidad_riesgo",
    labels={"fecha": "Fecha", "probabilidad_riesgo": "Probabilidad de riesgo"},
    range_y=[0, 1],
)
fig_prob.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="umbral de decision (50%)")
st.plotly_chart(fig_prob, width='stretch')

# ---------------------------------------------------------------------------
# Panorama de todas las regiones
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Panorama general (todas las regiones)")
resumen = (
    df.sort_values("fecha")
    .groupby("region_display")
    .agg(
        temperatura_actual=("temperatura_c", "last"),
        alertas_ultimos_30d=("riesgo_predicho_ml", lambda s: int(s.tail(240).sum())),
    )
    .reset_index()
    .rename(columns={"region_display": "Region"})
)
st.dataframe(resumen, width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Panel de desempeno estadistico del modelo (transparencia)
# ---------------------------------------------------------------------------
st.divider()
st.header("Desempeno del modelo (validacion estadistica)")
st.caption(
    "Metricas calculadas sobre datos que el modelo NUNCA vio durante el "
    "entrenamiento (division cronologica: se entrena con el pasado, se "
    "evalua con el futuro). Esto es lo que respalda tecnicamente al sistema."
)

if metricas is None:
    st.warning("No se encontraron metricas del modelo. Corre python models/train_model.py primero.")
else:
    colg1, colg2, colg3, colg4 = st.columns(4)
    colg1.metric("Exactitud global (accuracy)", f"{metricas['global']['accuracy']:.1%}")
    colg2.metric("Recall en Alerta", f"{metricas['global']['recall_alerta']:.1%}")
    colg3.metric("Precision en Alerta", f"{metricas['global']['precision_alerta']:.1%}")
    colg4.metric("F1-score en Alerta", f"{metricas['global']['f1_alerta']:.1%}")

    st.caption(
        f"Comparado contra un modelo de 'persistencia' (asumir que el riesgo "
        f"futuro sera igual al actual): ese metodo ingenuo logra "
        f"{metricas['baseline_persistencia']['recall_alerta']:.1%} de recall, "
        f"contra {metricas['global']['recall_alerta']:.1%} del modelo entrenado. "
        f"Evaluado sobre {metricas['casos_prueba']:,} casos de prueba, "
        f"prediciendo {metricas['horizonte_pasos']} lecturas hacia adelante."
    )

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        st.markdown("**Matriz de confusion (conjunto de prueba)**")
        mc = metricas["matriz_confusion"]
        tabla_mc = pd.DataFrame(
            [
                ["Normal (real)", mc["normal_predicho_normal"], mc["normal_predicho_alerta"]],
                ["Alerta (real)", mc["alerta_predicho_normal"], mc["alerta_predicho_alerta"]],
            ],
            columns=["Real \\ Predicho", "Predijo Normal", "Predijo Alerta"],
        )
        st.dataframe(tabla_mc, width='stretch', hide_index=True)

    with col_mc2:
        st.markdown("**Variables mas influyentes en la prediccion**")
        var_importancia = pd.DataFrame(
            list(metricas["variables_mas_influyentes"].items()),
            columns=["Variable", "Importancia"],
        )
        st.dataframe(var_importancia, width='stretch', hide_index=True)

    st.markdown("**Confiabilidad desglosada por region**")
    st.caption(
        "El desempeno global promedia todas las regiones, pero la confiabilidad "
        "real varia segun cuantos casos de riesgo tuvo cada una para validar el "
        "modelo. Mostrar esto sin filtrar es una decision deliberada de "
        "transparencia: un numero global oculta estas diferencias."
    )
    filas_region = []
    for rid, datos in metricas["por_region"].items():
        filas_region.append(
            {
                "Region": NOMBRES_REGION.get(rid, rid),
                "Casos de riesgo en prueba": datos["casos_alerta_reales"],
                "Recall": f"{datos['recall_alerta']:.0%}",
                "Precision": f"{datos['precision_alerta']:.0%}",
                "Confiabilidad": "Alta" if datos["casos_alerta_reales"] >= 20 else "Baja (pocos datos)",
            }
        )
    tabla_region = pd.DataFrame(filas_region).sort_values(
        "Casos de riesgo en prueba", ascending=False
    )
    st.dataframe(tabla_region, width='stretch', hide_index=True)

st.caption(
    "Datos simulados con fines de demostracion. Cada region representa una "
    "ciudad de referencia, no la cobertura del pais completo. En produccion, "
    "estos valores provendrian de sensores fisicos instalados en campo, y la "
    "cobertura geografica se ampliaria progresivamente con datos reales."
)