# AgroClima IA

Sistema de alerta temprana climática para agricultura de pequeña escala en
Latinoamérica y África.

## Descripción

AgroClima IA es un sistema de monitoreo y detección de riesgo térmico
orientado a cultivos sensibles en etapa de germinación. Combina hardware de
bajo costo (microcontrolador + sensores de temperatura/humedad) con un
modelo de reconocimiento de patrones climáticos, diseñado para operar sin
depender de infraestructura estatal centralizada: los datos permanecen bajo
control de la cooperativa o comunidad local que opera el sistema.

El proyecto no esta acotado a una sola región. El módulo de simulación
incluye perfiles climáticos de referencia de América Latina (Perú, Bolivia,
México) y África (Kenia, Burkina Faso, Madagascar) como base de prueba antes
de integrar datos de campo reales.

## Objetivo

Proveer alertas tempranas a pequeños agricultores y cooperativas agrícolas,
reduciendo pérdidas por eventos térmicos extremos (heladas fuera de
temporada, olas de calor cortas) en cultivos de ciclo corto.

## Tecnologías

- Python 3.12+
- NumPy / Pandas para procesamiento de datos
- Scikit-learn para el modelo de deteccion de riesgo (fase 2)
- ESP32 + DHT22 como hardware de sensores (fase 3)
- EasyEDA para el diseño de circuito del sensor
- Webots para la simulacion previa del comportamiento del sistema
- Streamlit para el panel de datos agregados (fase 4)

## Estructura del proyecto

```
agroclima-ia/
├── data/
│   ├── raw/          # Datos de campo sin procesar (fase posterior)
│   ├── processed/    # Datos limpios y listos para modelado
│   └── synthetic/     # Datos simulados generados por main.py
├── models/
│   ├── trained/       # Modelos entrenados serializados
│   └── notebooks/     # Notebooks de exploracion y prototipado
├── src/                # Codigo fuente modular (fase 2 en adelante)
├── hardware/
│   ├── easyeda/        # Archivos de diseño del circuito
│   └── webots/         # Mundos y controladores de simulacion
├── docs/                # Reportes generados y documentacion
├── main.py
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone https://github.com/DominiqueAVN/agroclima-ia.git
cd agroclima-ia
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Uso

Simular una región específica:

```bash
python main.py --region arequipa_pe --cultivo germinado --dias 30
```

Simular todas las regiones definidas:

```bash
python main.py --region todas --cultivo germinado
```

Regiones disponibles: `arequipa_pe`, `cochabamba_bo`, `oaxaca_mx`,
`nairobi_ke`, `ouagadougou_bf`, `antananarivo_mg`.

Cultivos disponibles: `germinado`, `lechuga`, `tomate`, `quinua`, `sorgo`,
`mijo`, `maiz`.

Cada ejecución genera un archivo CSV en `data/synthetic/` y un reporte de
texto en `docs/`.

## Estado del proyecto

Fase actual: simulación de datos y logica de detección de riesgo térmico.

Próximos pasos: entrenamiento de un modelo de predicción con scikit-learn,
diseño del circuito de sensores en EasyEDA, validación lógica en Webots
antes de la construcción física.

## Referencias bibliográficas

- FuXi Weather: data-to-forecast machine learning system for global weather
  (arXiv:2408.05472)
- MASK-CNN-Transformer for real-time multi-label weather recognition
  (arXiv:2304.14857)
- DEWASAT-2: a 6U CubeSat platform for low Earth remote sensing (Aerospace,
  2023)
- Utilizing low-cost Linux micro-computer and Android phone solutions on
  CubeSats (arXiv:2205.08255)

## Licencia

MIT License. Ver archivo LICENSE para detalles.