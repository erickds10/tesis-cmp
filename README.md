# Detección no supervisada de anomalías en trayectorias AIS (Galápagos)

Repositorio de tesis (Erick D. Suárez — USFQ). Contiene **todo** el desarrollo teórico y práctico:
datos de ejemplo, scripts de procesamiento, modelos, notebooks, resultados y documentos.

## Objetivo
Modelar trayectorias AIS como series de tiempo para detectar anomalías indicativas de posible pesca ilegal
alrededor de la Reserva Marina de Galápagos (GMR), entrenando con comportamiento normal y validando con eventos marcados como `is_suspicious = 1`.

## Estructura
```
.
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── .gitattributes
├── CITATION.cff
├── configs/
│   ├── paths.yaml
│   └── models/
│       ├── iforest.yaml
│       ├── hdbscan.yaml
│       └── lstm_autoencoder.yaml
├── data/                 # (no se suben datos pesados; usar LFS/DVC)
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   └── windows/.gitkeep
├── docs/
│   ├── planning/planificacion_proyecto.pdf
│   ├── thesis/00_resumen.md
│   ├── thesis/01_introduccion.md
│   ├── thesis/02_estado_del_arte.md
│   ├── thesis/03_metodologia.md
│   ├── thesis/04_resultados.md
│   ├── thesis/05_conclusiones.md
│   └── thesis/refs.bib
├── notebooks/
│   ├── 01_eda_ais.ipynb
│   ├── 02_features_geotemporales.ipynb
│   └── 03_experimentos_modelos.ipynb
├── scripts/
│   ├── procesar_ais_gal.py
│   ├── generar_features.py
│   ├── hacer_ventanas.py
│   └── entrenar_iforest.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loaders.py
│   │   └── utils_geo.py
│   ├── features/
│   │   ├── engineering.py
│   │   └── windows.py
│   ├── models/
│   │   ├── iforest.py
│   │   ├── hdbscan.py
│   │   └── lstm_autoencoder.py
│   └── evaluation/
│       ├── metrics.py
│       └── plots.py
├── tests/
│   ├── test_windows.py
│   └── test_metrics.py
└── .github/workflows/ci.yml
```

## Cómo empezar
1. Crear ambiente (opcional): `conda env create -f environment.yml && conda activate tesis-ais`  
2. Instalar deps: `pip install -r requirements.txt`
3. Ejecutar un flujo mínimo:
   ```bash
   python scripts/procesar_ais_gal.py --input data/raw/sample_ais.csv --out data/interim/ais_clean.parquet
   python scripts/generar_features.py --input data/interim/ais_clean.parquet --out data/processed/ais_feat.parquet
   python scripts/hacer_ventanas.py --input data/processed/ais_feat.parquet --out data/windows/ventanas_tabular.npz --format tabular
   python scripts/entrenar_iforest.py --windows data/windows/ventanas_tabular.npz --out runs/iforest_baseline
   ```

## Convenciones
- **Datos pesados** → Git LFS (ver `.gitattributes`) o DVC.
- **Ramas**: `main` (estable), `dev` (trabajo), `feat/*`, `fix/*`.
- **Commits**: estilo conventional commits (feat:, fix:, docs:, chore:, refactor:, test:).
