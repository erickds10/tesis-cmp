# 🧭 Tesis CMP – Detección de Anomalías en Trayectorias AIS (Galápagos)

**Autor:** Erick D. Suárez  
**Año:** 2025  
**Repositorio:** https://github.com/erickds10/tesis-cmp

---

## 🎯 Objetivo
Detectar **anomalías no supervisadas** en trayectorias AIS de pesqueros en/near GMR (Galápagos), integrando AIS con capas geoespaciales y modelando dinámicas por ventanas.

---

## ⚙️ Metodología general

El proyecto sigue un flujo modular basado en los entregables A3–A6 definidos en la planificación:

1. **A3 – Integración y Limpieza de Datos**  
   - Fuentes: AIS, GMR (polígonos), Batimetría GEBCO y distancias a costa/puertos.  
   - Reglas: eliminación de duplicados `(mmsi, timestamp)`, validación de coordenadas y velocidades, interpolación de valores faltantes y segmentación de trayectorias por gaps.  
   - **Output:** `ais_clean.parquet`.

2. **A4 – Ingeniería de Variables y Enriquecimiento**  
   - Derivadas dinámicas: `speed_calc`, `acceleration`, `heading_change`, `angular_velocity`.  
   - Lags y diferencias (`t-1`, `t-2`), medias móviles (SMA/EMA), y ratios físicos (`speed/depth`, `dist_coast/port`).  
   - **Output:** `ais_features.parquet`.

3. **A5 – Serialización por Ventanas Temporales**  
   - Ventanas deslizantes de tamaño `T=20`, paso `10`.  
   - Generación de matrices `N×(T·F)` listas para modelado no supervisado.  
   - **Output:** `ais_serialized_tabular/`.

4. **A6 – Modelado de Detección de Anomalías (OC-SVM)**  
   - **One-Class SVM** con kernel RBF optimizado para grandes volúmenes (27.8M ventanas).  
   - Implementación escalable con:
     - **PCA (16 comps)** → reducción de dimensionalidad.  
     - **Random Fourier Features (8192 comps)** → aproximación del kernel RBF.  
     - **SGDOneClassSVM** → entrenamiento lineal rápido.  
   - **Scoring por lotes (memmap)** sobre dataset completo.

---

## 📂 Estructura
> **Nota:** Los archivos gigantes (memmaps `.dat`, parquets grandes, modelos `.pkl`) están ignorados por `.gitignore`. Se incluyen **configuraciones, CSVs y snapshots ligeros**.
>
> tesis-cmp/
├── notebooks/
│   ├── 01_eda_limpieza.ipynb
│   ├── 02_features_enriquecimiento.ipynb
│   ├── 03_serializacion_ventanas.ipynb
│   ├── 04_ocsvm_modelado.ipynb
│
├── scripts/
│   ├── preprocess_features.py
│   ├── ocsvm_train_eval.py
│   ├── utils_eval.py
│
├── data/
│   └── ocsvm_runs/
│       ├── ocsvm_rbf_config.json
│       ├── ocsvm_rbf_summary_metrics.csv
│       ├── ocsvm_rbf_top1pct_detailed.csv
│       ├── ocsvm_rbf_top1pct_smooth.csv
│       ├── ocsvm_rbf_mmsi_agg.parquet
│       └── final_run_20251027_0238/    ← snapshot completo del modelo final
│
├── requirements.txt
└── README.md

---

## 🧠 Modelo OC-SVM (RBF)

**Arquitectura final ejecutada en Lightning AI Studio:**

| Módulo | Descripción |
|--------|--------------|
| **Input** | 27,789,660 ventanas × 19 features |
| **Prepro** | RobustScaler + imputación por medianas |
| **PCA** | 16 componentes |
| **Random Fourier Features (RFF)** | D = 8192, γ = 0.3 |
| **SGDOneClassSVM** | ν = 0.08, kernel aproximado lineal |
| **Scoring** | batch size = 1.2M filas (memmap de 27.7M) |
| **Runtime total** |  ~20–25 min (CPU 32 threads, RAM 135 GB) |

---

## 🧩 Pipeline (resumen por entregable)
**A3 – Integración/Limpieza:** AIS + GMR + GEBCO + distancias costa/puerto; deduplicación por `(mmsi, timestamp)`, validación de velocidades, segmentación por gaps.  
**A4 – Ingeniería:** variables dinámicas (velocidad, aceleración, giro), lags/diffs, SMA/EMA, ratios con contexto (depth, distancias).  
**A5 – Serialización:** ventanas fijas (T, stride) → matriz tabular (*N×T·F*).  
**A6 – Modelado (este entregable):**  
- **OC-SVM** escalable: **PCA(16) → RFF(8192, γ=0.3) → SGDOneClassSVM(ν=0.08)**.  
- Scoring por lotes con **memmap** sobre **27.8M** ventanas.

**Reconstrucción de etiquetas (eval):** JOIN por `window_id` contra `labels_anom.parquet`.

---

## 📈 Métricas finales

Dataset eval: **27,789,660** ventanas (positivos: **1,003,200** ≈ **3.61%**)

| Variante | ROC-AUC | PR-AUC/AP | P@1% | R@1% | F1@1% |
|----------|---------|-----------|------|------|-------|
| **RAW** | 0.5300 | 0.0464 | 0.0865 | 0.0240 | 0.0375 |
| **SMOOTH (rolling=3 por MMSI)** | **0.5463** | **0.0497** | **0.1060** | **0.0294** | **0.0460** |

**Interpretación:**  
- Lift vs. azar (prevalencia ~0.036): AP~0.0464 (**1.28×**), P@1% = 0.0865 (**2.4×**); con suavizado P@1% = **0.1060** (**2.9×**).  
- El suavizado por MMSI mejora la priorización operativa en el top-1%.

---

## 🔧 Reproducibilidad (alto nivel)

1) Instala dependencias:
bash
  pip install -r requirements.txt
2)	Colocar los parquets limpios y de ventanas en data/ o ajustar CFG["external_data_dir"].
3)	Ejecutar el notebook 04_ocsvm_modelado.ipynb o el script equivalente:
python scripts/ocsvm_train_eval.py
4)	Los resultados se guardan automáticamente en data/ocsvm_runs/.

## requirements.txt
pandas
numpy
scikit-learn
pyarrow
joblib
matplotlib
tqdm
psutil

📦 Snapshot final

/data/ocsvm_runs/final_run_20251027_0238/
Contiene todos los artefactos, configuraciones y métricas reproducibles del modelo final.



