# 🧭 Tesis CMP – Detección de Anomalías en Trayectorias AIS (Galápagos)

**Autor:** Erick D. Suárez  
**Año:** 2025  
**Repositorio:** https://github.com/erickds10/tesis-cmp

---

## 🎯 Objetivo
Detectar **anomalías no supervisadas** en trayectorias AIS de pesqueros en/near GMR (Galápagos), integrando AIS con capas geoespaciales y modelando dinámicas por ventanas.

---

## 📂 Estructura
> **Nota:** Los archivos gigantes (memmaps `.dat`, parquets grandes, modelos `.pkl`) están ignorados por `.gitignore`. Se incluyen **configuraciones, CSVs y snapshots ligeros**.

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
```bash
pip install -r requirements.txt
MIT © 2025 – Erick D. Suárez
