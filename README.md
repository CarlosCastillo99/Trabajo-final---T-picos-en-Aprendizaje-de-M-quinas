# Predicción de ausentismo a citas médicas con aprendizaje de máquinas

Proyecto final del curso **TPAM** (2025-II).  
El objetivo es predecir la probabilidad de **no-show** (ausentismo) en citas médicas ambulatorias en Vitória (Brasil), usando modelos tabulares (regresión logística y Random Forest) y métricas enfocadas en priorización operativa.

---

## 1. Descripción general

- **Problema**: dadas las características observables de la cita en el momento del agendamiento (edad, barrio, día y hora de la cita, tiempo de espera, etc.), predecir si el paciente **no asistirá**.
- **Dataset**: `Medical Appointment No Shows` (110 527 registros, 14 variables) publicado en Kaggle.
- **Etiquetas**:  
  - `no_show = 1`: el paciente **no asistió**.  
  - `no_show = 0`: el paciente **asistió**.
- **Meta**: construir un pipeline reproducible que:
  - Logre buen desempeño fuera de muestra (ROC-AUC, PR-AUC).
  - Sea útil para **priorizar** pacientes de alto riesgo cuando la capacidad de intervención (recordatorios, llamadas, reprogramaciones) es limitada.
  - Separe un **score descriptivo** (política histórica) de un **score de política ex-ante** basado en IPW.

---

## 2. Estructura del repositorio

```text
.
├── Proyecto_final_TPAM.ipynb      # Notebook principal con todo el pipeline
├── data/
│   └── noshowappointments.csv     # Dataset descargado desde Kaggle
├── figs/                          # Figuras exportadas usadas en el paper
│   ├── fig_distribuciones.png
│   ├── fig_corr_spearman.png
│   ├── fig_bivariadas.png
│   ├── fig_no_show_por_dia.png
│   ├── fig_no_show_por_sms.png
│   ├── fig_pr_test.png
│   ├── fig_gains_test.png
│   ├── fig_pr_policy_p0_D0test.png
│   ├── fig_gains_policy_p0_D0test.png
│   ├── fig_propensity_overlap.png
│   ├── fig_ipw_weights.png
│   ├── fig_calibration_test.png
│   ├── fig_calibration_policy_D0.png
│   └── fig_perm_importance_policy_rf.png
├── paper/
│   ├── topicos_proyecto_final.tex # Manuscrito en LaTeX
│   └── topicos_proyecto_final.pdf # Versión final del artículo
└── README.md

3. Requisitos

Python ≥ 3.9

Librerías principales:

numpy

pandas

matplotlib

seaborn

scikit-learn

jupyter / notebook

Instalación sugerida (una vez creado un entorno virtual):

pip install -r requirements.txt
# o manualmente:
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

4. Datos

Crear la carpeta data/ en la raíz del repositorio.

Descargar el dataset Medical Appointment No Shows desde Kaggle.

Guardar el archivo CSV original como data/noshowappointments.csv (o actualizar la ruta correspondiente en el notebook).

El notebook realiza la limpieza mínima:

Estandarización de nombres de columnas.

Conversión de fechas a datetime.

Construcción de variables derivadas:

wait_days (días entre programación y cita).

appt_weekday (día de la semana de la cita).

sched_hour (hora de programación).

Eliminación de la única observación con edad negativa.

Creación de la etiqueta binaria no_show.

5. Cómo reproducir los resultados

Clonar el repositorio:

git clone <URL_DEL_REPO>
cd <carpeta_del_repo>


Crear y activar un entorno virtual (opcional, pero recomendado).

Instalar dependencias (ver sección 3).

Lanzar Jupyter:

jupyter notebook


Abrir Proyecto_final_TPAM.ipynb y ejecutar todas las celdas en orden.

El notebook:

Carga y limpia los datos.

Genera las figuras de EDA (distribuciones, correlaciones, relaciones bivariadas, tasas por día y por SMS).

Construye el pipeline de preprocesamiento (ColumnTransformer + Pipeline).

Ajusta y evalúa:

Regresión logística (baseline).

Random Forest (modelo no lineal).

Calcula métricas en validación cruzada y en prueba:

ROC-AUC.

PR-AUC (Average Precision).

P@10% y P@20% (Precision@k).

Curvas Precision–Recall y curvas de ganancias acumuladas.

Implementa el score de política con IPW (entrenamiento en D=0 con pesos).

Genera diagnósticos de:

Overlap del propensity score.

Distribución de pesos IPW y ESS.

Sensibilidad a diferentes caps de pesos.

Calibración (curvas de confiabilidad y Brier score).

Importancia de variables por permutation importance.

Todas las figuras se guardan en la carpeta figs/ y las tablas principales se imprimen en el notebook.

6. Resultados principales (resumen)

Desbalance: prevalencia de no-show ≈ 20 %.

Modelos estándar (toda la muestra):

Regresión logística:

ROC-AUC ≈ 0.67

PR-AUC ≈ 0.31

P@10% ≈ 0.36, P@20% ≈ 0.35

Random Forest:

ROC-AUC ≈ 0.74

PR-AUC ≈ 0.37

P@10% ≈ 0.42, P@20% ≈ 0.39

El Random Forest domina a la logística tanto en AUC como en capacidad de concentrar no-shows en el top-k de riesgo.

Score de política (estimación de $p_0(X) = \Pr(Y=1 \mid X, D=0)$ con IPW y cap p99):

Policy-LogReg:

ROC-AUC ≈ 0.64, PR-AUC ≈ 0.27

Policy-RF:

ROC-AUC ≈ 0.78, PR-AUC ≈ 0.37

P@10% ≈ 0.43, P@20% ≈ 0.38

Interpretación operativa:

En un problema con prevalencia ≈20 %, P@10% ≈ 0.42–0.43 implica lifts de alrededor de 2–2.6 veces la línea base, es decir, el modelo permite concentrar buena parte de los no-shows en una fracción pequeña de pacientes priorizados.

El score de política $\hat{p}_0(X)$ está pensado para decisiones ex-ante (riesgo sin SMS), mientras que el modelo estándar refleja el régimen histórico de asignación de SMS.

7. Uso de IA y fuentes adicionales

Los conceptos de propensity score, tamaño efectivo de muestra (ESS) y winsorización de pesos IPW no se trabajaron en detalle en el curso. Para poder explorar estas ideas y redactar los apéndices técnicos:

Revisé bibliografía estándar de inferencia causal (IPW, overlap, ESS).

Utilicé asistentes de inteligencia artificial (por ejemplo, Gemini / ChatGPT) como apoyo en:

Búsqueda de referencias.

Borradores de explicaciones conceptuales.

Sugerencias de gráficos/tablas de robustez.

El procesamiento de datos, la implementación concreta en el notebook, la elección de escenarios de sensibilidad (sin cap, p95, p99, cap 20) y la validación de resultados fueron realizados por mí. Esta parte del proyecto debe verse como material complementario de aprendizaje y no como contenido central del curso.

8. Créditos

Autor del proyecto: Carlos Castillo.

Curso: TPAM (Proyecto final, 2025-II).

Dataset:

J. Aroba – Medical Appointment No Shows (Kaggle).

Software: Python, scikit-learn y ecosistema científico estándar.

Este README condensa lo que cuentas en el artículo (datos, pipeline, métricas, score de política, IPW, calibración) y deja explícito el uso de IA como apoyo en la parte más avanzada del apéndice. 
::contentReference[oaicite:1]{index=1}
