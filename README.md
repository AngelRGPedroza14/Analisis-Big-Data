# Predicción del Rendimiento Escolar mediante Regresión Lineal Múltiple (Big Data & IA)

Este proyecto implementa un modelo predictivo basado en **Regresión Lineal Múltiple (RLM)** utilizando técnicas de **Big Data** y **Machine Learning Supervisado**. Su objetivo es predecir la calificación final (`G3`) de los estudiantes a partir de variables asociadas a su contexto educativo, hábitos de estudio, disciplina y estado de salud general.

La aplicación incluye un pipeline automatizado de **limpieza exhaustiva de datos** (eliminación de ruido y outliers semánticos) y una **interfaz gráfica web interactiva** desarrollada con **Gradio** para facilitar las predicciones en tiempo real.

---

## Arquitectura y Descripción del Proyecto

El rendimiento académico es un fenómeno multifactorial. Este proyecto aborda la problemática desde una perspectiva puramente cuantitativa, analizando el impacto de múltiples variables independientes para generar una ecuación predictiva de la calificación final.

### Algoritmo Utilizado: Regresión Lineal Múltiple

La relación se modela bajo la siguiente ecuación matemática formal:

$$
\text{G3} = \beta_0 + \beta_1(\text{G1}) + \beta_2(\text{G2}) + \beta_3(\text{studytime}) + \beta_4(\text{failures}) + \beta_5(\text{absences}) + \beta_6(\text{health}) + \epsilon
$$

Donde:

- **$\beta_0$**: Intercepto del modelo  
- **$\beta_1 \dots \beta_6$**: Coeficientes de regresión (peso de cada variable)  
- **$\epsilon$**: Término de error aleatorio  

---

## Variables del Dataset (`student-mat2.csv`)

El dataset utilizado se basa en el estándar UCI Student Performance. Se seleccionaron las características con mayor correlación e importancia estadística.

| Tipo | Variable | Descripción | Rango Válido | Signo Esperado |
|------|----------|-------------|--------------|----------------|
| **Dependiente (Y)** | `G3` | Calificación final del curso | `[0, 20]` | — |
| **Independiente (X)** | `G1` | Calificación del primer periodo | `[0, 20]` | `+` |
| **Independiente (X)** | `G2` | Calificación del segundo periodo | `[0, 20]` | `+` |
| **Independiente (X)** | `studytime` | Tiempo de estudio semanal (1–4) | `[1, 4]` | `+` |
| **Independiente (X)** | `failures` | Materias reprobadas previamente | `[0, 4]` | `-` |
| **Independiente (X)** | `absences` | Número de faltas escolares | `[0, 50+]` | `-` |
| **Independiente (X)** | `health` | Estado de salud (1–5) | `[1, 5]` | `+` |

---

## Pipeline de Limpieza de Datos (Optimización)

Para maximizar el coeficiente de determinación ($R^2$) y reducir el error cuadrático medio ($RMSE$), el sistema ejecuta la función `cargar_y_limpiar()`:

1. **Remoción de columnas irrelevantes**: Eliminación de variables categóricas no numéricas del dataset original.
2. **Tratamiento de nulos**: Conversión de valores vacíos a `NA` y eliminación de registros incompletos.
3. **Conversión de tipos**: Asegura compatibilidad total con `scikit-learn`.
4. **Validación de rangos**: Filtrado de datos fuera de los rangos académicos válidos.
5. **Eliminación de outliers semánticos**: Corrección de inconsistencias lógicas en el rendimiento académico.

---

## Instalación y Uso

### Prerrequisitos

Asegúrate de tener instalado Python 3.8 o superior:

```bash
pip install gradio pandas scikit-learn numpy matplotlib seaborn
```

---

### Estructura de Archivos Recomendada

```
├── bigdata.py          # Script principal
└── student-mat2.csv    # Dataset limpio
```

---

### Ejecución del Proyecto

Ejecuta el siguiente comando:

```bash
python bigdata.py
```

Al iniciar, el sistema entrenará automáticamente el modelo y mostrará una URL local:

```
Running on local URL: http://127.0.0.1:7860
```

---

## Evaluación del Modelo y Resultados

El modelo se entrena con división **80% entrenamiento / 20% prueba** (`test_size=0.2`).

Métricas generadas:

- **Coeficiente de determinación ($R^2$)**: porcentaje de varianza explicada por el modelo.
- **Error cuadrático medio ($RMSE$)**: magnitud promedio del error de predicción.

---

## Interfaz de Usuario (Gradio)

La interfaz se compone de dos paneles:

- **Panel izquierdo (inputs):** sliders para G1, G2, estudio, faltas, salud y reprobaciones.
- **Panel derecho (outputs):** botón de predicción ` PREDECIR NOTA FINAL` con resultado visual dinámico.

---

*Sistema diseñado para analítica predictiva educativa y detección temprana de riesgo académico.*

---
## Autor

Angel Rodrigo Gutiérrez Pedroza

Egresado en proceso de Residencias Profesionales de Ingeniería en Sistemas Computacionales con especialidad en Redes.
