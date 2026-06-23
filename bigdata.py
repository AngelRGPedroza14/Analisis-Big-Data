import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # backend sin Tkinter
import seaborn as sns

# -------------------------------------------------
# Función para cargar y LIMPIAR el dataset
# (modelo optimizado sin outliers semánticos)
# -------------------------------------------------
def cargar_y_limpiar(path="student-mat2.csv"):
    print("\n====================")
    print("Cargando dataset...")
    print("====================")

    # 1. Cargar dataset original UCI
    df = pd.read_csv(path, sep=';')
    print(f"Dataset cargado: {len(df)} registros")

    # 2. Seleccionar columnas útiles
    cols = ["studytime", "failures", "absences", "G1", "G2", "health", "G3"]
    df = df[cols]
    print("Columnas seleccionadas:", cols)

    # 3. Quitar vacíos / nulos
    before_dropna = len(df)
    df = df.replace("", pd.NA).dropna()
    print(f"Registros eliminados por nulos/vacíos: {before_dropna - len(df)}")

    # 4. A numérico
    df = df.astype(float)
    print("Conversión numérica realizada")

    # 5. Rangos válidos (reglas UCI)
    before_ranges = len(df)
    df = df[
        (df["studytime"].between(1, 4)) &
        (df["failures"].between(0, 4)) &
        (df["absences"] >= 0) &
        (df["G1"].between(0, 20)) &
        (df["G2"].between(0, 20)) &
        (df["G3"].between(0, 20)) &
        (df["health"].between(1, 5))
    ]
    print(f"Registros eliminados por rangos inválidos: {before_ranges - len(df)}")

    # 6. Eliminar outliers semánticos:
    print("\n🧹 Eliminando outliers semánticos...")
    cond_outlier = (df["G3"] == 0) & (((df["G1"] + df["G2"]) / 2) >= 5)
    outliers_sem_count = cond_outlier.sum()
    
    df_clean = df[~cond_outlier].copy()
    print(f"Outliers semánticos eliminados: {outliers_sem_count}")

    # 7. Guardar CSV limpio
    df_clean.to_csv("student-mat-clean.csv", index=False, sep=';')
    print(f"\nCSV limpio guardado como: student-mat-clean.csv")
    print(f"Registros finales después de limpieza: {len(df_clean)}\n")

    return df_clean


# -------------------------------------------------
# ENTRENAR EL MODELO CON EL CSV LIMPIO
# -------------------------------------------------
def entrenar_modelo():
    print("==========================")
    print("Entrenando modelo RLM...")
    print("==========================\n")

    df = cargar_y_limpiar("student-mat2.csv")

    X = df[['G1', 'G2', 'studytime', 'failures', 'absences', 'health']]
    y = df['G3']

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"División entrenamiento/prueba: {len(X_train)} / {len(X_test)}")

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print("Modelo entrenado correctamente")

    # Métricas
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² del modelo: {r2:.4f}")
    print(f"RMSE del modelo: {rmse:.4f}\n")

    # =====================================================
    # 1) GRÁFICO DE DISPERSIÓN SIMPLE
    # =====================================================
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred, color='royalblue', s=70, edgecolor='white')
    plt.plot([0, 20], [0, 20], '--', color='red', label='Línea perfecta (y=x)')
    plt.xlabel('Notas reales (G3)')
    plt.ylabel('Notas predichas')
    plt.title(f'Dispersión Simple\nR²={r2:.2f} | RMSE={rmse:.2f}', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig("grafico_simple.png")
    plt.close()

    print("Gráfico simple guardado: grafico_simple.png")

    # =====================================================
    # 2) GRÁFICO DE DISPERSIÓN SEGMENTADO EN 6 ÁREAS
    # =====================================================
    y_real = y_test.values
    y_pred_np = y_pred

    x = y_real
    bins_x = np.digitize(x, [5, 15])  # <5, [5,15), >=15

    over = y_pred_np >= y_real   # sobreestima
    under = ~over                # subestima

    mask1 = (bins_x == 0) & over
    mask2 = (bins_x == 0) & under
    mask3 = (bins_x == 1) & over
    mask4 = (bins_x == 1) & under
    mask5 = (bins_x == 2) & over
    mask6 = (bins_x == 2) & under

    plt.figure(figsize=(7, 5))

    plt.scatter(x[mask1], y_pred_np[mask1], s=70, edgecolor='white', label="1: Bajo + sobreestima")
    plt.scatter(x[mask2], y_pred_np[mask2], s=70, edgecolor='white', label="2: Bajo + subestima")
    plt.scatter(x[mask3], y_pred_np[mask3], s=70, edgecolor='white', label="3: Medio + sobreestima")
    plt.scatter(x[mask4], y_pred_np[mask4], s=70, edgecolor='white', label="4: Medio + subestima")
    plt.scatter(x[mask5], y_pred_np[mask5], s=70, edgecolor='white', label="5: Alto + sobreestima")
    plt.scatter(x[mask6], y_pred_np[mask6], s=70, edgecolor='white', label="6: Alto + subestima")

    plt.plot([0, 20], [0, 20], '--', color='red')

    plt.axvline(x=5, color='black', linestyle='--')
    plt.axvline(x=15, color='black', linestyle='--')

    plt.xlabel('Notas reales (G3)')
    plt.ylabel('Notas predichas')
    plt.title(f'Dispersión Segmentada (6 áreas)\nR²={r2:.2f} | RMSE={rmse:.2f}')
    plt.legend(fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 20)
    plt.ylim(0, 20)

    plt.tight_layout()
    plt.savefig("grafico_segmentado.png")
    plt.close()

    print("Gráfico segmentado guardado: grafico_segmentado.png\n")

    # DEVOLVEMOS AMBAS RUTAS
    return modelo, r2, rmse, "grafico_simple.png", "grafico_segmentado.png"

# Entrenar al inicio
modelo, r2, rmse, grafico_simple, grafico_segmentado = entrenar_modelo()


# -------------------------------------------------
# FUNCIÓN DE PREDICCIÓN
# -------------------------------------------------
def predecir(G1, G2, studytime, failures, absences, health):
    try:
        datos = pd.DataFrame({
            'G1': [G1],
            'G2': [G2],
            'studytime': [studytime],
            'failures': [failures],
            'absences': [absences],
            'health': [health]
        })

        nota = float(modelo.predict(datos)[0])

        # Clasificación simple
        if nota < 10:
            nivel = "🔴 Desempeño bajo"
            clase_nivel = "bajo"
        elif nota < 15:
            nivel = "🟡 Desempeño medio"
            clase_nivel = "medio"
        else:
            nivel = "🟢 Buen desempeño"
            clase_nivel = "alto"

        html = f"""
        <h3>Resultados de la Predicción</h3>

        <div class="nota-final">{nota:.2f}</div>

        <div class="nivel {clase_nivel}">
            {nivel}
        </div>

        <hr>

        <div class="metricas">
            <strong>R²:</strong> {r2:.2f}<br>
            <strong>RMSE:</strong> {rmse:.2f}<br>
            <strong>Tiempo de estudio:</strong> {studytime}<br>
            <strong>Materias reprobadas:</strong> {failures}
        </div>
        """

        return html, grafico_simple, grafico_segmentado

    except Exception as e:
        return f"<pre>Error: {repr(e)}</pre>", None, None


# -------------------------------------------------
# UI DE GRADIO
# -------------------------------------------------
print("===============================")
print("Lanzando interfaz Gradio...")
print("===============================\n")

CSS = """
#resultado_card {
    padding: 14px 18px;
    background: #1e293b; /* azul grisáceo suave */
    border: 1px solid #475569;
    border-radius: 12px;
    color: #f1f5f9;
    font-family: 'Segoe UI', sans-serif;
    max-width: 780px;
    max-height: 480px
}

#resultado_card h3 {
    margin-bottom: 8px;
    font-size: 18px;
    font-weight: 600;
}

#resultado_card .nota-final {
    font-size: 32px;
    font-weight: bold;
    margin: 8px 0;
    color: #38bdf8; /* azul bonito */
}

#resultado_card .nivel {
    padding: 6px 10px;
    border-radius: 8px;
    display: inline-block;
    margin-top: 6px;
    font-size: 14px;
    font-weight: 600;
}

#resultado_card .bajo {
    background: rgba(239,68,68,0.2);
    color: #fca5a5;
}

#resultado_card .medio {
    background: rgba(234,179,8,0.2);
    color: #fcd34d;
}

#resultado_card .alto {
    background: rgba(34,197,94,0.2);
    color: #86efac;
}

#resultado_card .metricas {
    margin-top: 12px;
    font-size: 20px;
    line-height: 30px;
}

#resultado_card hr {
    border: none;
    border-top: 1px solid #334155;
    margin: 10px 0;
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        """
        # Predicción del Rendimiento Escolar
        Usa este modelo de **Regresión Lineal Múltiple** para predecir la nota final (G3) de un estudiante.
        El modelo se entrena automáticamente al iniciar la app con un dataset limpio y optimizado.
        """
    )

    with gr.Row():
        with gr.Column():
            G1 = gr.Slider(0, 20, value=10, step=1, label="G1 (Nota 1er periodo)")
            G2 = gr.Slider(0, 20, value=10, step=1, label="G2 (Nota 2do periodo)")
            studytime = gr.Slider(1, 4, value=2, step=1, label="Tiempo de estudio semanal (1-4)")
            failures = gr.Slider(0, 5, value=0, step=1, label="Número de materias reprobadas")
            absences = gr.Slider(0, 50, value=3, step=1, label="Número de ausencias")
            health = gr.Slider(1, 5, value=3, step=1, label="Salud (1=mala, 5=excelente)")

        with gr.Column():
            btn_predecir = gr.Button("🔮  PREDECIR NOTA FINAL")
            salida = gr.HTML(elem_id="resultado_card")

    with gr.Row():
        with gr.Column():
            grafico1 = gr.Image(label="Dispersión simple", type="filepath")
        with gr.Column():
            grafico2 = gr.Image(label="Dispersión segmentada", type="filepath")


    btn_predecir.click(
        predecir,
        inputs=[G1, G2, studytime, failures, absences, health],
        outputs=[salida, grafico1, grafico2]
    )

demo.launch(share=True)
