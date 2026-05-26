"""
markov.py
─────────
Construye y analiza la matriz de transición de Markov desde tabla_const.

La cadena tiene 4 estados (etapas 1–4) y es absorbente hacia adelante:
- Solo se puede avanzar (k → k+1) o quedarse (k → k)
- El estado 4 es absorbente (k → k siempre)

La matriz se estima empíricamente desde la distribución de etapas
observada en tabla_const, usando la proporción de hogares en cada
etapa como proxy de las probabilidades de transición base.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ── CONSTRUIR MATRIZ DE TRANSICIÓN ────────────────────────────────────────────
def construir_matriz_markov(df_hogares, col_etapa="etapa", col_factor="factor_viv"):
    """
    Estima la matriz de transición P[k, k'] desde los datos observados.

    Lógica:
    - La proporción de hogares en cada etapa refleja la 'resistencia'
      a avanzar: si muchos hogares están en etapa k, la prob. de quedarse
      en k es alta.
    - p(k → k+1) se calibra como la razón entre hogares en etapa k+1
      y hogares en etapa k (flujo observado).
    - p(k → k) = 1 - p(k → k+1)
    - El estado 4 es absorbente: p(4 → 4) = 1

    Retorna:
    - P: matriz 4×4 numpy array
    - dist_inicial: distribución de probabilidad inicial por etapa
    - resumen: DataFrame con conteos y probabilidades por etapa
    """
    etapas = [1, 2, 3, 4]

    # Contar hogares por etapa (ponderado por factor de expansión)
    if col_factor in df_hogares.columns:
        conteos = (df_hogares
                   .groupby(col_etapa)[col_factor]
                   .sum()
                   .reindex(etapas, fill_value=0))
    else:
        conteos = (df_hogares
                   .groupby(col_etapa)
                   .size()
                   .reindex(etapas, fill_value=0))

    total = conteos.sum()

    # Distribución inicial (estado en t=0)
    dist_inicial = (conteos / total).values

    # Construir matriz P (4×4)
    P = np.zeros((4, 4))

    for i, k in enumerate(etapas):
        n_k    = conteos.get(k,   0)
        n_k1   = conteos.get(k+1, 0) if k < 4 else 0

        if k == 4:
            # Estado absorbente
            P[i, i] = 1.0
        elif n_k > 0:
            # p(avanzar) = proporción relativa de hogares en etapa k+1 vs k
            # Calibración: si hay muchos en k+1 relativo a k → transición fluida
            p_avanzar = n_k1 / (n_k + n_k1) if (n_k + n_k1) > 0 else 0.1
            # Clampeamos para evitar extremos
            p_avanzar = np.clip(p_avanzar, 0.05, 0.60)
            P[i, i]     = 1 - p_avanzar   # quedarse
            P[i, i + 1] = p_avanzar        # avanzar
        else:
            P[i, i] = 1.0

    # Resumen legible
    resumen = pd.DataFrame({
        "etapa":          etapas,
        "n_hogares_pond": [conteos.get(k, 0) for k in etapas],
        "pct_dist":       [dist_inicial[i] * 100 for i in range(4)],
        "p_quedarse":     [P[i, i]     for i in range(4)],
        "p_avanzar":      [P[i, i+1] if i < 3 else 0.0 for i in range(4)],
    })

    return P, dist_inicial, resumen


# ── PROYECCIÓN ANALÍTICA DE LA CADENA ────────────────────────────────────────
def proyectar_markov(P, dist_inicial, T=30):
    """
    Proyecta la distribución de estados a lo largo de T pasos.

    Retorna DataFrame con distribución por etapa en cada paso t.
    """
    dist = dist_inicial.copy()
    registros = [{"t": 0, **{f"etapa_{k}": dist[i] for i, k in enumerate([1,2,3,4])}}]

    for t in range(1, T + 1):
        dist = dist @ P
        registros.append({
            "t": t,
            **{f"etapa_{k}": dist[i] for i, k in enumerate([1,2,3,4])}
        })

    return pd.DataFrame(registros)


# ── REZAGO ESPERADO ───────────────────────────────────────────────────────────
def rezago_esperado(df_proyeccion, rezago_por_etapa=None):
    """
    Calcula el rezago continuo esperado en cada paso t.

    rezago_por_etapa: dict {etapa: rezago_promedio_observado}
    Si no se provee, usa valores teóricos decrecientes.
    """
    if rezago_por_etapa is None:
        # Valores teóricos: a mayor etapa, menor rezago
        rezago_por_etapa = {1: 0.85, 2: 0.60, 3: 0.35, 4: 0.10}

    df = df_proyeccion.copy()
    df["rezago_esperado"] = sum(
        df[f"etapa_{k}"] * rezago_por_etapa[k]
        for k in [1, 2, 3, 4]
    )
    return df


# ── TIEMPO ESPERADO PARA ALCANZAR ETAPA 4 ────────────────────────────────────
def tiempo_esperado_absorcion(P, T=50):
    """
    Calcula el tiempo esperado (en años) para que un hogar
    alcance el estado absorbente (etapa 4) desde cada estado inicial.

    Usa la fórmula de tiempos de absorción de cadenas de Markov.
    """
    # Submatriz Q (estados transitorios: etapas 1, 2, 3)
    Q = P[:3, :3]
    I = np.eye(3)

    # Matriz fundamental N = (I - Q)^{-1}
    # N[i,j] = número esperado de visitas al estado j partiendo de i
    try:
        N = np.linalg.inv(I - Q)
        # Tiempo esperado de absorción = suma de filas de N
        t_esperado = N.sum(axis=1)
        return {
            "etapa_1": round(t_esperado[0], 2),
            "etapa_2": round(t_esperado[1], 2),
            "etapa_3": round(t_esperado[2], 2),
            "etapa_4": 0.0
        }
    except np.linalg.LinAlgError:
        return {"etapa_1": np.nan, "etapa_2": np.nan,
                "etapa_3": np.nan, "etapa_4": 0.0}


# ── VISUALIZACIÓN DE LA MATRIZ ────────────────────────────────────────────────
def visualizar_matriz(P, dist_inicial, proyeccion, rezago_col, output_path=None):
    """
    Genera una figura con:
    1. Heatmap de la matriz de transición
    2. Diagrama de flujo de estados
    3. Proyección de distribución por etapa
    4. Evolución del rezago esperado
    """
    AZUL   = "#0BA1DA"
    VERDE  = "#BFD602"
    COLORES_ETAPA = ["#D0021B", "#F5A623", "#0BA1DA", "#BFD602"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Cadena de Markov — Autoproducción de vivienda\n"
        "Quintil 1 de ingresos | ENIGH 2024",
        fontsize=13, fontweight="bold"
    )

    etiquetas = ["Etapa 1\nInicial", "Etapa 2\nConsolidación",
                 "Etapa 3\nTerminaciones", "Etapa 4\nServicios"]

    # ── 1. Heatmap de la matriz ───────────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f"E{k}" for k in [1,2,3,4]])
    ax.set_yticklabels([f"E{k}" for k in [1,2,3,4]])
    ax.set_xlabel("Estado destino (t+1)")
    ax.set_ylabel("Estado origen (t)")
    ax.set_title("Matriz de transición P", fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center",
                    fontsize=11, color="white" if P[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ── 2. Diagrama de flujo ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Diagrama de transición de estados", fontweight="bold")

    xs = [1, 3.5, 6, 8.5]
    for i, (x, label, color) in enumerate(zip(xs, etiquetas, COLORES_ETAPA)):
        circle = plt.Circle((x, 2), 0.7, color=color, alpha=0.85)
        ax.add_patch(circle)
        ax.text(x, 2, f"E{i+1}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(x, 0.8, label, ha="center", va="center", fontsize=8)
        # Distribución inicial
        ax.text(x, 3.1, f"{dist_inicial[i]*100:.1f}%",
                ha="center", fontsize=9, color=color, fontweight="bold")

        # Flecha de avance
        if i < 3:
            p_av = P[i, i+1]
            ax.annotate("", xy=(xs[i+1]-0.7, 2), xytext=(x+0.7, 2),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
            ax.text((x + xs[i+1]) / 2, 2.4, f"p={p_av:.2f}",
                    ha="center", fontsize=8, color="gray")

        # Flecha de permanencia (loop)
        if i < 3:
            p_qd = P[i, i]
            ax.annotate("", xy=(x - 0.5, 2.65), xytext=(x + 0.5, 2.65),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        connectionstyle="arc3,rad=-0.5", lw=1.2))
            ax.text(x, 3.55, f"1-p={p_qd:.2f}",
                    ha="center", fontsize=7, color=color)

    ax.text(8.5, 3.1, "Absorbente", ha="center", fontsize=8,
            color=COLORES_ETAPA[3], style="italic")

    # ── 3. Proyección de distribución ────────────────────────────────────────
    ax = axes[1, 0]
    for i, (k, color) in enumerate(zip([1,2,3,4], COLORES_ETAPA)):
        ax.plot(proyeccion["t"], proyeccion[f"etapa_{k}"] * 100,
                color=color, linewidth=2.5, marker="o", markersize=4,
                label=f"Etapa {k}")
    ax.set_title("Proyección analítica de distribución\npor etapa", fontweight="bold")
    ax.set_xlabel("Año")
    ax.set_ylabel("% de hogares")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── 4. Rezago esperado ────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(rezago_col["t"], rezago_col["rezago_esperado"] * 100,
            color=AZUL, linewidth=2.5, marker="s", markersize=5)
    ax.fill_between(rezago_col["t"], rezago_col["rezago_esperado"] * 100,
                    alpha=0.15, color=AZUL)
    ax.set_title("Rezago habitacional esperado\n(índice continuo 0–100)", fontweight="bold")
    ax.set_xlabel("Año")
    ax.set_ylabel("Índice de rezago promedio")
    ax.grid(axis="y", alpha=0.3)

    # Anotar inicio y fin
    r0 = rezago_col["rezago_esperado"].iloc[0] * 100
    rT = rezago_col["rezago_esperado"].iloc[-1] * 100
    ax.annotate(f"t=0: {r0:.1f}", xy=(0, r0), fontsize=9,
                xytext=(0.5, r0 + 3), color=AZUL, fontweight="bold")
    ax.annotate(f"t=30: {rT:.1f}", xy=(30, rT), fontsize=9,
                xytext=(27, rT + 3), color=AZUL, fontweight="bold")

    fig.text(0.5, 0.01,
             "Fuente: Elaboración propia con datos de la ENIGH 2024, INEGI.",
             ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figura guardada en {output_path}")

    return fig


# ── FUNCIÓN PRINCIPAL ─────────────────────────────────────────────────────────
def analisis_markov(df_hogares, T=30, rezago_por_etapa=None,
                    col_etapa="etapa", col_factor="factor_viv",
                    output_path=None):
    """
    Ejecuta el análisis completo de la cadena de Markov.

    Parámetros:
    - df_hogares       : DataFrame con un registro por hogar (tabla_const)
    - T                : horizonte temporal en años
    - rezago_por_etapa : dict {etapa: rezago_promedio_observado}
                         Si None, calcula desde rezago_continuo si existe
    - col_etapa        : nombre de la columna de etapa
    - col_factor       : nombre del factor de expansión
    - output_path      : ruta para guardar la figura

    Retorna dict con: P, dist_inicial, resumen, proyeccion, rezago, t_absorcion
    """
    # Si existen datos de rezago observado por etapa, usarlos
    if rezago_por_etapa is None and "rezago_continuo" in df_hogares.columns:
        rezago_por_etapa = (df_hogares
                            .groupby(col_etapa)["rezago_continuo"]
                            .mean()
                            .to_dict())
        # Asegurar que las 4 etapas están presentes
        defaults = {1: 0.85, 2: 0.60, 3: 0.35, 4: 0.10}
        for k in [1, 2, 3, 4]:
            if k not in rezago_por_etapa:
                rezago_por_etapa[k] = defaults[k]

    # Construir matriz
    P, dist_inicial, resumen = construir_matriz_markov(
        df_hogares, col_etapa, col_factor)

    # Proyectar
    proyeccion  = proyectar_markov(P, dist_inicial, T)
    rezago_df   = rezago_esperado(proyeccion, rezago_por_etapa)
    t_absorcion = tiempo_esperado_absorcion(P)

    # Imprimir resumen
    print("═" * 55)
    print("CADENA DE MARKOV — Autoproducción de vivienda")
    print("═" * 55)
    print("\nDistribución inicial y probabilidades de transición:")
    print(resumen.to_string(index=False))
    print(f"\nTiempo esperado para alcanzar etapa 4 (años):")
    for k, t in t_absorcion.items():
        print(f"  Desde {k}: {t} años")
    print(f"\nRezago esperado inicial: {rezago_df['rezago_esperado'].iloc[0]:.3f}")
    print(f"Rezago esperado año {T}: {rezago_df['rezago_esperado'].iloc[-1]:.3f}")
    reduccion = (1 - rezago_df['rezago_esperado'].iloc[-1] /
                     rezago_df['rezago_esperado'].iloc[0]) * 100
    print(f"Reducción esperada del rezago en {T} años: {reduccion:.1f}%")
    print("═" * 55)

    # Visualizar
    fig = visualizar_matriz(P, dist_inicial, proyeccion, rezago_df, output_path)

    return {
        "P":            P,
        "dist_inicial": dist_inicial,
        "resumen":      resumen,
        "proyeccion":   proyeccion,
        "rezago":       rezago_df,
        "t_absorcion":  t_absorcion,
        "figura":       fig
    }


# ── INTEGRACIÓN CON ABM ───────────────────────────────────────────────────────
def calibrar_alphas_desde_markov(P):
    """
    Calibra los parámetros alpha de la función logística del ABM
    para que sean consistentes con las probabilidades de transición
    empíricas de la cadena de Markov.

    La idea: p(avanzar) empírica ≈ p(logística) con presupuesto promedio.
    Ajusta alpha_0 (intercepto) para cada etapa.
    """
    calibracion = {}
    for i in range(3):  # etapas 1, 2, 3
        p_emp = P[i, i+1]
        if p_emp > 0 and p_emp < 1:
            # Con B=C (presupuesto = costo), el logit sin covariables es α0
            # 1/(1+exp(-α0)) = p_emp → α0 = log(p_emp/(1-p_emp))
            alpha_0_k = np.log(p_emp / (1 - p_emp))
            calibracion[f"etapa_{i+1}"] = round(alpha_0_k, 3)

    print("\nCalibración sugerida de alpha_0 por etapa:")
    for k, v in calibracion.items():
        print(f"  {k}: alpha_0 = {v}")
    print("  (Usar el promedio como alpha_0 en abm_main.py)")
    alpha_0_promedio = np.mean(list(calibracion.values()))
    print(f"  Promedio: {alpha_0_promedio:.3f}")

    return calibracion, alpha_0_promedio


if __name__ == "__main__":
    # Ejemplo con datos sintéticos si no hay hogares.parquet
    from pathlib import Path
    import numpy as np

    data_path = Path("data/hogares.parquet")
    if data_path.exists():
        import pandas as pd
        df = pd.read_parquet(data_path)
        print(f"Usando hogares.parquet: {len(df):,} hogares")
    else:
        print("Usando datos sintéticos (sin hogares.parquet)")
        rng = np.random.default_rng(42)
        n   = 500
        df  = pd.DataFrame({
            "etapa":           rng.integers(1, 5, n),
            "factor_viv":      rng.uniform(100, 500, n),
            "rezago_continuo": rng.uniform(0, 1, n),
        })
        # Asignar rezago coherente con etapa
        df["rezago_continuo"] = df["etapa"].map(
            {1: 0.85, 2: 0.60, 3: 0.35, 4: 0.10}
        ) + rng.normal(0, 0.05, n)
        df["rezago_continuo"] = df["rezago_continuo"].clip(0, 1)

    Path("outputs").mkdir(exist_ok=True)
    resultados = analisis_markov(
        df_hogares  = df,
        T           = 30,
        output_path = "outputs/markov_analisis.png"
    )
    calibrar_alphas_desde_markov(resultados["P"])
