"""
Visualización del ABM de autoproducción de vivienda
Corre el modelo y genera gráficas de resultados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from abm_main import AutoproduccionModel

# ── COLORES ────────────────────────────────────────────────────────────────────
AZUL    = "#0BA1DA"
VERDE   = "#BFD602"
NARANJA = "#F5A623"
ROJO    = "#D0021B"
GRIS    = "#4A4A4A"
ESCENARIOS_COLORES = {
    "conservador": VERDE,
    "base":        AZUL,
    "critico":     ROJO,
}


def correr_escenarios(T=30, seed=42):
    """Corre los tres escenarios y devuelve resultados"""
    resultados = {}
    for esc in ["conservador", "base", "critico"]:
        print(f"  Corriendo escenario: {esc}...")
        modelo = AutoproduccionModel(escenario=esc, seed=seed)
        modelo.run(T=T)
        resultados[esc] = {
            "modelo":   modelo.resultados_modelo(),
            "agentes":  modelo.resultados_agentes(),
            "n_agentes": len(list(modelo.agents))
        }
    return resultados


def grafica_distribucion_etapas(resultados, ax):
    """Distribución de etapas a lo largo del tiempo por escenario"""
    esc = "base"
    df  = resultados[esc]["modelo"]
    n   = resultados[esc]["n_agentes"]

    pasos = df.index.tolist()
    for etapa, color, label in [
        ("Hogares_etapa_1", "#D0021B",  "Etapa 1: Inicial"),
        ("Hogares_etapa_2", "#F5A623",  "Etapa 2: Consolidación"),
        ("Hogares_etapa_3", AZUL,       "Etapa 3: Terminaciones"),
        ("Hogares_etapa_4", VERDE,      "Etapa 4: Servicios"),
    ]:
        ax.plot(pasos, df[etapa] / n * 100,
                color=color, linewidth=2, label=label, marker="o", markersize=4)

    ax.set_title("Distribución de hogares por etapa constructiva\n(Escenario base)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Año de simulación")
    ax.set_ylabel("% de hogares")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0, max(pasos))


def grafica_rezago_escenarios(resultados, ax):
    """Evolución del rezago promedio por escenario"""
    for esc, color in ESCENARIOS_COLORES.items():
        df    = resultados[esc]["modelo"]
        pasos = df.index.tolist()
        ax.plot(pasos, df["Pct_rezago"] * 100,
                color=color, linewidth=2.5, label=esc.capitalize(), marker="s", markersize=4)

    ax.set_title("Índice de rezago habitacional promedio\npor escenario", fontsize=11, fontweight="bold")
    ax.set_xlabel("Año de simulación")
    ax.set_ylabel("Índice de rezago (0–100)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0, max(pasos))


def grafica_pct_avanzaron(resultados, ax):
    """% de hogares que avanzaron al menos una etapa"""
    for esc, color in ESCENARIOS_COLORES.items():
        df    = resultados[esc]["modelo"]
        pasos = df.index.tolist()
        ax.plot(pasos, df["Pct_avanzaron"] * 100,
                color=color, linewidth=2.5, label=esc.capitalize(), marker="^", markersize=4)

    ax.set_title("% de hogares que avanzaron\nal menos una etapa", fontsize=11, fontweight="bold")
    ax.set_xlabel("Año de simulación")
    ax.set_ylabel("% de hogares")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0, max(pasos))


def grafica_ingreso(resultados, ax):
    """Evolución del ingreso promedio por escenario"""
    for esc, color in ESCENARIOS_COLORES.items():
        df    = resultados[esc]["modelo"]
        pasos = df.index.tolist()
        ax.plot(pasos, df["Ingreso_promedio"] / 1000,
                color=color, linewidth=2.5, label=esc.capitalize())

    ax.set_title("Ingreso anual promedio de los hogares\npor escenario", fontsize=11, fontweight="bold")
    ax.set_xlabel("Año de simulación")
    ax.set_ylabel("Miles de pesos MXN")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0, max(pasos))


def grafica_distribucion_final(resultados, ax):
    """Distribución final de etapas por escenario (barras agrupadas)"""
    escenarios = list(resultados.keys())
    etapas     = [1, 2, 3, 4]
    x          = np.arange(len(etapas))
    width      = 0.25

    for i, (esc, color) in enumerate(ESCENARIOS_COLORES.items()):
        df = resultados[esc]["modelo"]
        n  = resultados[esc]["n_agentes"]
        # Último paso
        ultimo = df.iloc[-1]
        vals = [
            ultimo["Hogares_etapa_1"] / n * 100,
            ultimo["Hogares_etapa_2"] / n * 100,
            ultimo["Hogares_etapa_3"] / n * 100,
            ultimo["Hogares_etapa_4"] / n * 100,
        ]
        bars = ax.bar(x + i * width, vals, width, label=esc.capitalize(),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_title("Distribución final de etapas\nal año 30 por escenario", fontsize=11, fontweight="bold")
    ax.set_xlabel("Etapa constructiva")
    ax.set_ylabel("% de hogares")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Etapa 1\nInicial", "Etapa 2\nConsolidación",
                         "Etapa 3\nTerminaciones", "Etapa 4\nServicios"])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def grafica_rural_urbano(resultados, ax):
    """Comparación de avance por ámbito rural/urbano (escenario base)"""
    df_agentes = resultados["base"]["agentes"]
    # Último paso disponible
    ultimo_paso = df_agentes.index.get_level_values("Step").max()
    df_final    = df_agentes.xs(ultimo_paso, level="Step")

    rural_avg   = df_final[df_final["rural"] == 1]["estado_viv"].mean()
    urbano_avg  = df_final[df_final["rural"] == 0]["estado_viv"].mean()

    ax.bar(["Rural", "Urbano"], [rural_avg, urbano_avg],
           color=[NARANJA, AZUL], alpha=0.85, width=0.5)
    ax.set_title("Etapa promedio al año 30\nRural vs Urbano (escenario base)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Etapa constructiva promedio")
    ax.set_ylim(0, 4.5)
    for i, val in enumerate([rural_avg, urbano_avg]):
        ax.text(i, val + 0.1, f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def generar_dashboard(resultados, T=30):
    """Genera el dashboard completo"""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    grafica_distribucion_etapas(resultados, ax1)
    grafica_rezago_escenarios(resultados,   ax2)
    grafica_pct_avanzaron(resultados,       ax3)
    grafica_ingreso(resultados,             ax4)
    grafica_distribucion_final(resultados,  ax5)
    grafica_rural_urbano(resultados,        ax6)

    # Título general
    fig.suptitle(
        "ABM: Autoproducción de vivienda en México — Quintil 1 de ingresos\n"
        "ENIGH 2024 | Horizonte: 30 años | Tres escenarios: Conservador, Base, Crítico",
        fontsize=13, fontweight="bold", y=0.98
    )

    # Nota al pie
    fig.text(0.5, 0.01,
             "Fuente: Elaboración propia con datos de la ENIGH 2024, INEGI. "
             "Modelo Basado en Agentes con cadena de Markov condicionada.",
             ha="center", fontsize=9, color="gray")

    return fig


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Iniciando simulación ABM...")
    print("Corriendo 3 escenarios × 30 años...")

    resultados = correr_escenarios(T=30, seed=42)

    print("\nGenerando dashboard...")
    fig = generar_dashboard(resultados, T=30)

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "abm_resultados.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Dashboard guardado en outputs/abm_resultados.png")

    # Guardar resultados en CSV
    for esc in ["conservador", "base", "critico"]:
        resultados[esc]["modelo"].to_csv(out / f"resultados_modelo_{esc}.csv")
        resultados[esc]["agentes"].to_csv(out / f"resultados_agentes_{esc}.csv")

    print("Resultados exportados a CSV.")
    print("\nResumen escenario base (último año):")
    df_base = resultados["base"]["modelo"]
    print(df_base.iloc[-1][["Hogares_etapa_1", "Hogares_etapa_2",
                              "Hogares_etapa_3", "Hogares_etapa_4",
                              "Pct_rezago", "Pct_avanzaron"]].to_string())
