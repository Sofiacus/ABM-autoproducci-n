"""
dashboard_completo.py
─────────────────────
Dashboard unificado del ABM de autoproducción de vivienda.

Genera dos archivos:
1. dashboard_completo.png  — todas las gráficas estáticas
2. animacion_hogares.gif   — animación de movimiento de hogares por etapa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from pathlib import Path
from abm_main import AutoproduccionModel
from markov import (construir_matriz_markov, proyectar_markov,
                    rezago_esperado, tiempo_esperado_absorcion)

# ── PALETA ────────────────────────────────────────────────────────────────────
AZUL    = "#0BA1DA"
VERDE   = "#BFD602"
NARANJA = "#F5A623"
ROJO    = "#D0021B"
GRIS    = "#6B6B6B"

COLORES_ETAPA = {
    1: "#D0021B",   # rojo — precario
    2: "#F5A623",   # naranja — consolidación
    3: "#0BA1DA",   # azul — terminaciones
    4: "#BFD602",   # verde — servicios completos
}
LABELS_ETAPA = {
    1: "Etapa 1\nInicial",
    2: "Etapa 2\nConsolidación",
    3: "Etapa 3\nTerminaciones",
    4: "Etapa 4\nServicios",
}
ESCENARIOS_COLORES = {
    "conservador": VERDE,
    "base":        AZUL,
    "critico":     ROJO,
}

# ── CORRER MODELOS ────────────────────────────────────────────────────────────
def correr_escenarios(T=30, seed=42):
    resultados = {}
    for esc in ["conservador", "base", "critico"]:
        print(f"  Escenario: {esc}...")
        m = AutoproduccionModel(escenario=esc, seed=seed)
        m.run(T=T)
        resultados[esc] = {
            "modelo":    m.resultados_modelo(),
            "agentes":   m.resultados_agentes(),
            "n":         len(list(m.agents)),
            "P_markov":  m.P_markov,
            "rezago_map":m.rezago_por_etapa,
            "hogares_df":m.hogares_df,
        }
    return resultados

# ── GRÁFICAS DEL ABM (las 6 originales) ──────────────────────────────────────
def g_dist_etapas(res, ax, esc="base"):
    df = res[esc]["modelo"]; n = res[esc]["n"]
    pasos = df.index.tolist()
    for etapa, color, label in [
        ("Hogares_etapa_1", COLORES_ETAPA[1], "Etapa 1"),
        ("Hogares_etapa_2", COLORES_ETAPA[2], "Etapa 2"),
        ("Hogares_etapa_3", COLORES_ETAPA[3], "Etapa 3"),
        ("Hogares_etapa_4", COLORES_ETAPA[4], "Etapa 4"),
    ]:
        ax.plot(pasos, df[etapa]/n*100, color=color, lw=2,
                label=label, marker="o", ms=4)
    ax.set_title("Distribución por etapa\n(Escenario base)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Año"); ax.set_ylabel("% hogares")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

def g_rezago_esc(res, ax):
    for esc, color in ESCENARIOS_COLORES.items():
        df = res[esc]["modelo"]
        ax.plot(df.index, df["Pct_rezago"]*100, color=color, lw=2.5,
                label=esc.capitalize(), marker="s", ms=4)
    ax.set_title("Rezago promedio\npor escenario", fontsize=10, fontweight="bold")
    ax.set_xlabel("Año"); ax.set_ylabel("Índice (0–100)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

def g_pct_avanzan(res, ax):
    for esc, color in ESCENARIOS_COLORES.items():
        df = res[esc]["modelo"]
        ax.plot(df.index, df["Pct_avanzaron"]*100, color=color, lw=2.5,
                label=esc.capitalize(), marker="^", ms=4)
    ax.set_title("% hogares que avanzaron\nal menos una etapa", fontsize=10, fontweight="bold")
    ax.set_xlabel("Año"); ax.set_ylabel("% hogares")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

def g_ingreso(res, ax):
    for esc, color in ESCENARIOS_COLORES.items():
        df = res[esc]["modelo"]
        ax.plot(df.index, df["Ingreso_promedio"]/1000, color=color, lw=2.5,
                label=esc.capitalize())
    ax.set_title("Ingreso anual promedio\npor escenario", fontsize=10, fontweight="bold")
    ax.set_xlabel("Año"); ax.set_ylabel("Miles MXN")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

def g_dist_final(res, ax):
    etapas = [1,2,3,4]; x = np.arange(4); w = 0.25
    for i, (esc, color) in enumerate(ESCENARIOS_COLORES.items()):
        df = res[esc]["modelo"]; n = res[esc]["n"]
        u = df.iloc[-1]
        vals = [u[f"Hogares_etapa_{k}"]/n*100 for k in etapas]
        bars = ax.bar(x + i*w, vals, w, label=esc.capitalize(),
                      color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_title("Distribución final\naño 30 por escenario", fontsize=10, fontweight="bold")
    ax.set_xticks(x+w); ax.set_xticklabels(["E1","E2","E3","E4"])
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

def g_rural_urb(res, ax):
    df_a = res["base"]["agentes"]
    last = df_a.index.get_level_values("Step").max()
    df_f = df_a.xs(last, level="Step")
    rv = df_f[df_f["rural"]==1]["estado_viv"].mean()
    uv = df_f[df_f["rural"]==0]["estado_viv"].mean()
    bars = ax.bar(["Rural","Urbano"], [rv,uv], color=[NARANJA,AZUL],
                  alpha=0.85, width=0.5)
    ax.set_title("Etapa promedio año 30\nRural vs Urbano", fontsize=10, fontweight="bold")
    ax.set_ylabel("Etapa promedio"); ax.set_ylim(0,4.5)
    for b, v in zip(bars, [rv,uv]):
        ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

# ── GRÁFICAS DE MARKOV ────────────────────────────────────────────────────────
def g_heatmap_markov(P, ax):
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["E1","E2","E3","E4"])
    ax.set_yticklabels(["E1","E2","E3","E4"])
    ax.set_xlabel("Estado destino (t+1)", fontsize=9)
    ax.set_ylabel("Estado origen (t)", fontsize=9)
    ax.set_title("Matriz de transición P\n(Cadena de Markov)", fontsize=10, fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if P[i,j]>0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

def g_proyeccion_markov(P, dist_inicial, rezago_map, ax1, ax2, T=30):
    proy   = proyectar_markov(P, dist_inicial, T)
    rez_df = rezago_esperado(proy, rezago_map)

    for k, color in COLORES_ETAPA.items():
        ax1.plot(proy["t"], proy[f"etapa_{k}"]*100, color=color,
                 lw=2, marker="o", ms=4, label=f"E{k}")
    ax1.set_title("Proyección analítica\n(Markov)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Año"); ax1.set_ylabel("% hogares")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(rez_df["t"], rez_df["rezago_esperado"]*100,
             color=AZUL, lw=2.5, marker="s", ms=5)
    ax2.fill_between(rez_df["t"], rez_df["rezago_esperado"]*100, alpha=0.15, color=AZUL)
    r0 = rez_df["rezago_esperado"].iloc[0]*100
    rT = rez_df["rezago_esperado"].iloc[-1]*100
    ax2.annotate(f"{r0:.1f}", xy=(0, r0), xytext=(0.3, r0+3), fontsize=9,
                 color=AZUL, fontweight="bold")
    ax2.annotate(f"{rT:.1f}", xy=(T, rT), xytext=(T-1.5, rT+3), fontsize=9,
                 color=AZUL, fontweight="bold")
    ax2.set_title("Rezago esperado\n(proyección Markov)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Año"); ax2.set_ylabel("Índice rezago (0–100)")
    ax2.grid(axis="y", alpha=0.3)

def g_tiempo_absorcion(P, ax):
    t_abs = tiempo_esperado_absorcion(P)
    etapas = ["Etapa 1", "Etapa 2", "Etapa 3"]
    tiempos = [t_abs["etapa_1"], t_abs["etapa_2"], t_abs["etapa_3"]]
    colors  = [COLORES_ETAPA[1], COLORES_ETAPA[2], COLORES_ETAPA[3]]
    bars = ax.barh(etapas, tiempos, color=colors, alpha=0.85)
    for b, v in zip(bars, tiempos):
        ax.text(v+0.1, b.get_y()+b.get_height()/2,
                f"{v:.1f} años", va="center", fontsize=10)
    ax.set_title("Tiempo esperado para\nalcanzar etapa 4", fontsize=10, fontweight="bold")
    ax.set_xlabel("Años"); ax.set_xlim(0, max(tiempos)*1.3)
    ax.grid(axis="x", alpha=0.3)

# ── ANIMACIÓN: MOVIMIENTO DE HOGARES ─────────────────────────────────────────
def crear_animacion(resultados, esc="base", n_muestra=120, output_path=None):
    """
    Crea una animación donde cada punto es un hogar.
    Los puntos cambian de color cuando cambian de etapa.
    Posición X = etapa, Posición Y = ingreso (normalizado).
    """
    df_agentes = resultados[esc]["agentes"].reset_index()
    pasos      = sorted(df_agentes["Step"].unique())
    n_pasos    = len(pasos)

    # Seleccionar muestra de agentes representativa
    agentes_ids = df_agentes["AgentID"].unique()
    if len(agentes_ids) > n_muestra:
        rng = np.random.default_rng(42)
        agentes_ids = rng.choice(agentes_ids, n_muestra, replace=False)

    df_sub = df_agentes[df_agentes["AgentID"].isin(agentes_ids)].copy()

    # Normalizar ingreso para posición Y
    max_ing = df_sub["ingreso_anual"].max()
    min_ing = df_sub["ingreso_anual"].min()
    df_sub["y_pos"] = (df_sub["ingreso_anual"] - min_ing) / (max_ing - min_ing + 1)

    # Añadir jitter en X para evitar superposición
    rng = np.random.default_rng(42)
    jitter_map = {aid: rng.uniform(-0.25, 0.25) for aid in agentes_ids}
    df_sub["x_jitter"] = df_sub["AgentID"].map(jitter_map)
    df_sub["x_pos"]    = df_sub["estado_viv"] + df_sub["x_jitter"]

    # Rastrear etapa anterior para detectar cambios
    df_sub = df_sub.sort_values(["AgentID", "Step"])
    df_sub["etapa_anterior"] = df_sub.groupby("AgentID")["estado_viv"].shift(1)
    df_sub["avanzo"] = df_sub["estado_viv"] > df_sub["etapa_anterior"].fillna(df_sub["estado_viv"])

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")

    # Leyenda fija
    patches = [mpatches.Patch(color=COLORES_ETAPA[k], label=LABELS_ETAPA[k].replace("\n"," "))
               for k in [1,2,3,4]]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.9)

    # Líneas divisoras de etapas
    for x in [1.5, 2.5, 3.5]:
        ax.axvline(x, color="gray", lw=0.8, ls="--", alpha=0.4)

    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Etapa constructiva", fontsize=11)
    ax.set_ylabel("Ingreso anual (normalizado)", fontsize=11)
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(["E1\nInicial","E2\nConsolidación",
                         "E3\nTerminaciones","E4\nServicios"], fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    titulo = ax.set_title("", fontsize=12, fontweight="bold")
    scatter_pts = {}

    def init():
        return []

    def update(frame_idx):
        paso = pasos[frame_idx]
        df_paso = df_sub[df_sub["Step"] == paso]

        # Limpiar puntos del frame anterior
        for s in scatter_pts.values():
            s.remove()
        scatter_pts.clear()

        for etapa in [1,2,3,4]:
            df_e = df_paso[df_paso["estado_viv"] == etapa]
            if df_e.empty:
                continue
            # Hogares que acaban de avanzar: más grandes y con borde
            avanzo_mask = df_e["avanzo"] == True
            # Puntos normales
            if (~avanzo_mask).any():
                s1 = ax.scatter(
                    df_e.loc[~avanzo_mask, "x_pos"],
                    df_e.loc[~avanzo_mask, "y_pos"],
                    c=COLORES_ETAPA[etapa], s=40, alpha=0.7,
                    edgecolors="none", zorder=3
                )
                scatter_pts[f"e{etapa}_normal"] = s1
            # Puntos que avanzaron este paso (más grandes, con borde blanco)
            if avanzo_mask.any():
                s2 = ax.scatter(
                    df_e.loc[avanzo_mask, "x_pos"],
                    df_e.loc[avanzo_mask, "y_pos"],
                    c=COLORES_ETAPA[etapa], s=120, alpha=1.0,
                    edgecolors="white", linewidths=1.5, zorder=5
                )
                scatter_pts[f"e{etapa}_avanzo"] = s2

        # Conteos por etapa
        conteos = df_paso.groupby("estado_viv").size()
        conteo_str = "  |  ".join([f"E{k}: {conteos.get(k,0)}" for k in [1,2,3,4]])
        titulo.set_text(
            f"Movimiento de hogares — Año {int(paso)}\n"
            f"(n={len(df_paso)} hogares)   {conteo_str}\n"
            f"⬤ grande = avanzó de etapa este año"
        )
        return list(scatter_pts.values()) + [titulo]

    ani = animation.FuncAnimation(
        fig, update, frames=n_pasos,
        init_func=init, blit=False,
        interval=800, repeat=True
    )

    if output_path:
        print(f"  Guardando animación en {output_path}...")
        ani.save(output_path, writer="pillow", fps=1.2, dpi=120)
        print("  Animacion guardada")

    plt.close(fig)
    return ani


# ── DASHBOARD COMPLETO ────────────────────────────────────────────────────────
def generar_dashboard_completo(resultados, T=30, output_path=None):
    """
    Dashboard de 3 filas × 4 columnas = 12 paneles:
    Fila 1: Las 6 gráficas del ABM (ocupa 2 columnas cada una en 2 subfigs)
    Fila 2-3: Las 4 gráficas de Markov
    """
    # Tomar datos del escenario base para Markov
    P          = resultados["base"]["P_markov"]
    rez_map    = resultados["base"]["rezago_map"]
    hdf        = resultados["base"]["hogares_df"]
    _, dist_0, _ = construir_matriz_markov(hdf, "etapa", "factor_viv")

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.40,
                           top=0.87, bottom=0.05)

    # Fila 0: ABM (6 gráficas en 4 columnas — las dos últimas comparten col)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3])

    # Fila 1: ABM (2 más) + Markov (2)
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])
    ax13 = fig.add_subplot(gs[1, 3])

    # Fila 2: Markov (2 más) + rural/urbano + shocks
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])
    ax23 = fig.add_subplot(gs[2, 3])

    # ── Fila 0: ABM ──────────────────────────────────────────────────────────
    g_dist_etapas(resultados, ax00)
    g_rezago_esc(resultados,  ax01)
    g_pct_avanzan(resultados, ax02)
    g_ingreso(resultados,     ax03)

    # ── Fila 1: ABM (2) + Markov (2) ─────────────────────────────────────────
    g_dist_final(resultados,  ax10)
    g_rural_urb(resultados,   ax11)
    g_heatmap_markov(P,       ax12)
    g_tiempo_absorcion(P,     ax13)

    # ── Fila 2: Markov (proyección + rezago) + shocks + rezago por rural/urb ─
    g_proyeccion_markov(P, dist_0, rez_map, ax20, ax21, T)

    # Shocks acumulados por escenario
    for esc, color in ESCENARIOS_COLORES.items():
        df_a = resultados[esc]["agentes"].reset_index()
        last = df_a["Step"].max()
        shocks_paso = (df_a.groupby("Step")["shocks_recibidos"]
                       .mean())
        ax22.plot(shocks_paso.index, shocks_paso.values,
                  color=color, lw=2.5, label=esc.capitalize(), marker="o", ms=4)
    ax22.set_title("Shocks negativos acumulados\npromedio por agente", fontsize=10, fontweight="bold")
    ax22.set_xlabel("Año"); ax22.set_ylabel("Shocks promedio")
    ax22.legend(fontsize=8); ax22.grid(axis="y", alpha=0.3)

    # Rezago final por ámbito y escenario
    ambitos = ["Urbano", "Rural"]
    x_pos   = np.arange(len(ambitos))
    w       = 0.25
    for i, (esc, color) in enumerate(ESCENARIOS_COLORES.items()):
        df_a = resultados[esc]["agentes"].reset_index()
        last = df_a["Step"].max()
        df_f = df_a[df_a["Step"] == last]
        vals = [
            df_f[df_f["rural"]==0]["rezago_actual"].mean() * 100,
            df_f[df_f["rural"]==1]["rezago_actual"].mean() * 100,
        ]
        bars = ax23.bar(x_pos + i*w, vals, w, color=color,
                        alpha=0.85, label=esc.capitalize())
        for b, v in zip(bars, vals):
            ax23.text(b.get_x()+b.get_width()/2, v+0.3,
                      f"{v:.1f}", ha="center", fontsize=7)
    ax23.set_title("Rezago final año 30\nUrbano vs Rural por escenario",
                   fontsize=10, fontweight="bold")
    ax23.set_xticks(x_pos + w); ax23.set_xticklabels(ambitos)
    ax23.set_ylabel("Índice rezago (0–100)")
    ax23.legend(fontsize=8); ax23.grid(axis="y", alpha=0.3)

    # ── Título y nota ─────────────────────────────────────────────────────────
    fig.suptitle(
        "ABM: Autoproducción de vivienda en México — Quintil 1 de ingresos\n"
        "ENIGH 2024 | Cadena de Markov condicionada | Horizonte: 30 años | "
        "Tres escenarios: Conservador, Base, Crítico",
        fontsize=12, fontweight="bold", y=0.975
    )
    fig.text(0.5, 0.01,
             "Fuente: Elaboración propia con datos de la ENIGH 2024, INEGI. "
             "ABM con cadena de Markov condicionada y función logística de transición.",
             ha="center", fontsize=9, color="gray")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Dashboard guardado en {output_path}")

    return fig


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    print("Corriendo escenarios...")
    resultados = correr_escenarios(T=30, seed=42)

    print("Generando dashboard completo...")
    generar_dashboard_completo(
        resultados,
        output_path=out / "dashboard_completo.png"
    )

    print("Generando animación de hogares (escenario base)...")
    crear_animacion(
        resultados,
        esc       = "base",
        n_muestra = 150,
        output_path = out / "animacion_hogares.gif"
    )

    print("\n=== Archivos generados ===")
    print("  outputs/dashboard_completo.png")
    print("  outputs/animacion_hogares.gif")
