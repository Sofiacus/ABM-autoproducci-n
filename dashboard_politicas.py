
"""dashboard_politicas.py
──────────────────────
Dashboard interactivo de políticas públicas para la autoproducción de vivienda.

Políticas simulables:
  A. Subsidio directo a construcción
  B. Microcrédito para vivienda
  C. Reducción de shocks económicos
  D. Densificación de oferentes
  E. Educación financiera / formal
  F. Paquete integral (A + C + E)

Ejecución:  python dashboard_politicas.py
Abre:       http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fast_sim import correr_sim as _sim_fast, P_efectiva
from markov import tiempo_esperado_absorcion

# ── PALETA ─────────────────────────────────────────────────────────────────────
AZUL    = "#0BA1DA"
VERDE   = "#BFD602"
NARANJA = "#F5A623"
ROJO    = "#D0021B"
GRIS    = "#6B6B6B"
NEGRO   = "#1A1A2E"
FONDO   = "#F7F9FC"

COL_ETAPA = {1: ROJO, 2: NARANJA, 3: AZUL, 4: VERDE}
LAB_ETAPA = {
    1: "E1 · Inicial",
    2: "E2 · Consolidación",
    3: "E3 · Terminaciones",
    4: "E4 · Servicios",
}

POLITICAS = {
    "ninguna":   {"label": "Sin política (línea base)",       "color": GRIS},
    "subsidio":  {"label": "A · Subsidio a construcción",     "color": "#E91E8C"},
    "credito":   {"label": "B · Microcrédito vivienda",       "color": "#9C27B0"},
    "shocks":    {"label": "C · Reducción de shocks",         "color": "#FF6F00"},
    "oferentes": {"label": "D · Densificación de oferentes",  "color": "#00897B"},
    "educacion": {"label": "E · Educación financiera",        "color": "#1565C0"},
    "integral":  {"label": "F · Paquete integral (A+C+E)",    "color": "#2E7D32"},
}

SEED = 42

# ── CACHÉ DE LÍNEAS BASE ───────────────────────────────────────────────────────
_cache_base: dict = {}


def _obtener_base(escenario: str, T: int) -> dict:
    key = (escenario, T)
    if key not in _cache_base:
        _cache_base[key] = _sim_fast("ninguna", 0.0, T, escenario)
    return _cache_base[key]


def _paquete(res: dict) -> dict:
    """Convierte resultados a tipos JSON-seguros para dcc.Store."""
    df = res["df"].copy()
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].astype(float)
    for col in df.select_dtypes(include=[np.integer]).columns:
        df[col] = df[col].astype(int)
    return {
        "df":  df.to_dict("records"),
        "n":   int(res["n"]),
        # P_inicial solo se usa para el heatmap de Markov; la absorción
        # usa P_efectiva calculada desde la historia dentro del callback.
        "P":   res["P_inicial"].tolist(),
    }


# ── PRE-CALENTAR ───────────────────────────────────────────────────────────────
try:
    print("Pre-calculando linea base (escenario=base, T=20)...")
    _obtener_base("base", 20)
    print("Listo.")
except Exception as _e:
    import traceback
    print(f"[ADVERTENCIA] Pre-calentamiento fallido: {_e}")
    traceback.print_exc()

# ── HELPERS UI ─────────────────────────────────────────────────────────────────
_lbl_style = {"fontSize": 13, "fontWeight": 600, "color": NEGRO,
              "marginBottom": 6, "display": "block"}


def _tarjeta(titulo: str, texto: str, color: str) -> html.Div:
    return html.Div(style={
        "flex": "1", "minWidth": 240, "background": "white",
        "borderRadius": 10, "padding": "14px 18px",
        "borderLeft": f"4px solid {color}", "boxShadow": "0 2px 6px #0001",
    }, children=[
        html.H4(titulo, style={"margin": "0 0 5px", "fontSize": 13, "color": color}),
        html.P(texto,  style={"margin": 0, "fontSize": 12, "color": "#444",
                               "lineHeight": 1.5}),
    ])


def _kpi(titulo: str, base: float, pol: float,
         unidad: str = "", mejor: str = "menor") -> html.Div:
    delta    = pol - base
    es_mejor = (delta < 0) if mejor == "menor" else (delta > 0)
    col_d    = VERDE if es_mejor else ROJO
    signo    = "+" if delta >= 0 else ""
    return html.Div(style={
        "background": "white", "borderRadius": 10, "padding": "12px 18px",
        "flex": "1", "minWidth": 150, "textAlign": "center",
        "boxShadow": "0 2px 6px #0001",
    }, children=[
        html.P(titulo, style={"margin": "0 0 3px", "fontSize": 11,
                               "color": GRIS, "fontWeight": 600}),
        html.P(f"{pol:.1f}{unidad}",
               style={"margin": 0, "fontSize": 20, "fontWeight": 700, "color": NEGRO}),
        html.P(f"Base {base:.1f}{unidad}  ·  Δ {signo}{delta:.1f}{unidad}",
               style={"margin": "3px 0 0", "fontSize": 11,
                      "color": col_d, "fontWeight": 600}),
    ])


def _hex2rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _loading(*children):
    """Envuelve elementos en un spinner de carga."""
    return dcc.Loading(
        children=list(children),
        type="circle",
        color=AZUL,
        style={"minHeight": 60},
    )


# ── LAYOUT ─────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="ABM · Políticas de autoproducción")
server = app.server   # requerido por gunicorn en producción

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif",
           "backgroundColor": FONDO, "minHeight": "100vh",
           "paddingBottom": 40},
    children=[

        # Encabezado
        html.Div(style={"background": NEGRO, "color": "white",
                        "padding": "20px 36px 16px"}, children=[
            html.H1("Simulador de políticas · Autoproducción de vivienda",
                    style={"margin": 0, "fontSize": 21, "fontWeight": 700}),
            html.P("ABM · Quintil 1 de ingresos · ENIGH 2024  |  "
                   "Diagnóstico: cuello de botella E3→E4 y rezago extremo en E1",
                   style={"margin": "3px 0 0", "fontSize": 12, "opacity": 0.7}),
        ]),

        # Tarjetas diagnóstico
        html.Div(style={"display": "flex", "gap": 14, "flexWrap": "wrap",
                        "padding": "18px 36px 0"}, children=[
            _tarjeta("Cuello de botella E3 → E4",
                     "Servicios básicos (agua, drenaje, electricidad) requieren "
                     "infraestructura externa que el hogar no puede autogestionar.",
                     AZUL),
            _tarjeta("Etapa 1 muy rezagada",
                     "Sin excedente de ingreso no hay ahorro posible. "
                     "Sin capital semilla la autoproducción nunca arranca.",
                     ROJO),
            _tarjeta("Proceso progresivo incompleto",
                     "La mayoría no alcanza E4 en 30 años. "
                     "Shocks económicos y altos costos en E3 son los factores clave.",
                     NARANJA),
        ]),

        # Panel de controles
        html.Div(style={"background": "white", "borderRadius": 10,
                        "padding": "20px 24px", "margin": "18px 36px 0",
                        "boxShadow": "0 2px 8px #0001"}, children=[
            html.H3("Configuración de la simulación",
                    style={"margin": "0 0 14px", "fontSize": 15, "color": NEGRO}),
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "2fr 1fr 1fr 1fr",
                            "gap": 24, "alignItems": "start"}, children=[

                # Política
                html.Div([
                    html.Label("Política pública", style=_lbl_style),
                    dcc.Dropdown(
                        id="dd-politica",
                        options=[{"label": v["label"], "value": k}
                                 for k, v in POLITICAS.items()],
                        value="ninguna", clearable=False,
                        style={"fontSize": 13}),
                ]),

                # Intensidad
                html.Div([
                    html.Label("Intensidad", style=_lbl_style),
                    dcc.Slider(id="sl-intensidad", min=0, max=1, step=0.05,
                               value=0.5,
                               marks={0: "Baja", 0.5: "Media", 1: "Alta"},
                               tooltip={"placement": "bottom",
                                        "always_visible": True}),
                ]),

                # Horizonte
                html.Div([
                    html.Label("Horizonte (años)", style=_lbl_style),
                    dcc.Slider(id="sl-horizonte", min=5, max=40, step=5,
                               value=20,
                               marks={5: "5", 10: "10", 20: "20",
                                      30: "30", 40: "40"},
                               tooltip={"placement": "bottom",
                                        "always_visible": True}),
                ]),

                # Escenario + botón
                html.Div([
                    html.Label("Escenario macro", style=_lbl_style),
                    dcc.RadioItems(
                        id="ri-escenario",
                        options=[{"label": " Conservador", "value": "conservador"},
                                 {"label": " Base",        "value": "base"},
                                 {"label": " Crítico",     "value": "critico"}],
                        value="base", inline=True,
                        style={"fontSize": 12, "gap": 10}),
                    html.Div(style={"height": 10}),
                    html.Button("▶  Simular política", id="btn-simular", n_clicks=0,
                                style={"background": AZUL, "color": "white",
                                       "border": "none", "borderRadius": 7,
                                       "padding": "9px 22px", "fontSize": 13,
                                       "cursor": "pointer", "fontWeight": 600,
                                       "width": "100%"}),
                    html.Div(id="div-status",
                             style={"marginTop": 6, "fontSize": 11, "color": GRIS}),
                ]),
            ]),
        ]),

        # KPIs
        _loading(
            html.Div(id="div-kpi",
                     style={"display": "flex", "gap": 12, "flexWrap": "wrap",
                            "padding": "18px 36px 0"}),
        ),

        # Gráfica 1: distribución de etapas
        html.Div(style={"padding": "18px 36px 0"}, children=[
            _loading(dcc.Graph(id="graf-etapas",
                               config={"displayModeBar": False},
                               style={"height": 400})),
        ]),

        # Gráficas 2 y 3
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": 18, "padding": "0 36px"}, children=[
            _loading(dcc.Graph(id="graf-rezago",
                               config={"displayModeBar": False},
                               style={"height": 320})),
            _loading(dcc.Graph(id="graf-absorcion",
                               config={"displayModeBar": False},
                               style={"height": 320})),
        ]),

        # Gráficas 4 y 5
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": 18, "padding": "18px 36px 0"}, children=[
            _loading(dcc.Graph(id="graf-markov",
                               config={"displayModeBar": False},
                               style={"height": 340})),
            _loading(dcc.Graph(id="graf-impacto",
                               config={"displayModeBar": False},
                               style={"height": 340})),
        ]),

        # Gráfica 6: embudo
        html.Div(style={"padding": "18px 36px 0"}, children=[
            _loading(dcc.Graph(id="graf-embudo",
                               config={"displayModeBar": False},
                               style={"height": 300})),
        ]),

        html.P(
            "Fuente: Elaboración propia — ABM con datos ENIGH 2024, INEGI. "
            "Transiciones modeladas con función logística condicionada "
            "por cadena de Markov empírica.",
            style={"textAlign": "center", "fontSize": 11, "color": GRIS,
                   "marginTop": 24, "padding": "0 36px"},
        ),

        # Store único con ambos resultados
        dcc.Store(id="store-resultados"),
    ]
)


# ── CALLBACK: SIMULAR ─────────────────────────────────────────────────────────
@app.callback(
    Output("store-resultados", "data"),
    Output("div-status",       "children"),
    Input("btn-simular",       "n_clicks"),
    State("dd-politica",       "value"),
    State("sl-intensidad",     "value"),
    State("sl-horizonte",      "value"),
    State("ri-escenario",      "value"),
    prevent_initial_call=True,
)
def simular(_, politica, intensidad, horizonte, escenario):
    base_raw = _obtener_base(escenario, horizonte)
    pol_raw  = _sim_fast(politica, intensidad, horizonte, escenario)

    payload = {
        "base": _paquete(base_raw),
        "pol":  _paquete(pol_raw),
        "meta": {
            "politica":   politica,
            "intensidad": intensidad,
            "T":          horizonte,
            "escenario":  escenario,
        },
    }
    lab    = POLITICAS[politica]["label"]
    status = f"OK: {lab} · {intensidad:.0%} · {horizonte} anios · {escenario}"
    return payload, status


# ── CALLBACK: VISUALIZAR ──────────────────────────────────────────────────────
@app.callback(
    Output("div-kpi",        "children"),
    Output("graf-etapas",    "figure"),
    Output("graf-rezago",    "figure"),
    Output("graf-absorcion", "figure"),
    Output("graf-markov",    "figure"),
    Output("graf-impacto",   "figure"),
    Output("graf-embudo",    "figure"),
    Input("store-resultados", "data"),
    prevent_initial_call=True,
)
def actualizar(data):
    if not data:
        raise dash.exceptions.PreventUpdate

    meta      = data["meta"]
    politica  = meta["politica"]
    T         = meta["T"]
    col_pol   = POLITICAS[politica]["color"]
    lab_pol   = POLITICAS[politica]["label"]

    df_pol  = pd.DataFrame(data["pol"]["df"])
    df_base = pd.DataFrame(data["base"]["df"])
    n       = data["pol"]["n"]
    P_pol   = np.array(data["pol"]["P"])
    P_base  = np.array(data["base"]["P"])

    xcol = "Paso" if "Paso" in df_pol.columns else df_pol.columns[0]
    x    = df_pol[xcol]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    def pct(df, k):
        return df.iloc[-1][f"Hogares_etapa_{k}"] / n * 100

    # Absorción: t_base anclado en P_inicial (ENIGH), que representa la
    # "resistencia estructural" (~25.9 años). t_pol escala P_inicial por
    # la mejora relativa observada en P_efectiva del ABM, preservando la
    # dirección del cambio sin que el vaciado tardío de etapas distorsione
    # la magnitud.
    P_ef_base = P_efectiva(df_base)
    P_ef_pol  = P_efectiva(df_pol)
    t_base = tiempo_esperado_absorcion(P_base)
    P_pol_adj = P_base.copy()
    for _i in range(3):
        _p_b = max(float(P_ef_base[_i, _i + 1]), 1e-4)
        _p_p = float(P_ef_pol[_i, _i + 1])
        _p_adj = float(np.clip(P_base[_i, _i + 1] * (_p_p / _p_b), 0.005, 0.95))
        P_pol_adj[_i, _i + 1] = _p_adj
        P_pol_adj[_i, _i]     = 1.0 - _p_adj
    t_pol = tiempo_esperado_absorcion(P_pol_adj)

    kpis = [
        _kpi("% en Etapa 4 (año final)",
             pct(df_base, 4), pct(df_pol, 4), "%", "mayor"),
        _kpi("% atrapados en Etapa 1",
             pct(df_base, 1), pct(df_pol, 1), "%", "menor"),
        _kpi("Índice rezago final",
             df_base.iloc[-1]["Pct_rezago"] * 100,
             df_pol.iloc[-1]["Pct_rezago"]  * 100, "%", "menor"),
        _kpi("% hogares que avanzaron",
             df_base.iloc[-1]["Pct_avanzaron"] * 100,
             df_pol.iloc[-1]["Pct_avanzaron"]  * 100, "%", "mayor"),
        _kpi("Años esperados E1 → E4",
             t_base["etapa_1"], t_pol["etapa_1"], " años", "menor"),
    ]

    # ── Graf 1: Distribución de etapas ────────────────────────────────────────
    fig_e = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Línea base (sin política)", lab_pol],
        shared_yaxes=True,
    )
    for et in [1, 2, 3, 4]:
        col_e = COL_ETAPA[et]
        y_b   = df_base[f"Hogares_etapa_{et}"] / n * 100
        y_p   = df_pol[f"Hogares_etapa_{et}"]  / n * 100
        fig_e.add_trace(go.Scatter(
            x=x, y=y_b, name=LAB_ETAPA[et],
            line=dict(color=col_e, width=2),
            legendgroup=f"e{et}", showlegend=True,
            hovertemplate="%{y:.1f}%<extra>" + LAB_ETAPA[et] + "</extra>",
        ), row=1, col=1)
        fig_e.add_trace(go.Scatter(
            x=x, y=y_p, name=LAB_ETAPA[et],
            line=dict(color=col_e, width=2.5),
            legendgroup=f"e{et}", showlegend=False,
            hovertemplate="%{y:.1f}%<extra>" + LAB_ETAPA[et] + "</extra>",
        ), row=1, col=2)
    fig_e.update_layout(
        title=dict(text="Distribución de hogares por etapa constructiva",
                   font=dict(size=14, color=NEGRO)),
        height=400, plot_bgcolor="white", paper_bgcolor=FONDO,
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center"),
        margin=dict(l=50, r=30, t=70, b=70),
        hovermode="x unified",
    )
    fig_e.update_yaxes(title_text="% de hogares", gridcolor="#EEE", col=1)
    fig_e.update_xaxes(title_text="Año", gridcolor="#EEE")

    # ── Graf 2: Rezago ────────────────────────────────────────────────────────
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=x, y=df_base["Pct_rezago"] * 100, name="Línea base",
        line=dict(color=GRIS, width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(107,107,107,0.07)",
    ))
    fig_r.add_trace(go.Scatter(
        x=x, y=df_pol["Pct_rezago"] * 100, name=lab_pol,
        line=dict(color=col_pol, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({_hex2rgb(col_pol)},0.10)",
    ))
    fig_r.update_layout(
        title="Índice de rezago habitacional promedio",
        xaxis_title="Año", yaxis_title="Rezago promedio (0–100)",
        plot_bgcolor="white", paper_bgcolor=FONDO, height=320,
        margin=dict(l=50, r=20, t=50, b=60),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        yaxis=dict(gridcolor="#EEE"), xaxis=dict(gridcolor="#EEE"),
        hovermode="x unified",
    )

    # ── Graf 3: Tiempo de absorción ───────────────────────────────────────────
    etiq_abs = ["Desde E1", "Desde E2", "Desde E3"]
    t_b = [t_base["etapa_1"], t_base["etapa_2"], t_base["etapa_3"]]
    t_p = [t_pol["etapa_1"],  t_pol["etapa_2"],  t_pol["etapa_3"]]
    fig_abs = go.Figure()
    fig_abs.add_trace(go.Bar(
        name="Línea base", x=etiq_abs, y=t_b, marker_color=GRIS,
        opacity=0.70, text=[f"{v:.1f} años" for v in t_b],
        textposition="outside",
    ))
    fig_abs.add_trace(go.Bar(
        name=lab_pol, x=etiq_abs, y=t_p, marker_color=col_pol,
        opacity=0.90, text=[f"{v:.1f} años" for v in t_p],
        textposition="outside",
    ))
    fig_abs.update_layout(
        title="Años esperados para alcanzar Etapa 4 (absorción de Markov)",
        barmode="group", plot_bgcolor="white", paper_bgcolor=FONDO,
        height=320, margin=dict(l=50, r=20, t=55, b=60),
        yaxis=dict(title="Años", gridcolor="#EEE"),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )

    # ── Graf 4: Diferencial de matrices Markov efectivas ─────────────────────
    diff_P = P_ef_pol - P_ef_base
    etiq_m = ["E1", "E2", "E3", "E4"]
    fig_mk = make_subplots(rows=1, cols=2,
                           subplot_titles=["P efectiva base", "Delta (Politica - Base)"])
    fig_mk.add_trace(go.Heatmap(
        z=P_ef_base, x=etiq_m, y=etiq_m, colorscale="Blues",
        zmin=0, zmax=1, showscale=False,
        text=[[f"{v:.3f}" for v in row] for row in P_ef_base],
        texttemplate="%{text}",
    ), row=1, col=1)
    fig_mk.add_trace(go.Heatmap(
        z=diff_P, x=etiq_m, y=etiq_m, colorscale="RdYlGn",
        zmid=0, zmin=-0.10, zmax=0.10, showscale=True,
        text=[[f"{v:+.3f}" for v in row] for row in diff_P],
        texttemplate="%{text}",
        colorbar=dict(title="Delta P", len=0.8),
    ), row=1, col=2)
    fig_mk.update_layout(
        title="Matrices de transición de Markov: base vs política",
        height=340, plot_bgcolor="white", paper_bgcolor=FONDO,
        margin=dict(l=40, r=50, t=65, b=40),
    )
    fig_mk.update_yaxes(autorange="reversed")

    # ── Graf 5: Impacto incremental ───────────────────────────────────────────
    delta_e4 = (df_pol["Hogares_etapa_4"] - df_base["Hogares_etapa_4"]).clip(lower=0)
    delta_e1 = (df_base["Hogares_etapa_1"] - df_pol["Hogares_etapa_1"]).clip(lower=0)
    fig_imp = make_subplots(specs=[[{"secondary_y": True}]])
    fig_imp.add_trace(go.Bar(
        x=x, y=delta_e4, name="Hogares adicionales en E4",
        marker_color=VERDE, opacity=0.80,
        hovertemplate="+%{y:.0f} hogares<extra>E4</extra>",
    ), secondary_y=False)
    fig_imp.add_trace(go.Scatter(
        x=x, y=delta_e1, name="Hogares rescatados de E1",
        line=dict(color=ROJO, width=2.5), mode="lines+markers",
        marker=dict(size=5),
        hovertemplate="%{y:.0f} hogares menos en E1<extra></extra>",
    ), secondary_y=True)
    fig_imp.update_layout(
        title="Impacto incremental de la política (respecto a línea base)",
        height=340, plot_bgcolor="white", paper_bgcolor=FONDO,
        margin=dict(l=55, r=55, t=55, b=60),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        xaxis=dict(title="Año", gridcolor="#EEE"),
        hovermode="x unified",
    )
    fig_imp.update_yaxes(title_text="Hogares extra en E4",
                         gridcolor="#EEE", secondary_y=False)
    fig_imp.update_yaxes(title_text="Hogares menos en E1", secondary_y=True)

    # ── Graf 6: Distribución final base vs política ───────────────────────────
    # Barras horizontales por etapa: muestra directamente cuántos hogares
    # quedaron en cada etapa al año T. Sensible a políticas en todas las etapas.
    etiq_emb = ["E1 · Inicial", "E2 · Consolidación",
                "E3 · Terminaciones", "E4 · Servicios"]
    col_etapas = [ROJO, NARANJA, AZUL, VERDE]
    vals_b = [df_base.iloc[-1][f"Hogares_etapa_{k}"] / n * 100 for k in [1, 2, 3, 4]]
    vals_p = [df_pol.iloc[-1][f"Hogares_etapa_{k}"]  / n * 100 for k in [1, 2, 3, 4]]
    deltas = [vp - vb for vp, vb in zip(vals_p, vals_b)]

    fig_emb = go.Figure()
    fig_emb.add_trace(go.Bar(
        name="Línea base", y=etiq_emb, x=vals_b,
        orientation="h",
        marker_color=col_etapas, opacity=0.35,
        text=[f"{v:.1f}%" for v in vals_b],
        textposition="inside", insidetextanchor="end",
        hovertemplate="%{y}: %{x:.1f}%<extra>Base</extra>",
    ))
    fig_emb.add_trace(go.Bar(
        name=lab_pol, y=etiq_emb, x=vals_p,
        orientation="h",
        marker_color=col_etapas, opacity=0.90,
        text=[f"{v:.1f}%  ({d:+.1f}%)" for v, d in zip(vals_p, deltas)],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra>" + lab_pol + "</extra>",
    ))
    fig_emb.update_layout(
        title=f"Distribución de hogares por etapa — año {T}  (base vs política)",
        barmode="overlay",
        height=300, paper_bgcolor=FONDO, plot_bgcolor="white",
        margin=dict(l=160, r=100, t=50, b=50),
        xaxis=dict(title="% de hogares", gridcolor="#EEE"),
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
    )

    return kpis, fig_e, fig_r, fig_abs, fig_mk, fig_imp, fig_emb


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=8050)
