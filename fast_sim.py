"""
fast_sim.py
───────────
ABM vectorizado (numpy) que replica la lógica de HogarAgent.step()
sobre arrays completos en lugar de iterar agente por agente.

Velocidad esperada: ~50x más rápido que la versión Mesa.
"""

import copy
import numpy as np
import pandas as pd
from abm_main import AutoproduccionModel, COSTOS_BASE_DEFAULT
from markov import construir_matriz_markov

SEED = 42

# ── CACHÉ: datos del modelo sin crear agentes de Mesa ─────────────────────────
_cache_ref: dict[str, dict] = {}


def _ref(escenario: str) -> dict:
    """
    Extrae y cachea parámetros y datos del modelo SIN crear los agentes de Mesa.
    Esto elimina los ~15s de inicialización de HogarAgent × 11k.
    """
    if escenario not in _cache_ref:
        print(f"  [fast_sim] cargando datos '{escenario}'…")
        # Creamos el modelo solo para acceder a sus métodos de carga
        # e inmediatamente extraemos lo que necesitamos.
        m = AutoproduccionModel.__new__(AutoproduccionModel)
        # Inicializar lo mínimo (sin super().__init__ ni agentes)
        m.hogares_df   = AutoproduccionModel._cargar_hogares(m)
        m.municipios_df = AutoproduccionModel._cargar_municipios(m)
        m.costos_etapa  = AutoproduccionModel._cargar_etapas(m)
        m.macro_por_ent, m.macro_nacional = AutoproduccionModel._cargar_macro(m)
        m.lineas_bienestar = AutoproduccionModel._cargar_lineas_bienestar(m)
        m.params        = AutoproduccionModel._cargar_parametros(m, escenario)

        if "rezago_continuo" in m.hogares_df.columns and "etapa" in m.hogares_df.columns:
            m.rezago_por_etapa = (
                m.hogares_df.groupby("etapa")["rezago_continuo"].mean().to_dict()
            )
        else:
            m.rezago_por_etapa = {1: 0.85, 2: 0.60, 3: 0.35, 4: 0.10}

        _cache_ref[escenario] = {
            "hogares_df":      m.hogares_df,
            "costos_etapa":    m.costos_etapa,
            "macro_por_ent":   m.macro_por_ent,
            "macro_nacional":  m.macro_nacional,
            "lineas_bienestar": m.lineas_bienestar,
            "params":          m.params,
            "rezago_por_etapa": m.rezago_por_etapa,
        }
        print(f"  [fast_sim] datos listos '{escenario}' — {len(m.hogares_df):,} hogares")
    return _cache_ref[escenario]


# ── POLÍTICAS ─────────────────────────────────────────────────────────────────

def aplicar_politica(params: dict, costos: dict,
                     politica: str, intensidad: float) -> tuple[dict, dict]:
    p = copy.deepcopy(params)
    c = copy.deepcopy(costos)
    s = intensidad

    if politica == "subsidio":
        for k in c:
            c[k] *= 1 - (0.40 * s if k == 3 else 0.28 * s)

    elif politica == "credito":
        p["theta_max"] = min(p["theta_max"] + 0.25 * s, 0.95)
        p["theta_min"] = min(p["theta_min"] + 0.05 * s, p["theta_max"] - 0.01)

    elif politica == "shocks":
        p["shock_prob_base"] *= (1 - 0.70 * s)
        p["shock_delta"]     *= (1 - 0.60 * s)

    elif politica == "oferentes":
        # dens_of=1 en el baseline → log(1)=0 → beta_oferta inerte.
        # La política eleva la densidad de oferentes directamente; el ABM
        # lee params["dens_of_factor"] para inicializar dens_of.
        p["dens_of_factor"] = 1.0 + 4.0 * s   # hasta 5x más oferentes
        p["beta_oferta"]    *= (1 + 1.5 * s)   # amplifica el efecto-costo

    elif politica == "educacion":
        p["alpha_4"] += 0.50 * s
        p["alpha_3"] += 0.30 * s

    elif politica == "integral":
        # Aplica A + C + E a plena intensidad s.
        # El nivel de inversión lo controla el slider; no hay descuento interno.
        for k in c:
            c[k] *= 1 - (0.40 * s if k == 3 else 0.28 * s)
        p["shock_prob_base"] *= (1 - 0.70 * s)
        p["shock_delta"]     *= (1 - 0.60 * s)
        p["alpha_4"]         += 0.50 * s
        p["alpha_3"]         += 0.30 * s

    return p, c


# ── ABM VECTORIZADO ───────────────────────────────────────────────────────────

class VectorizedABM:
    """Replcia la lógica de HogarAgent.step() con operaciones numpy."""

    def __init__(self, df: pd.DataFrame, params: dict, costos: dict,
                 lineas: dict, macro_nacional: dict, macro_por_ent: dict,
                 P_markov: np.ndarray, rezago_por_etapa: dict,
                 seed: int = SEED):

        self.rng     = np.random.default_rng(seed)
        self.params  = params
        self.costos  = costos
        self.lineas  = lineas
        self.macro   = macro_nacional
        self.P       = P_markov
        self.rez_map = rezago_por_etapa
        n = len(df)
        self.n = n

        # ── Lectura robusta de columnas (siempre copias escribibles) ─────────
        def _int(col, default=0):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int).values.copy()
            return np.full(n, default, dtype=int)

        def _flt(col, default=0.0):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float).values.copy()
            return np.full(n, default)

        # ── Estado inicial ────────────────────────────────────────────────────
        self.etapas   = _int("etapa", 1)
        self.ingresos = _flt("ingreso_hogar", 5_000.0) * 4.0
        self.ahorros  = np.zeros(n)

        # ── Atributos fijos ───────────────────────────────────────────────────
        self.n_integ    = _flt("n_integrantes", 4.0)
        rural_col       = "rural_si" if "rural_si" in df.columns else "rural"
        self.rural      = _int(rural_col, 0).astype(float)
        self.fin_formal = _int("fin_formal_si", 0).astype(float)

        if "nivelaprob" in df.columns:
            def _edu(x):
                try:
                    return min(int(str(x).strip().lstrip("0") or "0") / 10, 1.0)
                except Exception:
                    return 0.2
            self.edu_norm = df["nivelaprob"].apply(_edu).values.astype(float)
        else:
            self.edu_norm = np.full(n, 0.2)

        # Municipal: densidad de oferentes. Baseline = 1 (log=0).
        # La política "oferentes" inyecta dens_of_factor > 1 vía params.
        dens_factor  = float(params.get("dens_of_factor", 1.0))
        self.dens_of = np.full(n, dens_factor)
        self.im_2020 = np.zeros(n)

        # Índice compuesto para el logit (igual que en HogarAgent)
        self.x_i = np.clip(
            0.3 * self.edu_norm + 0.3 * self.fin_formal - 0.4 * self.im_2020 / 100,
            -1.0, 1.0,
        )

        # Tasas macro por agente (por entidad federativa)
        self.crecimiento = np.full(n, macro_nacional.get("crecimiento_salario", 0.03))
        self.inflacion_g = np.full(n, macro_nacional.get("inflacion_general",   0.05))
        if macro_por_ent and "cve_ent" in df.columns:
            cve_arr = df["cve_ent"].astype(str).str.zfill(2).values
            for cve, vals in macro_por_ent.items():
                m = cve_arr == str(cve).zfill(2)
                self.crecimiento[m] = vals.get("crecimiento_salario", 0.03)
                self.inflacion_g[m] = vals.get("inflacion_general",   0.05)

        # Seguimiento acumulado
        self.ever_advanced       = np.zeros(n, dtype=bool)
        self.inflacion_acumulada = 1.0
        self.demanda_local       = 0.0
        self.history: list       = []

    def step(self) -> None:
        p = self.params
        n = self.n
        absorbing = int(p.get("estado_absorbente", 4))

        # 1. Actualizar ingresos
        self.ingresos = np.maximum(
            self.ingresos * (1.0 + self.crecimiento - self.inflacion_g), 1.0
        )

        # 2. Shocks económicos
        shock = self.rng.random(n) < p["shock_prob_base"]
        self.ingresos[shock] *= (1.0 - p["shock_delta"])
        self.ahorros[shock]  *= 0.5

        # 3. Presupuesto de construcción
        lb      = np.where(self.rural > 0,
                           self.lineas["lb_rur_anual"],
                           self.lineas["lb_urb_anual"])
        consumo   = lb * self.n_integ
        excedente = np.maximum(self.ingresos - consumo, 0.0)
        theta = np.clip(
            p["theta_min"] + (p["theta_max"] - p["theta_min"]) *
            (excedente / np.maximum(self.ingresos, 1.0)),
            p["theta_min"], p["theta_max"],
        )
        self.ahorros += np.maximum(excedente * theta, 0.0)

        # 4. Inflación de construcción acumulada
        self.inflacion_acumulada *= (1.0 + self.macro.get("inflacion_construccion", 0.07))

        # 5. Costo por etapa (vectorizado por grupo)
        cost = np.full(n, np.inf)
        for k in [1, 2, 3]:
            m = self.etapas == k
            if not m.any():
                continue
            c_base   = float(self.costos.get(k, COSTOS_BASE_DEFAULT.get(k, 40_000)))
            rural_f  = 1.0 + p["tau_ruralidad"] * self.rural[m]
            supply_f = np.maximum(
                1.0 - p["beta_oferta"] * np.log(np.maximum(self.dens_of[m], 0.1)),
                0.5,
            )
            cong_f   = 1.0 + p["lambda_congestion"] * self.demanda_local
            cost[m]  = c_base * rural_f * supply_f * cong_f * self.inflacion_acumulada

        # 6. Probabilidad de transición (logit × Markov)
        not_abs = self.etapas < absorbing
        valid   = not_abs & (self.ahorros > 0) & np.isfinite(cost)

        p_final = np.zeros(n)
        if valid.any():
            ln_B  = np.log(np.maximum(self.ahorros[valid], 1.0))
            ln_C  = np.log(np.maximum(cost[valid],         1.0))
            logit = (
                p["alpha_0"]
                + p["alpha_1"] * ln_B
                - p["alpha_2"] * ln_C
                + p["alpha_3"] * np.log(np.maximum(self.dens_of[valid], 0.1))
                + p["alpha_4"] * self.x_i[valid]
            )
            p_logit = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))

            # Componente Markov por etapa
            p_mk  = np.zeros(valid.sum())
            etv   = self.etapas[valid]
            for i in range(3):
                m2 = etv == (i + 1)
                if m2.any():
                    p_mk[m2] = self.P[i, i + 1]

            p_final[valid] = p_mk * p_logit

        # 7. Transiciones — contar por etapa ANTES de actualizar
        trans = (self.rng.random(n) < p_final) & not_abs
        trans_por_k = {k: int((trans & (self.etapas == k)).sum()) for k in [1, 2, 3]}

        self.ahorros[trans]   = np.maximum(self.ahorros[trans] - cost[trans], 0.0)
        self.etapas[trans]   += 1
        self.ever_advanced   |= trans
        self.demanda_local    = float(trans.sum()) / n

        # 8. Registro de paso
        counts = np.bincount(self.etapas, minlength=5)
        rezago  = sum(counts[k] * self.rez_map.get(k, 0.0) for k in range(1, 5)) / n
        self.history.append({
            "Hogares_etapa_1":  int(counts[1]),
            "Hogares_etapa_2":  int(counts[2]),
            "Hogares_etapa_3":  int(counts[3]),
            "Hogares_etapa_4":  int(counts[4]),
            "Pct_rezago":       float(rezago),
            "Pct_avanzaron":    float(self.ever_advanced.sum() / n),
            "Ingreso_promedio": float(self.ingresos.mean()),
            # Transiciones reales por etapa (para P_efectiva precisa)
            "Trans_1_2":        trans_por_k[1],
            "Trans_2_3":        trans_por_k[2],
            "Trans_3_4":        trans_por_k[3],
        })

    def run(self, T: int) -> pd.DataFrame:
        for _ in range(T):
            self.step()
        df = pd.DataFrame(self.history)
        df.insert(0, "Paso", range(1, T + 1))
        return df


# ── FUNCIÓN DE SIMULACIÓN PÚBLICA ─────────────────────────────────────────────

def correr_sim(politica: str, intensidad: float,
               T: int, escenario: str = "base") -> dict:
    """
    Corre la simulación vectorizada con la política indicada.

    Retorna:
        df          : DataFrame con evolución + columnas Trans_k_{k+1}
        n           : número total de hogares
        P_inicial   : matriz de Markov empírica del estado inicial (4×4)
    """
    datos = _ref(escenario)

    params_mod, costos_mod = aplicar_politica(
        datos["params"], datos["costos_etapa"], politica, intensidad
    )
    P_markov, _, _ = construir_matriz_markov(
        datos["hogares_df"], "etapa", "factor_viv"
    )

    sim = VectorizedABM(
        df               = datos["hogares_df"],
        params           = params_mod,
        costos           = costos_mod,
        lineas           = datos["lineas_bienestar"],
        macro_nacional   = datos["macro_nacional"],
        macro_por_ent    = datos["macro_por_ent"],
        P_markov         = P_markov,
        rezago_por_etapa = datos["rezago_por_etapa"],
    )

    historia = sim.run(T)
    return {
        "df":       historia,
        "n":        int(datos["hogares_df"].shape[0]),
        "P_inicial": P_markov,
    }


# ── P EFECTIVA DESDE LA EVOLUCIÓN DEL ABM ─────────────────────────────────────

def P_efectiva(historia_df: pd.DataFrame) -> np.ndarray:
    """
    Estima la matriz de transición efectiva desde las transiciones reales
    registradas en cada paso del ABM.

    Usa las columnas Trans_k_{k+1} (transiciones exactas contadas en step())
    en lugar de flujos netos, que son imprecisos cuando hay entradas y salidas
    simultáneas en la misma etapa.
    """
    P = np.eye(4)
    for k in range(1, 4):
        trans_col = f"Trans_{k}_{k+1}"
        stage_col = f"Hogares_etapa_{k}"

        if trans_col in historia_df.columns and stage_col in historia_df.columns:
            trans  = historia_df[trans_col].values.astype(float)
            n_k    = historia_df[stage_col].shift(1).fillna(
                         historia_df[stage_col].iloc[0]).values.astype(float)
            # MLE: total_transiciones / total_exposición. Evita que períodos
            # tardíos con n_k≈0 arrastren el promedio al piso → 200 años.
            p_av   = float(np.clip(
                np.sum(trans) / np.maximum(np.sum(n_k), 1.0), 0.005, 0.80))
        else:
            vals   = historia_df[stage_col].values.astype(float)
            out    = np.maximum(vals[:-1] - vals[1:], 0.0)
            n_prev = np.maximum(vals[:-1], 1.0)
            p_av   = float(np.clip(
                np.sum(out) / np.maximum(np.sum(n_prev), 1.0), 0.005, 0.80))

        P[k-1, k-1] = 1.0 - p_av
        P[k-1, k]   = p_av
    P[3, 3] = 1.0
    return P
