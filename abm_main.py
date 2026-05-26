"""
ABM: Autoproducción de vivienda en México
Quintil 1 de ingresos — ENIGH 2024
"""

import mesa
import numpy as np
import pandas as pd
from pathlib import Path
from markov import construir_matriz_markov, tiempo_esperado_absorcion

DATA_DIR = Path("data")

COSTOS_BASE_DEFAULT = {
    0: 10_666,
    1: 16_910,
    2: 59_800,
    3: 40_780,
    4: 36_350,
}

class HogarAgent(mesa.Agent):
    def __init__(self, model, datos_hogar, params, datos_municipio=None):
        super().__init__(model)
        self.ingreso_anual    = datos_hogar.get("ingreso_hogar", 0) * 4
        self.n_integrantes    = datos_hogar.get("n_integrantes", 1)
        def _int(val, default=0):
            try:
                return int(val) if val == val else default  # val != val iff NaN
            except (TypeError, ValueError):
                return default

        self.rural            = _int(datos_hogar.get("rural", datos_hogar.get("rural_si", 0)))
        self.estado_viv       = _int(datos_hogar.get("etapa", 1), default=1)
        self.factor_expansion = datos_hogar.get("factor_viv", 1)
        self.fin_formal       = _int(datos_hogar.get("fin_formal_si", 0))
        self.nivelaprob       = datos_hogar.get("nivelaprob", "02")
        self.rezago_0         = datos_hogar.get("rezago_continuo", 0.5)
        self.cve_ent          = str(datos_hogar.get("cve_ent", "00")).zfill(2)

        if datos_municipio is not None:
            self.dens_oferentes = max(float(datos_municipio.get("dens_empleo_10k", 1.0)), 0.1)
            self.im_2020        = float(datos_municipio.get("IM_2020", 0.0))
            self.pct_rural      = float(datos_municipio.get("pct_rural", 0.5))
        else:
            self.dens_oferentes = 1.0
            self.im_2020        = 0.0
            self.pct_rural      = 0.5

        self.params             = params
        self.ahorro_acumulado   = 0.0
        self.historial_etapas   = [self.estado_viv]
        self.historial_ingresos = [self.ingreso_anual]
        self.shocks_recibidos   = 0
        self.avances_realizados = 0
        self.rezago_actual      = self.rezago_0

    def consumo_minimo(self, lineas_bienestar):
        if self.rural:
            lb = lineas_bienestar.get("lb_rur_anual", 39_389)
        else:
            lb = lineas_bienestar.get("lb_urb_anual", 54_591)
        return lb * self.n_integrantes

    def presupuesto_construccion(self, lineas_bienestar):
        excedente = self.ingreso_anual - self.consumo_minimo(lineas_bienestar)
        theta = np.clip(
            self.params["theta_min"] +
            (self.params["theta_max"] - self.params["theta_min"]) *
            (excedente / max(self.ingreso_anual, 1)),
            self.params["theta_min"],
            self.params["theta_max"]
        )
        B = max(excedente * theta, 0)
        self.ahorro_acumulado += B
        return self.ahorro_acumulado

    def costo_etapa(self):
        k = self.estado_viv
        if k >= int(self.params.get("estado_absorbente", 4)):
            return np.inf
        c_base = self.model.costos_etapa.get(k, COSTOS_BASE_DEFAULT.get(k, 40_000))
        c_base *= (1 + self.params["tau_ruralidad"] * self.rural)
        c_base *= max(
            1 - self.params["beta_oferta"] * np.log(max(self.dens_oferentes, 0.1)),
            0.5
        )
        demanda_local = self.model.hogares_construyendo / max(len(self.model.agents), 1)
        c_base *= (1 + self.params["lambda_congestion"] * demanda_local)
        c_base *= self.model.inflacion_acumulada
        return c_base

    def prob_transicion(self, B, C):
        if B <= 0 or C == np.inf:
            return 0.0
        ln_B = np.log(max(B, 1))
        ln_C = np.log(max(C, 1))
        try:
            edu_norm = int(str(self.nivelaprob).strip().lstrip("0") or "0") / 10
        except (ValueError, AttributeError):
            edu_norm = 0.2
        edu_norm = np.clip(edu_norm, 0, 1)
        im_norm  = np.clip(self.im_2020 / 100, 0, 1)
        x_i = 0.3 * edu_norm + 0.3 * self.fin_formal - 0.4 * im_norm
        logit = (self.params["alpha_0"] +
                 self.params["alpha_1"] * ln_B -
                 self.params["alpha_2"] * ln_C +
                 self.params["alpha_3"] * np.log(max(self.dens_oferentes, 0.1)) +
                 self.params["alpha_4"] * x_i)
        return 1 / (1 + np.exp(-np.clip(logit, -20, 20)))

    def aplicar_shock(self):
        if self.model.random.random() < self.params["shock_prob_base"]:
            self.ingreso_anual    *= (1 - self.params["shock_delta"])
            self.ahorro_acumulado *= 0.5
            self.shocks_recibidos += 1
            return True
        return False

    def actualizar_ingreso(self):
        macro_row = self.model.macro_por_ent.get(self.cve_ent, self.model.macro_nacional)
        crecimiento = macro_row.get("crecimiento_salario", 0.03)
        inflacion   = macro_row.get("inflacion_general",   0.05)
        self.ingreso_anual = max(self.ingreso_anual * (1 + crecimiento - inflacion), 1)

    def step(self):
        estado_abs = int(self.params.get("estado_absorbente", 4))
        if self.estado_viv >= estado_abs:
            self.historial_etapas.append(self.estado_viv)
            return

        self.actualizar_ingreso()
        self.aplicar_shock()

        B = self.presupuesto_construccion(self.model.lineas_bienestar)
        C = self.costo_etapa()

        idx_etapa = np.clip(self.estado_viv - 1, 0, 2)
        p_markov  = self.model.P_markov[idx_etapa, idx_etapa + 1]
        p_logit   = self.prob_transicion(B, C)
        p_final   = p_markov * p_logit

        if self.model.random.random() < p_final:
            self.ahorro_acumulado   = max(self.ahorro_acumulado - C, 0)
            self.estado_viv        += 1
            self.avances_realizados += 1
            self.model.hogares_construyendo += 1

        k = self.estado_viv
        self.rezago_actual = self.model.rezago_por_etapa.get(
            k, max(0.0, self.rezago_0 - self.avances_realizados * 0.25)
        )
        self.historial_etapas.append(self.estado_viv)
        self.historial_ingresos.append(self.ingreso_anual)


class AutoproduccionModel(mesa.Model):

    def __init__(self, escenario="base", seed=42):
        super().__init__(rng=seed)

        self.hogares_df                    = self._cargar_hogares()
        self.municipios_df                 = self._cargar_municipios()
        self.costos_etapa                  = self._cargar_etapas()
        self.macro_por_ent, \
            self.macro_nacional            = self._cargar_macro()
        self.lineas_bienestar              = self._cargar_lineas_bienestar()
        self.params                        = self._cargar_parametros(escenario)

        self.paso_actual          = 0
        self.hogares_construyendo = 0
        self.inflacion_acumulada  = 1.0

        self.P_markov, _, _ = construir_matriz_markov(
            self.hogares_df, col_etapa="etapa", col_factor="factor_viv"
        )

        if "rezago_continuo" in self.hogares_df.columns and "etapa" in self.hogares_df.columns:
            self.rezago_por_etapa = (
                self.hogares_df.groupby("etapa")["rezago_continuo"].mean().to_dict()
            )
        else:
            self.rezago_por_etapa = {1: 0.85, 2: 0.60, 3: 0.35, 4: 0.10}

        self._crear_agentes()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Paso":                 lambda m: m.paso_actual,
                "Hogares_etapa_1":      lambda m: sum(1 for a in m.agents if a.estado_viv == 1),
                "Hogares_etapa_2":      lambda m: sum(1 for a in m.agents if a.estado_viv == 2),
                "Hogares_etapa_3":      lambda m: sum(1 for a in m.agents if a.estado_viv == 3),
                "Hogares_etapa_4":      lambda m: sum(1 for a in m.agents if a.estado_viv == 4),
                "Pct_rezago":           lambda m: np.mean([a.rezago_actual for a in m.agents]),
                "Pct_avanzaron":        lambda m: sum(1 for a in m.agents if a.avances_realizados > 0) / max(len(list(m.agents)), 1),
                "Ingreso_promedio":     lambda m: np.mean([a.ingreso_anual for a in m.agents]),
                "Hogares_construyendo": lambda m: m.hogares_construyendo,
                "Inflacion_acumulada":  lambda m: m.inflacion_acumulada,
            },
            agent_reporters={
                "estado_viv":         "estado_viv",
                "ingreso_anual":      "ingreso_anual",
                "ahorro_acumulado":   "ahorro_acumulado",
                "rezago_actual":      "rezago_actual",
                "shocks_recibidos":   "shocks_recibidos",
                "avances_realizados": "avances_realizados",
                "rural":              "rural",
                "n_integrantes":      "n_integrantes",
            }
        )

    def _cargar_hogares(self):
        path = DATA_DIR / "hogares.parquet"
        if path.exists():
            return pd.read_parquet(path)
        print("⚠️  hogares.parquet no encontrado. Usando datos sintéticos.")
        return self._datos_sinteticos()

    def _cargar_municipios(self):
        path = DATA_DIR / "municipios.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="latin1")
            for col, width in [("cve_ent", 2), ("cve_mun", 3)]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.zfill(width)
            return df
        print("⚠️  municipios.csv no encontrado.")
        return pd.DataFrame()

    def _cargar_etapas(self):
        """etapas.csv: etapa (0-4), nombre, costo_base"""
        path = DATA_DIR / "etapas.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="latin1")
            if "etapa" in df.columns and "costo_base" in df.columns:
                return dict(zip(df["etapa"].astype(int), df["costo_base"].astype(float)))
        return COSTOS_BASE_DEFAULT.copy()

    def _cargar_macro(self):
        """
        macro.csv: estado, cve_ent, anio, inflacion_general,
                   inflacion_construccion, crecimiento_salario, tasa_desocupacion
        Promedia los últimos 5 años por entidad.
        """
        path = DATA_DIR / "macro.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="latin1")
            anio_max = df["anio"].max()
            df = df[df["anio"] >= anio_max - 4]
            cols_num = ["inflacion_general", "inflacion_construccion",
                        "crecimiento_salario", "tasa_desocupacion"]
            # El CSV tiene outliers extremos (ej. 166% de inflación en una entidad)
            # que distorsionan la media. Se aplican topes económicos fijos:
            # inflación 0–25%, crecimiento salarial 0–20%.
            _clips = {
                "inflacion_general":      (0.00, 0.25),
                "inflacion_construccion": (0.00, 0.25),
                "crecimiento_salario":    (0.00, 0.20),
            }
            for col, (lo, hi) in _clips.items():
                if col in df.columns:
                    df[col] = df[col].clip(lo, hi)
            macro_ent = df.groupby("cve_ent")[cols_num].mean().reset_index()
            macro_por_ent = {
                str(int(row["cve_ent"])).zfill(2): row[cols_num].to_dict()
                for _, row in macro_ent.iterrows()
            }
            macro_nacional = df[cols_num].mean().to_dict()
            return macro_por_ent, macro_nacional
        macro_nacional = {
            "inflacion_general": 0.05, "inflacion_construccion": 0.07,
            "crecimiento_salario": 0.03, "tasa_desocupacion": 0.04,
        }
        return {}, macro_nacional

    def _cargar_lineas_bienestar(self):
        """
        lineas_bienestar.csv: anio, lb_urb_anual, lb_rur_anual,
                               lbm_urb_anual, lbm_rur_anual
        Los valores ya están en escala mensual per cápita — multiplicar × 12
        para obtener escala anual.
        """
        path = DATA_DIR / "lineas_bienestar.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="latin1")
            if "anio" in df.columns:
                df = df[df["anio"] == df["anio"].max()]
            if not df.empty:
                row = df.iloc[0].to_dict()
                # Los valores del CSV son mensuales → convertir a anuales
                return {
                    "lb_urb_anual":  row.get("lb_urb_anual",  4549.31) * 12,
                    "lb_rur_anual":  row.get("lb_rur_anual",  3282.43) * 12,
                    "lbm_urb_anual": row.get("lbm_urb_anual", 2328.22) * 12,
                    "lbm_rur_anual": row.get("lbm_rur_anual", 1781.34) * 12,
                }
        return {
            "lb_urb_anual":  54_591,
            "lb_rur_anual":  39_389,
            "lbm_urb_anual": 27_939,
            "lbm_rur_anual": 21_376,
        }

    def _cargar_parametros(self, escenario):
        """
        parametros.csv en formato LARGO: escenario, param_name, value, description
        Pivotea a dict para el escenario solicitado.
        estado_absorbente viene como 5 en el CSV → se normaliza a 4.
        """
        path = DATA_DIR / "parametros.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="latin1")
            df_esc = df[df["escenario"] == escenario]
            if not df_esc.empty:
                params = dict(zip(df_esc["param_name"], df_esc["value"].astype(float)))
                # El CSV marca estado_absorbente=5 (conceptual), en código usamos 4
                params["estado_absorbente"] = 4
                # Alphas no están en el CSV, agregar defaults
                params.setdefault("alpha_0", -2.0)
                params.setdefault("alpha_1",  0.8)
                params.setdefault("alpha_2",  0.6)
                params.setdefault("alpha_3",  0.3)
                params.setdefault("alpha_4",  0.5)
                return params

        defaults = {
            "conservador": dict(T=30, beta_oferta=0.05, tau_ruralidad=0.15,
                lambda_congestion=0.02, theta_min=0.01, theta_max=0.50,
                shock_prob_base=0.05, shock_delta=0.20, hacinamiento_umbral=2.5,
                estado_absorbente=4, alpha_0=-2.0, alpha_1=0.8,
                alpha_2=0.6, alpha_3=0.3, alpha_4=0.5),
            "base": dict(T=30, beta_oferta=0.15, tau_ruralidad=0.30,
                lambda_congestion=0.05, theta_min=0.01, theta_max=0.60,
                shock_prob_base=0.10, shock_delta=0.30, hacinamiento_umbral=2.5,
                estado_absorbente=4, alpha_0=-2.0, alpha_1=0.8,
                alpha_2=0.6, alpha_3=0.3, alpha_4=0.5),
            "critico": dict(T=30, beta_oferta=0.30, tau_ruralidad=0.50,
                lambda_congestion=0.10, theta_min=0.01, theta_max=0.70,
                shock_prob_base=0.20, shock_delta=0.40, hacinamiento_umbral=2.5,
                estado_absorbente=4, alpha_0=-2.0, alpha_1=0.8,
                alpha_2=0.6, alpha_3=0.3, alpha_4=0.5),
        }
        return defaults.get(escenario, defaults["base"])

    def _datos_sinteticos(self, n=200):
        rng = np.random.default_rng(42)
        etapas  = rng.integers(1, 5, n)
        rezagos = np.array([{1: 0.85, 2: 0.60, 3: 0.35, 4: 0.10}[e] for e in etapas])
        rezagos += rng.normal(0, 0.05, n)
        return pd.DataFrame({
            "ingreso_hogar":   rng.normal(5_000, 1_500, n).clip(1_000),
            "n_integrantes":   rng.integers(2, 8, n),
            "rural_si":        rng.integers(0, 2, n),
            "etapa":           etapas,
            "factor_viv":      rng.uniform(100, 500, n),
            "fin_formal_si":   rng.integers(0, 2, n),
            "nivelaprob":      rng.choice(["02", "03", "04", "06"], n),
            "rezago_continuo": np.clip(rezagos, 0, 1),
            "cve_ent":         rng.choice([str(x).zfill(2) for x in range(1, 33)], n),
            "cve_mun":         rng.choice([str(x).zfill(3) for x in range(1, 50)], n),
        })

    def _crear_agentes(self):
        # Pre-indexar municipios por (cve_ent, cve_mun)
        mun_index = {}
        if not self.municipios_df.empty:
            for _, row in self.municipios_df.iterrows():
                key = (str(row.get("cve_ent", "")).zfill(2),
                       str(row.get("cve_mun", "")).zfill(3))
                mun_index[key] = row.to_dict()

        for _, row in self.hogares_df.iterrows():
            datos_hogar = row.to_dict()
            datos_mun = None
            if mun_index:
                cve_ent = str(datos_hogar.get("cve_ent", "")).zfill(2)
                cve_mun = str(datos_hogar.get("cve_mun", "")).zfill(3)
                datos_mun = mun_index.get((cve_ent, cve_mun))
                if datos_mun is None:  # fallback: primer municipio de la entidad
                    datos_mun = next(
                        (v for k, v in mun_index.items() if k[0] == cve_ent), None
                    )
            HogarAgent(
                model=self, datos_hogar=datos_hogar,
                params=self.params, datos_municipio=datos_mun,
            )

    def step(self):
        self.paso_actual          += 1
        self.hogares_construyendo  = 0
        inf_const = self.macro_nacional.get("inflacion_construccion", 0.07)
        self.inflacion_acumulada  *= (1 + inf_const)
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def run(self, T=None):
        if T is None:
            T = int(self.params.get("T", 30))
        self.datacollector.collect(self)
        for _ in range(T):
            self.step()
        return self

    def resultados_modelo(self):
        return self.datacollector.get_model_vars_dataframe()

    def resultados_agentes(self):
        return self.datacollector.get_agent_vars_dataframe()
