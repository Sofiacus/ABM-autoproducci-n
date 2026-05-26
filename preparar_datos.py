"""
preparar_datos.py
─────────────────
Verifica que todos los archivos del ABM estén presentes y tengan
la estructura correcta.

Desde R, exportar tabla_const así:
  library(arrow)
  tabla_const_hogar <- tabla_const %>%
    distinct(folioviv, foliohog, .keep_all = TRUE) %>%
    mutate(across(where(is.factor), ~ as.numeric(as.character(.))),
           cve_ent = substr(ubica_geo, 1, 3),
           cve_mun = substr(ubica_geo, 4, 6))
  write_parquet(tabla_const_hogar, "abm_autoproduccion/data/hogares.parquet")
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

# ── Columnas requeridas en hogares.parquet ─────────────────────────────────────
COLUMNAS_REQUERIDAS = {
    "ingreso_hogar":   "ingreso trimestral del hogar (se multiplica ×4 en el ABM)",
    "n_integrantes":   "número de integrantes del hogar",
    "rural_si":        "1=rural, 0=urbano (3er dígito folioviv==6)",
    "etapa":           "etapa constructiva 1–4",
    "factor_viv":      "factor de expansión vivienda",
    "fin_formal_si":   "financiamiento formal 0/1",
    "nivelaprob":      "nivel educativo jefe formato '00'–'10'",
    "rezago_continuo": "índice de rezago habitacional 0–1",
    "cve_ent":         "clave entidad federativa (2 o 3 dígitos)",
    "cve_mun":         "clave municipio (3 dígitos)",
}

COLUMNAS_OPCIONALES = {
    "informal_si": "trabajador informal 0/1",
    "edad":        "edad del jefe de hogar",
    "sexo":        "sexo del jefe (1=hombre, 2=mujer)",
}

# ── Estructura esperada de cada CSV ───────────────────────────────────────────
ESTRUCTURA_CSVS = {
    "etapas.csv": {
        "cols": ["etapa", "nombre", "costo_base"],
        "desc": "etapas 0–4 con costos base en pesos MXN 2024",
    },
    "lineas_bienestar.csv": {
        "cols": ["anio", "lb_urb_anual", "lb_rur_anual", "lbm_urb_anual", "lbm_rur_anual"],
        "desc": "líneas de bienestar mensuales per cápita (CONEVAL), se multiplican ×12",
    },
    "macro.csv": {
        "cols": ["estado", "cve_ent", "anio", "inflacion_general",
                 "inflacion_construccion", "crecimiento_salario", "tasa_desocupacion"],
        "desc": "indicadores macroeconómicos por entidad y año",
    },
    "municipios.csv": {
        "cols": ["cve_ent", "cve_mun", "pob_mun", "pct_rural",
                 "dens_empleo_10k", "IM_2020"],
        "desc": "datos municipales para ajuste de costos y marginación",
    },
    "parametros.csv": {
        "cols": ["escenario", "param_name", "value", "description"],
        "desc": "parámetros del ABM en formato largo (escenarios: conservador, base, critico)",
    },
}


def verificar_hogares():
    path = DATA_DIR / "hogares.parquet"
    print("─── hogares.parquet ───────────────────────────────────────")
    if not path.exists():
        print(f"  ❌ No encontrado en {path}")
        print("  Exporta desde R:")
        print("    write_parquet(tabla_const_hogar, 'data/hogares.parquet')")
        return False

    df = pd.read_parquet(path)
    print(f"  ✅ {df.shape[0]:,} hogares × {df.shape[1]} columnas")

    faltantes = []
    for col, desc in COLUMNAS_REQUERIDAS.items():
        if col not in df.columns:
            faltantes.append(col)
            print(f"  ❌ '{col}' — {desc}")
        else:
            n_na = df[col].isna().sum()
            marca = "⚠️ " if n_na > 0 else "  ✅"
            print(f"  {marca} '{col}': {df[col].dtype} | NAs: {n_na:,}")

    for col, desc in COLUMNAS_OPCIONALES.items():
        if col in df.columns:
            print(f"  ✅ '{col}' (opcional): {desc}")
        else:
            print(f"  ⚠️  '{col}' no disponible (opcional): {desc}")

    # Verificar rango de etapas
    if "etapa" in df.columns:
        vals = df["etapa"].dropna().unique()
        if not all(v in [1, 2, 3, 4] for v in vals):
            print(f"  ⚠️  Valores de 'etapa' fuera de [1,2,3,4]: {sorted(vals)}")

    # Verificar rezago_continuo en [0,1]
    if "rezago_continuo" in df.columns:
        fuera = ((df["rezago_continuo"] < 0) | (df["rezago_continuo"] > 1)).sum()
        if fuera > 0:
            print(f"  ⚠️  {fuera} valores de 'rezago_continuo' fuera de [0,1]")

    return len(faltantes) == 0


def verificar_csvs():
    for nombre, info in ESTRUCTURA_CSVS.items():
        print(f"─── {nombre} ───────────────────────────────────────────")
        path = DATA_DIR / nombre
        if not path.exists():
            print(f"  ❌ No encontrado")
            continue

        try:
            df = pd.read_csv(path, encoding="latin1")
        except Exception as e:
            print(f"  ❌ Error al leer: {e}")
            continue

        print(f"  ✅ {df.shape[0]:,} filas × {df.shape[1]} columnas — {info['desc']}")
        faltantes = [c for c in info["cols"] if c not in df.columns]
        if faltantes:
            print(f"  ⚠️  Columnas esperadas faltantes: {faltantes}")
            print(f"     Columnas presentes: {df.columns.tolist()}")
        else:
            print(f"  ✅ Todas las columnas requeridas presentes")

        # Verificaciones específicas
        if nombre == "parametros.csv":
            esc = df["escenario"].unique().tolist() if "escenario" in df.columns else []
            falt_esc = [e for e in ["conservador", "base", "critico"] if e not in esc]
            if falt_esc:
                print(f"  ⚠️  Escenarios faltantes: {falt_esc}")
            else:
                print(f"  ✅ Tres escenarios presentes: conservador, base, crítico")

        if nombre == "macro.csv":
            if "cve_ent" in df.columns:
                n_ent = df["cve_ent"].nunique()
                print(f"  ℹ️  {n_ent} entidades en macro.csv")
            if "anio" in df.columns:
                print(f"  ℹ️  Años: {df['anio'].min()}–{df['anio'].max()}")

        if nombre == "lineas_bienestar.csv":
            if "lb_urb_anual" in df.columns:
                ult = df.iloc[-1]
                print(f"  ℹ️  Último año: {ult.get('anio', '?')} | "
                      f"lb_urb={ult.get('lb_urb_anual', 0):.2f} (mensual × 12 = "
                      f"{ult.get('lb_urb_anual', 0)*12:,.0f} anual)")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║   Verificación de datos — ABM Autoproducción    ║")
    print("╚══════════════════════════════════════════════════╝\n")

    DATA_DIR.mkdir(exist_ok=True)

    ok_parquet = verificar_hogares()
    print()
    verificar_csvs()

    print("\n" + "═" * 52)
    if ok_parquet:
        print("✅ hogares.parquet listo — puedes correr dashboard_completo.py")
    else:
        print("⚠️  Exporta hogares.parquet desde R antes de continuar")
    print("═" * 52)
