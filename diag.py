import numpy as np
from abm_main import AutoproduccionModel

m = AutoproduccionModel(escenario="base", seed=42)
m.run(T=30)

df_m = m.resultados_modelo()
df_a = m.resultados_agentes().reset_index()

print("=== Distribucion por etapa por anno ===")
for t in [0, 5, 10, 15, 20, 25, 30]:
    row = df_m.iloc[t]
    print(f"Anno {t:2d}: E1={int(row.Hogares_etapa_1):4d} E2={int(row.Hogares_etapa_2):4d} E3={int(row.Hogares_etapa_3):4d} E4={int(row.Hogares_etapa_4):4d}")

print()
print("=== Ahorro vs ingreso promedio (agentes no absorbidos) ===")
for t in [1, 5, 10, 15, 20, 25, 30]:
    df_t = df_a[(df_a["Step"] == t) & (df_a["estado_viv"] < 4)]
    if len(df_t) > 0:
        print(f"Anno {t:2d}: ahorro={df_t['ahorro_acumulado'].mean():9.0f}  ingreso={df_t['ingreso_anual'].mean():9.0f}  n_activos={len(df_t)}")
    else:
        print(f"Anno {t:2d}: sin agentes activos")

print()
print(f"Inflacion acumulada anno 30: {m.inflacion_acumulada:.2f}x")

# Costo de cada etapa al anno 30
from abm_main import COSTOS_BASE_DEFAULT
params = m.params
for k in [1, 2, 3]:
    c_base = COSTOS_BASE_DEFAULT[k]
    costo_final = c_base * m.inflacion_acumulada
    print(f"Costo etapa {k} al anno 30: {costo_final:,.0f} MXN (base: {c_base:,.0f})")
