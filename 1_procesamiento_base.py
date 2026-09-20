from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN
# =========================================================
RUTA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_procesamiento_final"
)

ARCH_CICLOS = RUTA / "ciclos_caminata_normalizados.csv"
ARCH_PROM_SUJ = RUTA / "promedio_por_sujeto.csv"

RUTA_SALIDA = RUTA / "seleccion_sujetos_robustos"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# filtros mínimos de robustez
MIN_CICLOS = 4
MIN_TRIALS = 2

# cuántos sujetos mostrar / exportar
TOP_K = 6

GENERAR_PLOTS = True


# =========================================================
# FUNCIONES
# =========================================================
def promedio_por_phase(df, xcol="x_norm", zcol="z_norm"):
    return (
        df.groupby("phase")[[xcol, zcol]]
        .mean()
        .reset_index()
        .sort_values("phase")
        .reset_index(drop=True)
    )


def dispersion_media(df_ciclos_subj, df_prom_subj, xcol="x_norm", zcol="z_norm"):
    vals = []

    for cycle_uid, g in df_ciclos_subj.groupby("cycle_uid"):
        g = g.sort_values("phase").reset_index(drop=True)
        p = df_prom_subj.sort_values("phase").reset_index(drop=True)

        if len(g) != len(p):
            continue

        d = np.sqrt(
            (g[xcol].to_numpy() - p["x_mean"].to_numpy()) ** 2 +
            (g[zcol].to_numpy() - p["z_mean"].to_numpy()) ** 2
        )
        vals.append(np.mean(d))

    if len(vals) == 0:
        return np.nan

    return float(np.mean(vals))


def normalizar_curva(df, xcol="x_mean", zcol="z_mean"):
    g = df.sort_values("phase").reset_index(drop=True).copy()

    # centrar X para comparar forma horizontal
    g["x_centered"] = g[xcol] - g[xcol].mean()

    # normalizar Z tomando el mínimo como referencia de suelo
    z_min = g[zcol].min()
    g["z_norm_suelo"] = g[zcol] - z_min

    return g


def distancia_curvas(df1, df2, xcol="x_centered", zcol="z_norm_suelo"):
    a = df1.sort_values("phase").reset_index(drop=True)
    b = df2.sort_values("phase").reset_index(drop=True)

    if len(a) != len(b):
        raise ValueError("Las curvas no tienen la misma cantidad de puntos")

    d = np.sqrt(
        (a[xcol].to_numpy() - b[xcol].to_numpy()) ** 2 +
        (a[zcol].to_numpy() - b[zcol].to_numpy()) ** 2
    )
    return float(np.mean(d))


def metricas_curva(df, xcol="x_mean", zcol="z_mean"):
    g = df.sort_values("phase").reset_index(drop=True)
    x = g[xcol].to_numpy()
    z = g[zcol].to_numpy()
    phase = g["phase"].to_numpy()

    return {
        "dx_norm": float(np.max(x) - np.min(x)),
        "dz_norm": float(np.max(z) - np.min(z)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "fase_xmax": float(phase[np.argmax(x)]),
        "fase_xmin": float(phase[np.argmin(x)]),
        "fase_zmax": float(phase[np.argmax(z)]),
        "fase_zmin": float(phase[np.argmin(z)]),
    }


# =========================================================
# CARGA
# =========================================================
df_ciclos = pd.read_csv(ARCH_CICLOS)
df_prom_suj = pd.read_csv(ARCH_PROM_SUJ)

# solo lado derecho
df_ciclos = df_ciclos[df_ciclos["lado"] == "r"].copy()
df_prom_suj = df_prom_suj[df_prom_suj["lado"] == "r"].copy()

if df_ciclos.empty or df_prom_suj.empty:
    raise ValueError("No hay datos del lado derecho para procesar.")

# =========================================================
# RESUMEN BASE POR SUJETO
# =========================================================
resumen = (
    df_ciclos.groupby("subject_id")
    .agg(
        n_ciclos=("cycle_uid", "nunique"),
        n_trials=("trial_num", "nunique"),
        longitud_funcional_mediana=("longitud_funcional", "median"),
    )
    .reset_index()
)

# =========================================================
# FILTRO DE ROBUSTEZ
# =========================================================
resumen_f = resumen[
    (resumen["n_ciclos"] >= MIN_CICLOS) &
    (resumen["n_trials"] >= MIN_TRIALS)
].copy()

if resumen_f.empty:
    raise ValueError(
        "No quedaron sujetos luego del filtro de robustez. "
        "Probá bajar MIN_CICLOS o MIN_TRIALS."
    )

ids_validos = resumen_f["subject_id"].tolist()

# =========================================================
# MÉTRICAS Y DISPERSIÓN POR SUJETO
# =========================================================
filas = []

for subj in ids_validos:
    gc = df_ciclos[df_ciclos["subject_id"] == subj].copy()
    gp = df_prom_suj[df_prom_suj["subject_id"] == subj].copy().sort_values("phase").reset_index(drop=True)

    disp = dispersion_media(gc, gp, xcol="x_norm", zcol="z_norm")
    met = metricas_curva(gp, xcol="x_mean", zcol="z_mean")

    fila = {
        "subject_id": subj,
        "dispersion_media": disp,
        **met
    }
    filas.append(fila)

df_rank = pd.DataFrame(filas)

# merge con resumen filtrado
df_rank = resumen_f.merge(df_rank, on="subject_id", how="left")

# score simple: la consistencia se decide por dispersión
df_rank["score_consistencia"] = df_rank["dispersion_media"]

df_rank = df_rank.sort_values(
    ["score_consistencia", "n_ciclos"],
    ascending=[True, False]
).reset_index(drop=True)

df_rank.to_csv(RUTA_SALIDA / "ranking_sujetos_robustos.csv", index=False, encoding="utf-8-sig")

# =========================================================
# TOP SUJETOS ROBUSTOS
# =========================================================
top_subjects = df_rank.head(TOP_K)["subject_id"].tolist()

# exportar curvas promedio top
df_top_curvas = df_prom_suj[df_prom_suj["subject_id"].isin(top_subjects)].copy()
df_top_curvas.to_csv(RUTA_SALIDA / "curvas_promedio_top_sujetos.csv", index=False, encoding="utf-8-sig")

# =========================================================
# CURVAS CENTRADAS PARA COMPARACIÓN DE FORMA
# =========================================================
curvas_centradas = []
for subj in top_subjects:
    g = df_prom_suj[df_prom_suj["subject_id"] == subj].copy()
    g = normalizar_curva(g, xcol="x_mean", zcol="z_mean")
    g["subject_id"] = subj
    curvas_centradas.append(g)

df_curvas_centradas = pd.concat(curvas_centradas, ignore_index=True)
df_curvas_centradas.to_csv(RUTA_SALIDA / "curvas_top_sujetos_centradas.csv", index=False, encoding="utf-8-sig")

# =========================================================
# MATRIZ DE DISTANCIAS ENTRE TOP SUJETOS
# =========================================================
filas_dist = []

curvas_dict = {}
for subj in top_subjects:
    curvas_dict[subj] = normalizar_curva(
        df_prom_suj[df_prom_suj["subject_id"] == subj].copy(),
        xcol="x_mean",
        zcol="z_mean"
    )

for si in top_subjects:
    for sj in top_subjects:
        d = distancia_curvas(
            curvas_dict[si],
            curvas_dict[sj],
            xcol="x_centered",
            zcol="z_norm_suelo"
        )
        filas_dist.append({
            "subject_i": si,
            "subject_j": sj,
            "distancia_forma": d
        })

df_dist = pd.DataFrame(filas_dist)
df_dist.to_csv(RUTA_SALIDA / "matriz_distancias_top_sujetos.csv", index=False, encoding="utf-8-sig")

# =========================================================
# MEDOID DENTRO DE LOS TOP SUJETOS
# =========================================================
dist_media = (
    df_dist[df_dist["subject_i"] != df_dist["subject_j"]]
    .groupby("subject_i")["distancia_forma"]
    .mean()
    .reset_index()
    .rename(columns={"subject_i": "subject_id", "distancia_forma": "dist_media_a_top"})
    .sort_values("dist_media_a_top")
    .reset_index(drop=True)
)

subject_medoid = int(dist_media.iloc[0]["subject_id"])

df_rank = df_rank.merge(dist_media, on="subject_id", how="left")
df_rank["es_medoid_top"] = df_rank["subject_id"] == subject_medoid
df_rank.to_csv(RUTA_SALIDA / "ranking_sujetos_robustos.csv", index=False, encoding="utf-8-sig")

# exportar curva medoid
curva_medoid = df_prom_suj[df_prom_suj["subject_id"] == subject_medoid].copy()
curva_medoid.to_csv(RUTA_SALIDA / f"curva_medoid_sujeto_{subject_medoid}.csv", index=False, encoding="utf-8-sig")

# promedio de top sujetos centrados (solo forma)
promedio_top_centrado = (
    df_curvas_centradas.groupby("phase")[["x_centered", "z_norm_suelo"]]
    .mean()
    .reset_index()
)

promedio_top_centrado.to_csv(RUTA_SALIDA / "promedio_top_sujetos_centrado.csv", index=False, encoding="utf-8-sig")

# =========================================================
# REPORTE
# =========================================================
print("\n=== RANKING DE SUJETOS ROBUSTOS ===")
print(df_rank.head(10)[[
    "subject_id",
    "n_ciclos",
    "n_trials",
    "longitud_funcional_mediana",
    "dispersion_media",
    "dx_norm",
    "dz_norm",
    "score_consistencia",
    "dist_media_a_top"
]].to_string(index=False))

print(f"\nSujetos top considerados: {top_subjects}")
print(f"Sujeto medoid dentro de los top: {subject_medoid}")

print(f"\nArchivos guardados en:\n{RUTA_SALIDA}")

# =========================================================
# GRÁFICOS
# =========================================================
if GENERAR_PLOTS:
    # 1. Curvas promedio originales de top sujetos
    plt.figure(figsize=(8, 6))
    for subj in top_subjects:
        g = df_prom_suj[df_prom_suj["subject_id"] == subj].sort_values("phase")
        lw = 2.8 if subj == subject_medoid else 1.8
        alpha = 1.0 if subj == subject_medoid else 0.85
        label = f"sujeto {subj}" + (" (medoid)" if subj == subject_medoid else "")
        plt.plot(g["x_mean"], g["z_mean"], linewidth=lw, alpha=alpha, label=label)

    plt.xlabel("X normalizada")
    plt.ylabel("Z normalizada")
    plt.title("Mejores promedios por sujeto - lado derecho")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2. Curvas centradas de top sujetos
    plt.figure(figsize=(8, 6))
    for subj in top_subjects:
        g = curvas_dict[subj].sort_values("phase")
        lw = 2.8 if subj == subject_medoid else 1.6
        alpha = 1.0 if subj == subject_medoid else 0.75
        label = f"sujeto {subj}" + (" (medoid)" if subj == subject_medoid else "")
        plt.plot(g["x_centered"], g["z_norm_suelo"], linewidth=lw, alpha=alpha, label=label)

    plt.xlabel("X centrada")
    plt.ylabel("Z normalizada al mínimo")
    plt.title("Top sujetos centrados para comparar forma")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3. Medoid vs promedio centrado
    plt.figure(figsize=(8, 6))
    gmed = curvas_dict[subject_medoid].sort_values("phase")
    plt.plot(gmed["x_centered"], gmed["z_norm_suelo"], linewidth=2.8, label=f"medoid {subject_medoid}")
    plt.plot(
        promedio_top_centrado["x_centered"],
        promedio_top_centrado["z_norm_suelo"],
        linewidth=2.8,
        label="promedio top centrado"
    )

    plt.xlabel("X centrada")
    plt.ylabel("Z normalizada al mínimo")
    plt.title("Medoid vs promedio de top sujetos (forma)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 4. X por fase
    plt.figure(figsize=(9, 4))
    for subj in top_subjects:
        g = df_prom_suj[df_prom_suj["subject_id"] == subj].sort_values("phase")
        lw = 2.8 if subj == subject_medoid else 1.5
        alpha = 1.0 if subj == subject_medoid else 0.75
        plt.plot(g["phase"], g["x_mean"], linewidth=lw, alpha=alpha, label=f"sujeto {subj}")
    plt.xlabel("Fase")
    plt.ylabel("X normalizada")
    plt.title("X por fase - top sujetos robustos")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 5. Z por fase
    plt.figure(figsize=(9, 4))
    for subj in top_subjects:
        g = df_prom_suj[df_prom_suj["subject_id"] == subj].sort_values("phase")
        lw = 2.8 if subj == subject_medoid else 1.5
        alpha = 1.0 if subj == subject_medoid else 0.75
        plt.plot(g["phase"], g["z_mean"], linewidth=lw, alpha=alpha, label=f"sujeto {subj}")
    plt.xlabel("Fase")
    plt.ylabel("Z normalizada")
    plt.title("Z por fase - top sujetos robustos")
    plt.grid(True)
    plt.legend()
    plt.show()