from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


RUTA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_procesamiento_final"
)

ARCH_CICLOS = RUTA / "ciclos_caminata_normalizados.csv"

SALIDA = RUTA / "seleccion_mejor_trial"
SALIDA.mkdir(parents=True, exist_ok=True)

LADO = "r"

L_MIN = 0.50
L_MAX = 0.65

MIN_CICLOS_TRIAL = 3
TOP_K = 10

VENTANA_SAVGOL = 11
ORDEN_SAVGOL = 3


def promedio_trial(df):
    return (
        df.groupby("phase")[["x_norm", "z_norm"]]
        .mean()
        .reset_index()
        .sort_values("phase")
        .reset_index(drop=True)
    )


def dispersion_trial(df_ciclos, df_prom):
    vals = []

    for cycle_uid, g in df_ciclos.groupby("cycle_uid"):
        g = g.sort_values("phase").reset_index(drop=True)
        p = df_prom.sort_values("phase").reset_index(drop=True)

        if len(g) != len(p):
            continue

        d = np.sqrt(
            (g["x_norm"].to_numpy() - p["x_norm"].to_numpy()) ** 2 +
            (g["z_norm"].to_numpy() - p["z_norm"].to_numpy()) ** 2
        )
        vals.append(np.mean(d))

    return np.nan if len(vals) == 0 else float(np.mean(vals))


def metricas_curva(df_prom):
    x = df_prom["x_norm"].to_numpy()
    z = df_prom["z_norm"].to_numpy()
    phase = df_prom["phase"].to_numpy()

    return {
        "dx_norm": float(x.max() - x.min()),
        "dz_norm": float(z.max() - z.min()),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "fase_xmax": float(phase[np.argmax(x)]),
        "fase_xmin": float(phase[np.argmin(x)]),
        "fase_zmax": float(phase[np.argmax(z)]),
        "fase_zmin": float(phase[np.argmin(z)]),
        "error_cierre": float(np.sqrt((x[-1] - x[0]) ** 2 + (z[-1] - z[0]) ** 2)),
    }


def suavizar_curva(df):
    df = df.copy()

    df["x_norm"] = savgol_filter(df["x_norm"], VENTANA_SAVGOL, ORDEN_SAVGOL)
    df["z_norm"] = savgol_filter(df["z_norm"], VENTANA_SAVGOL, ORDEN_SAVGOL)

    return df


def refasear_por_contacto(df):
    df = df.copy()

    idx_contacto = df["z_norm"].idxmin()

    df = pd.concat([
        df.loc[idx_contacto:],
        df.loc[:idx_contacto - 1]
    ]).reset_index(drop=True)

    df["phase"] = np.linspace(0, 1, len(df))

    return df


def forzar_cierre(df):
    df = df.copy()

    df.loc[len(df) - 1, "x_norm"] = df.loc[0, "x_norm"]
    df.loc[len(df) - 1, "z_norm"] = df.loc[0, "z_norm"]

    return df

def limpiar_rulo_cierre(df, radio=0.035):
    df = df.copy().reset_index(drop=True)

    x0 = df.loc[0, "x_norm"]
    z0 = df.loc[0, "z_norm"]

    dist = np.sqrt((df["x_norm"] - x0)**2 + (df["z_norm"] - z0)**2)

    # conservar primer punto, eliminar puntos intermedios que vuelven demasiado cerca del inicio
    mask = np.ones(len(df), dtype=bool)
    mask[1:-1] = dist.iloc[1:-1] > radio

    df = df[mask].reset_index(drop=True)

    # reinterpolar a 100 puntos
    t_old = np.linspace(0, 1, len(df))
    t_new = np.linspace(0, 1, 100)

    out = pd.DataFrame({
        "phase": t_new,
        "x_norm": np.interp(t_new, t_old, df["x_norm"]),
        "z_norm": np.interp(t_new, t_old, df["z_norm"]),
    })

    out.loc[len(out)-1, "x_norm"] = out.loc[0, "x_norm"]
    out.loc[len(out)-1, "z_norm"] = out.loc[0, "z_norm"]

    return out

def matriz_fourier(phase, orden):
    theta = 2 * np.pi * phase
    cols = [np.ones(len(phase))]
    for k in range(1, orden + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    return np.column_stack(cols)


def suavizado_fourier_cerrado(df, orden=3, npts=200):
    df = df.sort_values("phase").reset_index(drop=True).copy()

    phase = df["phase"].to_numpy()
    x = df["x_norm"].to_numpy()
    z = df["z_norm"].to_numpy()

    A = matriz_fourier(phase, orden)

    coef_x, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    coef_z, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    phase_new = np.linspace(0, 1, npts)
    A_new = matriz_fourier(phase_new, orden)

    x_new = A_new @ coef_x
    z_new = A_new @ coef_z

    out = pd.DataFrame({
        "phase": phase_new,
        "x_norm": x_new,
        "z_norm": z_new
    })

    # cierre exacto
    out.loc[len(out)-1, "x_norm"] = out.loc[0, "x_norm"]
    out.loc[len(out)-1, "z_norm"] = out.loc[0, "z_norm"]

    return out


# =========================================================
# CARGA Y FILTRO
# =========================================================

df = pd.read_csv(ARCH_CICLOS)

df = df[df["lado"] == LADO].copy()

df = df[
    (df["longitud_funcional"] >= L_MIN) &
    (df["longitud_funcional"] <= L_MAX)
].copy()

if df.empty:
    raise ValueError("No quedaron ciclos luego del filtro de lado/tamaño.")


# =========================================================
# RANKING DE TRIALS
# =========================================================

filas = []
curvas_trial = {}

grupos = df.groupby(["subject_id", "trial_num", "direction", "archivo"])

for key, g in grupos:
    subject_id, trial_num, direction, archivo = key

    n_ciclos = g["cycle_uid"].nunique()
    if n_ciclos < MIN_CICLOS_TRIAL:
        continue

    prom = promedio_trial(g)
    disp = dispersion_trial(g, prom)
    met = metricas_curva(prom)

    L_med = float(g["longitud_funcional"].median())

    fila = {
        "subject_id": subject_id,
        "trial_num": trial_num,
        "direction": direction,
        "archivo": archivo,
        "n_ciclos": n_ciclos,
        "longitud_funcional_mediana": L_med,
        "dispersion_media": disp,
        **met
    }

    filas.append(fila)
    curvas_trial[key] = prom

ranking = pd.DataFrame(filas)

if ranking.empty:
    raise ValueError("No quedaron trials con suficientes ciclos. Bajá MIN_CICLOS_TRIAL.")

ranking["penalizacion_dz_bajo"] = np.where(ranking["dz_norm"] < 0.02, 0.05, 0.0)
ranking["bonus_ciclos"] = 1 / np.sqrt(ranking["n_ciclos"])

ranking["score"] = (
    ranking["dispersion_media"]
    + ranking["penalizacion_dz_bajo"]
    + 0.01 * ranking["bonus_ciclos"]
)

ranking = ranking.sort_values("score").reset_index(drop=True)
ranking.to_csv(SALIDA / "ranking_trials.csv", index=False, encoding="utf-8-sig")

mejor = ranking.iloc[0]

key_mejor = (
    mejor["subject_id"],
    mejor["trial_num"],
    mejor["direction"],
    mejor["archivo"]
)

# =========================================================
# CURVA OBJETIVO FINAL
# =========================================================

curva_mejor_original = curvas_trial[key_mejor].copy()

# 1) suavizar primero
curva_mejor = suavizar_curva(curva_mejor_original)

# 2) refasear después del suavizado
curva_mejor = refasear_por_contacto(curva_mejor)

# 3) forzar cierre exacto
# cierre preliminar
curva_mejor = forzar_cierre(curva_mejor)

# suavizado global cerrado por Fourier
curva_mejor = suavizado_fourier_cerrado(curva_mejor, orden=2, npts=200)

# refasear otra vez porque el suavizado puede mover levemente el mínimo
curva_mejor = refasear_por_contacto(curva_mejor)

# cierre final
curva_mejor = forzar_cierre(curva_mejor)

# 4) recalcular métricas finales ya sobre la curva procesada
metricas_finales = metricas_curva(curva_mejor)

print("Fase del contacto después del refaseo:",
      curva_mejor.loc[curva_mejor["z_norm"].idxmin(), "phase"])

curva_mejor.to_csv(
    SALIDA / "trayectoria_objetivo_mejor_trial.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([metricas_finales]).to_csv(
    SALIDA / "metricas_trayectoria_objetivo.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# REPORTE
# =========================================================

print("\n=== MEJOR TRIAL SELECCIONADO ===")
print(mejor.to_string())

print("\n=== MÉTRICAS FINALES DE LA CURVA OBJETIVO ===")
for k, v in metricas_finales.items():
    print(f"{k}: {v:.6f}")

print("\n=== TOP TRIALS ===")
print(
    ranking.head(TOP_K)[[
        "subject_id", "trial_num", "direction", "n_ciclos",
        "longitud_funcional_mediana", "dispersion_media",
        "dx_norm", "dz_norm", "score"
    ]].to_string(index=False)
)

print(f"\nArchivos guardados en:\n{SALIDA}")


# =========================================================
# GRÁFICOS
# =========================================================

top = ranking.head(TOP_K)

plt.figure(figsize=(8, 6))
for _, row in top.iterrows():
    key = (row["subject_id"], row["trial_num"], row["direction"], row["archivo"])
    c = curvas_trial[key]
    label = f"d{row['subject_id']}_t{row['trial_num']}_{row['direction']}"
    lw = 3 if row.name == 0 else 1.5
    alpha = 1.0 if row.name == 0 else 0.55
    plt.plot(c["x_norm"], c["z_norm"], linewidth=lw, alpha=alpha, label=label)

plt.xlabel("X normalizada")
plt.ylabel("Z normalizada")
plt.title("Top trials candidatos - trayectoria promedio")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(
    curva_mejor_original["x_norm"],
    curva_mejor_original["z_norm"],
    "--",
    linewidth=1.5,
    label="Original sin refasear"
)
plt.plot(
    curva_mejor["x_norm"],
    curva_mejor["z_norm"],
    linewidth=3,
    label="Final suavizada y refaseada"
)
plt.xlabel("X normalizada")
plt.ylabel("Z normalizada")
plt.title("Trayectoria objetivo seleccionada")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(9, 4))
plt.plot(curva_mejor["phase"], curva_mejor["x_norm"], label="X")
plt.plot(curva_mejor["phase"], curva_mejor["z_norm"], label="Z")
plt.xlabel("Fase")
plt.ylabel("Posición normalizada")
plt.title("Trayectoria objetivo final por fase")
plt.grid(True)
plt.legend()
plt.show()


# ciclos individuales del mejor trial
df_mejor = df[
    (df["subject_id"] == mejor["subject_id"]) &
    (df["trial_num"] == mejor["trial_num"]) &
    (df["direction"] == mejor["direction"]) &
    (df["archivo"] == mejor["archivo"])
].copy()

plt.figure(figsize=(8, 6))
for cycle_uid, g in df_mejor.groupby("cycle_uid"):
    g = g.sort_values("phase")
    plt.plot(g["x_norm"], g["z_norm"], color="gray", alpha=0.25)

plt.plot(
    curva_mejor["x_norm"],
    curva_mejor["z_norm"],
    color="red",
    linewidth=3,
    label="Curva objetivo final"
)

plt.xlabel("X normalizada")
plt.ylabel("Z normalizada")
plt.title("Ciclos individuales del mejor trial + curva objetivo final")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()