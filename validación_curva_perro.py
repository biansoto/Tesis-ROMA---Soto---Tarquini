from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# RUTAS
# =========================================================

# Archivo con TODOS los ciclos reinterpolados
# Debe contener una fila por punto del ciclo
# y una columna que identifique el ciclo
RUTA_CICLOS = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\dataset\ciclos_perro_reinterpolados.csv"
)

# Curva Fourier final del perro
RUTA_FOURIER = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\dataset\3DDogsLab2024\Data\Optical\Sync_Align_v2023_11_16b\salidas_caminata_global\curva_cerrada_suave\fourier_lado_derecho\trayectoria_fourier_suave.csv"
)

SALIDA = RUTA_FOURIER.parent / "validacion_curva_perro"
SALIDA.mkdir(exist_ok=True)

MOSTRAR_GRAFICOS = True


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def detectar_columnas(df):
    """
    Detecta nombres de columnas típicos.
    Ajustá esta función si tu archivo usa otros nombres.
    """
    cols = {c.lower(): c for c in df.columns}

    # phase
    col_phase = None
    for k in ["phase", "fase"]:
        if k in cols:
            col_phase = cols[k]
            break

    # X
    col_x = None
    for k in ["x", "x_rel", "x_mean", "x_fourier"]:
        if k in cols:
            col_x = cols[k]
            break

    # Z
    col_z = None
    for k in ["z", "z_rel", "y_rel", "z_mean", "z_fourier", "y"]:
        if k in cols:
            col_z = cols[k]
            break

    # ciclo id
    col_cycle = None
    for k in ["ciclo_id", "cycle_id", "trial", "pasada", "cycle", "ciclo"]:
        if k in cols:
            col_cycle = cols[k]
            break

    return col_phase, col_x, col_z, col_cycle


def normalizar_isotropico(x, z):
    """
    Centra y escala isotrópicamente preservando forma geométrica.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    x = x - np.mean(x)
    z = z - np.mean(z)

    escala = max(x.max() - x.min(), z.max() - z.min())
    if escala == 0:
        raise ValueError("Escala nula al normalizar.")

    x = x / escala
    z = z / escala
    return x, z


def error_punto_a_punto(x1, z1, x2, z2):
    return np.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


# =========================================================
# CARGA DE DATOS
# =========================================================

df_ciclos = pd.read_csv(RUTA_CICLOS)
df_fourier = pd.read_csv(RUTA_FOURIER)

col_phase, col_x, col_z, col_cycle = detectar_columnas(df_ciclos)

if col_phase is None or col_x is None or col_z is None or col_cycle is None:
    raise ValueError(
        "No pude detectar automáticamente las columnas del archivo de ciclos.\n"
        f"Columnas encontradas: {list(df_ciclos.columns)}\n"
        "Necesitás tener algo equivalente a: phase, x/z y cycle_id."
    )

print("Columnas detectadas en ciclos:")
print("phase  ->", col_phase)
print("x      ->", col_x)
print("z      ->", col_z)
print("ciclo  ->", col_cycle)

# Curva Fourier
if not {"phase", "x_fourier", "z_fourier"}.issubset(df_fourier.columns):
    raise ValueError(
        "El archivo Fourier debe contener: phase, x_fourier, z_fourier"
    )

phase_ref = df_fourier["phase"].to_numpy()
x_fourier = df_fourier["x_fourier"].to_numpy()
z_fourier = df_fourier["z_fourier"].to_numpy()

# Normalizo Fourier
x_fourier_n, z_fourier_n = normalizar_isotropico(x_fourier, z_fourier)

# =========================================================
# PREPARAR CICLOS
# =========================================================

# Me quedo solo con columnas útiles
df_ciclos = df_ciclos[[col_phase, col_x, col_z, col_cycle]].copy()
df_ciclos.columns = ["phase", "x", "z", "cycle_id"]

for c in ["phase", "x", "z"]:
    df_ciclos[c] = pd.to_numeric(df_ciclos[c], errors="coerce")

df_ciclos = df_ciclos.dropna(subset=["phase", "x", "z", "cycle_id"]).reset_index(drop=True)

# Si la fase no está exactamente en los mismos puntos, reinterpolamos cada ciclo
ciclos_ids = df_ciclos["cycle_id"].unique()

ciclos_interp = []
errores_ciclos = []

for cid in ciclos_ids:
    sub = df_ciclos[df_ciclos["cycle_id"] == cid].sort_values("phase").copy()

    if len(sub) < 5:
        continue

    phase_in = sub["phase"].to_numpy()
    x_in = sub["x"].to_numpy()
    z_in = sub["z"].to_numpy()

    # reinterpolación al phase_ref de la Fourier
    x_i = np.interp(phase_ref, phase_in, x_in)
    z_i = np.interp(phase_ref, phase_in, z_in)

    # normalización isotrópica por ciclo
    x_i, z_i = normalizar_isotropico(x_i, z_i)

    tmp = pd.DataFrame({
        "cycle_id": cid,
        "phase": phase_ref,
        "x": x_i,
        "z": z_i
    })
    ciclos_interp.append(tmp)

if not ciclos_interp:
    raise ValueError("No pude construir ciclos interpolados válidos.")

df_interp = pd.concat(ciclos_interp, ignore_index=True)

# =========================================================
# CURVA PROMEDIO Y DISPERSIÓN
# =========================================================

g = df_interp.groupby("phase")

media = g[["x", "z"]].mean().rename(columns={"x": "x_mean", "z": "z_mean"})
std = g[["x", "z"]].std().rename(columns={"x": "x_std", "z": "z_std"})
n_by_phase = g.size().rename("n")

df_stats = media.join(std).join(n_by_phase).reset_index()

x_mean = df_stats["x_mean"].to_numpy()
z_mean = df_stats["z_mean"].to_numpy()
x_std = df_stats["x_std"].fillna(0).to_numpy()
z_std = df_stats["z_std"].fillna(0).to_numpy()

# =========================================================
# ERROR FOURIER VS MEDIA
# =========================================================

error_fourier_media = error_punto_a_punto(x_fourier_n, z_fourier_n, x_mean, z_mean)

metricas_fourier = {
    "error_medio_fourier_vs_media": float(np.mean(error_fourier_media)),
    "error_rms_fourier_vs_media": float(np.sqrt(np.mean(error_fourier_media**2))),
    "error_max_fourier_vs_media": float(np.max(error_fourier_media)),
}

# =========================================================
# ERROR DE CADA CICLO VS MEDIA
# =========================================================

resumen_ciclos = []

for cid in df_interp["cycle_id"].unique():
    sub = df_interp[df_interp["cycle_id"] == cid].sort_values("phase")

    xc = sub["x"].to_numpy()
    zc = sub["z"].to_numpy()

    err = error_punto_a_punto(xc, zc, x_mean, z_mean)

    resumen_ciclos.append({
        "cycle_id": cid,
        "error_medio_vs_media": float(np.mean(err)),
        "error_rms_vs_media": float(np.sqrt(np.mean(err**2))),
        "error_max_vs_media": float(np.max(err)),
    })

df_resumen_ciclos = pd.DataFrame(resumen_ciclos)

metricas_globales = {
    "n_ciclos": int(df_resumen_ciclos.shape[0]),
    "error_medio_ciclos_vs_media": float(df_resumen_ciclos["error_medio_vs_media"].mean()),
    "error_rms_ciclos_vs_media": float(np.sqrt(np.mean(df_resumen_ciclos["error_rms_vs_media"]**2))),
    "error_max_ciclos_vs_media": float(df_resumen_ciclos["error_max_vs_media"].max()),
    "x_std_medio": float(np.nanmean(x_std)),
    "z_std_medio": float(np.nanmean(z_std)),
    "x_std_max": float(np.nanmax(x_std)),
    "z_std_max": float(np.nanmax(z_std)),
    **metricas_fourier
}

df_metricas = pd.DataFrame([metricas_globales])

# =========================================================
# BOOTSTRAP DE LA CURVA PROMEDIO
# =========================================================

N_BOOT = 200
rng = np.random.default_rng(1234)

boot_x = []
boot_z = []

ids_validos = df_interp["cycle_id"].unique()

for _ in range(N_BOOT):
    sample_ids = rng.choice(ids_validos, size=len(ids_validos), replace=True)

    partes = []
    for sid in sample_ids:
        partes.append(df_interp[df_interp["cycle_id"] == sid])

    boot_df = pd.concat(partes, ignore_index=True)

    boot_g = boot_df.groupby("phase")[["x", "z"]].mean().reset_index()

    boot_x.append(boot_g["x"].to_numpy())
    boot_z.append(boot_g["z"].to_numpy())

boot_x = np.array(boot_x)
boot_z = np.array(boot_z)

x_boot_mean = boot_x.mean(axis=0)
z_boot_mean = boot_z.mean(axis=0)

x_boot_std = boot_x.std(axis=0)
z_boot_std = boot_z.std(axis=0)

# =========================================================
# EXPORTAR
# =========================================================

df_stats.to_csv(SALIDA / "curva_media_y_std.csv", index=False, encoding="utf-8-sig")
df_resumen_ciclos.to_csv(SALIDA / "errores_por_ciclo_vs_media.csv", index=False, encoding="utf-8-sig")
df_metricas.to_csv(SALIDA / "metricas_validacion_curva_perro.csv", index=False, encoding="utf-8-sig")

df_boot = pd.DataFrame({
    "phase": phase_ref,
    "x_boot_mean": x_boot_mean,
    "z_boot_mean": z_boot_mean,
    "x_boot_std": x_boot_std,
    "z_boot_std": z_boot_std
})
df_boot.to_csv(SALIDA / "bootstrap_curva_media.csv", index=False, encoding="utf-8-sig")

# =========================================================
# REPORTE
# =========================================================

print("\n=== VALIDACIÓN CURVA DEL PERRO ===")
for k, v in metricas_globales.items():
    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICOS
# =========================================================

if MOSTRAR_GRAFICOS:

    # 1. Todos los ciclos + media + Fourier
    plt.figure(figsize=(7, 6))

    for cid in df_interp["cycle_id"].unique():
        sub = df_interp[df_interp["cycle_id"] == cid]
        plt.plot(sub["x"], sub["z"], alpha=0.15)

    plt.plot(x_mean, z_mean, linewidth=3, label="Media ciclos")
    plt.plot(x_fourier_n, z_fourier_n, linewidth=3, label="Fourier")

    plt.xlabel("X normalizada")
    plt.ylabel("Z normalizada")
    plt.title("Ciclos individuales vs media vs Fourier")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2. Comparación por fase
    plt.figure(figsize=(9, 4))
    plt.plot(phase_ref, x_mean, label="X media")
    plt.plot(phase_ref, z_mean, label="Z media")
    plt.plot(phase_ref, x_fourier_n, "--", label="X Fourier")
    plt.plot(phase_ref, z_fourier_n, "--", label="Z Fourier")

    plt.xlabel("Fase")
    plt.ylabel("Posición normalizada")
    plt.title("Media vs Fourier")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3. Error Fourier vs media
    plt.figure(figsize=(9, 4))
    plt.plot(phase_ref, error_fourier_media)

    plt.xlabel("Fase")
    plt.ylabel("Error")
    plt.title("Error Fourier vs media")
    plt.grid(True)
    plt.show()

    # 4. Banda de variabilidad por fase
    plt.figure(figsize=(9, 4))
    plt.plot(phase_ref, x_mean, label="X media")
    plt.fill_between(
        phase_ref,
        x_mean - x_std,
        x_mean + x_std,
        alpha=0.25,
        label="±1 std X"
    )

    plt.plot(phase_ref, z_mean, label="Z media")
    plt.fill_between(
        phase_ref,
        z_mean - z_std,
        z_mean + z_std,
        alpha=0.25,
        label="±1 std Z"
    )

    plt.xlabel("Fase")
    plt.ylabel("Posición normalizada")
    plt.title("Variabilidad entre ciclos")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 5. Bootstrap: estabilidad de la media
    plt.figure(figsize=(9, 4))
    plt.plot(phase_ref, x_boot_mean, label="X bootstrap media")
    plt.fill_between(
        phase_ref,
        x_boot_mean - x_boot_std,
        x_boot_mean + x_boot_std,
        alpha=0.25,
        label="±1 std bootstrap X"
    )

    plt.plot(phase_ref, z_boot_mean, label="Z bootstrap media")
    plt.fill_between(
        phase_ref,
        z_boot_mean - z_boot_std,
        z_boot_mean + z_boot_std,
        alpha=0.25,
        label="±1 std bootstrap Z"
    )

    plt.xlabel("Fase")
    plt.ylabel("Posición normalizada")
    plt.title("Estabilidad bootstrap de la curva media")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 6. Histograma de error RMS de ciclos
    plt.figure(figsize=(8, 4))
    plt.hist(df_resumen_ciclos["error_rms_vs_media"], bins=20)

    plt.xlabel("Error RMS ciclo vs media")
    plt.ylabel("Frecuencia")
    plt.title("Distribución del error de los ciclos")
    plt.grid(True)
    plt.show()