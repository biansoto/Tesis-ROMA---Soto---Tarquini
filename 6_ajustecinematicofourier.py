from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN
# =========================================================
RUTA_CURVA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_caminata_global\curva_cerrada_suave\fourier_lado_derecho\trayectoria_fourier_suave.csv"
)

RUTA_SALIDA = RUTA_CURVA.parent / "analisis_cinematico"
RUTA_SALIDA.mkdir(exist_ok=True)

# Umbrales para detección aproximada de apoyo
# fracción de la amplitud vertical desde el mínimo
FRACCION_Z_APOYO = 0.20

# percentil para considerar velocidad vertical "baja"
PERCENTIL_VZ_BAJA = 40

GENERAR_PLOTS = True


# =========================================================
# FUNCIONES
# =========================================================
def derivada_central(y, x):
    return np.gradient(y, x)


def calcular_curvatura(x, z, phase):
    dx = derivada_central(x, phase)
    dz = derivada_central(z, phase)

    ddx = derivada_central(dx, phase)
    ddz = derivada_central(dz, phase)

    numerador = np.abs(dx * ddz - dz * ddx)
    denominador = (dx**2 + dz**2) ** 1.5

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denominador > 1e-12, numerador / denominador, np.nan)

    return dx, dz, ddx, ddz, kappa


def detectar_apoyo(df):
    """
    Detecta una región aproximada de apoyo usando:
    1) Z cerca del mínimo
    2) velocidad vertical pequeña
    """
    z = df["z_fourier"].to_numpy()
    dz = df["dz_dphase"].to_numpy()

    z_min = np.min(z)
    z_max = np.max(z)
    amp_z = z_max - z_min

    umbral_z = z_min + FRACCION_Z_APOYO * amp_z
    umbral_vz = np.percentile(np.abs(dz), PERCENTIL_VZ_BAJA)

    mask_bajo = z <= umbral_z
    mask_vz_baja = np.abs(dz) <= umbral_vz

    mask_apoyo = mask_bajo & mask_vz_baja

    return mask_apoyo, {
        "z_min": float(z_min),
        "z_max": float(z_max),
        "amp_z": float(amp_z),
        "umbral_z_apoyo": float(umbral_z),
        "umbral_abs_dz_apoyo": float(umbral_vz),
    }


def extraer_segmentos_contiguos(mask, phase):
    """
    Devuelve segmentos contiguos donde mask=True
    """
    segmentos = []
    idx = np.where(mask)[0]

    if len(idx) == 0:
        return segmentos

    inicio = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segmentos.append((inicio, prev))
            inicio = i
            prev = i

    segmentos.append((inicio, prev))

    out = []
    for a, b in segmentos:
        out.append({
            "idx_ini": int(a),
            "idx_fin": int(b),
            "phase_ini": float(phase[a]),
            "phase_fin": float(phase[b]),
            "duracion_fase": float(phase[b] - phase[a]),
        })

    return out


def elegir_segmento_principal(segmentos):
    if not segmentos:
        return None
    return max(segmentos, key=lambda s: s["duracion_fase"])


# =========================================================
# CARGA
# =========================================================
df = pd.read_csv(RUTA_CURVA).sort_values("phase").reset_index(drop=True)

phase = df["phase"].to_numpy()
x = df["x_fourier"].to_numpy()
z = df["z_fourier"].to_numpy()

# =========================================================
# DERIVADAS, VELOCIDAD, CURVATURA
# =========================================================
dx, dz, ddx, ddz, kappa = calcular_curvatura(x, z, phase)

vel_mod = np.sqrt(dx**2 + dz**2)

df["dx_dphase"] = dx
df["dz_dphase"] = dz
df["ddx_dphase2"] = ddx
df["ddz_dphase2"] = ddz
df["velocidad_mod"] = vel_mod
df["curvatura"] = kappa

# =========================================================
# DETECCIÓN APROXIMADA DE APOYO
# =========================================================
mask_apoyo, info_apoyo = detectar_apoyo(df)
df["mask_apoyo"] = mask_apoyo

segmentos = extraer_segmentos_contiguos(mask_apoyo, phase)
segmento_principal = elegir_segmento_principal(segmentos)

if segmento_principal is not None:
    phase_ini_apoyo = segmento_principal["phase_ini"]
    phase_fin_apoyo = segmento_principal["phase_fin"]
    duracion_apoyo = segmento_principal["duracion_fase"]
else:
    phase_ini_apoyo = np.nan
    phase_fin_apoyo = np.nan
    duracion_apoyo = np.nan

# =========================================================
# MÉTRICAS GLOBALES
# =========================================================
metricas = {
    "n_puntos": len(df),
    "x_min": float(np.min(x)),
    "x_max": float(np.max(x)),
    "z_min": float(np.min(z)),
    "z_max": float(np.max(z)),
    "vel_media": float(np.nanmean(vel_mod)),
    "vel_max": float(np.nanmax(vel_mod)),
    "curvatura_media": float(np.nanmean(kappa)),
    "curvatura_max": float(np.nanmax(kappa)),
    "phase_ini_apoyo": phase_ini_apoyo,
    "phase_fin_apoyo": phase_fin_apoyo,
    "duracion_apoyo_fase": duracion_apoyo,
    **info_apoyo
}

df_metricas = pd.DataFrame([metricas])
df_metricas.to_csv(RUTA_SALIDA / "metricas_cinematica_apoyo.csv", index=False, encoding="utf-8-sig")

df.to_csv(RUTA_SALIDA / "trayectoria_fourier_con_cinematica.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(segmentos).to_csv(RUTA_SALIDA / "segmentos_apoyo_detectados.csv", index=False, encoding="utf-8-sig")

# =========================================================
# REPORTE
# =========================================================
print("\n=== MÉTRICAS CINEMÁTICAS ===")
for k, v in metricas.items():
    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")

if segmento_principal is not None:
    print("\n=== APOYO PRINCIPAL DETECTADO ===")
    print(f"phase_ini_apoyo: {phase_ini_apoyo:.6f}")
    print(f"phase_fin_apoyo: {phase_fin_apoyo:.6f}")
    print(f"duracion_apoyo_fase: {duracion_apoyo:.6f}")
else:
    print("\nNo se detectó un segmento principal de apoyo.")

print(f"\nArchivos guardados en:\n{RUTA_SALIDA}")

# =========================================================
# GRÁFICOS
# =========================================================
if GENERAR_PLOTS:

    # 1. Trayectoria con apoyo resaltado
    plt.figure(figsize=(7, 6))
    plt.plot(x, z, label="Trayectoria Fourier", linewidth=2)

    if np.any(mask_apoyo):
        plt.scatter(x[mask_apoyo], z[mask_apoyo], s=18, label="Apoyo detectado")

    plt.xlabel("X normalizada")
    plt.ylabel("Z normalizada")
    plt.title("Trayectoria del pie con fase de apoyo estimada")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2. X y Z vs fase
    plt.figure(figsize=(9, 4))
    plt.plot(phase, x, label="X")
    plt.plot(phase, z, label="Z")

    if segmento_principal is not None:
        plt.axvspan(phase_ini_apoyo, phase_fin_apoyo, alpha=0.2, label="Apoyo principal")

    plt.xlabel("Fase")
    plt.ylabel("Posición")
    plt.title("Posición del pie vs fase")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3. Velocidad
    plt.figure(figsize=(9, 4))
    plt.plot(phase, vel_mod, label="|v|")
    plt.plot(phase, np.abs(dz), label="|dz/dphase|")

    if segmento_principal is not None:
        plt.axvspan(phase_ini_apoyo, phase_fin_apoyo, alpha=0.2, label="Apoyo principal")

    plt.xlabel("Fase")
    plt.ylabel("Magnitud")
    plt.title("Velocidad respecto de la fase")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 4. Curvatura
    plt.figure(figsize=(9, 4))
    plt.plot(phase, kappa, label="Curvatura")

    if segmento_principal is not None:
        plt.axvspan(phase_ini_apoyo, phase_fin_apoyo, alpha=0.2, label="Apoyo principal")

    plt.xlabel("Fase")
    plt.ylabel("Curvatura")
    plt.title("Curvatura de la trayectoria")
    plt.grid(True)
    plt.legend()
    plt.show()