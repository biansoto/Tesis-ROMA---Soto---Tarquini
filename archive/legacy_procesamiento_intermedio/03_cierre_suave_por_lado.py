from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN
# =========================================================
RUTA_BASE = Path(r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset")

ARCH_PROM_GLOBAL = RUTA_BASE / "salidas_caminata_global\promedio_global_caminata.csv"

# lado a corregir: "l" o "r"
LADO = "r"

# carpeta de salida
RUTA_SALIDA = RUTA_BASE / "curva_cerrada_suave"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# FUNCIONES
# =========================================================
def calcular_error_cierre(df, xcol="x_mean", zcol="z_mean", phasecol="phase"):
    df = df.sort_values(phasecol).copy()

    x0 = df.iloc[0][xcol]
    z0 = df.iloc[0][zcol]
    x1 = df.iloc[-1][xcol]
    z1 = df.iloc[-1][zcol]

    err = np.sqrt((x1 - x0)**2 + (z1 - z0)**2)

    return {
        "x_inicio": x0,
        "z_inicio": z0,
        "x_final": x1,
        "z_final": z1,
        "dx_cierre": x1 - x0,
        "dz_cierre": z1 - z0,
        "error_cierre": err
    }


def cierre_suave_lineal(df, xcol="x_mean", zcol="z_mean", phasecol="phase"):
    """
    Corrige el cierre distribuyendo linealmente la diferencia
    entre el punto final y el inicial a lo largo de toda la fase.
    """
    df = df.sort_values(phasecol).copy()

    phase = df[phasecol].to_numpy()
    x = df[xcol].to_numpy()
    z = df[zcol].to_numpy()

    dx = x[-1] - x[0]
    dz = z[-1] - z[0]

    x_corr = x - phase * dx
    z_corr = z - phase * dz

    out = df.copy()
    out["x_cerrado"] = x_corr
    out["z_cerrado"] = z_corr

    return out


def metricas_trayectoria(df, xcol, zcol, phasecol="phase"):
    df = df.sort_values(phasecol).copy()

    x = df[xcol].to_numpy()
    z = df[zcol].to_numpy()
    phase = df[phasecol].to_numpy()

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


def exportar_metricas(path_out, lado, err_orig, err_corr, met_orig, met_corr):
    fila = {
        "lado": lado,

        "x_inicio_original": err_orig["x_inicio"],
        "z_inicio_original": err_orig["z_inicio"],
        "x_final_original": err_orig["x_final"],
        "z_final_original": err_orig["z_final"],
        "dx_cierre_original": err_orig["dx_cierre"],
        "dz_cierre_original": err_orig["dz_cierre"],
        "error_cierre_original": err_orig["error_cierre"],

        "x_inicio_cerrado": err_corr["x_inicio"],
        "z_inicio_cerrado": err_corr["z_inicio"],
        "x_final_cerrado": err_corr["x_final"],
        "z_final_cerrado": err_corr["z_final"],
        "dx_cierre_cerrado": err_corr["dx_cierre"],
        "dz_cierre_cerrado": err_corr["dz_cierre"],
        "error_cierre_cerrado": err_corr["error_cierre"],

        "dx_norm_original": met_orig["dx_norm"],
        "dz_norm_original": met_orig["dz_norm"],
        "x_min_original": met_orig["x_min"],
        "x_max_original": met_orig["x_max"],
        "z_min_original": met_orig["z_min"],
        "z_max_original": met_orig["z_max"],

        "dx_norm_cerrado": met_corr["dx_norm"],
        "dz_norm_cerrado": met_corr["dz_norm"],
        "x_min_cerrado": met_corr["x_min"],
        "x_max_cerrado": met_corr["x_max"],
        "z_min_cerrado": met_corr["z_min"],
        "z_max_cerrado": met_corr["z_max"],

        "fase_xmax_cerrado": met_corr["fase_xmax"],
        "fase_xmin_cerrado": met_corr["fase_xmin"],
        "fase_zmax_cerrado": met_corr["fase_zmax"],
        "fase_zmin_cerrado": met_corr["fase_zmin"],
    }

    df_out = pd.DataFrame([fila])
    df_out.to_csv(path_out, index=False, encoding="utf-8-sig")


# =========================================================
# CARGA
# =========================================================
df_global = pd.read_csv(ARCH_PROM_GLOBAL)

sub = df_global[df_global["lado"] == LADO].copy()

if sub.empty:
    raise ValueError(f"No hay datos para el lado '{LADO}' en {ARCH_PROM_GLOBAL}")

sub = sub.sort_values("phase").reset_index(drop=True)

# =========================================================
# ERROR ORIGINAL
# =========================================================
err_orig = calcular_error_cierre(sub, xcol="x_mean", zcol="z_mean", phasecol="phase")
met_orig = metricas_trayectoria(sub, xcol="x_mean", zcol="z_mean", phasecol="phase")

print("=== ERROR DE CIERRE ORIGINAL ===")
print(f"Inicio: ({err_orig['x_inicio']:.6f}, {err_orig['z_inicio']:.6f})")
print(f"Final : ({err_orig['x_final']:.6f}, {err_orig['z_final']:.6f})")
print(f"dx cierre: {err_orig['dx_cierre']:.6f}")
print(f"dz cierre: {err_orig['dz_cierre']:.6f}")
print(f"Error original: {err_orig['error_cierre']:.6f}")

# =========================================================
# CIERRE SUAVE
# =========================================================
sub_corr = cierre_suave_lineal(sub, xcol="x_mean", zcol="z_mean", phasecol="phase")

err_corr = calcular_error_cierre(sub_corr, xcol="x_cerrado", zcol="z_cerrado", phasecol="phase")
met_corr = metricas_trayectoria(sub_corr, xcol="x_cerrado", zcol="z_cerrado", phasecol="phase")

print("\n=== ERROR DE CIERRE DESPUÉS DEL AJUSTE ===")
print(f"Inicio: ({err_corr['x_inicio']:.6f}, {err_corr['z_inicio']:.6f})")
print(f"Final : ({err_corr['x_final']:.6f}, {err_corr['z_final']:.6f})")
print(f"dx cierre: {err_corr['dx_cierre']:.6f}")
print(f"dz cierre: {err_corr['dz_cierre']:.6f}")
print(f"Error final: {err_corr['error_cierre']:.6f}")

# =========================================================
# EXPORTAR CURVA CORREGIDA
# =========================================================
archivo_curva = RUTA_SALIDA / f"promedio_global_caminata_lado_{LADO}_cerrado.csv"
sub_corr.to_csv(archivo_curva, index=False, encoding="utf-8-sig")

archivo_metricas = RUTA_SALIDA / f"metricas_cierre_suave_lado_{LADO}.csv"
exportar_metricas(archivo_metricas, LADO, err_orig, err_corr, met_orig, met_corr)

print(f"\nCurva corregida guardada en:\n{archivo_curva}")
print(f"Métricas guardadas en:\n{archivo_metricas}")

# =========================================================
# GRÁFICOS
# =========================================================
plt.figure(figsize=(7, 6))
plt.plot(sub["x_mean"], sub["z_mean"], label="Original", linewidth=2)
plt.plot(sub_corr["x_cerrado"], sub_corr["z_cerrado"], label="Cierre suave", linewidth=2)
plt.scatter([sub.iloc[0]["x_mean"]], [sub.iloc[0]["z_mean"]], label="Inicio original")
plt.scatter([sub.iloc[-1]["x_mean"]], [sub.iloc[-1]["z_mean"]], label="Final original")
plt.xlabel("X normalizada")
plt.ylabel("Z normalizada")
plt.title(f"Curva original vs corregida - lado {LADO}")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(sub["phase"], sub["x_mean"], label="X original")
plt.plot(sub["phase"], sub_corr["x_cerrado"], label="X cerrado")
plt.plot(sub["phase"], sub["z_mean"], label="Z original")
plt.plot(sub["phase"], sub_corr["z_cerrado"], label="Z cerrado")
plt.xlabel("Fase")
plt.ylabel("Posición normalizada")
plt.title(f"Corrección por fase - lado {LADO}")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(7, 6))
plt.plot(sub_corr["x_cerrado"], sub_corr["z_cerrado"], label="Curva cerrada", linewidth=2)
plt.scatter([sub_corr.iloc[0]["x_cerrado"]], [sub_corr.iloc[0]["z_cerrado"]], label="Inicio cerrado")
plt.scatter([sub_corr.iloc[-1]["x_cerrado"]], [sub_corr.iloc[-1]["z_cerrado"]], label="Final cerrado")
plt.xlabel("X normalizada")
plt.ylabel("Z normalizada")
plt.title(f"Verificación del cierre - lado {LADO}")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()