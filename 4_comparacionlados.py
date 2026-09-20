from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN
# =========================================================
RUTA_BASE = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_caminata_global\curva_cerrada_suave"
)

ARCH_DER = RUTA_BASE / "promedio_global_caminata_lado_r_cerrado.csv"
ARCH_IZQ = RUTA_BASE / "promedio_global_caminata_lado_l_cerrado.csv"

RUTA_SALIDA = RUTA_BASE / "comparacion_lados"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

GENERAR_PLOTS = True


# =========================================================
# FUNCIONES
# =========================================================
def cargar_curva(path_csv):
    df = pd.read_csv(path_csv)
    df = df.sort_values("phase").reset_index(drop=True)
    return df


def calcular_error_cierre(df, xcol="x_cerrado", zcol="z_cerrado"):
    x0 = df.iloc[0][xcol]
    z0 = df.iloc[0][zcol]
    x1 = df.iloc[-1][xcol]
    z1 = df.iloc[-1][zcol]

    dx = x1 - x0
    dz = z1 - z0
    err = np.sqrt(dx**2 + dz**2)

    return {
        "x_inicio": float(x0),
        "z_inicio": float(z0),
        "x_final": float(x1),
        "z_final": float(z1),
        "dx_cierre": float(dx),
        "dz_cierre": float(dz),
        "error_cierre": float(err),
    }


def longitud_trayectoria(df, xcol="x_cerrado", zcol="z_cerrado"):
    x = df[xcol].to_numpy()
    z = df[zcol].to_numpy()
    ds = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
    return float(np.sum(ds))


def metricas_trayectoria(df, lado, xcol="x_cerrado", zcol="z_cerrado", phasecol="phase"):
    x = df[xcol].to_numpy()
    z = df[zcol].to_numpy()
    phase = df[phasecol].to_numpy()

    err_cierre = calcular_error_cierre(df, xcol=xcol, zcol=zcol)
    long_tray = longitud_trayectoria(df, xcol=xcol, zcol=zcol)

    out = {
        "lado": lado,
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
        "longitud_trayectoria": long_tray,
        "x_inicio": err_cierre["x_inicio"],
        "z_inicio": err_cierre["z_inicio"],
        "x_final": err_cierre["x_final"],
        "z_final": err_cierre["z_final"],
        "dx_cierre": err_cierre["dx_cierre"],
        "dz_cierre": err_cierre["dz_cierre"],
        "error_cierre": err_cierre["error_cierre"],
    }
    return out


def comparar_curvas(df_r, df_l, reflejar_x_izq=False):
    """
    Compara las curvas punto a punto en la misma fase.
    Si reflejar_x_izq=True, compara x_r con -x_l.
    """
    # Asegurar mismo largo y misma fase
    dr = df_r.sort_values("phase").reset_index(drop=True).copy()
    dl = df_l.sort_values("phase").reset_index(drop=True).copy()

    if len(dr) != len(dl):
        raise ValueError("Las curvas no tienen la misma cantidad de puntos.")

    xr = dr["x_cerrado"].to_numpy()
    zr = dr["z_cerrado"].to_numpy()

    xl = dl["x_cerrado"].to_numpy()
    zl = dl["z_cerrado"].to_numpy()

    if reflejar_x_izq:
        xl_comp = -xl
    else:
        xl_comp = xl

    dx = xr - xl_comp
    dz = zr - zl
    dist = np.sqrt(dx**2 + dz**2)

    out = {
        "reflejar_x_izq": reflejar_x_izq,
        "error_medio_punto_a_punto": float(np.mean(dist)),
        "error_rms_punto_a_punto": float(np.sqrt(np.mean(dist**2))),
        "error_max_punto_a_punto": float(np.max(dist)),
        "dif_media_x": float(np.mean(dx)),
        "dif_media_z": float(np.mean(dz)),
    }

    return out, dist, dx, dz


def comparar_metricas(metric_r, metric_l):
    out = {
        "dif_dx_norm": float(metric_r["dx_norm"] - metric_l["dx_norm"]),
        "dif_dz_norm": float(metric_r["dz_norm"] - metric_l["dz_norm"]),
        "dif_x_min": float(metric_r["x_min"] - metric_l["x_min"]),
        "dif_x_max": float(metric_r["x_max"] - metric_l["x_max"]),
        "dif_z_min": float(metric_r["z_min"] - metric_l["z_min"]),
        "dif_z_max": float(metric_r["z_max"] - metric_l["z_max"]),
        "dif_fase_xmax": float(metric_r["fase_xmax"] - metric_l["fase_xmax"]),
        "dif_fase_xmin": float(metric_r["fase_xmin"] - metric_l["fase_xmin"]),
        "dif_fase_zmax": float(metric_r["fase_zmax"] - metric_l["fase_zmax"]),
        "dif_fase_zmin": float(metric_r["fase_zmin"] - metric_l["fase_zmin"]),
        "dif_longitud_trayectoria": float(metric_r["longitud_trayectoria"] - metric_l["longitud_trayectoria"]),
    }
    return out


# =========================================================
# CARGA
# =========================================================
if not ARCH_DER.exists():
    raise FileNotFoundError(f"No existe el archivo del lado derecho:\n{ARCH_DER}")

if not ARCH_IZQ.exists():
    raise FileNotFoundError(
        f"No existe el archivo del lado izquierdo:\n{ARCH_IZQ}\n\n"
        "Primero generá el cierre suave del lado izquierdo."
    )

df_r = cargar_curva(ARCH_DER)
df_l = cargar_curva(ARCH_IZQ)

# =========================================================
# MÉTRICAS POR LADO
# =========================================================
metric_r = metricas_trayectoria(df_r, lado="r", xcol="x_cerrado", zcol="z_cerrado")
metric_l = metricas_trayectoria(df_l, lado="l", xcol="x_cerrado", zcol="z_cerrado")

df_metricas = pd.DataFrame([metric_r, metric_l])
df_metricas.to_csv(RUTA_SALIDA / "metricas_por_lado.csv", index=False, encoding="utf-8-sig")

# =========================================================
# COMPARACIÓN ENTRE LADOS
# =========================================================
comp_sin_reflejo, dist_sin, dx_sin, dz_sin = comparar_curvas(df_r, df_l, reflejar_x_izq=False)
comp_con_reflejo, dist_ref, dx_ref, dz_ref = comparar_curvas(df_r, df_l, reflejar_x_izq=True)

comp_metricas = comparar_metricas(metric_r, metric_l)

df_comparacion = pd.DataFrame([{
    **{f"sin_reflejo_{k}": v for k, v in comp_sin_reflejo.items() if k != "reflejar_x_izq"},
    **{f"con_reflejo_{k}": v for k, v in comp_con_reflejo.items() if k != "reflejar_x_izq"},
    **comp_metricas
}])

df_comparacion.to_csv(RUTA_SALIDA / "comparacion_entre_lados.csv", index=False, encoding="utf-8-sig")

# =========================================================
# EXPORTAR COMPARACIÓN PUNTO A PUNTO
# =========================================================
df_p2p = pd.DataFrame({
    "phase": df_r["phase"],
    "x_r": df_r["x_cerrado"],
    "z_r": df_r["z_cerrado"],
    "x_l": df_l["x_cerrado"],
    "z_l": df_l["z_cerrado"],
    "x_l_reflejado": -df_l["x_cerrado"],
    "dist_sin_reflejo": dist_sin,
    "dist_con_reflejo": dist_ref,
    "dx_sin_reflejo": dx_sin,
    "dz_sin_reflejo": dz_sin,
    "dx_con_reflejo": dx_ref,
    "dz_con_reflejo": dz_ref,
})
df_p2p.to_csv(RUTA_SALIDA / "comparacion_punto_a_punto_lados.csv", index=False, encoding="utf-8-sig")

# =========================================================
# REPORTE EN CONSOLA
# =========================================================
print("=== MÉTRICAS LADO DERECHO ===")
for k, v in metric_r.items():
    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")

print("\n=== MÉTRICAS LADO IZQUIERDO ===")
for k, v in metric_l.items():
    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")

print("\n=== COMPARACIÓN ENTRE LADOS (sin reflejar izquierdo) ===")
for k, v in comp_sin_reflejo.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== COMPARACIÓN ENTRE LADOS (reflejando X del izquierdo) ===")
for k, v in comp_con_reflejo.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== DIFERENCIAS DE MÉTRICAS ENTRE LADOS (r - l) ===")
for k, v in comp_metricas.items():
    print(f"{k}: {v:.6f}")

print(f"\nArchivos guardados en:\n{RUTA_SALIDA}")

# =========================================================
# GRÁFICOS
# =========================================================
if GENERAR_PLOTS:
    # 1. Curvas superpuestas sin reflejo
    plt.figure(figsize=(8, 6))
    plt.plot(df_r["x_cerrado"], df_r["z_cerrado"], label="Derecho")
    plt.plot(df_l["x_cerrado"], df_l["z_cerrado"], label="Izquierdo")
    plt.xlabel("X normalizada")
    plt.ylabel("Z normalizada")
    plt.title("Comparación entre lados - sin reflejo")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2. Curvas superpuestas con reflejo del izquierdo
    plt.figure(figsize=(8, 6))
    plt.plot(df_r["x_cerrado"], df_r["z_cerrado"], label="Derecho")
    plt.plot(-df_l["x_cerrado"], df_l["z_cerrado"], label="Izquierdo reflejado")
    plt.xlabel("X normalizada")
    plt.ylabel("Z normalizada")
    plt.title("Comparación entre lados - izquierdo reflejado en X")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 3. X y Z vs fase
    plt.figure(figsize=(9, 4))
    plt.plot(df_r["phase"], df_r["x_cerrado"], label="X derecho")
    plt.plot(df_r["phase"], df_r["z_cerrado"], label="Z derecho")
    plt.plot(df_l["phase"], df_l["x_cerrado"], "--", label="X izquierdo")
    plt.plot(df_l["phase"], df_l["z_cerrado"], "--", label="Z izquierdo")
    plt.xlabel("Fase")
    plt.ylabel("Posición normalizada")
    plt.title("Posición vs fase - ambos lados")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 4. Distancia punto a punto
    plt.figure(figsize=(9, 4))
    plt.plot(df_r["phase"], dist_sin, label="Distancia sin reflejo")
    plt.plot(df_r["phase"], dist_ref, label="Distancia con reflejo")
    plt.xlabel("Fase")
    plt.ylabel("Distancia entre lados")
    plt.title("Error punto a punto entre lados")
    plt.grid(True)
    plt.legend()
    plt.show()