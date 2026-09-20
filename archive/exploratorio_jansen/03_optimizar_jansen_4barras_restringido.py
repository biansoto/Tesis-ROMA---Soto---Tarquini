from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# =========================================================
# RUTAS
# =========================================================

RUTA_OBJETIVO = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_procesamiento_final\seleccion_mejor_trial\modelo_matematico_trayectoria\trayectoria_escalada_cm\trayectoria_objetivo_cm_suelo.csv"
)

SALIDA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_jansen_4b_restringido"
)
SALIDA.mkdir(parents=True, exist_ok=True)

NPTS = 200
ALTURA_CADERA_CM = 43.0


# =========================================================
# FUNCIONES
# =========================================================

def cargar_objetivo():
    df = pd.read_csv(RUTA_OBJETIVO).sort_values("phase").reset_index(drop=True)

    phase_old = df["phase"].to_numpy()
    x_old = df["x_cm"].to_numpy()
    z_old = df["z_cm"].to_numpy()

    phase = np.linspace(0, 1, NPTS)
    x = np.interp(phase, phase_old, x_old)
    z = np.interp(phase, phase_old, z_old)

    return phase, x, z


def interseccion_circulos(B, D, r1, r2, rama=1):
    B = np.asarray(B, dtype=float)
    D = np.asarray(D, dtype=float)

    dx, dz = D - B
    d = np.sqrt(dx**2 + dz**2)

    if d < 1e-9:
        return None

    if d > r1 + r2 or d < abs(r1 - r2):
        return None

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h2 = r1**2 - a**2

    if h2 < -1e-8:
        return None

    h = np.sqrt(max(h2, 0))

    P2 = B + a * (D - B) / d

    rx = -(D[1] - B[1]) / d
    rz = (D[0] - B[0]) / d

    C1 = P2 + h * np.array([rx, rz])
    C2 = P2 - h * np.array([rx, rz])

    return C1 if rama >= 0 else C2


def normalizar_para_comparar(x, z):
    """
    Centra X y pone suelo en Z = 0.
    """
    x = x - np.mean(x)
    z = z - np.min(z)
    return x, z


def aplicar_shift_fase(x, z, shift):
    phase = np.linspace(0, 1, len(x))
    phase_shift = (phase + shift) % 1.0
    orden = np.argsort(phase_shift)

    x_shift = np.interp(phase, phase_shift[orden], x[orden])
    z_shift = np.interp(phase, phase_shift[orden], z[orden])

    return x_shift, z_shift


def trayectoria_mecanismo_4b(params, theta):
    """
    Mecanismo 4 barras con punto acoplador:
    A fijo
    D fijo
    AB = manivela
    BC = acoplador
    CD = balancín
    P = pie unido rígidamente al acoplador BC

    params:
    r_ab, l_bc, l_cd, d_ad,
    px, pz,
    x0, z0,
    rama,
    shift_fase
    """

    (
        r_ab, l_bc, l_cd, d_ad,
        px, pz,
        x0, z0,
        rama_cont,
        shift_fase
    ) = params

    rama = 1 if rama_cont >= 0 else -1

    A = np.array([0.0, ALTURA_CADERA_CM])
    D = np.array([d_ad, ALTURA_CADERA_CM])

    xs = []
    zs = []

    for th in theta:
        B = A + r_ab * np.array([np.cos(th), np.sin(th)])

        C = interseccion_circulos(B, D, l_bc, l_cd, rama=rama)

        if C is None:
            return None, None, False

        v = C - B
        L = np.linalg.norm(v)

        if L < 1e-9:
            return None, None, False

        e1 = v / L
        e2 = np.array([-e1[1], e1[0]])

        P = B + px * e1 + pz * e2

        xs.append(P[0] + x0)
        zs.append(P[1] + z0)

    return np.array(xs), np.array(zs), True


def error_mecanismo(params, theta, x_obj, z_obj):
    (
        r_ab, l_bc, l_cd, d_ad,
        px, pz,
        x0, z0,
        rama_cont,
        shift_fase
    ) = params

    x_mec, z_mec, ok = trayectoria_mecanismo_4b(params, theta)

    if not ok:
        return 1e6

    if np.any(~np.isfinite(x_mec)) or np.any(~np.isfinite(z_mec)):
        return 1e6

    x_mec, z_mec = normalizar_para_comparar(x_mec, z_mec)
    x_mec, z_mec = aplicar_shift_fase(x_mec, z_mec, shift_fase)

    d = np.sqrt((x_obj - x_mec) ** 2 + (z_obj - z_mec) ** 2)
    rms = np.sqrt(np.mean(d ** 2))

    dx = np.max(x_mec) - np.min(x_mec)
    dz = np.max(z_mec) - np.min(z_mec)

    dx_obj = np.max(x_obj) - np.min(x_obj)
    dz_obj = np.max(z_obj) - np.min(z_obj)

    penal = 0.0

    # Amplitudes similares a la curva objetivo
    penal += 1.0 * abs(dx - dx_obj)
    penal += 5.0 * abs(dz - dz_obj)

    # Evitar barras/offsets absurdos
    if abs(z0) > 25:
        penal += 20.0 * (abs(z0) - 25)

    if abs(x0) > 25:
        penal += 10.0 * (abs(x0) - 25)

    if abs(px) > 50:
        penal += 5.0 * (abs(px) - 50)

    if abs(pz) > 45:
        penal += 5.0 * (abs(pz) - 45)

    # Evitar trayectorias demasiado altas o demasiado planas
    if dz > 5:
        penal += 20.0 * (dz - 5)

    if dz < 1.5:
        penal += 20.0 * (1.5 - dz)

    # Fase de apoyo aproximada: zona cercana al suelo
    umbral_apoyo = np.min(z_obj) + 0.25 * dz_obj
    mask_apoyo = z_obj <= umbral_apoyo

    if np.any(mask_apoyo):
        error_z_apoyo = np.mean(np.abs(z_mec[mask_apoyo] - z_obj[mask_apoyo]))
        penal += 5.0 * error_z_apoyo

        dz_dphase = np.gradient(z_mec, theta)
        pendiente_apoyo = np.mean(np.abs(dz_dphase[mask_apoyo]))
        penal += 0.4 * pendiente_apoyo

    return rms + penal


def metricas_error(x_obj, z_obj, x_mec, z_mec):
    d = np.sqrt((x_obj - x_mec) ** 2 + (z_obj - z_mec) ** 2)

    return {
        "error_medio_cm": float(np.mean(d)),
        "error_rms_cm": float(np.sqrt(np.mean(d ** 2))),
        "error_max_cm": float(np.max(d)),
        "dx_obj_cm": float(np.max(x_obj) - np.min(x_obj)),
        "dz_obj_cm": float(np.max(z_obj) - np.min(z_obj)),
        "dx_mec_cm": float(np.max(x_mec) - np.min(x_mec)),
        "dz_mec_cm": float(np.max(z_mec) - np.min(z_mec)),
    }, d


# =========================================================
# CARGA OBJETIVO
# =========================================================

phase, x_obj, z_obj = cargar_objetivo()
x_obj, z_obj = normalizar_para_comparar(x_obj, z_obj)

theta = 2 * np.pi * phase


# =========================================================
# OPTIMIZACIÓN
# =========================================================

bounds = [
    (1.5, 10.0),      # r_ab manivela
    (8.0, 35.0),      # l_bc
    (8.0, 35.0),      # l_cd
    (5.0, 35.0),      # d_ad

    (-50.0, 65.0),    # px punto pie sobre acoplador
    (-45.0, 10.0),    # pz offset perpendicular

    (-25.0, 25.0),    # x0
    (-25.0, 10.0),    # z0

    (-1.0, 1.0),      # rama
    (0.0, 1.0),       # shift_fase
]

resultado = differential_evolution(
    error_mecanismo,
    bounds=bounds,
    args=(theta, x_obj, z_obj),
    maxiter=1200,
    popsize=22,
    tol=1e-8,
    polish=True,
    seed=42,
    workers=1
)

params_opt = resultado.x
shift_fase = params_opt[-1]

x_mec, z_mec, ok = trayectoria_mecanismo_4b(params_opt, theta)

if not ok:
    raise RuntimeError("La solución optimizada no generó trayectoria válida.")

x_mec, z_mec = normalizar_para_comparar(x_mec, z_mec)
x_mec, z_mec = aplicar_shift_fase(x_mec, z_mec, shift_fase)

metricas, error = metricas_error(x_obj, z_obj, x_mec, z_mec)


# =========================================================
# REPORTE
# =========================================================

nombres = [
    "r_ab_cm",
    "l_bc_cm",
    "l_cd_cm",
    "d_ad_cm",
    "px_pie_cm",
    "pz_pie_cm",
    "x0_cm",
    "z0_cm",
    "rama",
    "shift_fase",
]

print("\n=== OPTIMIZACIÓN JANSEN REDUCIDO 4 BARRAS RESTRINGIDO ===")
for nombre, valor in zip(nombres, params_opt):
    print(f"{nombre}: {valor:.4f}")

print("\n=== MÉTRICAS ===")
for k, v in metricas.items():
    print(f"{k}: {v:.4f}")

print(f"\nFunción objetivo final: {resultado.fun:.6f}")


# =========================================================
# EXPORTAR
# =========================================================

df_comp = pd.DataFrame({
    "phase": phase,
    "x_obj_cm": x_obj,
    "z_obj_cm": z_obj,
    "x_mecanismo_cm": x_mec,
    "z_mecanismo_cm": z_mec,
    "error_cm": error,
})

df_comp.to_csv(
    SALIDA / "comparacion_jansen_4_barras_restringido.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([{
    **{nombre: valor for nombre, valor in zip(nombres, params_opt)},
    **metricas,
    "funcion_objetivo": resultado.fun
}]).to_csv(
    SALIDA / "parametros_jansen_4_barras_restringido.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# GRÁFICOS
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(x_obj, z_obj, linewidth=3, label="Trayectoria objetivo")
plt.plot(x_mec, z_mec, linewidth=3, label="4 barras restringido")
plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Jansen reducido 4 barras restringido vs trayectoria objetivo")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase, error)
plt.xlabel("Fase")
plt.ylabel("Error [cm]")
plt.title("Error punto a punto")
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase, x_obj, label="X objetivo")
plt.plot(phase, x_mec, label="X mecanismo")
plt.plot(phase, z_obj, label="Z objetivo")
plt.plot(phase, z_mec, label="Z mecanismo")
plt.xlabel("Fase")
plt.ylabel("Posición [cm]")
plt.title("Coordenadas por fase")
plt.grid(True)
plt.legend()
plt.show()

print(f"\nArchivos guardados en:\n{SALIDA}")