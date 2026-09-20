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
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_jansen_completo"
)
SALIDA.mkdir(parents=True, exist_ok=True)

NPTS = 160

# =========================================================
# HOLY NUMBERS THEO JANSEN
# =========================================================

HOLY = {
    "x": 38.0,
    "y": 7.8,
    "b": 41.5,
    "c": 39.3,
    "d": 40.1,
    "e": 55.8,
    "f": 39.4,
    "g": 36.7,
    "h": 65.7,
    "i": 49.0,
    "j": 50.0,
    "k": 61.9,
    "m": 15.0,
}

LINK_NAMES = ["x", "y", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "m"]


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


def normalizar_para_comparar(x, z):
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


def safe_acos(valor):
    if not np.isfinite(valor):
        return None
    if valor < -1.000001 or valor > 1.000001:
        return None
    return np.arccos(np.clip(valor, -1.0, 1.0))


def construir_links(params):
    """
    params:
    scale_global,
    multipliers para x,y,b,c,d,e,f,g,h,i,j,k,m,
    shift_fase
    """
    scale_global = params[0]
    mults = params[1:1 + len(LINK_NAMES)]

    links = {}
    for name, mult in zip(LINK_NAMES, mults):
        links[name] = HOLY[name] * scale_global * mult

    shift = params[-1]
    return links, shift


def trayectoria_jansen_completo(params, theta):
    links, shift_fase = construir_links(params)

    x = links["x"]
    y = links["y"]
    b = links["b"]
    c = links["c"]
    d = links["d"]
    e = links["e"]
    f = links["f"]
    g = links["g"]
    h = links["h"]
    i = links["i"]
    j = links["j"]
    k = links["k"]
    m = links["m"]

    px = []
    pz = []

    # Ángulo de entrada
    for th in theta:
        # Punto de crank
        X_m = x + m * np.cos(th)
        Y_m = y + m * np.sin(th)

        R = np.sqrt(X_m**2 + Y_m**2)
        if R < 1e-9:
            return None, None, False

        alpha = np.arctan2(Y_m, X_m)

        # Upper four-bar: n-m-b-j
        acos_j = safe_acos((j**2 - b**2 + R**2) / (2 * j * R))
        acos_b = safe_acos((b**2 - j**2 + R**2) / (2 * b * R))
        if acos_j is None or acos_b is None:
            return None, None, False

        theta_j = acos_j + alpha
        theta_b = acos_b + alpha

        # Lower four-bar: n-m-c-k
        acos_k = safe_acos((k**2 - c**2 + R**2) / (2 * k * R))
        acos_c = safe_acos((c**2 - k**2 + R**2) / (2 * c * R))
        if acos_k is None or acos_c is None:
            return None, None, False

        theta_k = acos_k + alpha
        theta_c = acos_c + alpha

        # Triangle b-d-e
        acos_d = safe_acos((b**2 + d**2 - e**2) / (2 * b * d))
        if acos_d is None:
            return None, None, False

        theta_d = acos_d + theta_b

        # Linkage d-c-f-g
        X3 = d * np.cos(theta_d) + c * np.cos(theta_c)
        Y3 = d * np.sin(theta_d) + c * np.sin(theta_c)

        R3 = np.sqrt(X3**2 + Y3**2)
        if R3 < 1e-9:
            return None, None, False

        alpha3 = np.arctan2(Y3, X3)

        acos_f = safe_acos((f**2 - g**2 + R3**2) / (2 * f * R3))
        acos_g = safe_acos((g**2 - f**2 + R3**2) / (2 * g * R3))
        if acos_f is None or acos_g is None:
            return None, None, False

        theta_f = acos_f + alpha3
        theta_g = acos_g + alpha3

        # Triangle g-h-i
        acos_i = safe_acos((g**2 + i**2 - h**2) / (2 * g * i))
        if acos_i is None:
            return None, None, False

        theta_i = theta_g - acos_i

        # Punto P / toe
        p_x = c * np.cos(theta_c) + i * np.cos(theta_i)
        p_y = c * np.sin(theta_c) + i * np.sin(theta_i)

        px.append(p_x)
        pz.append(p_y)

    px = np.array(px)
    pz = np.array(pz)

    # Pasar a sistema comparable: X centrado, suelo en Z=0
    px, pz = normalizar_para_comparar(px, pz)

    # Shift de fase
    px, pz = aplicar_shift_fase(px, pz, shift_fase)

    return px, pz, True


def error_mecanismo(params, theta, x_obj, z_obj):
    x_mec, z_mec, ok = trayectoria_jansen_completo(params, theta)

    if not ok:
        return 1e6

    if np.any(~np.isfinite(x_mec)) or np.any(~np.isfinite(z_mec)):
        return 1e6

    d = np.sqrt((x_obj - x_mec)**2 + (z_obj - z_mec)**2)
    rms = np.sqrt(np.mean(d**2))

    dx = np.max(x_mec) - np.min(x_mec)
    dz = np.max(z_mec) - np.min(z_mec)

    dx_obj = np.max(x_obj) - np.min(x_obj)
    dz_obj = np.max(z_obj) - np.min(z_obj)

    penal = 0.0

    # Ajustar amplitudes globales
    penal += 1.0 * abs(dx - dx_obj)
    penal += 6.0 * abs(dz - dz_obj)

    # Evitar trayectoria demasiado alta
    if dz > 6.0:
        penal += 15.0 * (dz - 6.0)

    # Evitar trayectoria demasiado plana
    if dz < 1.0:
        penal += 15.0 * (1.0 - dz)

    # Evitar links absurdamente distintos entre sí
    links, _ = construir_links(params)
    valores = np.array([links[n] for n in LINK_NAMES])

    if np.max(valores) > 45:
        penal += 5.0 * (np.max(valores) - 45)

    if np.min(valores) < 1.0:
        penal += 10.0 * (1.0 - np.min(valores))

    # Penalizar deformación excesiva respecto de holy numbers
    mults = np.array(params[1:1 + len(LINK_NAMES)])
    penal += 0.2 * np.mean((mults - 1.0)**2)

    return rms + penal


def metricas_error(x_obj, z_obj, x_mec, z_mec):
    d = np.sqrt((x_obj - x_mec)**2 + (z_obj - z_mec)**2)

    return {
        "error_medio_cm": float(np.mean(d)),
        "error_rms_cm": float(np.sqrt(np.mean(d**2))),
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

# scale_global + 13 multipliers + shift_fase
bounds = []

bounds.append((0.25, 0.85))  # escala global

for name in LINK_NAMES:
    if name == "y":
        bounds.append((0.5, 1.5))
    elif name == "m":
        bounds.append((0.5, 1.5))
    else:
        bounds.append((0.65, 1.35))

bounds.append((0.0, 1.0))  # shift_fase

print("Número de variables:", len(bounds))

resultado = differential_evolution(
    error_mecanismo,
    bounds=bounds,
    args=(theta, x_obj, z_obj),
    maxiter=900,
    popsize=18,
    tol=1e-7,
    polish=True,
    seed=42,
    workers=1
)

params_opt = resultado.x

x_mec, z_mec, ok = trayectoria_jansen_completo(params_opt, theta)

if not ok:
    raise RuntimeError("La solución optimizada no generó trayectoria válida.")

metricas, error = metricas_error(x_obj, z_obj, x_mec, z_mec)

links_opt, shift_fase = construir_links(params_opt)

# =========================================================
# REPORTE
# =========================================================

print("\n=== OPTIMIZACIÓN JANSEN COMPLETO ===")
print(f"scale_global: {params_opt[0]:.4f}")

for name in LINK_NAMES:
    print(f"{name}_cm: {links_opt[name]:.4f}")

print(f"shift_fase: {shift_fase:.4f}")

print("\n=== MÉTRICAS ===")
for k_name, v in metricas.items():
    print(f"{k_name}: {v:.4f}")

print(f"\nFunción objetivo final: {resultado.fun:.6f}")

# =========================================================
# EXPORTAR
# =========================================================

df_comp = pd.DataFrame({
    "phase": phase,
    "x_obj_cm": x_obj,
    "z_obj_cm": z_obj,
    "x_jansen_completo_cm": x_mec,
    "z_jansen_completo_cm": z_mec,
    "error_cm": error,
})

df_comp.to_csv(
    SALIDA / "comparacion_jansen_completo.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([{
    "scale_global": params_opt[0],
    **{f"{name}_cm": links_opt[name] for name in LINK_NAMES},
    "shift_fase": shift_fase,
    **metricas,
    "funcion_objetivo": resultado.fun,
}]).to_csv(
    SALIDA / "parametros_jansen_completo.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# GRÁFICOS
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(x_obj, z_obj, linewidth=3, label="Trayectoria objetivo")
plt.plot(x_mec, z_mec, linewidth=3, label="Jansen completo optimizado")
plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Jansen completo optimizado vs trayectoria objetivo")
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
plt.plot(phase, x_mec, label="X Jansen completo")
plt.plot(phase, z_obj, label="Z objetivo")
plt.plot(phase, z_mec, label="Z Jansen completo")
plt.xlabel("Fase")
plt.ylabel("Posición [cm]")
plt.title("Coordenadas por fase")
plt.grid(True)
plt.legend()
plt.show()

print(f"\nArchivos guardados en:\n{SALIDA}")