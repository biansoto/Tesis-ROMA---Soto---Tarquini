from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN
# =========================================================

SALIDA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_fourier_final"
)
SALIDA.mkdir(parents=True, exist_ok=True)

NPTS = 400

# Longitud funcional usada para escalar
L_FUNCIONAL_CM = 53.5

# Coeficientes Fourier obtenidos
# x(phi) = c0 + a1 cos(2pi phi) + b1 sin(2pi phi)
#          + a2 cos(4pi phi) + b2 sin(4pi phi)

COEF_X = {
    "c0": 0.0523439241,
    "cos1": -0.0766145947,
    "sin1": -0.2639054638,
    "cos2": 0.0012291096,
    "sin2": -0.0402470305,
}

COEF_Z = {
    "c0": -0.1182807980,
    "cos1": -0.0237654068,
    "sin1": -0.0003613511,
    "cos2": -0.0091445417,
    "sin2": -0.0000102314,
}


# =========================================================
# FUNCIONES
# =========================================================

def evaluar_fourier(phi, coef):
    return (
        coef["c0"]
        + coef["cos1"] * np.cos(2 * np.pi * phi)
        + coef["sin1"] * np.sin(2 * np.pi * phi)
        + coef["cos2"] * np.cos(4 * np.pi * phi)
        + coef["sin2"] * np.sin(4 * np.pi * phi)
    )


# =========================================================
# GENERAR CURVA NORMALIZADA
# =========================================================

phi = np.linspace(0, 1, NPTS)

x_norm = evaluar_fourier(phi, COEF_X)
z_norm = evaluar_fourier(phi, COEF_Z)

# cierre exacto
x_norm[-1] = x_norm[0]
z_norm[-1] = z_norm[0]

# =========================================================
# ESCALAR A CM Y PONER SUELO EN Z = 0
# =========================================================

x_cm = x_norm * L_FUNCIONAL_CM
z_cm = (z_norm - np.min(z_norm)) * L_FUNCIONAL_CM

# centrar X para diseño
x_cm_centrada = x_cm - np.mean(x_cm)

# =========================================================
# MÉTRICAS
# =========================================================

metricas = {
    "L_funcional_cm": L_FUNCIONAL_CM,
    "x_min_cm": float(np.min(x_cm_centrada)),
    "x_max_cm": float(np.max(x_cm_centrada)),
    "z_min_cm": float(np.min(z_cm)),
    "z_max_cm": float(np.max(z_cm)),
    "avance_horizontal_cm": float(np.max(x_cm_centrada) - np.min(x_cm_centrada)),
    "altura_maxima_cm": float(np.max(z_cm)),
}

# =========================================================
# GUARDAR
# =========================================================

df = pd.DataFrame({
    "phase": phi,
    "x_norm": x_norm,
    "z_norm": z_norm,
    "x_cm": x_cm,
    "x_cm_centrada": x_cm_centrada,
    "z_cm_suelo": z_cm,
})

df.to_csv(
    SALIDA / "trayectoria_fourier_final_cm.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([metricas]).to_csv(
    SALIDA / "metricas_trayectoria_fourier_final.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(SALIDA / "ecuaciones_fourier_final.txt", "w", encoding="utf-8") as f:
    f.write("Modelo Fourier final de trayectoria objetivo\n")
    f.write("phi pertenece a [0, 1]\n\n")

    f.write("x_norm(phi) = ")
    f.write(
        f"{COEF_X['c0']:.10f} "
        f"+ ({COEF_X['cos1']:.10f}) cos(2*pi*phi) "
        f"+ ({COEF_X['sin1']:.10f}) sin(2*pi*phi) "
        f"+ ({COEF_X['cos2']:.10f}) cos(4*pi*phi) "
        f"+ ({COEF_X['sin2']:.10f}) sin(4*pi*phi)\n\n"
    )

    f.write("z_norm(phi) = ")
    f.write(
        f"{COEF_Z['c0']:.10f} "
        f"+ ({COEF_Z['cos1']:.10f}) cos(2*pi*phi) "
        f"+ ({COEF_Z['sin1']:.10f}) sin(2*pi*phi) "
        f"+ ({COEF_Z['cos2']:.10f}) cos(4*pi*phi) "
        f"+ ({COEF_Z['sin2']:.10f}) sin(4*pi*phi)\n\n"
    )

    f.write("Escalado:\n")
    f.write("x_cm = x_norm * 53.5\n")
    f.write("z_cm_suelo = (z_norm - min(z_norm)) * 53.5\n\n")

    f.write("Metricas:\n")
    for k, v in metricas.items():
        f.write(f"{k}: {v:.6f}\n")

# =========================================================
# REPORTE
# =========================================================

print("\n=== TRAYECTORIA FOURIER FINAL ===")
for k, v in metricas.items():
    print(f"{k}: {v:.3f}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICOS
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(x_cm_centrada, z_cm, linewidth=3)
plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Trayectoria objetivo final - modelo Fourier")
plt.axis("equal")
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phi, x_cm_centrada, label="X [cm]")
plt.plot(phi, z_cm, label="Z [cm]")
plt.xlabel("Fase")
plt.ylabel("Posición [cm]")
plt.title("Trayectoria objetivo Fourier por fase")
plt.grid(True)
plt.legend()
plt.show()