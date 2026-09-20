from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# CONFIGURACIÓN
# ============================================

RUTA_CURVA = Path(
r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_caminata_global\curva_cerrada_suave\promedio_global_caminata_lado_r_cerrado.csv"
)

RUTA_SALIDA = RUTA_CURVA.parent / "fourier_lado_derecho"
RUTA_SALIDA.mkdir(exist_ok=True)

ORDEN_FOURIER = 4

# ============================================
# FUNCIONES
# ============================================

def construir_matriz_fourier(theta, orden):

    n = len(theta)

    columnas = [np.ones(n)]

    for k in range(1, orden + 1):

        columnas.append(np.cos(k * theta))
        columnas.append(np.sin(k * theta))

    return np.column_stack(columnas)


def ajustar_fourier(theta, y, orden):

    A = construir_matriz_fourier(theta, orden)

    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    y_fit = A @ coef

    return coef, y_fit


def evaluar_fourier(theta, coef, orden):

    A = construir_matriz_fourier(theta, orden)

    return A @ coef


# ============================================
# CARGAR CURVA
# ============================================

df = pd.read_csv(RUTA_CURVA)

phase = df["phase"].values
x = df["x_cerrado"].values
z = df["z_cerrado"].values

theta = 2 * np.pi * phase


# ============================================
# AJUSTE FOURIER
# ============================================

coef_x, x_fit = ajustar_fourier(theta, x, ORDEN_FOURIER)
coef_z, z_fit = ajustar_fourier(theta, z, ORDEN_FOURIER)


# ============================================
# CURVA SUAVE DE ALTA RESOLUCIÓN
# ============================================

phase_suave = np.linspace(0, 1, 400)
theta_suave = 2 * np.pi * phase_suave

x_suave = evaluar_fourier(theta_suave, coef_x, ORDEN_FOURIER)
z_suave = evaluar_fourier(theta_suave, coef_z, ORDEN_FOURIER)


# ============================================
# ERROR DEL AJUSTE
# ============================================

error = np.sqrt((x_fit - x)**2 + (z_fit - z)**2)

print("\n=== ERROR DEL AJUSTE FOURIER ===")

print(f"error medio: {np.mean(error):.6f}")
print(f"error RMS: {np.sqrt(np.mean(error**2)):.6f}")
print(f"error máximo: {np.max(error):.6f}")


# ============================================
# EXPORTAR COEFICIENTES
# ============================================

coef_table = []

for i, c in enumerate(coef_x):
    coef_table.append({"coef": f"x_{i}", "valor": c})

for i, c in enumerate(coef_z):
    coef_table.append({"coef": f"z_{i}", "valor": c})

coef_df = pd.DataFrame(coef_table)

coef_df.to_csv(RUTA_SALIDA / "coeficientes_fourier.csv", index=False)


# ============================================
# EXPORTAR CURVA SUAVE
# ============================================

df_suave = pd.DataFrame({
    "phase": phase_suave,
    "x_fourier": x_suave,
    "z_fourier": z_suave
})

df_suave.to_csv(RUTA_SALIDA / "trayectoria_fourier_suave.csv", index=False)


# ============================================
# GRÁFICOS
# ============================================

plt.figure(figsize=(7,6))

plt.plot(x, z, label="Curva original", linewidth=2)
plt.plot(x_suave, z_suave, label=f"Ajuste Fourier orden {ORDEN_FOURIER}", linewidth=2)

plt.xlabel("X normalizada")
plt.ylabel("Z normalizada")

plt.title("Trayectoria del pie - ajuste Fourier (lado derecho)")

plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()


plt.figure(figsize=(8,4))

plt.plot(phase, x, label="X original")
plt.plot(phase, x_fit, label="X Fourier")

plt.plot(phase, z, label="Z original")
plt.plot(phase, z_fit, label="Z Fourier")

plt.xlabel("Fase")
plt.ylabel("Posición")

plt.title("Ajuste por fase")

plt.grid(True)
plt.legend()

plt.show()


plt.figure(figsize=(8,4))

plt.plot(phase, error)

plt.xlabel("Fase")
plt.ylabel("Error")

plt.title("Error del ajuste Fourier")

plt.grid(True)

plt.show()


print("\nArchivos generados en:")

print(RUTA_SALIDA)