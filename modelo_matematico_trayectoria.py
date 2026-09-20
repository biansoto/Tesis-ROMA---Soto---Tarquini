from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# RUTAS
# =========================================================

RUTA_CURVA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_procesamiento_final\seleccion_mejor_trial\trayectoria_objetivo_mejor_trial.csv"
)

SALIDA = RUTA_CURVA.parent / "modelo_matematico_trayectoria"
SALIDA.mkdir(parents=True, exist_ok=True)

ORDEN_FOURIER = 2
NPTS = 400


# =========================================================
# FOURIER
# =========================================================

def matriz_fourier(phase, orden):
    theta = 2 * np.pi * phase

    cols = [np.ones(len(phase))]

    for k in range(1, orden + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))

    return np.column_stack(cols)


def ajustar_fourier(phase, y, orden):
    A = matriz_fourier(phase, orden)
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_fit = A @ coef
    return coef, y_fit


def evaluar_fourier(phase, coef, orden):
    A = matriz_fourier(phase, orden)
    return A @ coef


def calcular_error(x, z, x_fit, z_fit):
    e = np.sqrt((x - x_fit) ** 2 + (z - z_fit) ** 2)

    return {
        "error_medio": float(np.mean(e)),
        "error_rms": float(np.sqrt(np.mean(e ** 2))),
        "error_max": float(np.max(e)),
    }, e


def texto_ecuacion(nombre, coef, orden):
    partes = [f"{coef[0]:.10f}"]

    idx = 1
    for k in range(1, orden + 1):
        a = coef[idx]
        b = coef[idx + 1]

        partes.append(f"({a:.10f}) cos({k}·2πφ)")
        partes.append(f"({b:.10f}) sin({k}·2πφ)")

        idx += 2

    return f"{nombre}(φ) = " + " + ".join(partes)


# =========================================================
# CARGA
# =========================================================

df = pd.read_csv(RUTA_CURVA).sort_values("phase").reset_index(drop=True)

phase = df["phase"].to_numpy()
x = df["x_norm"].to_numpy()
z = df["z_norm"].to_numpy()

# =========================================================
# AJUSTE FOURIER
# =========================================================

coef_x, x_fit = ajustar_fourier(phase, x, ORDEN_FOURIER)
coef_z, z_fit = ajustar_fourier(phase, z, ORDEN_FOURIER)

metricas, error = calcular_error(x, z, x_fit, z_fit)

# curva de alta resolución
phase_suave = np.linspace(0, 1, NPTS)
x_suave = evaluar_fourier(phase_suave, coef_x, ORDEN_FOURIER)
z_suave = evaluar_fourier(phase_suave, coef_z, ORDEN_FOURIER)

# cierre exacto
x_suave[-1] = x_suave[0]
z_suave[-1] = z_suave[0]

# =========================================================
# EXPORTAR
# =========================================================

df_coef = pd.DataFrame({
    "coeficiente": (
        ["c0"] +
        [f"cos_{k}" for k in range(1, ORDEN_FOURIER + 1)] +
        [f"sin_{k}" for k in range(1, ORDEN_FOURIER + 1)]
    )
})

# mejor armar coeficientes explícitos
filas_coef = []

filas_coef.append({"variable": "x", "termino": "c0", "valor": coef_x[0]})
filas_coef.append({"variable": "z", "termino": "c0", "valor": coef_z[0]})

idx = 1
for k in range(1, ORDEN_FOURIER + 1):
    filas_coef.append({"variable": "x", "termino": f"cos_{k}", "valor": coef_x[idx]})
    filas_coef.append({"variable": "x", "termino": f"sin_{k}", "valor": coef_x[idx + 1]})
    filas_coef.append({"variable": "z", "termino": f"cos_{k}", "valor": coef_z[idx]})
    filas_coef.append({"variable": "z", "termino": f"sin_{k}", "valor": coef_z[idx + 1]})
    idx += 2

pd.DataFrame(filas_coef).to_csv(
    SALIDA / "coeficientes_fourier_trayectoria.csv",
    index=False,
    encoding="utf-8-sig"
)

df_modelo = pd.DataFrame({
    "phase": phase_suave,
    "x_modelo": x_suave,
    "z_modelo": z_suave,
})

df_modelo.to_csv(
    SALIDA / "trayectoria_modelo_fourier.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame([{
    "orden_fourier": ORDEN_FOURIER,
    **metricas
}]).to_csv(
    SALIDA / "metricas_ajuste_fourier.csv",
    index=False,
    encoding="utf-8-sig"
)

ecuacion_x = texto_ecuacion("x", coef_x, ORDEN_FOURIER)
ecuacion_z = texto_ecuacion("z", coef_z, ORDEN_FOURIER)

with open(SALIDA / "ecuaciones_fourier.txt", "w", encoding="utf-8") as f:
    f.write("Modelo matemático de trayectoria objetivo\n")
    f.write("Variable de fase: φ ∈ [0,1]\n\n")
    f.write(ecuacion_x + "\n\n")
    f.write(ecuacion_z + "\n\n")
    f.write("Métricas de ajuste:\n")
    for k, v in metricas.items():
        f.write(f"{k}: {v:.10f}\n")

# =========================================================
# REPORTE
# =========================================================

print("\n=== MODELO MATEMÁTICO FOURIER ===")
print(f"Orden Fourier: {ORDEN_FOURIER}")

print("\nEcuación X:")
print(ecuacion_x)

print("\nEcuación Z:")
print(ecuacion_z)

print("\n=== ERROR DEL MODELO ===")
for k, v in metricas.items():
    print(f"{k}: {v:.6f}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICOS
# =========================================================

plt.figure(figsize=(8, 6))
plt.plot(x, z, "--", linewidth=2, label="Curva objetivo")
plt.plot(x_suave, z_suave, linewidth=3, label=f"Modelo Fourier orden {ORDEN_FOURIER}")
plt.xlabel("X normalizada")
plt.ylabel("Z normalizada")
plt.title("Modelo matemático de trayectoria objetivo")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase, x, "--", label="X objetivo")
plt.plot(phase, x_fit, label="X modelo")
plt.plot(phase, z, "--", label="Z objetivo")
plt.plot(phase, z_fit, label="Z modelo")
plt.xlabel("Fase φ")
plt.ylabel("Posición normalizada")
plt.title("Ajuste Fourier por fase")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase, error)
plt.xlabel("Fase φ")
plt.ylabel("Error")
plt.title("Error del modelo Fourier")
plt.grid(True)
plt.show()