from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# RUTAS
# =========================================================

RUTA_CURVA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_procesamiento_final\seleccion_mejor_trial\modelo_matematico_trayectoria\trayectoria_modelo_fourier.csv"
)

SALIDA = RUTA_CURVA.parent / "trayectoria_escalada_cm"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# MEDIDAS ANATÓMICAS EN CM
# =========================================================

FEMUR_CM = 22
TIBIA_CM = 21
METATARSO_CM = 10.5
FALANGES_CM = 5

L_FUNCIONAL_CM = FEMUR_CM + TIBIA_CM + METATARSO_CM + FALANGES_CM

# =========================================================
# CARGA
# =========================================================

df = pd.read_csv(RUTA_CURVA)

x_norm = df["x_modelo"].to_numpy()
z_norm = df["z_modelo"].to_numpy()
phase = df["phase"].to_numpy()

# =========================================================
# ESCALADO A CM
# =========================================================

x_cm = x_norm * L_FUNCIONAL_CM

# suelo en z = 0
z_cm = (z_norm - np.min(z_norm)) * L_FUNCIONAL_CM

df_out = pd.DataFrame({
    "phase": phase,
    "x_cm": x_cm,
    "z_cm": z_cm
})

df_out.to_csv(
    SALIDA / "trayectoria_objetivo_cm_suelo.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# MÉTRICAS
# =========================================================

metricas = {
    "longitud_funcional_cm": L_FUNCIONAL_CM,
    "x_min_cm": float(np.min(x_cm)),
    "x_max_cm": float(np.max(x_cm)),
    "z_min_cm": float(np.min(z_cm)),
    "z_max_cm": float(np.max(z_cm)),
    "avance_horizontal_cm": float(np.max(x_cm) - np.min(x_cm)),
    "altura_maxima_cm": float(np.max(z_cm)),
}

pd.DataFrame([metricas]).to_csv(
    SALIDA / "metricas_trayectoria_cm.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== TRAYECTORIA ESCALADA A CM ===")
for k, v in metricas.items():
    print(f"{k}: {v:.3f}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICO
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(x_cm, z_cm, linewidth=3)
plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Trayectoria objetivo escalada - suelo en Z = 0")
plt.axis("equal")
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase, x_cm, label="X [cm]")
plt.plot(phase, z_cm, label="Z [cm]")
plt.xlabel("Fase")
plt.ylabel("Posición [cm]")
plt.title("Trayectoria objetivo en cm por fase")
plt.grid(True)
plt.legend()
plt.show()