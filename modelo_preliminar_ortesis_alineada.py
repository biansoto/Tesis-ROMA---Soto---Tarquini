from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# RUTAS
# =========================================================

RUTA_CINEMATICA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_fourier_final\cinematica_inversa\cinematica_inversa_fourier.csv"
)

SALIDA = RUTA_CINEMATICA.parent / "modelo_ortesis_anatomica"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARÁMETROS ANATÓMICOS [cm]
# =========================================================

L_FEMUR = 22
L_DISTAL = 36.5

ALTURA_CADERA = 48.0

# =========================================================
# PARÁMETROS DE DISEÑO DE LA ÓRTESIS
# =========================================================

# Offset lateral: distancia de la órtesis respecto a la pata.
# En este modelo 2D no modifica la trayectoria sagital, pero queda documentado.
OFFSET_LATERAL_CM = 3.0

# Asistencia buscada
PESO_PERRO_KG = 20.0
G = 9.81

PORC_ASISTENCIA_MIN = 0.10
PORC_ASISTENCIA_MAX = 0.20

# Compresión útil estimada del resorte
COMPRESION_RESORTE_CM = 1.0

# Fuerzas de diseño para evaluar resortes [N]
FUERZAS_DISENO_N = [5, 10, 15, 20, 25]

# Fases que se van a graficar
FASES_A_GRAFICAR = [0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

# =========================================================
# CARGA DE DATOS
# =========================================================

df = pd.read_csv(RUTA_CINEMATICA)

phase = df["phase"].to_numpy()
x_pie = df["x_pie_cm"].to_numpy()
z_pie = df["z_pie_cm"].to_numpy()

theta_cadera = df["theta_cadera_rad"].to_numpy()
theta_rodilla = df["theta_rodilla_rad"].to_numpy()

theta_cadera_deg = df["theta_cadera_deg"].to_numpy()
theta_rodilla_deg = df["theta_rodilla_deg"].to_numpy()

valido = df["valido"].to_numpy().astype(bool)

# =========================================================
# RECONSTRUCCIÓN GEOMÉTRICA
# =========================================================

# Cadera fija en sistema global
x_cadera = np.zeros_like(phase)
z_cadera = np.ones_like(phase) * ALTURA_CADERA

# Rodilla calculada desde la cadera
x_rodilla = x_cadera + L_FEMUR * np.cos(theta_cadera)
z_rodilla = z_cadera + L_FEMUR * np.sin(theta_cadera)

# Metatarso / punto distal desde los datos originales
x_distal = x_pie
z_distal = z_pie

# Longitud real reconstruida entre rodilla y distal
L_distal_reconstruida = np.sqrt(
    (x_distal - x_rodilla) ** 2 + (z_distal - z_rodilla) ** 2
)

# Error respecto a longitud distal esperada
error_L_distal = L_distal_reconstruida - L_DISTAL

# =========================================================
# CÁLCULOS DE CARGA Y RESORTE
# =========================================================

peso_total_N = PESO_PERRO_KG * G

# Aproximación simple: cada pata trasera soporta 1/4 del peso total en estática
carga_estatica_pata_N = peso_total_N / 4

asistencia_min_N = PORC_ASISTENCIA_MIN * carga_estatica_pata_N
asistencia_max_N = PORC_ASISTENCIA_MAX * carga_estatica_pata_N

compresion_m = COMPRESION_RESORTE_CM / 100

resortes = []

for F in FUERZAS_DISENO_N:
    k_N_m = F / compresion_m
    k_N_cm = F / COMPRESION_RESORTE_CM

    resortes.append({
        "fuerza_N": F,
        "compresion_cm": COMPRESION_RESORTE_CM,
        "k_N_m": k_N_m,
        "k_N_cm": k_N_cm,
    })

df_resortes = pd.DataFrame(resortes)

# =========================================================
# MÉTRICAS GENERALES
# =========================================================

metricas = {
    "L_femur_cm": L_FEMUR,
    "L_distal_cm": L_DISTAL,
    "altura_cadera_cm": ALTURA_CADERA,
    "offset_lateral_cm": OFFSET_LATERAL_CM,

    "peso_perro_kg": PESO_PERRO_KG,
    "peso_total_N": peso_total_N,
    "carga_estatica_pata_N": carga_estatica_pata_N,
    "asistencia_min_N": asistencia_min_N,
    "asistencia_max_N": asistencia_max_N,

    "theta_cadera_min_deg": float(np.nanmin(theta_cadera_deg)),
    "theta_cadera_max_deg": float(np.nanmax(theta_cadera_deg)),
    "theta_cadera_rango_deg": float(np.nanmax(theta_cadera_deg) - np.nanmin(theta_cadera_deg)),

    "theta_rodilla_min_deg": float(np.nanmin(theta_rodilla_deg)),
    "theta_rodilla_max_deg": float(np.nanmax(theta_rodilla_deg)),
    "theta_rodilla_rango_deg": float(np.nanmax(theta_rodilla_deg) - np.nanmin(theta_rodilla_deg)),

    "L_distal_reconstruida_min_cm": float(np.nanmin(L_distal_reconstruida)),
    "L_distal_reconstruida_max_cm": float(np.nanmax(L_distal_reconstruida)),
    "error_L_distal_medio_cm": float(np.nanmean(error_L_distal)),
    "error_L_distal_max_abs_cm": float(np.nanmax(np.abs(error_L_distal))),
}

df_metricas = pd.DataFrame([metricas])

# =========================================================
# EXPORTAR DATOS
# =========================================================

df_modelo = pd.DataFrame({
    "phase": phase,

    "x_cadera_cm": x_cadera,
    "z_cadera_cm": z_cadera,

    "x_rodilla_cm": x_rodilla,
    "z_rodilla_cm": z_rodilla,

    "x_distal_cm": x_distal,
    "z_distal_cm": z_distal,

    "theta_cadera_deg": theta_cadera_deg,
    "theta_rodilla_deg": theta_rodilla_deg,

    "L_distal_reconstruida_cm": L_distal_reconstruida,
    "error_L_distal_cm": error_L_distal,

    "valido": valido,
})

df_modelo.to_csv(
    SALIDA / "modelo_ortesis_anatomica.csv",
    index=False,
    encoding="utf-8-sig"
)

df_metricas.to_csv(
    SALIDA / "metricas_modelo_ortesis_anatomica.csv",
    index=False,
    encoding="utf-8-sig"
)

df_resortes.to_csv(
    SALIDA / "dimensionamiento_resorte.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# REPORTE EN CONSOLA
# =========================================================

print("\n=== MODELO PRELIMINAR DE ÓRTESIS ANATÓMICA ===")
for k, v in metricas.items():
    print(f"{k}: {v}")

print("\n=== DIMENSIONAMIENTO PRELIMINAR DE RESORTE ===")
print(df_resortes)

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICO 1: TRAYECTORIA Y ARTICULACIONES
# =========================================================

plt.figure(figsize=(9, 6))

plt.plot(x_distal, z_distal, linewidth=3, label="Trayectoria metatarso")
plt.plot(x_rodilla, z_rodilla, linewidth=2, label="Trayectoria rodilla reconstruida")

plt.scatter([0], [ALTURA_CADERA], s=80, label="Cadera")

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Trayectorias articulares del modelo anatómico")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "trayectorias_articulares.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 2: ESQUEMA DE ÓRTESIS EN VARIAS FASES
# =========================================================

plt.figure(figsize=(10, 7))

for fase_obj in FASES_A_GRAFICAR:
    idx = np.argmin(np.abs(phase - fase_obj))

    xs = [x_cadera[idx], x_rodilla[idx], x_distal[idx]]
    zs = [z_cadera[idx], z_rodilla[idx], z_distal[idx]]

    plt.plot(xs, zs, marker="o", linewidth=2, label=f"fase {phase[idx]:.2f}")

plt.axhline(0, linestyle="--", linewidth=1, label="suelo")

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Esquema 2D de órtesis anatómica en distintas fases")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "esquema_ortesis_fases.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 3: ÁNGULOS ARTICULARES
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, theta_cadera_deg, label="Cadera")
plt.plot(phase, theta_rodilla_deg, label="Rodilla")

plt.xlabel("Fase")
plt.ylabel("Ángulo [deg]")
plt.title("Rangos articulares requeridos para la órtesis")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "angulos_articulares_ortesis.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 4: ERROR DE LONGITUD DISTAL
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, error_L_distal)

plt.xlabel("Fase")
plt.ylabel("Error [cm]")
plt.title("Error de reconstrucción de longitud distal")
plt.grid(True)
plt.tight_layout()
plt.savefig(SALIDA / "error_longitud_distal.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 5: FUERZA VS CONSTANTE ELÁSTICA
# =========================================================

plt.figure(figsize=(7, 4))

plt.plot(df_resortes["fuerza_N"], df_resortes["k_N_m"], marker="o")

plt.xlabel("Fuerza asistiva [N]")
plt.ylabel("Constante elástica k [N/m]")
plt.title(f"Dimensionamiento de resorte para compresión de {COMPRESION_RESORTE_CM} cm")
plt.grid(True)
plt.tight_layout()
plt.savefig(SALIDA / "dimensionamiento_resorte.png", dpi=300)
plt.show()