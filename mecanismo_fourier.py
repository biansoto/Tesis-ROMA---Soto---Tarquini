from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# RUTAS
# =========================================================

RUTA_FOURIER = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_fourier_final\trayectoria_fourier_final_cm.csv"
)

SALIDA = RUTA_FOURIER.parent / "cinematica_inversa"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# MEDIDAS ANATÓMICAS [cm]
# =========================================================

FEMUR = 17
TIBIA = 21
PIE_DISTAL = 10.5 + 5.0

ALTURA_CADERA = 48.0

# Longitud efectiva desde rodilla hasta punto distal aproximado
L1 = FEMUR
L2 = TIBIA + PIE_DISTAL

# =========================================================
# CARGA CURVA FOURIER
# =========================================================

df = pd.read_csv(RUTA_FOURIER)

phase = df["phase"].to_numpy()
x_pie = df["x_cm_centrada"].to_numpy()
z_pie = df["z_cm_suelo"].to_numpy()

# Pasar a sistema con origen en cadera
# Cadera en (0, ALTURA_CADERA), suelo en z=0
x_rel = x_pie
z_rel = z_pie - ALTURA_CADERA

# =========================================================
# CINEMÁTICA INVERSA 2 ESLABONES
# =========================================================

theta_cadera = []
theta_rodilla = []
valido = []

for x, z in zip(x_rel, z_rel):
    r = np.sqrt(x**2 + z**2)

    # verificar alcanzabilidad
    if r > (L1 + L2) or r < abs(L1 - L2):
        theta_cadera.append(np.nan)
        theta_rodilla.append(np.nan)
        valido.append(False)
        continue

    cos_knee = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_knee = np.clip(cos_knee, -1, 1)

    knee = np.arccos(cos_knee)

    alpha = np.arctan2(z, x)

    beta = np.arccos(
        np.clip((L1**2 + r**2 - L2**2) / (2 * L1 * r), -1, 1)
    )

    hip = alpha + beta

    theta_cadera.append(hip)
    theta_rodilla.append(knee)
    valido.append(True)

theta_cadera = np.array(theta_cadera)
theta_rodilla = np.array(theta_rodilla)
valido = np.array(valido)

# pasar a grados
theta_cadera_deg = np.degrees(theta_cadera)
theta_rodilla_deg = np.degrees(theta_rodilla)

# =========================================================
# EXPORTAR
# =========================================================

df_out = pd.DataFrame({
    "phase": phase,
    "x_pie_cm": x_pie,
    "z_pie_cm": z_pie,
    "x_rel_cadera_cm": x_rel,
    "z_rel_cadera_cm": z_rel,
    "theta_cadera": theta_cadera,
    "theta_rodilla": theta_rodilla,
    "theta_cadera_deg": theta_cadera_deg,
    "theta_rodilla_deg": theta_rodilla_deg,
    "valido": valido,
})

df_out.to_csv(
    SALIDA / "cinematica_inversa_fourier.csv",
    index=False,
    encoding="utf-8-sig"
)

metricas = {
    "L1_femur_cm": L1,
    "L2_tibia_pie_cm": L2,
    "altura_cadera_cm": ALTURA_CADERA,
    "puntos_validos": int(np.sum(valido)),
    "puntos_totales": int(len(valido)),
    "theta_cadera_min_deg": float(np.nanmin(theta_cadera_deg)),
    "theta_cadera_max_deg": float(np.nanmax(theta_cadera_deg)),
    "theta_rodilla_min_deg": float(np.nanmin(theta_rodilla_deg)),
    "theta_rodilla_max_deg": float(np.nanmax(theta_rodilla_deg)),
}

pd.DataFrame([metricas]).to_csv(
    SALIDA / "metricas_cinematica_inversa.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# REPORTE
# =========================================================

print("\n=== CINEMÁTICA INVERSA FOURIER ===")
for k, v in metricas.items():
    print(f"{k}: {v}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICOS
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(x_pie, z_pie, linewidth=3)
plt.xlabel("X pie [cm]")
plt.ylabel("Z pie [cm]")
plt.title("Trayectoria Fourier del pie")
plt.axis("equal")
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase, theta_cadera_deg, label="Ángulo cadera")
plt.plot(phase, theta_rodilla_deg, label="Ángulo rodilla")
plt.xlabel("Fase")
plt.ylabel("Ángulo [deg]")
plt.title("Cinemática inversa requerida")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase, x_pie, label="X pie [cm]")
plt.plot(phase, z_pie, label="Z pie [cm]")
plt.xlabel("Fase")
plt.ylabel("Posición [cm]")
plt.title("Trayectoria del pie por fase")
plt.grid(True)
plt.legend()
plt.show()

# =========================================================
# VALIDACIÓN: RECONSTRUCCIÓN DE TRAYECTORIA MEDIANTE IK
# =========================================================

# Cadera fija
x_cadera = 0.0
z_cadera = ALTURA_CADERA

# Reconstrucción de rodilla desde ángulo de cadera
x_rodilla_rec = x_cadera + L1 * np.cos(theta_cadera)
z_rodilla_rec = z_cadera + L1 * np.sin(theta_cadera)

# Reconstrucción del extremo distal usando ángulo absoluto del segmento distal
# IMPORTANTE:
# En este modelo theta_rodilla es el ángulo interno de rodilla.
# El ángulo absoluto del segmento distal se obtiene a partir de:
# =========================================================
# VALIDACIÓN: RECONSTRUCCIÓN DE TRAYECTORIA MEDIANTE IK
# Prueba automática de convenciones angulares
# =========================================================

x_cadera = 0.0
z_cadera = ALTURA_CADERA

x_rodilla_rec = x_cadera + L1 * np.cos(theta_cadera)
z_rodilla_rec = z_cadera + L1 * np.sin(theta_cadera)

convenciones = {
    "hip + knee": theta_cadera + theta_rodilla,
    "hip - knee": theta_cadera - theta_rodilla,
    "hip + pi - knee": theta_cadera + (np.pi - theta_rodilla),
    "hip - pi + knee": theta_cadera - np.pi + theta_rodilla,
    "hip + pi + knee": theta_cadera + np.pi + theta_rodilla,
    "hip - pi - knee": theta_cadera - np.pi - theta_rodilla,
}

mejor = None

print("\n=== PRUEBA DE CONVENCIONES ANGULARES ===")

for nombre, theta_distal_abs in convenciones.items():
    x_test = x_rodilla_rec + L2 * np.cos(theta_distal_abs)
    z_test = z_rodilla_rec + L2 * np.sin(theta_distal_abs)

    error_test = np.sqrt((x_test - x_pie)**2 + (z_test - z_pie)**2)
    rmse_test = np.sqrt(np.mean(error_test**2))
    error_max_test = np.max(error_test)

    print(f"{nombre}: RMSE = {rmse_test:.6e} cm | error_max = {error_max_test:.6e} cm")

    if mejor is None or rmse_test < mejor["rmse"]:
        mejor = {
            "nombre": nombre,
            "theta_distal_abs": theta_distal_abs,
            "rmse": rmse_test,
            "error_max": error_max_test,
        }

print(f"\nConvención seleccionada: {mejor['nombre']}")

theta_distal_abs = mejor["theta_distal_abs"]

x_pie_rec = x_rodilla_rec + L2 * np.cos(theta_distal_abs)
z_pie_rec = z_rodilla_rec + L2 * np.sin(theta_distal_abs)

error_x = x_pie_rec - x_pie
error_z = z_pie_rec - z_pie
error_euclidiano = np.sqrt(error_x**2 + error_z**2)

rmse_x = np.sqrt(np.mean(error_x**2))
rmse_z = np.sqrt(np.mean(error_z**2))
rmse_total = np.sqrt(np.mean(error_euclidiano**2))
error_max = np.max(error_euclidiano)

print("\n=== VALIDACIÓN RECONSTRUCCIÓN IK ===")
print(f"RMSE_x_cm: {rmse_x}")
print(f"RMSE_z_cm: {rmse_z}")
print(f"RMSE_total_cm: {rmse_total}")
print(f"error_max_cm: {error_max}")

x_pie_rec = x_rodilla_rec + L2 * np.cos(theta_distal_abs)
z_pie_rec = z_rodilla_rec + L2 * np.sin(theta_distal_abs)

# Errores de reconstrucción
error_x = x_pie_rec - x_pie
error_z = z_pie_rec - z_pie
error_euclidiano = np.sqrt(error_x**2 + error_z**2)

rmse_x = np.sqrt(np.mean(error_x**2))
rmse_z = np.sqrt(np.mean(error_z**2))
rmse_total = np.sqrt(np.mean(error_euclidiano**2))
error_max = np.max(error_euclidiano)

print("\n=== VALIDACIÓN RECONSTRUCCIÓN IK ===")
print(f"RMSE_x_cm: {rmse_x}")
print(f"RMSE_z_cm: {rmse_z}")
print(f"RMSE_total_cm: {rmse_total}")
print(f"error_max_cm: {error_max}")

# =========================================================
# GRÁFICO: TRAYECTORIA OBJETIVO VS RECONSTRUIDA
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    x_pie,
    z_pie,
    linewidth=3,
    label="Trayectoria objetivo"
)

plt.plot(
    x_pie_rec,
    z_pie_rec,
    linestyle="--",
    linewidth=2,
    label="Reconstrucción mediante IK"
)

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Validación de reconstrucción de la trayectoria objetivo")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()

plt.savefig(SALIDA / "validacion_trayectoria_objetivo_vs_reconstruida.png", dpi=300)
plt.savefig(SALIDA / "validacion_trayectoria_objetivo_vs_reconstruida.svg")
plt.show()

# =========================================================
# GRÁFICO: ERROR DE RECONSTRUCCIÓN VS FASE
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(
    phase,
    error_euclidiano,
    linewidth=2,
    label="Error euclidiano"
)

plt.xlabel("Fase")
plt.ylabel("Error [cm]")
plt.title("Error de reconstrucción mediante cinemática inversa")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(SALIDA / "error_reconstruccion_ik_vs_fase.png", dpi=300)
plt.savefig(SALIDA / "error_reconstruccion_ik_vs_fase.svg")
plt.show()

# =========================================================
# GUARDAR CSV DE VALIDACIÓN
# =========================================================

df_validacion = pd.DataFrame({
    "phase": phase,
    "x_objetivo_cm": x_pie,
    "z_objetivo_cm": z_pie,
    "x_reconstruido_cm": x_pie_rec,
    "z_reconstruido_cm": z_pie_rec,
    "error_x_cm": error_x,
    "error_z_cm": error_z,
    "error_euclidiano_cm": error_euclidiano,
})

df_validacion.to_csv(SALIDA / "validacion_reconstruccion_ik.csv", index=False)