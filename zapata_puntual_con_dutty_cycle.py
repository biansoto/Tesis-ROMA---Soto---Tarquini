from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# RUTAS
# =========================================================

RUTA_MODELO = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_fourier_final\cinematica_inversa\modelo_ortesis_anatomica\modelo_ortesis_anatomica.csv"
)

SALIDA = RUTA_MODELO.parent / "diseno_funcional_final"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARÁMETROS
# =========================================================

PESO_PERRO_KG = 20.0
G = 9.81

CARGA_PATA_N = PESO_PERRO_KG * G / 4

ASISTENCIA_MIN_N = 0.10 * CARGA_PATA_N
ASISTENCIA_MAX_N = 0.20 * CARGA_PATA_N
ASISTENCIA_OBJ_N = 0.15 * CARGA_PATA_N

DUTY_FACTOR = 0.55

K_RESORTE_N_M = 1000.0
K_RESORTE_N_CM = K_RESORTE_N_M / 100.0

LONGITUD_LIBRE_RESORTE_CM = 2.0
COMPRESION_MAX_CM = 2.0
ELEVACION_RETRACCION_CM = 1.2

TOL_CONTACTO_CM = 0.05

# =========================================================
# CARGA DE DATOS
# =========================================================

df = pd.read_csv(RUTA_MODELO)

phase = df["phase"].to_numpy()

x_cadera = df["x_cadera_cm"].to_numpy()
z_cadera = df["z_cadera_cm"].to_numpy()

x_rodilla = df["x_rodilla_cm"].to_numpy()
z_rodilla = df["z_rodilla_cm"].to_numpy()

x_distal = df["x_distal_cm"].to_numpy()
z_distal = df["z_distal_cm"].to_numpy()

theta_cadera_deg = df["theta_cadera_deg"].to_numpy()
theta_rodilla_deg = df["theta_rodilla_deg"].to_numpy()

# =========================================================
# FASE DE APOYO POR DUTY FACTOR
# =========================================================

mitad_apoyo = DUTY_FACTOR / 2

mask_apoyo_inicio = phase <= mitad_apoyo
mask_apoyo_final = phase >= 1 - mitad_apoyo

en_apoyo = mask_apoyo_inicio | mask_apoyo_final

fase_local = np.zeros_like(phase)

fase_local[mask_apoyo_inicio] = phase[mask_apoyo_inicio] / mitad_apoyo
fase_local[mask_apoyo_final] = (
    phase[mask_apoyo_final] - (1 - mitad_apoyo)
) / mitad_apoyo

# Perfil suave de fuerza: 0 en los bordes, máximo en mitad de cada tramo
perfil_apoyo = np.sin(np.pi * fase_local)
perfil_apoyo = np.clip(perfil_apoyo, 0, 1)

fuerza_asistiva_N = np.zeros_like(phase)
fuerza_asistiva_N[en_apoyo] = ASISTENCIA_OBJ_N * perfil_apoyo[en_apoyo]

# =========================================================
# RESORTE Y ZAPATA
# =========================================================

compresion_resorte_cm = fuerza_asistiva_N / K_RESORTE_N_CM
compresion_resorte_cm = np.clip(
    compresion_resorte_cm,
    0,
    COMPRESION_MAX_CM
)

fuerza_real_N = K_RESORTE_N_CM * compresion_resorte_cm

longitud_resorte_cm = LONGITUD_LIBRE_RESORTE_CM - compresion_resorte_cm
longitud_resorte_cm = np.clip(
    longitud_resorte_cm,
    0.2,
    LONGITUD_LIBRE_RESORTE_CM
)

x_zapata = x_distal.copy()

# =========================================================
# ZAPATA CON RETRACCIÓN SUAVE
# =========================================================

retraccion_suave = np.zeros_like(phase)

# En balanceo central, la zapata se eleva suavemente.
# La fase de balanceo va desde mitad_apoyo hasta 1 - mitad_apoyo.
inicio_balanceo = mitad_apoyo
fin_balanceo = 1 - mitad_apoyo
duracion_balanceo = fin_balanceo - inicio_balanceo

mask_balanceo = (~en_apoyo)

fase_balanceo = np.zeros_like(phase)
fase_balanceo[mask_balanceo] = (
    (phase[mask_balanceo] - inicio_balanceo) / duracion_balanceo
)

# Perfil suave: 0 en inicio/fin del balanceo, máximo en el centro
retraccion_suave[mask_balanceo] = (
    ELEVACION_RETRACCION_CM *
    np.sin(np.pi * fase_balanceo[mask_balanceo])
)

# En apoyo: la zapata toca el suelo
z_zapata_apoyo = np.zeros_like(z_distal)

# En balanceo: sigue al distal con retracción suave
z_zapata_balanceo = (
    z_distal
    - LONGITUD_LIBRE_RESORTE_CM
    + retraccion_suave
)

# Evitar que la zapata quede bajo el suelo
z_zapata_balanceo = np.maximum(z_zapata_balanceo, 0.0)

z_zapata = np.where(en_apoyo, z_zapata_apoyo, z_zapata_balanceo)

# Longitud geométrica real distal-zapata
longitud_resorte_geom_cm = z_distal - z_zapata

# Compresión geométrica
compresion_geom_cm = LONGITUD_LIBRE_RESORTE_CM - longitud_resorte_geom_cm
compresion_geom_cm = np.clip(compresion_geom_cm, 0, COMPRESION_MAX_CM)

# =========================================================
# VALIDACIONES
# =========================================================

contacto_zapata = z_zapata <= TOL_CONTACTO_CM

apoyo_sin_contacto = en_apoyo & (~contacto_zapata)
balanceo_con_contacto = (~en_apoyo) & contacto_zapata

porcentaje_apoyo = 100 * np.sum(en_apoyo) / len(en_apoyo)

porcentaje_apoyo_sin_contacto = (
    100 * np.sum(apoyo_sin_contacto) / np.sum(en_apoyo)
)

porcentaje_arrastre_balanceo = (
    100 * np.sum(balanceo_con_contacto) / np.sum(~en_apoyo)
)

# =========================================================
# MÉTRICAS
# =========================================================

metricas = {
    "peso_perro_kg": PESO_PERRO_KG,
    "carga_estimada_por_pata_N": CARGA_PATA_N,
    "asistencia_min_10porc_N": ASISTENCIA_MIN_N,
    "asistencia_max_20porc_N": ASISTENCIA_MAX_N,
    "asistencia_obj_15porc_N": ASISTENCIA_OBJ_N,

    "duty_factor": DUTY_FACTOR,
    "porcentaje_fase_apoyo_modelada": porcentaje_apoyo,

    "k_resorte_N_m": K_RESORTE_N_M,
    "k_resorte_N_cm": K_RESORTE_N_CM,

    "longitud_libre_resorte_cm": LONGITUD_LIBRE_RESORTE_CM,
    "elevacion_retraccion_cm": ELEVACION_RETRACCION_CM,

    "fuerza_max_real_N": float(np.max(fuerza_real_N)),
    "fuerza_media_en_apoyo_N": float(np.mean(fuerza_real_N[en_apoyo])),

    "compresion_max_cm": float(np.max(compresion_resorte_cm)),
    "compresion_media_en_apoyo_cm": float(np.mean(compresion_resorte_cm[en_apoyo])),

    "longitud_resorte_min_cm": float(np.min(longitud_resorte_cm)),
    "longitud_resorte_max_cm": float(np.max(longitud_resorte_cm)),

    "porcentaje_apoyo_sin_contacto_zapata": porcentaje_apoyo_sin_contacto,
    "porcentaje_arrastre_zapata_en_balanceo": porcentaje_arrastre_balanceo,

    "theta_cadera_min_deg": float(np.nanmin(theta_cadera_deg)),
    "theta_cadera_max_deg": float(np.nanmax(theta_cadera_deg)),
    "theta_rodilla_min_deg": float(np.nanmin(theta_rodilla_deg)),
    "theta_rodilla_max_deg": float(np.nanmax(theta_rodilla_deg)),
}

df_metricas = pd.DataFrame([metricas])

# =========================================================
# EXPORTAR
# =========================================================

df_salida = pd.DataFrame({
    "phase": phase,

    "x_cadera_cm": x_cadera,
    "z_cadera_cm": z_cadera,

    "x_rodilla_cm": x_rodilla,
    "z_rodilla_cm": z_rodilla,

    "x_distal_cm": x_distal,
    "z_distal_cm": z_distal,

    "x_zapata_cm": x_zapata,
    "z_zapata_cm": z_zapata,

    "en_apoyo": en_apoyo,
    "contacto_zapata": contacto_zapata,

    "fuerza_asistiva_N": fuerza_asistiva_N,
    "fuerza_real_N": fuerza_real_N,

    "compresion_resorte_cm": compresion_resorte_cm,
    "longitud_resorte_cm": longitud_resorte_cm,

    "theta_cadera_deg": theta_cadera_deg,
    "theta_rodilla_deg": theta_rodilla_deg,
})

df_salida.to_csv(
    SALIDA / "diseno_funcional_final.csv",
    index=False,
    encoding="utf-8-sig"
)

df_metricas.to_csv(
    SALIDA / "metricas_diseno_funcional_final.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== DISEÑO FUNCIONAL FINAL ===")
for k, v in metricas.items():
    print(f"{k}: {v}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICO 1: FASE DE APOYO
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, z_distal, label="Z metatarso")
plt.fill_between(
    phase,
    0,
    z_distal,
    where=en_apoyo,
    alpha=0.3,
    label="fase de apoyo modelada"
)

plt.xlabel("Fase")
plt.ylabel("Z [cm]")
plt.title("Fase de apoyo modelada por duty factor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "fase_apoyo_duty_factor.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 2: FUERZA ASISTIVA
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, fuerza_real_N, label="Fuerza asistiva")
plt.axhline(ASISTENCIA_MIN_N, linestyle="--", label="10% carga pata")
plt.axhline(ASISTENCIA_MAX_N, linestyle="--", label="20% carga pata")
plt.axhline(ASISTENCIA_OBJ_N, linestyle=":", label="objetivo 15%")

plt.xlabel("Fase")
plt.ylabel("Fuerza [N]")
plt.title("Fuerza asistiva durante el ciclo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "fuerza_asistiva_final.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 3: COMPRESIÓN DEL RESORTE
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, compresion_resorte_cm)

plt.xlabel("Fase")
plt.ylabel("Compresión [cm]")
plt.title("Compresión del resorte")
plt.grid(True)
plt.tight_layout()
plt.savefig(SALIDA / "compresion_resorte_final.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 4: METATARSO Y ZAPATA
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(x_distal, z_distal, linewidth=3, label="Metatarso")
plt.plot(x_zapata, z_zapata, linewidth=2, label="Zapata")
plt.axhline(0, linestyle="--", label="suelo")

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Trayectoria del metatarso y zapata")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "trayectoria_metatarso_zapata_final.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 5: ESQUEMA DEL MECANISMO EN FASES
# =========================================================

FASES_A_GRAFICAR = [0.00, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]

plt.figure(figsize=(10, 7))

for fase_obj in FASES_A_GRAFICAR:
    idx = np.argmin(np.abs(phase - fase_obj))

    xs_pata = [
        x_cadera[idx],
        x_rodilla[idx],
        x_distal[idx],
    ]

    zs_pata = [
        z_cadera[idx],
        z_rodilla[idx],
        z_distal[idx],
    ]

    xs_resorte = [
        x_distal[idx],
        x_zapata[idx],
    ]

    zs_resorte = [
        z_distal[idx],
        z_zapata[idx],
    ]

    plt.plot(xs_pata, zs_pata, marker="o", linewidth=2, label=f"fase {phase[idx]:.2f}")
    plt.plot(xs_resorte, zs_resorte, marker="s", linestyle="--", linewidth=1)

plt.axhline(0, linestyle="--", linewidth=1, label="suelo")

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Órtesis anatómica con zapata y retracción pasiva")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "esquema_mecanismo_final.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 6: CONTACTO DE ZAPATA
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, contacto_zapata.astype(int), label="contacto zapata")
plt.plot(phase, en_apoyo.astype(int), linestyle="--", label="fase apoyo modelada")

plt.xlabel("Fase")
plt.ylabel("Estado")
plt.title("Verificación de contacto de zapata")
plt.yticks([0, 1], ["No", "Sí"])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "verificacion_contacto_zapata.png", dpi=300)
plt.show()