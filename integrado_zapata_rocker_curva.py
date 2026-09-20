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

SALIDA = RUTA_MODELO.parent / "ortesis_con_zapata_rocker"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARÁMETROS FUNCIONALES
# =========================================================

PESO_PERRO_KG = 20.0
G = 9.81

CARGA_PATA_N = PESO_PERRO_KG * G / 4
ASISTENCIA_OBJ_N = 0.15 * CARGA_PATA_N

DUTY_FACTOR = 0.55

K_RESORTE_N_M = 1000.0
K_RESORTE_N_CM = K_RESORTE_N_M / 100.0

LONGITUD_LIBRE_RESORTE_CM = 2.0
COMPRESION_MAX_CM = 2.0
ELEVACION_RETRACCION_CM = 1.2

# =========================================================
# PARÁMETROS ZAPATA ROCKER
# =========================================================

LARGO_ZAPATA_CM = 6.0
ANCHO_ZAPATA_CM = 2.5
ALTO_ZAPATA_CM = 1.2
RADIO_CURVATURA_CM = 8.0

NPTS_ZAPATA = 100

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
# FASE DE APOYO
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

perfil_apoyo = np.sin(np.pi * fase_local)
perfil_apoyo = np.clip(perfil_apoyo, 0, 1)

fuerza_asistiva_N = np.zeros_like(phase)
fuerza_asistiva_N[en_apoyo] = ASISTENCIA_OBJ_N * perfil_apoyo[en_apoyo]

compresion_resorte_cm = fuerza_asistiva_N / K_RESORTE_N_CM
compresion_resorte_cm = np.clip(compresion_resorte_cm, 0, COMPRESION_MAX_CM)

longitud_resorte_cm = LONGITUD_LIBRE_RESORTE_CM - compresion_resorte_cm
longitud_resorte_cm = np.clip(longitud_resorte_cm, 0.2, LONGITUD_LIBRE_RESORTE_CM)

# =========================================================
# CENTRO DE ZAPATA
# =========================================================

x_zapata_centro = x_distal.copy()

inicio_balanceo = mitad_apoyo
fin_balanceo = 1 - mitad_apoyo
duracion_balanceo = fin_balanceo - inicio_balanceo

mask_balanceo = ~en_apoyo

fase_balanceo = np.zeros_like(phase)
fase_balanceo[mask_balanceo] = (
    (phase[mask_balanceo] - inicio_balanceo) / duracion_balanceo
)

retraccion_suave = np.zeros_like(phase)
retraccion_suave[mask_balanceo] = (
    ELEVACION_RETRACCION_CM * np.sin(np.pi * fase_balanceo[mask_balanceo])
)

# En apoyo: la base inferior toca el suelo.
z_zapata_centro_apoyo = np.zeros_like(z_distal)

# En balanceo: se retrae.
z_zapata_centro_balanceo = (
    z_distal
    - LONGITUD_LIBRE_RESORTE_CM
    + retraccion_suave
)

z_zapata_centro_balanceo = np.maximum(z_zapata_centro_balanceo, 0.0)

z_zapata_centro = np.where(
    en_apoyo,
    z_zapata_centro_apoyo,
    z_zapata_centro_balanceo
)

# =========================================================
# PERFIL LOCAL DE ZAPATA ROCKER
# =========================================================

x_local = np.linspace(
    -LARGO_ZAPATA_CM / 2,
    LARGO_ZAPATA_CM / 2,
    NPTS_ZAPATA
)

z_inferior_local = RADIO_CURVATURA_CM - np.sqrt(
    np.maximum(RADIO_CURVATURA_CM**2 - x_local**2, 0)
)

z_inferior_local = z_inferior_local - np.min(z_inferior_local)

z_superior_local = z_inferior_local + ALTO_ZAPATA_CM

# =========================================================
# EXPORTAR PERFIL Y MÉTRICAS
# =========================================================

df_perfil = pd.DataFrame({
    "x_local_cm": x_local,
    "z_inferior_local_cm": z_inferior_local,
    "z_superior_local_cm": z_superior_local,
})

df_perfil.to_csv(
    SALIDA / "perfil_zapata_rocker.csv",
    index=False,
    encoding="utf-8-sig"
)

contacto_zapata = z_zapata_centro <= 0.05
balanceo_con_contacto = (~en_apoyo) & contacto_zapata

metricas = {
    "duty_factor": DUTY_FACTOR,
    "porcentaje_fase_apoyo_modelada": 100 * np.sum(en_apoyo) / len(en_apoyo),
    "k_resorte_N_m": K_RESORTE_N_M,
    "fuerza_objetivo_N": ASISTENCIA_OBJ_N,
    "compresion_max_cm": float(np.max(compresion_resorte_cm)),
    "longitud_resorte_min_cm": float(np.min(longitud_resorte_cm)),
    "longitud_resorte_max_cm": float(np.max(longitud_resorte_cm)),
    "largo_zapata_cm": LARGO_ZAPATA_CM,
    "ancho_zapata_cm": ANCHO_ZAPATA_CM,
    "alto_zapata_cm": ALTO_ZAPATA_CM,
    "radio_curvatura_cm": RADIO_CURVATURA_CM,
    "altura_maxima_curvatura_cm": float(np.max(z_inferior_local)),
    "porcentaje_arrastre_balanceo": 100 * np.sum(balanceo_con_contacto) / np.sum(~en_apoyo),
}

pd.DataFrame([metricas]).to_csv(
    SALIDA / "metricas_ortesis_zapata_rocker.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== ÓRTESIS CON ZAPATA ROCKER ===")
for k, v in metricas.items():
    print(f"{k}: {v}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICO 1: TRAYECTORIA METATARSO Y CENTRO DE ZAPATA
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(x_distal, z_distal, linewidth=3, label="Metatarso")
plt.plot(x_zapata_centro, z_zapata_centro, linewidth=2, label="Centro zapata")
plt.axhline(0, linestyle="--", label="suelo")

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Trayectoria del metatarso y centro de zapata rocker")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "trayectoria_metatarso_centro_zapata_rocker.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 2: ESQUEMA COMPLETO CON ZAPATA ROCKER
# =========================================================

FASES_A_GRAFICAR = [0.00, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]

plt.figure(figsize=(12, 7))

for fase_obj in FASES_A_GRAFICAR:
    idx = np.argmin(np.abs(phase - fase_obj))

    # Pata / órtesis
    xs_pata = [x_cadera[idx], x_rodilla[idx], x_distal[idx]]
    zs_pata = [z_cadera[idx], z_rodilla[idx], z_distal[idx]]

    plt.plot(
        xs_pata,
        zs_pata,
        marker="o",
        linewidth=2,
        label=f"fase {phase[idx]:.2f}"
    )

    # Resorte telescópico
    plt.plot(
        [x_distal[idx], x_zapata_centro[idx]],
        [z_distal[idx], z_zapata_centro[idx] + ALTO_ZAPATA_CM],
        linestyle="--",
        linewidth=1
    )

    # Zapata rocker: trasladar perfil local al centro global
    x_global = x_zapata_centro[idx] + x_local

    # En apoyo, el punto más bajo toca el suelo.
    # En balanceo, el perfil se eleva con el centro.
    z_inf_global = z_zapata_centro[idx] + z_inferior_local
    z_sup_global = z_zapata_centro[idx] + z_superior_local

    plt.plot(x_global, z_inf_global, linewidth=3)
    plt.plot(x_global, z_sup_global, linewidth=1)

plt.axhline(0, linestyle="--", linewidth=1, label="suelo")

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Órtesis anatómica con zapata rocker en distintas fases")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "esquema_ortesis_zapata_rocker_fases.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 3: FUERZA ASISTIVA
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, fuerza_asistiva_N, label="Fuerza asistiva")
plt.axhline(ASISTENCIA_OBJ_N, linestyle=":", label="objetivo 15%")

plt.xlabel("Fase")
plt.ylabel("Fuerza [N]")
plt.title("Fuerza asistiva con zapata rocker")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "fuerza_asistiva_zapata_rocker.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 4: COMPRESIÓN DEL RESORTE
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, compresion_resorte_cm)

plt.xlabel("Fase")
plt.ylabel("Compresión [cm]")
plt.title("Compresión del resorte con zapata rocker")
plt.grid(True)
plt.tight_layout()
plt.savefig(SALIDA / "compresion_resorte_zapata_rocker.png", dpi=300)
plt.show()