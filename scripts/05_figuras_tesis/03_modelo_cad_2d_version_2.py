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

SALIDA = RUTA_MODELO.parent / "modelo_opcion_C_pie_pasivo"
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

# =========================================================
# PARÁMETROS MECÁNICOS
# =========================================================

ANCHO_BARRA_CM = 1.6
RADIO_ARTICULACION_CM = 1.0

# Articulación distal pasiva
RADIO_ARTICULACION_DISTAL_CM = 0.7

# Zapata rocker
LARGO_ZAPATA_CM = 6.0
ANCHO_ZAPATA_CM = 2.5
ALTO_ZAPATA_CM = 1.2
RADIO_CURVATURA_CM = 8.0

# Rotación pasiva de zapata
ANGULO_ZAPATA_MAX_DEG = 15.0
ANGULO_ZAPATA_MAX_RAD = np.radians(ANGULO_ZAPATA_MAX_DEG)

# Retracción en balanceo
ELEVACION_RETRACCION_CM = 1.2

# Fases a dibujar
FASES_A_GRAFICAR = [0.00, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90]

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

# Fuerza suave: 0 en bordes, máxima en mitad del apoyo
perfil_apoyo = np.sin(np.pi * fase_local)
perfil_apoyo = np.clip(perfil_apoyo, 0, 1)

fuerza_asistiva_N = np.zeros_like(phase)
fuerza_asistiva_N[en_apoyo] = ASISTENCIA_OBJ_N * perfil_apoyo[en_apoyo]

compresion_resorte_cm = fuerza_asistiva_N / K_RESORTE_N_CM
compresion_resorte_cm = np.clip(compresion_resorte_cm, 0, COMPRESION_MAX_CM)

longitud_resorte_cm = LONGITUD_LIBRE_RESORTE_CM - compresion_resorte_cm
longitud_resorte_cm = np.clip(longitud_resorte_cm, 0.2, LONGITUD_LIBRE_RESORTE_CM)

# =========================================================
# ARTICULACIÓN DISTAL PASIVA
# =========================================================

# En este modelo, la articulación distal coincide inicialmente con el punto distal.
x_art_distal = x_distal.copy()
z_art_distal = z_distal.copy()

# =========================================================
# POSICIÓN DEL CENTRO DE ZAPATA
# =========================================================

x_zapata_centro = x_art_distal.copy()

# En apoyo, la zapata toca suelo.
z_zapata_centro_apoyo = np.zeros_like(z_art_distal)

# En balanceo, la zapata se retrae hacia arriba.
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

z_zapata_centro_balanceo = (
    z_art_distal
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
# ROTACIÓN PASIVA DE LA ZAPATA
# =========================================================

# =========================================================
# ROTACIÓN PASIVA DE LA ZAPATA CON RETORNO SUAVE
# =========================================================

angulo_zapata_rad = np.zeros_like(phase)

# En apoyo: rota suavemente de +max a -max, simulando rodadura.
angulo_zapata_rad[en_apoyo] = (
    ANGULO_ZAPATA_MAX_RAD *
    np.cos(np.pi * fase_local[en_apoyo])
)

# En balanceo: vuelve suavemente desde -max a +max pasando por 0.
# Esto evita el salto brusco de -15° a 0°.
fase_balanceo_rot = np.zeros_like(phase)

fase_balanceo_rot[mask_balanceo] = (
    (phase[mask_balanceo] - inicio_balanceo) / duracion_balanceo
)

angulo_zapata_rad[mask_balanceo] = (
    -ANGULO_ZAPATA_MAX_RAD *
    np.cos(np.pi * fase_balanceo_rot[mask_balanceo])
)

angulo_zapata_deg = np.degrees(angulo_zapata_rad)

# =========================================================
# PERFIL LOCAL ZAPATA ROCKER
# =========================================================

NPTS_ZAPATA = 100

x_local = np.linspace(
    -LARGO_ZAPATA_CM / 2,
    LARGO_ZAPATA_CM / 2,
    NPTS_ZAPATA
)

z_inferior_local = RADIO_CURVATURA_CM - np.sqrt(
    np.maximum(RADIO_CURVATURA_CM**2 - x_local**2, 0)
)

z_inferior_local -= np.min(z_inferior_local)

z_superior_local = z_inferior_local + ALTO_ZAPATA_CM

# =========================================================
# FUNCIONES DE DIBUJO
# =========================================================

def rotar_puntos(x, z, ang):
    xr = x * np.cos(ang) - z * np.sin(ang)
    zr = x * np.sin(ang) + z * np.cos(ang)
    return xr, zr


def dibujar_barra(ax, x1, z1, x2, z2, ancho, color=None, label=None):
    ax.plot(
        [x1, x2],
        [z1, z2],
        linewidth=ancho * 5,
        solid_capstyle="round",
        color=color,
        alpha=0.7,
        label=label
    )
    ax.plot([x1, x2], [z1, z2], linewidth=1.5, color="black", alpha=0.8)


def dibujar_articulacion(ax, x, z, radio, label=None):
    circ = plt.Circle((x, z), radio, fill=False, linewidth=2, color="black")
    ax.add_patch(circ)
    ax.scatter([x], [z], s=30, color="black")
    if label:
        ax.text(x + 0.5, z + 0.5, label, fontsize=9)


def dibujar_resorte(ax, x1, z1, x2, z2, n=8, amp=0.18, color="black"):
    dx = x2 - x1
    dz = z2 - z1
    L = np.sqrt(dx**2 + dz**2)

    if L < 1e-6:
        return

    ux = dx / L
    uz = dz / L

    nx = -uz
    nz = ux

    t = np.linspace(0, 1, 120)
    zig = np.sin(2 * np.pi * n * t)

    xs = x1 + dx * t + amp * zig * nx
    zs = z1 + dz * t + amp * zig * nz

    ax.plot(xs, zs, linewidth=1.5, color=color)


def dibujar_zapata_rotada(ax, x_c, z_c, ang, color=None):
    # Perfil inferior
    xi, zi = rotar_puntos(x_local, z_inferior_local, ang)
    xs, zs = rotar_puntos(x_local, z_superior_local, ang)

    ax.plot(x_c + xi, z_c + zi, linewidth=3, color=color)
    ax.plot(x_c + xs, z_c + zs, linewidth=2, color=color)

    # extremos
    ax.plot(
        [x_c + xi[0], x_c + xs[0]],
        [z_c + zi[0], z_c + zs[0]],
        linewidth=2,
        color=color
    )
    ax.plot(
        [x_c + xi[-1], x_c + xs[-1]],
        [z_c + zi[-1], z_c + zs[-1]],
        linewidth=2,
        color=color
    )

# =========================================================
# VALIDACIONES SIMPLES
# =========================================================

contacto_zapata = z_zapata_centro <= 0.05
balanceo_con_contacto = (~en_apoyo) & contacto_zapata

porcentaje_arrastre_balanceo = (
    100 * np.sum(balanceo_con_contacto) / np.sum(~en_apoyo)
)

# =========================================================
# EXPORTAR
# =========================================================

df_salida = pd.DataFrame({
    "phase": phase,

    "x_cadera_cm": x_cadera,
    "z_cadera_cm": z_cadera,

    "x_rodilla_cm": x_rodilla,
    "z_rodilla_cm": z_rodilla,

    "x_art_distal_cm": x_art_distal,
    "z_art_distal_cm": z_art_distal,

    "x_zapata_centro_cm": x_zapata_centro,
    "z_zapata_centro_cm": z_zapata_centro,

    "angulo_zapata_deg": angulo_zapata_deg,

    "en_apoyo": en_apoyo,

    "fuerza_asistiva_N": fuerza_asistiva_N,
    "compresion_resorte_cm": compresion_resorte_cm,
    "longitud_resorte_cm": longitud_resorte_cm,
})

df_salida.to_csv(
    SALIDA / "modelo_opcion_C_pie_pasivo.csv",
    index=False,
    encoding="utf-8-sig"
)

metricas = {
    "duty_factor": DUTY_FACTOR,
    "porcentaje_fase_apoyo_modelada": 100 * np.sum(en_apoyo) / len(en_apoyo),
    "k_resorte_N_m": K_RESORTE_N_M,
    "fuerza_objetivo_N": ASISTENCIA_OBJ_N,
    "fuerza_max_N": float(np.max(fuerza_asistiva_N)),
    "fuerza_media_apoyo_N": float(np.mean(fuerza_asistiva_N[en_apoyo])),
    "compresion_max_cm": float(np.max(compresion_resorte_cm)),
    "longitud_resorte_min_cm": float(np.min(longitud_resorte_cm)),
    "longitud_resorte_max_cm": float(np.max(longitud_resorte_cm)),
    "angulo_zapata_max_deg": ANGULO_ZAPATA_MAX_DEG,
    "angulo_zapata_min_real_deg": float(np.min(angulo_zapata_deg)),
    "angulo_zapata_max_real_deg": float(np.max(angulo_zapata_deg)),
    "largo_zapata_cm": LARGO_ZAPATA_CM,
    "ancho_zapata_cm": ANCHO_ZAPATA_CM,
    "alto_zapata_cm": ALTO_ZAPATA_CM,
    "radio_curvatura_cm": RADIO_CURVATURA_CM,
    "altura_max_curvatura_cm": float(np.max(z_inferior_local)),
    "porcentaje_arrastre_balanceo": porcentaje_arrastre_balanceo,
}

pd.DataFrame([metricas]).to_csv(
    SALIDA / "metricas_modelo_opcion_C.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== MODELO OPCIÓN C: PIE PASIVO ARTICULADO ===")
for k, v in metricas.items():
    print(f"{k}: {v}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICO 1: MODELO EN VARIAS FASES
# =========================================================

plt.figure(figsize=(13, 8))
ax = plt.gca()

for fase_obj in FASES_A_GRAFICAR:
    idx = np.argmin(np.abs(phase - fase_obj))

    color = None

    dibujar_barra(
        ax,
        x_cadera[idx],
        z_cadera[idx],
        x_rodilla[idx],
        z_rodilla[idx],
        ANCHO_BARRA_CM,
        color=color,
        label=f"fase {phase[idx]:.2f}"
    )

    dibujar_barra(
        ax,
        x_rodilla[idx],
        z_rodilla[idx],
        x_art_distal[idx],
        z_art_distal[idx],
        ANCHO_BARRA_CM * 0.85,
        color=color
    )

    dibujar_articulacion(ax, x_cadera[idx], z_cadera[idx], RADIO_ARTICULACION_CM)
    dibujar_articulacion(ax, x_rodilla[idx], z_rodilla[idx], RADIO_ARTICULACION_CM * 0.8)
    dibujar_articulacion(ax, x_art_distal[idx], z_art_distal[idx], RADIO_ARTICULACION_DISTAL_CM)

    # Resorte hacia parte superior de zapata
    z_union_zapata = z_zapata_centro[idx] + ALTO_ZAPATA_CM

    dibujar_resorte(
        ax,
        x_art_distal[idx],
        z_art_distal[idx],
        x_zapata_centro[idx],
        z_union_zapata,
        n=7,
        amp=0.18
    )

    ax.plot(
        [x_art_distal[idx], x_zapata_centro[idx]],
        [z_art_distal[idx], z_union_zapata],
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.5
    )

    dibujar_zapata_rotada(
        ax,
        x_zapata_centro[idx],
        z_zapata_centro[idx],
        angulo_zapata_rad[idx],
        color=color
    )

ax.axhline(0, linestyle="--", linewidth=1.5, label="suelo")

ax.plot(
    x_distal,
    z_distal,
    linewidth=2,
    color="gray",
    alpha=0.5,
    label="trayectoria metatarso"
)

ax.set_xlabel("X [cm]")
ax.set_ylabel("Z [cm]")
ax.set_title("Opción C: órtesis con pie pasivo articulado y zapata rocker")
ax.axis("equal")
ax.grid(True)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(SALIDA / "modelo_opcion_C_varias_fases.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 2: FASE REPRESENTATIVA
# =========================================================

FASE_REPRESENTATIVA = 0.10
idx = np.argmin(np.abs(phase - FASE_REPRESENTATIVA))

plt.figure(figsize=(8, 8))
ax = plt.gca()

dibujar_barra(
    ax,
    x_cadera[idx],
    z_cadera[idx],
    x_rodilla[idx],
    z_rodilla[idx],
    ANCHO_BARRA_CM,
    label="barra proximal"
)

dibujar_barra(
    ax,
    x_rodilla[idx],
    z_rodilla[idx],
    x_art_distal[idx],
    z_art_distal[idx],
    ANCHO_BARRA_CM * 0.85,
    label="barra distal"
)

dibujar_articulacion(ax, x_cadera[idx], z_cadera[idx], RADIO_ARTICULACION_CM, "cadera")
dibujar_articulacion(ax, x_rodilla[idx], z_rodilla[idx], RADIO_ARTICULACION_CM * 0.8, "rodilla")
dibujar_articulacion(ax, x_art_distal[idx], z_art_distal[idx], RADIO_ARTICULACION_DISTAL_CM, "art. distal")

z_union_zapata = z_zapata_centro[idx] + ALTO_ZAPATA_CM

dibujar_resorte(
    ax,
    x_art_distal[idx],
    z_art_distal[idx],
    x_zapata_centro[idx],
    z_union_zapata,
    n=7,
    amp=0.18
)

ax.plot(
    [x_art_distal[idx], x_zapata_centro[idx]],
    [z_art_distal[idx], z_union_zapata],
    linestyle="--",
    linewidth=1,
    color="black",
    label="guía telescópica"
)

dibujar_zapata_rotada(
    ax,
    x_zapata_centro[idx],
    z_zapata_centro[idx],
    angulo_zapata_rad[idx]
)

ax.axhline(0, linestyle="--", linewidth=1.5, label="suelo")

ax.set_xlabel("X [cm]")
ax.set_ylabel("Z [cm]")
ax.set_title(f"Opción C - fase {phase[idx]:.2f}")
ax.axis("equal")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(SALIDA / "modelo_opcion_C_fase_apoyo.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 3: ÁNGULO DE ZAPATA
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, angulo_zapata_deg)
plt.axhline(ANGULO_ZAPATA_MAX_DEG, linestyle="--")
plt.axhline(-ANGULO_ZAPATA_MAX_DEG, linestyle="--")

plt.xlabel("Fase")
plt.ylabel("Ángulo zapata [deg]")
plt.title("Rotación pasiva de la zapata")
plt.grid(True)
plt.tight_layout()
plt.savefig(SALIDA / "angulo_zapata_opcion_C.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 4: FUERZA Y COMPRESIÓN
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, fuerza_asistiva_N, label="Fuerza asistiva [N]")
plt.plot(phase, compresion_resorte_cm, label="Compresión [cm]")

plt.xlabel("Fase")
plt.title("Fuerza asistiva y compresión del resorte")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "fuerza_compresion_opcion_C.png", dpi=300)
plt.show()
