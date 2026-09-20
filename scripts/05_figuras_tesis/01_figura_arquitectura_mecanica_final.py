from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =========================================================
# RUTAS
# =========================================================

RUTA_MODELO = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_fourier_final\cinematica_inversa\modelo_ortesis_anatomica\modelo_ortesis_anatomica.csv"
)

SALIDA = RUTA_MODELO.parent / "figuras_arquitectura_mecanica"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARÁMETROS
# =========================================================

FASE_REPRESENTATIVA = 0.10

DUTY_FACTOR = 0.55

LARGO_ZAPATA_CM = 6.0
ALTO_ZAPATA_CM = 1.2
RADIO_CURVATURA_CM = 8.0

LONGITUD_LIBRE_RESORTE_CM = 2.0
ELEVACION_RETRACCION_CM = 1.2

ANGULO_ZAPATA_MAX_DEG = 15.0
ANGULO_ZAPATA_MAX_RAD = np.radians(ANGULO_ZAPATA_MAX_DEG)

ANCHO_BARRA_VISUAL = 6
RADIO_CADERA = 1.0
RADIO_RODILLA = 0.8
RADIO_DISTAL = 0.7

# =========================================================
# CARGA
# =========================================================

df = pd.read_csv(RUTA_MODELO)

phase = df["phase"].to_numpy()

x_cadera = df["x_cadera_cm"].to_numpy()
z_cadera = df["z_cadera_cm"].to_numpy()

x_rodilla = df["x_rodilla_cm"].to_numpy()
z_rodilla = df["z_rodilla_cm"].to_numpy()

x_distal = df["x_distal_cm"].to_numpy()
z_distal = df["z_distal_cm"].to_numpy()

# =========================================================
# FASE DE APOYO Y ZAPATA
# =========================================================

mitad_apoyo = DUTY_FACTOR / 2
en_apoyo = (phase <= mitad_apoyo) | (phase >= 1 - mitad_apoyo)

inicio_balanceo = mitad_apoyo
fin_balanceo = 1 - mitad_apoyo
duracion_balanceo = fin_balanceo - inicio_balanceo

mask_balanceo = ~en_apoyo

fase_local = np.zeros_like(phase)
fase_local[phase <= mitad_apoyo] = phase[phase <= mitad_apoyo] / mitad_apoyo
fase_local[phase >= 1 - mitad_apoyo] = (
    phase[phase >= 1 - mitad_apoyo] - (1 - mitad_apoyo)
) / mitad_apoyo

fase_balanceo = np.zeros_like(phase)
fase_balanceo[mask_balanceo] = (
    (phase[mask_balanceo] - inicio_balanceo) / duracion_balanceo
)

retraccion_suave = np.zeros_like(phase)
retraccion_suave[mask_balanceo] = (
    ELEVACION_RETRACCION_CM * np.sin(np.pi * fase_balanceo[mask_balanceo])
)

x_zapata = x_distal.copy()

z_zapata_apoyo = np.zeros_like(z_distal)
z_zapata_balanceo = z_distal - LONGITUD_LIBRE_RESORTE_CM + retraccion_suave
z_zapata_balanceo = np.maximum(z_zapata_balanceo, 0.0)
z_zapata = np.where(en_apoyo, z_zapata_apoyo, z_zapata_balanceo)

angulo_zapata = np.zeros_like(phase)
angulo_zapata[en_apoyo] = ANGULO_ZAPATA_MAX_RAD * np.cos(np.pi * fase_local[en_apoyo])
angulo_zapata[mask_balanceo] = (
    -ANGULO_ZAPATA_MAX_RAD * np.cos(np.pi * fase_balanceo[mask_balanceo])
)

# =========================================================
# PERFIL ZAPATA ROCKER
# =========================================================

x_local = np.linspace(-LARGO_ZAPATA_CM / 2, LARGO_ZAPATA_CM / 2, 120)

z_inferior = RADIO_CURVATURA_CM - np.sqrt(
    np.maximum(RADIO_CURVATURA_CM**2 - x_local**2, 0)
)
z_inferior -= np.min(z_inferior)
z_superior = z_inferior + ALTO_ZAPATA_CM

def rotar(x, z, ang):
    xr = x * np.cos(ang) - z * np.sin(ang)
    zr = x * np.sin(ang) + z * np.cos(ang)
    return xr, zr

def dibujar_articulacion(ax, x, z, r, etiqueta):
    circ = Circle((x, z), r, fill=False, linewidth=2)
    ax.add_patch(circ)
    ax.scatter([x], [z], s=40)
    ax.text(x + 0.7, z + 0.7, etiqueta, fontsize=10)

def dibujar_resorte(ax, x1, z1, x2, z2, n=7, amp=0.18):
    dx = x2 - x1
    dz = z2 - z1
    L = np.sqrt(dx**2 + dz**2)
    if L < 1e-6:
        return
    nx = -dz / L
    nz = dx / L
    t = np.linspace(0, 1, 150)
    zig = np.sin(2 * np.pi * n * t)
    xs = x1 + dx * t + amp * zig * nx
    zs = z1 + dz * t + amp * zig * nz
    ax.plot(xs, zs, linewidth=1.5, label="resorte/guía telescópica")

# =========================================================
# FIGURA
# =========================================================

idx = np.argmin(np.abs(phase - FASE_REPRESENTATIVA))

fig, ax = plt.subplots(figsize=(8, 8))

# Trayectoria objetivo
ax.plot(
    x_distal,
    z_distal,
    color="gray",
    alpha=0.35,
    linewidth=2,
    label="trayectoria objetivo"
)

# Barras
ax.plot(
    [x_cadera[idx], x_rodilla[idx]],
    [z_cadera[idx], z_rodilla[idx]],
    linewidth=ANCHO_BARRA_VISUAL,
    solid_capstyle="round",
    label="barra proximal"
)

ax.plot(
    [x_rodilla[idx], x_distal[idx]],
    [z_rodilla[idx], z_distal[idx]],
    linewidth=ANCHO_BARRA_VISUAL,
    solid_capstyle="round",
    label="barra distal"
)

# Articulaciones
dibujar_articulacion(ax, x_cadera[idx], z_cadera[idx], RADIO_CADERA, "cadera")
dibujar_articulacion(ax, x_rodilla[idx], z_rodilla[idx], RADIO_RODILLA, "rodilla")
dibujar_articulacion(ax, x_distal[idx], z_distal[idx], RADIO_DISTAL, "art. distal")

# Guía/resorte
z_union_zapata = z_zapata[idx] + ALTO_ZAPATA_CM

ax.plot(
    [x_distal[idx], x_zapata[idx]],
    [z_distal[idx], z_union_zapata],
    linestyle="--",
    linewidth=1.3,
    color="black",
    label="guía telescópica"
)

dibujar_resorte(
    ax,
    x_distal[idx],
    z_distal[idx],
    x_zapata[idx],
    z_union_zapata
)

# Zapata rocker
xi, zi = rotar(x_local, z_inferior, angulo_zapata[idx])
xs, zs = rotar(x_local, z_superior, angulo_zapata[idx])

ax.plot(x_zapata[idx] + xi, z_zapata[idx] + zi, linewidth=3, label="zapata rocker")
ax.plot(x_zapata[idx] + xs, z_zapata[idx] + zs, linewidth=2)

# Suelo
ax.axhline(0, linestyle="--", linewidth=1.2, label="suelo")

# Etiquetas extra
ax.text(
    0.02,
    0.03,
    f"Fase representativa = {phase[idx]:.2f}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)

ax.set_xlabel("X [cm]")
ax.set_ylabel("Z [cm]")
ax.set_title("Arquitectura mecánica conceptual de la órtesis")
ax.axis("equal")
ax.grid(True)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(SALIDA / "arquitectura_mecanica_conceptual.png", dpi=300)
plt.savefig(SALIDA / "arquitectura_mecanica_conceptual.svg")
plt.show()

print("Figura guardada en:")
print(SALIDA)