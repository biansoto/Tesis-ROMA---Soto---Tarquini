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

SALIDA = RUTA_MODELO.parent / "modelo_cad_conceptual_2d"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARÁMETROS GEOMÉTRICOS
# =========================================================

# Barras de órtesis
ANCHO_BARRA_CM = 1.6
RADIO_ARTICULACION_CM = 1.0

# Zapata rocker
LARGO_ZAPATA_CM = 6.0
ALTO_ZAPATA_CM = 1.2
RADIO_CURVATURA_CM = 8.0

# Resorte / guía telescópica
LONGITUD_LIBRE_RESORTE_CM = 2.0
ELEVACION_RETRACCION_CM = 1.2

# Fuerza / apoyo
DUTY_FACTOR = 0.55

# Fases para visualizar
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

# =========================================================
# FASE DE APOYO
# =========================================================

mitad_apoyo = DUTY_FACTOR / 2

en_apoyo = (phase <= mitad_apoyo) | (phase >= 1 - mitad_apoyo)

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

# Centro de zapata
x_zapata = x_distal.copy()

z_zapata_apoyo = np.zeros_like(z_distal)

z_zapata_balanceo = (
    z_distal
    - LONGITUD_LIBRE_RESORTE_CM
    + retraccion_suave
)

z_zapata_balanceo = np.maximum(z_zapata_balanceo, 0.0)

z_zapata = np.where(en_apoyo, z_zapata_apoyo, z_zapata_balanceo)

# =========================================================
# PERFIL LOCAL DE ZAPATA ROCKER
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

def dibujar_barra(ax, x1, z1, x2, z2, ancho, color=None, label=None):
    """
    Dibuja una barra como línea gruesa.
    """
    ax.plot(
        [x1, x2],
        [z1, z2],
        linewidth=ancho * 5,
        solid_capstyle="round",
        color=color,
        alpha=0.7,
        label=label
    )

    ax.plot(
        [x1, x2],
        [z1, z2],
        linewidth=1.5,
        color="black",
        alpha=0.8
    )


def dibujar_articulacion(ax, x, z, radio, label=None):
    circ = plt.Circle(
        (x, z),
        radio,
        fill=False,
        linewidth=2,
        color="black"
    )
    ax.add_patch(circ)
    ax.scatter([x], [z], s=30, color="black")

    if label is not None:
        ax.text(x + 0.5, z + 0.5, label, fontsize=9)


def dibujar_zapata(ax, x_c, z_c, color=None):
    xg = x_c + x_local
    zg_inf = z_c + z_inferior_local
    zg_sup = z_c + z_superior_local

    ax.plot(xg, zg_inf, linewidth=3, color=color)
    ax.plot(xg, zg_sup, linewidth=2, color=color)

    # extremos
    ax.plot(
        [xg[0], xg[0]],
        [zg_inf[0], zg_sup[0]],
        linewidth=2,
        color=color
    )
    ax.plot(
        [xg[-1], xg[-1]],
        [zg_inf[-1], zg_sup[-1]],
        linewidth=2,
        color=color
    )


def dibujar_resorte(ax, x1, z1, x2, z2, n=8, amp=0.25, color="black"):
    """
    Dibuja un resorte simplificado entre dos puntos.
    """
    # Vector
    dx = x2 - x1
    dz = z2 - z1
    L = np.sqrt(dx**2 + dz**2)

    if L < 1e-6:
        return

    ux = dx / L
    uz = dz / L

    # normal
    nx = -uz
    nz = ux

    t = np.linspace(0, 1, 120)
    zig = np.sin(2 * np.pi * n * t)

    xs = x1 + dx * t + amp * zig * nx
    zs = z1 + dz * t + amp * zig * nz

    ax.plot(xs, zs, linewidth=1.5, color=color)


# =========================================================
# GRÁFICO PRINCIPAL: VARIAS FASES
# =========================================================

plt.figure(figsize=(13, 8))
ax = plt.gca()

for fase_obj in FASES_A_GRAFICAR:
    idx = np.argmin(np.abs(phase - fase_obj))

    color = None

    # Barras principales
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
        x_distal[idx],
        z_distal[idx],
        ANCHO_BARRA_CM * 0.85,
        color=color
    )

    # Articulaciones
    dibujar_articulacion(ax, x_cadera[idx], z_cadera[idx], RADIO_ARTICULACION_CM)
    dibujar_articulacion(ax, x_rodilla[idx], z_rodilla[idx], RADIO_ARTICULACION_CM * 0.8)

    # Resorte desde distal hacia parte superior de zapata
    z_union_zapata = z_zapata[idx] + ALTO_ZAPATA_CM

    dibujar_resorte(
        ax,
        x_distal[idx],
        z_distal[idx],
        x_zapata[idx],
        z_union_zapata,
        n=7,
        amp=0.18
    )

    # Guía telescópica externa
    ax.plot(
        [x_distal[idx], x_zapata[idx]],
        [z_distal[idx], z_union_zapata],
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.5
    )

    # Zapata rocker
    dibujar_zapata(ax, x_zapata[idx], z_zapata[idx], color=color)

# Suelo
ax.axhline(0, linestyle="--", linewidth=1.5, label="suelo")

# Trayectoria metatarso
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
ax.set_title("Modelo CAD conceptual 2D de órtesis lateral con zapata rocker")
ax.axis("equal")
ax.grid(True)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(SALIDA / "modelo_cad_conceptual_varias_fases.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO: UNA FASE REPRESENTATIVA DE APOYO
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
    x_distal[idx],
    z_distal[idx],
    ANCHO_BARRA_CM * 0.85,
    label="barra distal"
)

dibujar_articulacion(ax, x_cadera[idx], z_cadera[idx], RADIO_ARTICULACION_CM, "cadera")
dibujar_articulacion(ax, x_rodilla[idx], z_rodilla[idx], RADIO_ARTICULACION_CM * 0.8, "rodilla")

z_union_zapata = z_zapata[idx] + ALTO_ZAPATA_CM

dibujar_resorte(
    ax,
    x_distal[idx],
    z_distal[idx],
    x_zapata[idx],
    z_union_zapata,
    n=7,
    amp=0.18
)

ax.plot(
    [x_distal[idx], x_zapata[idx]],
    [z_distal[idx], z_union_zapata],
    linestyle="--",
    linewidth=1,
    color="black",
    label="guía telescópica"
)

dibujar_zapata(ax, x_zapata[idx], z_zapata[idx])

ax.axhline(0, linestyle="--", linewidth=1.5, label="suelo")

ax.set_xlabel("X [cm]")
ax.set_ylabel("Z [cm]")
ax.set_title(f"Modelo CAD conceptual - fase {phase[idx]:.2f}")
ax.axis("equal")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(SALIDA / "modelo_cad_conceptual_fase_apoyo.png", dpi=300)
plt.show()

# =========================================================
# EXPORTAR PUNTOS CLAVE PARA CAD
# =========================================================

df_cad = pd.DataFrame({
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
})

df_cad.to_csv(
    SALIDA / "puntos_clave_modelo_cad_conceptual.csv",
    index=False,
    encoding="utf-8-sig"
)

metricas = {
    "ancho_barra_cm": ANCHO_BARRA_CM,
    "radio_articulacion_cm": RADIO_ARTICULACION_CM,
    "largo_zapata_cm": LARGO_ZAPATA_CM,
    "alto_zapata_cm": ALTO_ZAPATA_CM,
    "radio_curvatura_zapata_cm": RADIO_CURVATURA_CM,
    "longitud_libre_resorte_cm": LONGITUD_LIBRE_RESORTE_CM,
    "elevacion_retraccion_cm": ELEVACION_RETRACCION_CM,
    "duty_factor": DUTY_FACTOR,
}

pd.DataFrame([metricas]).to_csv(
    SALIDA / "metricas_modelo_cad_conceptual.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== MODELO CAD CONCEPTUAL 2D ===")
for k, v in metricas.items():
    print(f"{k}: {v}")

print(f"\nArchivos guardados en:\n{SALIDA}")