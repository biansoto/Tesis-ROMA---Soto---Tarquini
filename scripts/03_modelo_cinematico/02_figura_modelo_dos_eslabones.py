from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# =========================================================
# RUTAS
# =========================================================

RUTA_CINEMATICA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_fourier_final\cinematica_inversa\cinematica_inversa_fourier.csv"
)

SALIDA = RUTA_CINEMATICA.parent / "figuras_modelo_dos_eslabones"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARÁMETROS ANATÓMICOS [cm]
# =========================================================

L1 = 17.0      # fémur
L2 = 36.5      # tibia + pie distal
ALTURA_CADERA = 43.0

# Fase representativa para mostrar el mecanismo
FASE_REPRESENTATIVA = 0.15

# =========================================================
# CARGA DE DATOS
# =========================================================

df = pd.read_csv(RUTA_CINEMATICA)

phase = df["phase"].to_numpy()
x_pie = df["x_pie_cm"].to_numpy()
z_pie = df["z_pie_cm"].to_numpy()

theta_cadera_rad = df["theta_cadera_rad"].to_numpy()
theta_rodilla_rad = df["theta_rodilla_rad"].to_numpy()

theta_cadera_deg = df["theta_cadera_deg"].to_numpy()
theta_rodilla_deg = df["theta_rodilla_deg"].to_numpy()

# =========================================================
# SELECCIÓN DE FASE REPRESENTATIVA
# =========================================================

idx = np.argmin(np.abs(phase - FASE_REPRESENTATIVA))

hip = theta_cadera_rad[idx]
knee = theta_rodilla_rad[idx]

# Cadera fija
x_cadera = 0.0
z_cadera = ALTURA_CADERA

# Reconstrucción del modelo de dos eslabones
# El ángulo hip está calculado respecto del eje X global.
x_rodilla = x_cadera + L1 * np.cos(hip)
z_rodilla = z_cadera + L1 * np.sin(hip)

x_distal = x_pie[idx]
z_distal = z_pie[idx]

# =========================================================
# FUNCIÓN PARA DIBUJAR ARCOS ANGULARES
# =========================================================

def dibujar_arco(ax, centro, radio, ang1, ang2, texto, offset=(0, 0)):
    arc = Arc(
        centro,
        2 * radio,
        2 * radio,
        angle=0,
        theta1=np.degrees(ang1),
        theta2=np.degrees(ang2),
        linewidth=1.5
    )
    ax.add_patch(arc)

    ang_mid = (ang1 + ang2) / 2
    x_text = centro[0] + (radio + offset[0]) * np.cos(ang_mid)
    z_text = centro[1] + (radio + offset[1]) * np.sin(ang_mid)

    ax.text(
        x_text,
        z_text,
        texto,
        fontsize=11,
        ha="center",
        va="center"
    )

# =========================================================
# FIGURA PRINCIPAL
# =========================================================

plt.figure(figsize=(8, 7))
ax = plt.gca()

# Trayectoria objetivo completa
ax.plot(
    x_pie,
    z_pie,
    linewidth=2.5,
    alpha=0.6,
    label="Trayectoria objetivo del metatarso"
)

# Modelo de dos eslabones
ax.plot(
    [x_cadera, x_rodilla],
    [z_cadera, z_rodilla],
    marker="o",
    linewidth=4,
    label=r"$L_1$ fémur"
)

ax.plot(
    [x_rodilla, x_distal],
    [z_rodilla, z_distal],
    marker="o",
    linewidth=4,
    label=r"$L_2$ segmento distal"
)

# Puntos anatómicos
ax.scatter([x_cadera], [z_cadera], s=90, zorder=5)
ax.scatter([x_rodilla], [z_rodilla], s=90, zorder=5)
ax.scatter([x_distal], [z_distal], s=90, zorder=5)

ax.text(x_cadera + 0.8, z_cadera + 0.8, "Cadera", fontsize=11)
ax.text(x_rodilla + 0.8, z_rodilla + 0.8, "Rodilla", fontsize=11)
ax.text(x_distal + 0.8, z_distal + 0.8, "Metatarso", fontsize=11)

# Etiquetas de longitudes
x_mid_L1 = (x_cadera + x_rodilla) / 2
z_mid_L1 = (z_cadera + z_rodilla) / 2
ax.text(x_mid_L1 - 2.0, z_mid_L1, r"$L_1$", fontsize=13)

x_mid_L2 = (x_rodilla + x_distal) / 2
z_mid_L2 = (z_rodilla + z_distal) / 2
ax.text(x_mid_L2 + 1.0, z_mid_L2, r"$L_2$", fontsize=13)

# Arco de ángulo de cadera
# Se muestra respecto al eje horizontal negativo/positivo según visualización.
dibujar_arco(
    ax,
    (x_cadera, z_cadera),
    radio=5.0,
    ang1=hip,
    ang2=0,
    texto=r"$\theta_{cadera}$"
)

# Arco de rodilla
# Dirección del segmento distal respecto de la rodilla
ang_distal = np.arctan2(z_distal - z_rodilla, x_distal - x_rodilla)
ang_femur = np.arctan2(z_cadera - z_rodilla, x_cadera - x_rodilla)

dibujar_arco(
    ax,
    (x_rodilla, z_rodilla),
    radio=4.0,
    ang1=ang_distal,
    ang2=ang_femur,
    texto=r"$\theta_{rodilla}$"
)

# Suelo
ax.axhline(0, linestyle="--", linewidth=1.2, label="Suelo")

# Texto con fase
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
ax.set_title("Modelo cinemático simplificado de dos eslabones")
ax.axis("equal")
ax.grid(True)
ax.legend(loc="upper right")
plt.tight_layout()

plt.savefig(SALIDA / "modelo_cinematico_dos_eslabones.png", dpi=300)
plt.savefig(SALIDA / "modelo_cinematico_dos_eslabones.svg")

plt.show()

# =========================================================
# FIGURA COMPLEMENTARIA: ÁNGULOS VS FASE
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, theta_cadera_deg, label=r"$\theta_{cadera}$")
plt.plot(phase, theta_rodilla_deg, label=r"$\theta_{rodilla}$")

plt.xlabel("Fase del ciclo")
plt.ylabel("Ángulo [°]")
plt.title("Ángulos articulares requeridos por cinemática inversa")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(SALIDA / "angulos_articulares_vs_fase.png", dpi=300)
plt.savefig(SALIDA / "angulos_articulares_vs_fase.svg")

plt.show()

print("\nFiguras guardadas en:")
print(SALIDA)