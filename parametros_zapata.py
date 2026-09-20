from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# SALIDA
# =========================================================

SALIDA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_fourier_final\diseno_zapata_rocker"
)
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARÁMETROS GEOMÉTRICOS DE LA ZAPATA
# =========================================================

LARGO_ZAPATA_CM = 6.0
ANCHO_ZAPATA_CM = 2.5
ALTO_ZAPATA_CM = 1.2

RADIO_CURVATURA_CM = 8.0

NPTS = 200

# Punto de unión del resorte respecto al centro de la zapata
X_UNION_RESORTE_CM = 0.0
Z_UNION_RESORTE_CM = ALTO_ZAPATA_CM

# =========================================================
# PERFIL ROCKER 2D
# =========================================================

x = np.linspace(-LARGO_ZAPATA_CM / 2, LARGO_ZAPATA_CM / 2, NPTS)

# Perfil inferior curvo tipo arco.
# Se normaliza para que el punto más bajo esté en z = 0.
z_inferior = RADIO_CURVATURA_CM - np.sqrt(
    np.maximum(RADIO_CURVATURA_CM**2 - x**2, 0)
)

z_inferior = z_inferior - np.min(z_inferior)

# Perfil superior: espesor constante aproximado
z_superior = z_inferior + ALTO_ZAPATA_CM

# =========================================================
# EXPORTAR COORDENADAS
# =========================================================

df_perfil = pd.DataFrame({
    "x_cm": x,
    "z_inferior_cm": z_inferior,
    "z_superior_cm": z_superior,
})

df_perfil.to_csv(
    SALIDA / "perfil_zapata_rocker.csv",
    index=False,
    encoding="utf-8-sig"
)

metricas = {
    "largo_zapata_cm": LARGO_ZAPATA_CM,
    "ancho_zapata_cm": ANCHO_ZAPATA_CM,
    "alto_zapata_cm": ALTO_ZAPATA_CM,
    "radio_curvatura_cm": RADIO_CURVATURA_CM,
    "altura_maxima_curvatura_cm": float(np.max(z_inferior)),
    "x_union_resorte_cm": X_UNION_RESORTE_CM,
    "z_union_resorte_cm": Z_UNION_RESORTE_CM,
}

pd.DataFrame([metricas]).to_csv(
    SALIDA / "metricas_zapata_rocker.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# REPORTE
# =========================================================

print("\n=== GEOMETRÍA ZAPATA ROCKER ===")
for k, v in metricas.items():
    print(f"{k}: {v}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICO 1: PERFIL DE LA ZAPATA
# =========================================================

plt.figure(figsize=(8, 4))

plt.plot(x, z_inferior, linewidth=3, label="Perfil inferior curvo")
plt.plot(x, z_superior, linewidth=2, label="Perfil superior")

plt.scatter(
    [X_UNION_RESORTE_CM],
    [Z_UNION_RESORTE_CM],
    s=80,
    label="Unión resorte"
)

plt.axhline(0, linestyle="--", label="suelo")

plt.xlabel("X local zapata [cm]")
plt.ylabel("Z local zapata [cm]")
plt.title("Perfil preliminar de zapata tipo rocker")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "perfil_zapata_rocker.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 2: COMPARACIÓN DE RADIOS
# =========================================================

radios = [6.0, 8.0, 10.0, 12.0]

plt.figure(figsize=(8, 4))

for R in radios:
    z = R - np.sqrt(np.maximum(R**2 - x**2, 0))
    z = z - np.min(z)
    plt.plot(x, z, label=f"R = {R} cm")

plt.axhline(0, linestyle="--", label="suelo")

plt.xlabel("X local zapata [cm]")
plt.ylabel("Elevación [cm]")
plt.title("Comparación de curvaturas de zapata")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "comparacion_radios_zapata.png", dpi=300)
plt.show()