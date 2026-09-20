from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
RUTA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_caminata_global"
)

ARCH = RUTA / "ciclos_caminata_normalizados.csv"

RUTA_SALIDA = RUTA / "seleccion_biomecanica"
RUTA_SALIDA.mkdir(exist_ok=True)

# valores de referencia (bibliografía)
APOYO_OBJETIVO = 0.62
TOLERANCIA = 0.10   # ±10%

TOP_K = 6

# =========================================================
# FUNCIONES
# =========================================================
def detectar_apoyo(x, z):

    # derivadas
    dz = np.gradient(z)

    z_min = np.min(z)
    z_max = np.max(z)

    amp = z_max - z_min

    # umbral de suelo
    umbral_z = z_min + 0.25 * amp

    # velocidad baja
    umbral_dz = np.percentile(np.abs(dz), 40)

    mask = (z <= umbral_z) & (np.abs(dz) <= umbral_dz)

    return mask


def calcular_fase_apoyo(mask):
    return np.sum(mask) / len(mask)


def score_biomecanico(fase_apoyo):
    return abs(fase_apoyo - APOYO_OBJETIVO)


# =========================================================
# CARGA
# =========================================================
df = pd.read_csv(ARCH)

# solo lado derecho
df = df[df["lado"] == "r"].copy()

# =========================================================
# PROMEDIO POR SUJETO
# =========================================================
g = df.groupby(["subject_id", "phase"])[["x_norm", "z_norm"]]

prom = g.mean().rename(columns={"x_norm": "x", "z_norm": "z"}).reset_index()

# =========================================================
# ANALISIS POR SUJETO
# =========================================================
resultados = []

for subj, gsub in prom.groupby("subject_id"):

    gsub = gsub.sort_values("phase")

    x = gsub["x"].values
    z = gsub["z"].values

    mask = detectar_apoyo(x, z)

    fase_apoyo = calcular_fase_apoyo(mask)
    fase_vuelo = 1 - fase_apoyo

    score = score_biomecanico(fase_apoyo)

    resultados.append({
        "subject_id": subj,
        "fase_apoyo": fase_apoyo,
        "fase_vuelo": fase_vuelo,
        "error_vs_biblio": score
    })

df_res = pd.DataFrame(resultados)

# ordenar por mejor ajuste
df_res = df_res.sort_values("error_vs_biblio").reset_index(drop=True)

df_res.to_csv(RUTA_SALIDA / "ranking_biomecanico.csv", index=False)

# =========================================================
# SELECCION
# =========================================================
mejores = df_res.head(TOP_K)["subject_id"].tolist()

print("\n=== MEJORES SUJETOS SEGÚN BIOMECÁNICA ===")
print(df_res.head(10))

print("\nSujetos seleccionados:", mejores)

# =========================================================
# PROMEDIO FINAL
# =========================================================
df_sel = prom[prom["subject_id"].isin(mejores)]

prom_final = (
    df_sel.groupby("phase")[["x", "z"]]
    .mean()
    .reset_index()
)

prom_final.to_csv(RUTA_SALIDA / "curva_promedio_biomecanico.csv", index=False)

# =========================================================
# GRAFICOS
# =========================================================
plt.figure(figsize=(7,6))

for subj in mejores:
    gsub = prom[prom["subject_id"] == subj]
    plt.plot(gsub["x"], gsub["z"], alpha=0.3)

plt.plot(prom_final["x"], prom_final["z"], color="black", linewidth=3, label="Promedio final")

plt.axis("equal")
plt.grid()
plt.title("Curvas seleccionadas por biomecánica")
plt.legend()

plt.show()

# fase apoyo promedio final
mask_final = detectar_apoyo(prom_final["x"].values, prom_final["z"].values)
fase_final = calcular_fase_apoyo(mask_final)

print("\n=== RESULTADO FINAL ===")
print(f"Fase de apoyo final: {fase_final:.3f}")
print(f"Error vs bibliografía: {abs(fase_final - APOYO_OBJETIVO):.3f}")