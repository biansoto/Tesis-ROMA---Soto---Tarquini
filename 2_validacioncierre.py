from pathlib import Path
import pandas as pd
import numpy as np

RUTA_SALIDA = Path(r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset")

ARCH_PROM_GLOBAL = RUTA_SALIDA / "salidas_caminata_global\promedio_global_caminata.csv"
ARCH_PROM_SUJETO = RUTA_SALIDA / "salidas_caminata_global\promedio_por_sujeto.csv"

# cargar datos
prom_global = pd.read_csv(ARCH_PROM_GLOBAL)
prom_sujeto = pd.read_csv(ARCH_PROM_SUJETO)

# elegir lado
lado = "r"

# --------------------------------------------------
# función de error de cierre
# --------------------------------------------------

def error_cierre(df):

    df = df.sort_values("phase")

    x0 = df.iloc[0]["x_mean"]
    z0 = df.iloc[0]["z_mean"]

    x1 = df.iloc[-1]["x_mean"]
    z1 = df.iloc[-1]["z_mean"]

    err = np.sqrt((x1-x0)**2 + (z1-z0)**2)

    return err, (x0,z0), (x1,z1)


# --------------------------------------------------
# promedio global
# --------------------------------------------------

glob = prom_global[prom_global["lado"] == lado]

err_g, start_g, end_g = error_cierre(glob)

print("\n=== CIERRE PROMEDIO GLOBAL ===")
print("inicio:", start_g)
print("final:", end_g)
print("error:", err_g)

# --------------------------------------------------
# sujeto representativo
# --------------------------------------------------

subject_id = 34

subj = prom_sujeto[
    (prom_sujeto["lado"] == lado) &
    (prom_sujeto["subject_id"] == subject_id)
]

err_s, start_s, end_s = error_cierre(subj)

print("\n=== CIERRE SUJETO REPRESENTATIVO ===")
print("inicio:", start_s)
print("final:", end_s)
print("error:", err_s)