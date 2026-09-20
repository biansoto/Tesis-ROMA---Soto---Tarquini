from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# RUTAS
# =========================================================

RUTA_OBJETIVO = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_procesamiento_final\seleccion_mejor_trial\trayectoria_objetivo_mejor_trial.csv"
)

RUTA_JANSEN = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\kinovea fragmentos\Jansen Linkage HD from Linkage Program.xlsx"
)

SALIDA = RUTA_OBJETIVO.parent / "comparacion_objetivo_jansen"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# FUNCIONES
# =========================================================

def normalizar_forma(x, z):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    x = x - np.mean(x)
    z = z - np.mean(z)

    escala = max(x.max() - x.min(), z.max() - z.min())

    x = x / escala
    z = z / escala

    return x, z


def reinterpolar(x, z, npts):
    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, npts)

    x_new = np.interp(t_new, t_old, x)
    z_new = np.interp(t_new, t_old, z)

    return t_new, x_new, z_new


def alinear_por_fase(x_ref, z_ref, x_mov, z_mov):
    mejor_rms = np.inf

    for k in range(len(x_ref)):
        xs = np.roll(x_mov, k)
        zs = np.roll(z_mov, k)

        error = np.sqrt((x_ref - xs) ** 2 + (z_ref - zs) ** 2)
        rms = np.sqrt(np.mean(error ** 2))

        if rms < mejor_rms:
            mejor_rms = rms
            mejor_shift = k
            mejor_x = xs
            mejor_z = zs
            mejor_error = error

    return mejor_x, mejor_z, mejor_error, mejor_shift, mejor_rms


# =========================================================
# CARGAR CURVA OBJETIVO
# =========================================================

df_obj = pd.read_csv(RUTA_OBJETIVO)

df_obj = df_obj.sort_values("phase").reset_index(drop=True)

phase_obj = df_obj["phase"].to_numpy()
x_obj = df_obj["x_norm"].to_numpy()
z_obj = df_obj["z_norm"].to_numpy()

# =========================================================
# CARGAR THEO JANSEN
# =========================================================

df_jansen = pd.read_excel(RUTA_JANSEN, header=3)

df_jansen = df_jansen.dropna(axis=1, how="all")
df_jansen = df_jansen.iloc[:, :4].copy()

df_jansen.columns = ["time", "x", "y_raw", "y_corr"]

for c in df_jansen.columns:
    df_jansen[c] = pd.to_numeric(df_jansen[c], errors="coerce")

df_jansen = df_jansen.dropna().reset_index(drop=True)

x_jansen = df_jansen["x"].to_numpy()
z_jansen = df_jansen["y_corr"].to_numpy()

# recortar a un ciclo (ejemplo simple)
x_jansen = x_jansen[:len(x_jansen)//2]
z_jansen = z_jansen[:len(z_jansen)//2]

# =========================================================
# REINTERPOLAR JANSEN AL MISMO NÚMERO DE PUNTOS
# =========================================================

N = len(x_obj)

_, x_jansen, z_jansen = reinterpolar(x_jansen, z_jansen, N)

# =========================================================
# NORMALIZACIÓN DE FORMA
# =========================================================

x_obj_n, z_obj_n = normalizar_forma(x_obj, z_obj)
x_jansen_n, z_jansen_n = normalizar_forma(x_jansen, z_jansen)

# =========================================================
# ALINEACIÓN DE FASE
# =========================================================

x_jansen_al, z_jansen_al, error, shift, rms = alinear_por_fase(
    x_obj_n,
    z_obj_n,
    x_jansen_n,
    z_jansen_n
)

# =========================================================
# MÉTRICAS
# =========================================================

print("\n=== COMPARACIÓN OBJETIVO VS THEO JANSEN ===")
print(f"shift óptimo: {shift}")
print(f"error medio: {np.mean(error):.6f}")
print(f"error RMS: {rms:.6f}")
print(f"error máximo: {np.max(error):.6f}")

# =========================================================
# GUARDAR RESULTADOS
# =========================================================

df_out = pd.DataFrame({
    "phase": phase_obj,
    "x_objetivo": x_obj_n,
    "z_objetivo": z_obj_n,
    "x_jansen": x_jansen_al,
    "z_jansen": z_jansen_al,
    "error": error
})

df_out.to_csv(
    SALIDA / "comparacion_objetivo_jansen.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# GRÁFICOS
# =========================================================

plt.figure(figsize=(7, 6))
plt.plot(x_obj_n, z_obj_n, linewidth=3, label="Trayectoria objetivo")
plt.plot(x_jansen_al, z_jansen_al, linewidth=3, label="Theo Jansen")
plt.xlabel("X normalizada")
plt.ylabel("Z normalizada")
plt.title("Comparación trayectoria objetivo vs Theo Jansen")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase_obj, error)
plt.xlabel("Fase")
plt.ylabel("Error")
plt.title("Error punto a punto")
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(phase_obj, x_obj_n, label="X objetivo")
plt.plot(phase_obj, x_jansen_al, label="X Jansen")
plt.plot(phase_obj, z_obj_n, label="Z objetivo")
plt.plot(phase_obj, z_jansen_al, label="Z Jansen")
plt.xlabel("Fase")
plt.ylabel("Posición normalizada")
plt.title("Coordenadas por fase")
plt.grid(True)
plt.legend()
plt.show()

print(f"\nArchivos guardados en:\n{SALIDA}")