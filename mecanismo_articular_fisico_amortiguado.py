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

SALIDA = RUTA_MODELO.parent / "apoyo_amortiguado"
SALIDA.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARÁMETROS DE DISEÑO
# =========================================================

PESO_PERRO_KG = 20.0
G = 9.81

# Carga aproximada por pata trasera en estática
CARGA_PATA_N = PESO_PERRO_KG * G / 4

# Asistencia objetivo: 10-20% de la carga de esa pata
ASISTENCIA_MIN_N = 0.10 * CARGA_PATA_N
ASISTENCIA_MAX_N = 0.20 * CARGA_PATA_N

# Fuerza objetivo nominal
ASISTENCIA_OBJ_N = 0.15 * CARGA_PATA_N

# Resorte propuesto
K_RESORTE_N_M = 1000.0  # equivalente a 10 N para 1 cm de compresión
K_RESORTE_N_CM = K_RESORTE_N_M / 100.0

# Umbral para considerar fase de apoyo
# Como z_min = 0 y z_max ≈ 2.7 cm, se considera apoyo cuando el pie está cerca del suelo.
UMBRAL_APOYO_CM = 0.5

# Separación vertical entre metatarso y zapata.
# Para primera versión se asume que la zapata está debajo del punto distal.
ALTURA_ZAPATA_REPOSO_CM = 0.0

# Compresión máxima permitida para no hacer el apoyo demasiado blando.
COMPRESION_MAX_CM = 2.0

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
# BARRIDO DE UMBRALES DE APOYO Y RESORTES
# =========================================================

UMBRALES_APOYO_CM = [0.5, 0.75, 1.0]
K_RESORTES_N_M = [500, 750, 1000, 1500, 2000]

COMPRESION_MAX_CM = 2.0

resultados_barrido = []

for umbral in UMBRALES_APOYO_CM:

    en_apoyo = z_distal <= umbral
    porcentaje_apoyo = 100 * np.sum(en_apoyo) / len(en_apoyo)

    for k_N_m in K_RESORTES_N_M:

        k_N_cm = k_N_m / 100.0

        # Fuerza objetivo solo en apoyo
        fuerza_obj_N = np.where(en_apoyo, ASISTENCIA_OBJ_N, 0.0)

        # Compresión necesaria: F = k*x
        compresion_cm = fuerza_obj_N / k_N_cm

        # Limitar compresión máxima
        compresion_cm_limitada = np.clip(compresion_cm, 0, COMPRESION_MAX_CM)

        # Fuerza real obtenida con esa compresión limitada
        fuerza_real_N = k_N_cm * compresion_cm_limitada

        supera_compresion = np.max(compresion_cm) > COMPRESION_MAX_CM

        resultados_barrido.append({
            "umbral_apoyo_cm": umbral,
            "k_resorte_N_m": k_N_m,
            "k_resorte_N_cm": k_N_cm,
            "porcentaje_apoyo_detectado": porcentaje_apoyo,
            "fuerza_objetivo_N": ASISTENCIA_OBJ_N,
            "compresion_necesaria_cm": float(np.max(compresion_cm)),
            "compresion_limitada_max_cm": float(np.max(compresion_cm_limitada)),
            "fuerza_real_max_N": float(np.max(fuerza_real_N)),
            "fuerza_real_media_en_apoyo_N": float(np.mean(fuerza_real_N[en_apoyo])),
            "supera_compresion_max": supera_compresion,
        })

df_barrido = pd.DataFrame(resultados_barrido)

df_barrido.to_csv(
    SALIDA / "barrido_umbral_resorte.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== BARRIDO UMBRAL DE APOYO + RESORTE ===")
print(df_barrido)

# =========================================================
# MODELO FINAL: RESORTE VERTICAL TELESCÓPICO
# =========================================================

UMBRAL_APOYO_CM = 1.0
K_RESORTE_N_M = 1000.0
K_RESORTE_N_CM = K_RESORTE_N_M / 100.0

# Longitud libre del conjunto resorte-zapata.
# Es la distancia vertical entre el extremo distal y la zapata cuando no hay carga.
LONGITUD_LIBRE_RESORTE_CM = 2.0


# Compresión máxima permitida
COMPRESION_MAX_CM = 2.0

# Detectar fase de apoyo
en_apoyo = z_distal <= UMBRAL_APOYO_CM

# Fuerza objetivo
fuerza_asistiva_N = np.zeros_like(z_distal)

fuerza_asistiva_N[en_apoyo] = ASISTENCIA_OBJ_N * (
    1 - z_distal[en_apoyo] / UMBRAL_APOYO_CM
)

fuerza_asistiva_N = np.clip(fuerza_asistiva_N, 0, ASISTENCIA_OBJ_N)

# Compresión necesaria por F = k*x
compresion_cm = fuerza_asistiva_N / K_RESORTE_N_CM
compresion_cm = np.clip(compresion_cm, 0, COMPRESION_MAX_CM)

# Fuerza real entregada
fuerza_real_N = K_RESORTE_N_CM * compresion_cm

# Longitud instantánea del resorte
longitud_resorte_cm = LONGITUD_LIBRE_RESORTE_CM - compresion_cm

# Evitar longitudes negativas
longitud_resorte_cm = np.clip(longitud_resorte_cm, 0.2, LONGITUD_LIBRE_RESORTE_CM)

# Posición de zapata
x_zapata = x_distal.copy()


# En balanceo: la zapata cuelga debajo del distal con longitud libre
# En apoyo: la zapata toca el suelo
ELEVACION_RETRACCION_CM = 1.2
z_zapata_balanceo = z_distal - LONGITUD_LIBRE_RESORTE_CM + ELEVACION_RETRACCION_CM
z_zapata = np.where(en_apoyo, 0.0, z_zapata_balanceo)

# Si en balanceo la zapata queda por debajo del suelo, la limitamos al suelo
z_zapata = np.maximum(z_zapata, 0.0)
arrastre_zapata = np.sum((~en_apoyo) & (z_zapata == 0.0))
porcentaje_arrastre = 100 * arrastre_zapata / np.sum(~en_apoyo)

# Recalcular longitud geométrica real entre distal y zapata
longitud_resorte_geom_cm = z_distal - z_zapata

# Compresión geométrica efectiva
compresion_geom_cm = LONGITUD_LIBRE_RESORTE_CM - longitud_resorte_geom_cm
compresion_geom_cm = np.clip(compresion_geom_cm, 0, COMPRESION_MAX_CM)

# =========================================================
# MÉTRICAS
# =========================================================

porcentaje_fase_apoyo = 100 * np.sum(en_apoyo) / len(en_apoyo)

metricas = {
    "peso_perro_kg": PESO_PERRO_KG,
    "carga_estimada_por_pata_N": CARGA_PATA_N,
    "asistencia_min_10porc_N": ASISTENCIA_MIN_N,
    "asistencia_max_20porc_N": ASISTENCIA_MAX_N,
    "asistencia_obj_15porc_N": ASISTENCIA_OBJ_N,

    "k_resorte_N_m": K_RESORTE_N_M,
    "k_resorte_N_cm": K_RESORTE_N_CM,

    "umbral_apoyo_cm": UMBRAL_APOYO_CM,
    "porcentaje_fase_apoyo_detectado": porcentaje_fase_apoyo,

    "compresion_max_calculada_cm": float(np.max(compresion_cm)),
    "compresion_media_en_apoyo_cm": float(np.mean(compresion_cm[en_apoyo])),
    "fuerza_max_real_N": float(np.max(fuerza_real_N)),
    "fuerza_media_en_apoyo_N": float(np.mean(fuerza_real_N[en_apoyo])),
    "porcentaje_arrastre_zapata_en_balanceo": porcentaje_arrastre,
}

df_metricas = pd.DataFrame([metricas])

# =========================================================
# EXPORTAR
# =========================================================

df_apoyo = pd.DataFrame({
    "phase": phase,

    "x_distal_cm": x_distal,
    "z_distal_cm": z_distal,

    "x_zapata_cm": x_zapata,
    "z_zapata_cm": z_zapata,

    "en_apoyo": en_apoyo,
    "fuerza_asistiva_obj_N": fuerza_asistiva_N,
    "fuerza_real_N": fuerza_real_N,
    "compresion_resorte_teorica_cm": compresion_cm,
    "longitud_resorte_teorica_cm": longitud_resorte_cm,
    "longitud_resorte_geom_cm": longitud_resorte_geom_cm,
    "compresion_resorte_geom_cm": compresion_geom_cm,

    "longitud_libre_resorte_cm": LONGITUD_LIBRE_RESORTE_CM,
    "longitud_resorte_min_cm": float(np.min(longitud_resorte_cm)),
    "longitud_resorte_max_cm": float(np.max(longitud_resorte_cm)),
    "compresion_geom_max_cm": float(np.max(compresion_geom_cm)),

    "theta_cadera_deg": theta_cadera_deg,
    "theta_rodilla_deg": theta_rodilla_deg,
})

df_apoyo.to_csv(
    SALIDA / "modelo_apoyo_amortiguado.csv",
    index=False,
    encoding="utf-8-sig"
)

df_metricas.to_csv(
    SALIDA / "metricas_apoyo_amortiguado.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== MODELO DE APOYO AMORTIGUADO ===")
for k, v in metricas.items():
    print(f"{k}: {v}")

print(f"\nArchivos guardados en:\n{SALIDA}")

# =========================================================
# GRÁFICO 1: DETECCIÓN DE APOYO
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, z_distal, label="Z metatarso")
plt.axhline(UMBRAL_APOYO_CM, linestyle="--", label="umbral apoyo")

plt.fill_between(
    phase,
    0,
    z_distal,
    where=en_apoyo,
    alpha=0.3,
    label="fase de apoyo detectada"
)

plt.xlabel("Fase")
plt.ylabel("Z [cm]")
plt.title("Detección preliminar de fase de apoyo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "deteccion_fase_apoyo.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 2: FUERZA ASISTIVA
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, fuerza_real_N, label="Fuerza asistiva real")
plt.axhline(ASISTENCIA_MIN_N, linestyle="--", label="10% carga pata")
plt.axhline(ASISTENCIA_MAX_N, linestyle="--", label="20% carga pata")

plt.xlabel("Fase")
plt.ylabel("Fuerza [N]")
plt.title("Fuerza asistiva estimada durante el ciclo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "fuerza_asistiva.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 3: COMPRESIÓN DEL RESORTE
# =========================================================

plt.figure(figsize=(9, 4))

plt.plot(phase, compresion_cm)

plt.xlabel("Fase")
plt.ylabel("Compresión [cm]")
plt.title("Compresión estimada del resorte")
plt.grid(True)
plt.tight_layout()
plt.savefig(SALIDA / "compresion_resorte.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 4: TRAYECTORIA DISTAL Y ZAPATA
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(x_distal, z_distal, linewidth=3, label="Metatarso")
plt.plot(x_zapata, z_zapata, linewidth=2, label="Zapata")
plt.axhline(0, linestyle="--", label="suelo")

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Trayectoria del metatarso y apoyo de zapata")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "trayectoria_metatarso_zapata.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO 5: ESQUEMA EN FASES CON ZAPATA
# =========================================================

FASES_A_GRAFICAR = [0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

plt.figure(figsize=(10, 7))

for fase_obj in FASES_A_GRAFICAR:
    idx = np.argmin(np.abs(phase - fase_obj))

    xs = [
        x_cadera[idx],
        x_rodilla[idx],
        x_distal[idx],
        x_zapata[idx]
    ]

    zs = [
        z_cadera[idx],
        z_rodilla[idx],
        z_distal[idx],
        z_zapata[idx]
    ]

    plt.plot(xs[:3], zs[:3], marker="o", linewidth=2, label=f"fase {phase[idx]:.2f}")
    plt.plot(xs[2:], zs[2:], marker="s", linestyle="--", linewidth=1)

plt.axhline(0, linestyle="--", linewidth=1, label="suelo")

plt.xlabel("X [cm]")
plt.ylabel("Z [cm]")
plt.title("Órtesis anatómica con zapata amortiguada")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "ortesis_con_zapata_fases.png", dpi=300)
plt.show()

# =========================================================
# GRÁFICO: PORCENTAJE DE APOYO SEGÚN UMBRAL
# =========================================================

df_apoyo_umbral = (
    df_barrido
    .groupby("umbral_apoyo_cm")["porcentaje_apoyo_detectado"]
    .first()
    .reset_index()
)

plt.figure(figsize=(7, 4))
plt.plot(
    df_apoyo_umbral["umbral_apoyo_cm"],
    df_apoyo_umbral["porcentaje_apoyo_detectado"],
    marker="o"
)

plt.xlabel("Umbral de apoyo [cm]")
plt.ylabel("Porcentaje del ciclo [%]")
plt.title("Fase de apoyo detectada según umbral")
plt.grid(True)
plt.tight_layout()
plt.savefig(SALIDA / "barrido_porcentaje_apoyo.png", dpi=300)
plt.show()


# =========================================================
# GRÁFICO: COMPRESIÓN NECESARIA SEGÚN RESORTE
# =========================================================

plt.figure(figsize=(8, 5))

for umbral in UMBRALES_APOYO_CM:
    df_tmp = df_barrido[df_barrido["umbral_apoyo_cm"] == umbral]

    plt.plot(
        df_tmp["k_resorte_N_m"],
        df_tmp["compresion_necesaria_cm"],
        marker="o",
        label=f"umbral {umbral} cm"
    )

plt.axhline(COMPRESION_MAX_CM, linestyle="--", label="compresión máxima")

plt.xlabel("Constante del resorte k [N/m]")
plt.ylabel("Compresión necesaria [cm]")
plt.title("Compresión requerida para la asistencia objetivo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "barrido_compresion_resorte.png", dpi=300)
plt.show()


# =========================================================
# GRÁFICO: FUERZA REAL MÁXIMA SEGÚN RESORTE
# =========================================================

plt.figure(figsize=(8, 5))

for umbral in UMBRALES_APOYO_CM:
    df_tmp = df_barrido[df_barrido["umbral_apoyo_cm"] == umbral]

    plt.plot(
        df_tmp["k_resorte_N_m"],
        df_tmp["fuerza_real_max_N"],
        marker="o",
        label=f"umbral {umbral} cm"
    )

plt.axhline(ASISTENCIA_MIN_N, linestyle="--", label="10% carga pata")
plt.axhline(ASISTENCIA_MAX_N, linestyle="--", label="20% carga pata")
plt.axhline(ASISTENCIA_OBJ_N, linestyle=":", label="objetivo 15%")

plt.xlabel("Constante del resorte k [N/m]")
plt.ylabel("Fuerza máxima [N]")
plt.title("Fuerza asistiva lograda según resorte")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / "barrido_fuerza_resorte.png", dpi=300)
plt.show()