
# ===========================================
#  Cargar Excel -> DataFrame por hoja
# ===========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter

# 1) Ruta del Excel (cambiá esto)
ruta_excel = r"C:/Users/bianc/Desktop/bioingenieria/tesis/procesamiento de trayectorias/procesamiento de trayectorias/milka con cinta 2.xlsx"

# 2) Cargar libro y preparar contenedor
xls = pd.ExcelFile(ruta_excel)
dfs = {}  # dict con todos los DataFrames por hoja

for hoja in xls.sheet_names:
    try:
        df = pd.read_excel(ruta_excel, sheet_name=hoja)
        dfs[hoja] = df           # acceso recomendado
        globals()[hoja] = df     # crea variable con el nombre de la hoja (opcional)
        print(f"✓ '{hoja}': {df.shape[0]} filas x {df.shape[1]} columnas")
    except Exception as e:
        print(f"⚠️ Error en hoja '{hoja}': {e}")

#resampleo de cada set de datos a 50 datos por paso
# ===========================================
#   Resamplear cada (pasada, paso) a 50 puntos
# ===========================================

# ------------------------------
# 🔹 Funciones auxiliares
# ------------------------------

def es_monotono_estricto(v):
    return np.all(np.diff(v) > 0)

def resamplear_segmento(t, x, y, N=50):
    """Interpola un segmento a N puntos."""
    t = np.asarray(t); x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[mask], x[mask], y[mask]
    if len(t) < 3:
        return np.linspace(0,1,N), np.full(N, np.nan), np.full(N, np.nan)

    if es_monotono_estricto(t):
        phase = (t - t[0]) / (t[-1] - t[0])
        phase_new = np.linspace(0, 1, N)
        x_new = np.interp(phase_new, phase, x)
        y_new = np.interp(phase_new, phase, y)
    else:
        s = np.linspace(0, 1, len(x))
        s_new = np.linspace(0, 1, N)
        x_new = np.interp(s_new, s, x)
        y_new = np.interp(s_new, s, y)
        phase_new = s_new
    return phase_new, x_new, y_new

def hoja_a_largo(df_hoja, nombre_hoja):
    """
    Formato largo con columnas:
    ['trial','pasada','idx_fila','Time','X','Y','paso'].
    Bloques de 4 columnas por pasada: Time, X, Y, paso.
    """
    cols = list(df_hoja.columns)
    if len(cols) % 4 != 0:
        raise ValueError(f"La hoja '{nombre_hoja}' no tiene múltiplo de 4 columnas.")
    n_pasadas = len(cols) // 4
    registros = []
    for k in range(n_pasadas):
        cT, cX, cY, cP = cols[4*k:4*k+4]
        sub = df_hoja[[cT, cX, cY, cP]].copy()
        sub.columns = ['Time', 'X', 'Y', 'paso']
        sub['trial'] = k + 1                 # 👈 ID numérico común
        sub['pasada'] = f"{nombre_hoja}{k+1}"  # opcional, informativo
        sub['idx_fila'] = np.arange(len(sub))
        registros.append(sub)
    largo = pd.concat(registros, axis=0, ignore_index=True)
    for c in ['Time','X','Y','paso']:
        largo[c] = pd.to_numeric(largo[c], errors='coerce')
    # Normaliza tipo de 'paso' para que matchee entre hojas
    largo['paso'] = largo['paso'].astype('Int64')  # o int si no hay NaN
    return largo[['trial','pasada','idx_fila','Time','X','Y','paso']]


def resamplear_por_pasada_y_paso(df_largo, N=50):
    out = []
    for (trial, ps), g in df_largo.groupby(['trial','paso'], dropna=False):
        g = g.sort_values('Time') if es_monotono_estricto(g['Time'].fillna(method='ffill').values) else g.sort_values('idx_fila')
        phase, x_new, y_new = resamplear_segmento(g['Time'].values, g['X'].values, g['Y'].values, N=N)
        tmp = pd.DataFrame({
            'trial': trial,        # 👈
            'paso': ps,
            'phase': phase,
            'X': x_new,
            'Y': y_new
        })
        out.append(tmp)
    return pd.concat(out, axis=0, ignore_index=True)


# ------------------------------
# 📥 Lectura del Excel y resampleo
# ------------------------------
NPTS = 50

resampled_dict = {}

for hoja, df_hoja in dfs.items():   # 👈 usa los dataframes en memoria
    print(f"Procesando hoja en memoria: {hoja} ...")
    largo = hoja_a_largo(df_hoja, hoja)
    resampled = resamplear_por_pasada_y_paso(largo, N=NPTS)
    resampled_dict[hoja] = resampled
    print(f"✓ {hoja}: {resampled.shape[0]} filas ({resampled['trial'].nunique()} pasadas)")

# ===========================================
#  Exportar resampled_dict a un nuevo Excel
# ===========================================

def exportar_resampleado_a_excel(resampled_dict, ruta_salida=None, ordenar=True):
    """
    Escribe un Excel con una hoja por entrada de resampled_dict.
    Cada hoja contiene columnas: ['pasada','paso','phase','X','Y'].
    
    Params
    ------
    resampled_dict : dict[str, pd.DataFrame]
        Diccionario con DataFrames resampleados por hoja (p.ej., 'sacroder', 'coxalder', etc.).
    ruta_salida : str | Path | None
        Ruta de salida del nuevo .xlsx. Si es None, crea una con timestamp en el cwd.
    ordenar : bool
        Si True, ordena por ['pasada','paso','phase'] antes de exportar.
    """
    # Nombre de salida por defecto con timestamp
    if ruta_salida is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_salida = Path(f"trayectorias_resampleadas_{ts}.xlsx")
    else:
        ruta_salida = Path(ruta_salida)
    
    # Asegurar carpeta
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        for hoja, df in resampled_dict.items():
            if df is None or df.empty:
                print(f"⚠️  Hoja '{hoja}' está vacía; se omite.")
                continue

            df_out = df.copy()

            # Ordenar y tipos
            if ordenar:
                cols_ord = [c for c in ['trial','paso','phase'] if c in df_out.columns]  # 👈
                if cols_ord:
                    df_out = df_out.sort_values(cols_ord, kind="mergesort").reset_index(drop=True)

            # Asegurar tipos numéricos cuando corresponda
            for c in ['paso','phase','X','Y']:
                if c in df_out.columns:
                    df_out[c] = pd.to_numeric(df_out[c], errors='coerce')

            # Limitar nombre de hoja a 31 caracteres (límite Excel)
            sheet_name = hoja[:31]

            # Escribir
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Exportado a: {ruta_salida.resolve()}")

def limpiar_pasos_vacios(df):
    """
    Elimina combinaciones (pasada, paso) donde X e Y son todos NaN
    o el número de puntos válidos es muy bajo.
    """
    out = []
    for (p, ps), g in df.groupby(['trial','paso']):  # 👈
        n_validos = g[['X','Y']].notna().sum().max()
        if n_validos > 0:
            out.append(g)

    return pd.concat(out, axis=0, ignore_index=True)


# ===== Ejemplo de uso =====
# Asumiendo que ya tenés resampled_dict (p.ej., salida de resamplear_por_pasada_y_paso):
# exportar_resampleado_a_excel(resampled_dict)
# o a una ruta específica:
for hoja in resampled_dict.keys():
    df = resampled_dict[hoja]
    resampled_dict[hoja] = limpiar_pasos_vacios(df)

#exportar_resampleado_a_excel(resampled_dict, ruta_salida=r"C:/Users/bianc/Desktop/bioingenieria/tesis/procesamiento de trayectorias/procesamiento de trayectorias/resampleado_50pts.xlsx")

#suavizar

def suavizar_resampleado(df, win=7, poly=2):
    df = df.copy()
    for col in ['X','Y']:
        df[col] = savgol_filter(df[col], window_length=win, polyorder=poly, mode='interp')
    return df

for hoja in resampled_dict:
    resampled_dict[hoja] = suavizar_resampleado(resampled_dict[hoja], win=7, poly=2)

#checkeo previo
for k in ("sacroder","coxalder"):
    print(k, resampled_dict[k].shape)
    print(resampled_dict[k][["trial","paso"]].drop_duplicates().head())

print("\n(sacroder) combos:", resampled_dict["sacroder"][["trial","paso"]].drop_duplicates().head(12))
print("(coxalder) combos:", resampled_dict["coxalder"][["trial","paso"]].drop_duplicates().head(12))


#referenciar al coxal
# ===============================
# Referencia por lado
# ===============================
right_markers = {"sacroder","coxalder","rodillader","tarsoder","metatarsoder","falangesder"}
left_markers  = {"sacroizq","coxalizq","rodillaizq","tarsoizq","metatarsoizq","falangesizq"}

# --- Coxales base
df_coxal_der = resampled_dict["coxalder"].copy()
df_coxal_der = df_coxal_der.rename(columns={"X":"X_coxal","Y":"Y_coxal"})
df_coxal_der["phase_r"] = df_coxal_der["phase"].round(6)
df_coxal_der["paso"] = df_coxal_der["paso"].astype('Int64')

df_coxal_izq = resampled_dict["coxalizq"].copy()
df_coxal_izq = df_coxal_izq.rename(columns={"X":"X_coxal","Y":"Y_coxal"})
df_coxal_izq["phase_r"] = df_coxal_izq["phase"].round(6)
df_coxal_izq["paso"] = df_coxal_izq["paso"].astype('Int64')

def referenciar(df_m, df_coxal_base):
    df = df_m.copy()
    if df.empty:
        return df
    df["phase_r"] = df["phase"].round(6)
    df["paso"] = df["paso"].astype('Int64')
    m = df.merge(
        df_coxal_base[["trial","paso","phase_r","X_coxal","Y_coxal"]],
        on=["trial","paso","phase_r"],
        how="inner"
    )
    m["X_rel"] = m["X"] - m["X_coxal"]
    m["Y_rel"] = m["Y"] - m["Y_coxal"]
    return m.drop(columns="phase_r")

# --- Aplica referencia correcta por lado
for hoja, df in resampled_dict.items():
    if hoja in right_markers:
        if hoja == "coxalder":
            # el coxal respecto de sí mismo → 0
            tmp = resampled_dict["coxalder"].copy()
            tmp["X_rel"] = 0.0
            tmp["Y_rel"] = 0.0
            resampled_dict[hoja] = tmp
        else:
            resampled_dict[hoja] = referenciar(df, df_coxal_der)
    elif hoja in left_markers:
        if hoja == "coxalizq":
            tmp = resampled_dict["coxalizq"].copy()
            tmp["X_rel"] = 0.0
            tmp["Y_rel"] = 0.0
            resampled_dict[hoja] = tmp
        else:
            resampled_dict[hoja] = referenciar(df, df_coxal_izq)
    else:
        # si hubiese otras hojas, las dejamos como están o avisamos
        pass


#checkeo post
dfp = resampled_dict["sacroder"]
print("Filas sacroder referenciado:", dfp.shape)
print("Pasos únicos:", dfp["paso"].unique())

#visualizar
dfp = resampled_dict["sacroder"].dropna(subset=["phase","X_rel"])

print("Pasos únicos:", dfp["paso"].unique())
for paso in sorted(dfp["paso"].dropna().unique()):
    sub = dfp[dfp["paso"] == paso].sort_values("phase")
    print(f"Paso {paso}: {len(sub)} puntos")


df = resampled_dict["sacroder"]

#for paso in df["paso"].unique():
#    subset = df[df["paso"] == paso]
#    plt.plot(subset["phase"], subset["X_rel"], label=f"Paso {paso}")

#plt.xlabel("Fase (0-1)")
#plt.ylabel("X relativa (px)")
#plt.title("Trayectorias X - sacroder")
#plt.legend()
#plt.show()

# 1) ¿Qué hojas resampleadas existen y cuántas filas tiene cada una?
for k, v in resampled_dict.items():
    print(k, v.shape)

# 2) ¿Tiene sacroder datos "antes" de referenciar?
print("\nSACRODER (crudo):")
print(resampled_dict["sacroder"].columns.tolist())
print(resampled_dict["sacroder"].head())

# 3) ¿Tiene coxalder datos?
print("\nCOXALDER (crudo):")
print(resampled_dict["coxalder"].columns.tolist())
print(resampled_dict["coxalder"].head())

# 4) (pasada, paso) disponibles en ambos
print("\n(sacroder) combos:", resampled_dict["sacroder"][["trial","paso"]].drop_duplicates().head(12))
print("(coxalder) combos:", resampled_dict["coxalder"][["trial","paso"]].drop_duplicates().head(12))


def normalizar_por_inicio(df):
    """Centra cada paso en (0,0) usando el primer frame de cada paso como referencia."""
    df = df.copy()
    for (trial, paso), g in df.groupby(["trial", "paso"]):
        if not g.empty:
            x0 = g.iloc[0]["X_rel"]
            y0 = g.iloc[0]["Y_rel"]
            idx = g.index
            df.loc[idx, "X_rel"] = g["X_rel"] - x0
            df.loc[idx, "Y_rel"] = g["Y_rel"] - y0
    return df

for m in resampled_dict.keys():
    resampled_dict[m] = normalizar_por_inicio(resampled_dict[m])

# --------------------------------------------
# Unifica todos los pasos en una sola curva promedio por marcador
# --------------------------------------------

def curva_promedio_unificada(df_marker, label):
    g = df_marker.groupby("phase")[["X_rel","Y_rel"]]
    mean = g.mean().rename(columns={"X_rel":"X_mean","Y_rel":"Y_mean"})
    std  = g.std().rename(columns={"X_rel":"X_std","Y_rel":"Y_std"})
    out = mean.join(std).reset_index()
    out["marker"] = label
    return out

# Marcadores que querés procesar (pueden ser todos o solo algunos)
marcadores_derecha = ["sacroder","coxalder","rodillader","tarsoder","metatarsoder","falangesder"]
marcadores_izquierda = ["sacroizq","coxalizq","rodillaizq","tarsoizq","metatarsoizq","falangesizq"]

curvas_derecha = []
curvas_izquierda = []

# Generar curvas promedio por marcador
for m in marcadores_derecha:
    dfm = resampled_dict[m]
    # salteá si faltan columnas (o está vacío)
    if not set(["phase","X_rel","Y_rel"]).issubset(dfm.columns) or dfm.empty:
        print(f"⚠️ '{m}' no tiene X_rel/Y_rel o está vacío. Se omite.")
        continue
    curvas_derecha.append(curva_promedio_unificada(dfm.dropna(subset=["phase","X_rel","Y_rel"]), m))

for m in marcadores_izquierda:
    dfm = resampled_dict[m]
    if not set(["phase","X_rel","Y_rel"]).issubset(dfm.columns) or dfm.empty:
        print(f"⚠️ '{m}' no tiene X_rel/Y_rel o está vacío. Se omite.")
        continue
    curvas_izquierda.append(curva_promedio_unificada(dfm.dropna(subset=["phase","X_rel","Y_rel"]), m))

curva_derecha_final  = pd.concat(curvas_derecha, axis=0, ignore_index=True) if curvas_derecha else pd.DataFrame()
curva_izquierda_final = pd.concat(curvas_izquierda, axis=0, ignore_index=True) if curvas_izquierda else pd.DataFrame()

#visualizar
for marcador in marcadores_derecha:
    cur = curva_derecha_final[curva_derecha_final["marker"] == marcador]
    if cur.empty:
        continue

    plt.figure(figsize=(6, 6))
    plt.plot(cur["X_mean"], cur["Y_mean"], label=marcador)
    plt.fill_betweenx(
        cur["Y_mean"],
        cur["X_mean"] - cur["X_std"],
        cur["X_mean"] + cur["X_std"],
        alpha=0.2
    )
    plt.xlabel("X relativa (px)")
    plt.ylabel("Y relativa (px)")
    plt.title(f"Trayectoria promedio 2D — {marcador}")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()

for marcador in marcadores_izquierda:
    cur = curva_izquierda_final[curva_izquierda_final["marker"] == marcador]
    if cur.empty:
        continue

    plt.figure(figsize=(6, 6))
    plt.plot(cur["X_mean"], cur["Y_mean"], label=marcador)
    plt.fill_betweenx(
        cur["Y_mean"],
        cur["X_mean"] - cur["X_std"],
        cur["X_mean"] + cur["X_std"],
        alpha=0.2
    )
    plt.xlabel("X relativa (px)")
    plt.ylabel("Y relativa (px)")
    plt.title(f"Trayectoria promedio 2D — {marcador}")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()


with pd.ExcelWriter("curvas_promedio_unificadas.xlsx", engine="openpyxl") as writer:
    curva_derecha_final.to_excel(writer, sheet_name="Derecha", index=False)
    curva_izquierda_final.to_excel(writer, sheet_name="Izquierda", index=False)
