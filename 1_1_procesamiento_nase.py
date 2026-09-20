from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================================================
# CONFIGURACIÓN
# =========================================================
RUTA_OPTICAL = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\dataset\3DDogsLab2024\Data\Optical\Sync_Align_v2023_11_16b"
)

RUTA_SALIDA = Path(
    r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_caminata_global"
)
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# cantidad de puntos por ciclo
NPTS = 100

# trials de caminata
TRIALS_CAMINATA = [1, 2, 3]

# parámetros de detección de ciclos
MIN_FRAMES_CICLO = 15
MAX_FRAMES_CICLO = 120
MIN_DIST_MINIMOS = 12

# referencia anatómica
MARCADOR_REF = "sacrum"

# lados a procesar
LADOS = ["l", "r"]

# filtros de calidad
MIN_DX_ABS = 0.03
MIN_DZ_ABS = 0.005

# evento de realineación
EVENTO_REALINEACION = "z_max"   # opciones: x_max, x_min, z_max, z_min

# generar plots
GENERAR_PLOTS = True


# =========================================================
# PARSEO DE NOMBRES
# =========================================================
def parsear_nombre_archivo(path_txt):
    """
    Espera nombres tipo:
    optical_sync_align_d16_t1_a
    o
    optical_sync_align_d16_t1_a.txt
    """
    nombre = path_txt.stem
    m = re.search(r"optical_sync_align_d(\d+)_t(\d+)_([ab])$", nombre)
    if not m:
        return None

    return {
        "subject_id": int(m.group(1)),
        "trial_num": int(m.group(2)),
        "direction": m.group(3),
        "basename": nombre
    }


def es_caminata(info_nombre):
    return info_nombre["trial_num"] in TRIALS_CAMINATA


# =========================================================
# LECTURA DE ARCHIVOS
# =========================================================
def leer_lineas(path_txt):
    with open(path_txt, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def parsear_archivo_global(path_txt):
    """
    Parsea archivos tipo:
    fps 60.000000
    frame_numbering_ref pass_start
    alignment_ref rgbd_gbl_ref
    frame_num poll withers sacrum ...
    0 x y z x y z ...
    """
    lineas = leer_lineas(path_txt)

    fps = None
    frame_numbering_ref = None
    alignment_ref = None
    idx_header = None

    for i, line in enumerate(lineas):
        if line.startswith("fps "):
            fps = float(line.split()[1])
        elif line.startswith("frame_numbering_ref "):
            frame_numbering_ref = line.split(maxsplit=1)[1]
        elif line.startswith("alignment_ref "):
            alignment_ref = line.split(maxsplit=1)[1]
        elif line.startswith("frame_num "):
            idx_header = i
            break

    if idx_header is None:
        raise ValueError("No se encontró la cabecera 'frame_num'")

    header_tokens = lineas[idx_header].split()
    if header_tokens[0] != "frame_num":
        raise ValueError("Cabecera inválida")

    marcadores = header_tokens[1:]
    filas_data = [line.split() for line in lineas[idx_header + 1:] if line.strip()]

    columnas = ["frame_num"]
    for m in marcadores:
        columnas.extend([f"{m}_x", f"{m}_y", f"{m}_z"])

    data_ajustada = []
    for row in filas_data:
        if len(row) < len(columnas):
            row = row + [np.nan] * (len(columnas) - len(row))
        elif len(row) > len(columnas):
            row = row[:len(columnas)]
        data_ajustada.append(row)

    df = pd.DataFrame(data_ajustada, columns=columnas)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    meta = {
        "fps": fps,
        "frame_numbering_ref": frame_numbering_ref,
        "alignment_ref": alignment_ref,
        "marcadores": marcadores
    }

    return df, meta


# =========================================================
# UTILIDADES
# =========================================================
def moving_average(x, win=7):
    x = np.asarray(x, dtype=float)
    if win < 2:
        return x.copy()

    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(xpad, kernel, mode="valid")


def detectar_minimos(signal, min_dist=12):
    s = np.asarray(signal, dtype=float)
    idxs = []

    for i in range(1, len(s) - 1):
        if np.isnan(s[i - 1]) or np.isnan(s[i]) or np.isnan(s[i + 1]):
            continue

        if s[i] <= s[i - 1] and s[i] < s[i + 1]:
            if not idxs or (i - idxs[-1]) >= min_dist:
                idxs.append(i)
            else:
                if s[i] < s[idxs[-1]]:
                    idxs[-1] = i

    return np.array(idxs, dtype=int)


def reinterpolar_ciclo(x, z, npts=100):
    t = np.linspace(0, 1, len(x))
    tnew = np.linspace(0, 1, npts)
    xnew = np.interp(tnew, t, x)
    znew = np.interp(tnew, t, z)
    return tnew, xnew, znew


def rotar_series(x, z, idx_inicio):
    x_rot = np.roll(np.asarray(x), -idx_inicio)
    z_rot = np.roll(np.asarray(z), -idx_inicio)
    return x_rot, z_rot


def realinear_ciclo_por_evento(x, z, evento="z_max"):
    x = np.asarray(x)
    z = np.asarray(z)

    if evento == "z_max":
        idx = int(np.argmax(x))
    elif evento == "z_max":
        idx = int(np.argmin(x))
    elif evento == "z_max":
        idx = int(np.argmax(z))
    elif evento == "z_max":
        idx = int(np.argmin(z))
    else:
        raise ValueError(f"Evento no soportado: {evento}")

    x_rot, z_rot = rotar_series(x, z, idx)
    return x_rot, z_rot, idx


def cierre_suave_ciclo(x, z):
    """
    Cierra suavemente un ciclo distribuyendo linealmente
    la diferencia entre el punto final y el inicial.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    n = len(x)
    if n < 2:
        return x.copy(), z.copy()

    t = np.linspace(0, 1, n)

    dx = x[-1] - x[0]
    dz = z[-1] - z[0]

    x_corr = x - t * dx
    z_corr = z - t * dz

    return x_corr, z_corr


def dist_3d(df, a, b):
    ax, ay, az = df[f"{a}_x"], df[f"{a}_y"], df[f"{a}_z"]
    bx, by, bz = df[f"{b}_x"], df[f"{b}_y"], df[f"{b}_z"]
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def longitud_funcional_pata(df, lado):
    iliac = f"{lado}_iliac"
    stifle = f"{lado}_stifle"
    hock = f"{lado}_hock"
    meta = f"{lado}_meta_tars"

    d1 = dist_3d(df, iliac, stifle)
    d2 = dist_3d(df, stifle, hock)
    d3 = dist_3d(df, hock, meta)

    L = d1 + d2 + d3
    L_valid = L[np.isfinite(L)]

    if len(L_valid) == 0:
        return np.nan

    return float(np.median(L_valid))


def preparar_lado(df, lado="l", ref="sacrum"):
    out = df.copy()
    out[f"{lado}_x_rel"] = out[f"{lado}_meta_tars_x"] - out[f"{ref}_x"]
    out[f"{lado}_z_rel"] = out[f"{lado}_meta_tars_z"] - out[f"{ref}_z"]
    return out


def extraer_ciclos_lado(df, lado="l", npts=100,
                        min_frames=15, max_frames=120, min_dist=12,
                        evento_realineacion="z_max"):
    """
    Segmenta por mínimos de z_rel, reinterpola y realinea cada ciclo por evento.
    """
    xcol = f"{lado}_x_rel"
    zcol = f"{lado}_z_rel"

    sub = df[["frame_num", xcol, zcol]].dropna().copy()
    sub = sub.rename(columns={xcol: "x_rel", zcol: "z_rel"})

    if len(sub) < 10:
        return sub, np.array([]), np.array([]), []

    z_smooth = moving_average(sub["z_rel"].values, win=7)
    mins = detectar_minimos(z_smooth, min_dist=min_dist)

    ciclos = []
    for i in range(len(mins) - 1):
        a = mins[i]
        b = mins[i + 1]
        largo = b - a

        if largo < min_frames or largo > max_frames:
            continue

        seg = sub.iloc[a:b + 1].copy()

        phase, xnew, znew = reinterpolar_ciclo(
            seg["x_rel"].values,
            seg["z_rel"].values,
            npts=npts
        )

        # 1) realinear por evento
        x_al, z_al, idx_evento = realinear_ciclo_por_evento(
            xnew, znew, evento=evento_realineacion
        )

        # 2) cerrar suavemente cada ciclo YA realineado
        x_cerr, z_cerr = cierre_suave_ciclo(x_al, z_al)

        ciclo = pd.DataFrame({
            "phase": np.linspace(0, 1, npts),
            "x_rel": x_cerr,
            "z_rel": z_cerr,
            "x_rel_sin_cerrar": x_al,
            "z_rel_sin_cerrar": z_al,
            "frame_start": int(seg["frame_num"].iloc[0]),
            "frame_end": int(seg["frame_num"].iloc[-1]),
            "n_frames": len(seg),
            "idx_evento_realign": idx_evento,
            "evento_realign": evento_realineacion
        })
        ciclos.append(ciclo)

    return sub, z_smooth, mins, ciclos


def calidad_ciclo(c):
    dx = c["x_rel"].max() - c["x_rel"].min()
    dz = c["z_rel"].max() - c["z_rel"].min()
    return dx, dz


# =========================================================
# MAIN
# =========================================================
def main():
    archivos = []

    for p in sorted(RUTA_OPTICAL.iterdir()):
        if p.is_file() and p.stem.startswith("optical_sync_align_"):
            info = parsear_nombre_archivo(p)
            if info is None:
                continue
            if es_caminata(info):
                archivos.append((p, info))

    print(f"Archivos candidatos de caminata: {len(archivos)}")

    todos_los_ciclos = []
    resumen_trials = []
    errores = []

    for path_txt, info in archivos:
        try:
            df, meta = parsear_archivo_global(path_txt)

            # solo globales
            if meta["alignment_ref"] != "rgbd_gbl_ref":
                continue

            for lado in LADOS:
                cols_necesarias = [
                    f"{MARCADOR_REF}_x", f"{MARCADOR_REF}_z",
                    f"{lado}_iliac_x", f"{lado}_iliac_y", f"{lado}_iliac_z",
                    f"{lado}_stifle_x", f"{lado}_stifle_y", f"{lado}_stifle_z",
                    f"{lado}_hock_x", f"{lado}_hock_y", f"{lado}_hock_z",
                    f"{lado}_meta_tars_x", f"{lado}_meta_tars_y", f"{lado}_meta_tars_z",
                ]

                if not set(cols_necesarias).issubset(df.columns):
                    resumen_trials.append({
                        "subject_id": info["subject_id"],
                        "trial_num": info["trial_num"],
                        "direction": info["direction"],
                        "lado": lado,
                        "archivo": path_txt.name,
                        "fps": meta["fps"],
                        "n_frames": len(df),
                        "n_ciclos": 0,
                        "longitud_funcional": np.nan,
                        "estado": "faltan_columnas"
                    })
                    continue

                L = longitud_funcional_pata(df, lado)

                if pd.isna(L) or L <= 0:
                    resumen_trials.append({
                        "subject_id": info["subject_id"],
                        "trial_num": info["trial_num"],
                        "direction": info["direction"],
                        "lado": lado,
                        "archivo": path_txt.name,
                        "fps": meta["fps"],
                        "n_frames": len(df),
                        "n_ciclos": 0,
                        "longitud_funcional": np.nan,
                        "estado": "sin_longitud_funcional"
                    })
                    continue

                dfr = preparar_lado(df, lado=lado, ref=MARCADOR_REF)

                sub, z_smooth, mins, ciclos = extraer_ciclos_lado(
                    dfr,
                    lado=lado,
                    npts=NPTS,
                    min_frames=MIN_FRAMES_CICLO,
                    max_frames=MAX_FRAMES_CICLO,
                    min_dist=MIN_DIST_MINIMOS,
                    evento_realineacion=EVENTO_REALINEACION
                )

                if not ciclos:
                    resumen_trials.append({
                        "subject_id": info["subject_id"],
                        "trial_num": info["trial_num"],
                        "direction": info["direction"],
                        "lado": lado,
                        "archivo": path_txt.name,
                        "fps": meta["fps"],
                        "n_frames": len(df),
                        "n_ciclos": 0,
                        "longitud_funcional": L,
                        "estado": "sin_ciclos"
                    })
                    continue

                ciclos_buenos = []
                for i, c in enumerate(ciclos, start=1):
                    dx, dz = calidad_ciclo(c)

                    if dx <= 0 or dz <= 0:
                        continue
                    if dx < MIN_DX_ABS or dz < MIN_DZ_ABS:
                        continue

                    c2 = c.copy()
                    c2["x_norm"] = c2["x_rel"] / L
                    c2["z_norm"] = c2["z_rel"] / L

                    dx_cierre_ind = float(c2["x_rel"].iloc[-1] - c2["x_rel"].iloc[0])
                    dz_cierre_ind = float(c2["z_rel"].iloc[-1] - c2["z_rel"].iloc[0])
                    err_cierre_ind = float(np.sqrt(dx_cierre_ind**2 + dz_cierre_ind**2))

                    c2["dx_cierre_individual"] = dx_cierre_ind
                    c2["dz_cierre_individual"] = dz_cierre_ind
                    c2["error_cierre_individual"] = err_cierre_ind

                    # mantener convención del eje de avance
                    if info["direction"] == "b":
                        c2["x_norm"] = -c2["x_norm"]
                        c2["x_rel"] = -c2["x_rel"]

                        if "x_rel_sin_cerrar" in c2.columns:
                            c2["x_rel_sin_cerrar"] = -c2["x_rel_sin_cerrar"]

                    c2["subject_id"] = info["subject_id"]
                    c2["trial_num"] = info["trial_num"]
                    c2["direction"] = info["direction"]
                    c2["lado"] = lado
                    c2["archivo"] = path_txt.name
                    c2["fps"] = meta["fps"]
                    c2["longitud_funcional"] = L
                    c2["cycle_id_local"] = i

                    todos_los_ciclos.append(c2)
                    ciclos_buenos.append(c2)

                resumen_trials.append({
                    "subject_id": info["subject_id"],
                    "trial_num": info["trial_num"],
                    "direction": info["direction"],
                    "lado": lado,
                    "archivo": path_txt.name,
                    "fps": meta["fps"],
                    "n_frames": len(df),
                    "n_ciclos": len(ciclos_buenos),
                    "longitud_funcional": L,
                    "estado": "ok"
                })

        except Exception as e:
            errores.append({
                "archivo": path_txt.name,
                "error": str(e)
            })

    # =====================================================
    # CONSOLIDAR
    # =====================================================
    df_ciclos = pd.concat(todos_los_ciclos, ignore_index=True) if todos_los_ciclos else pd.DataFrame()
    df_resumen = pd.DataFrame(resumen_trials)
    df_errores = pd.DataFrame(errores)

    df_resumen.to_csv(RUTA_SALIDA / "resumen_trials_caminata.csv", index=False, encoding="utf-8-sig")
    df_errores.to_csv(RUTA_SALIDA / "errores_procesamiento.csv", index=False, encoding="utf-8-sig")

    if not df_ciclos.empty:
        df_ciclos.to_csv(RUTA_SALIDA / "datos_ciclos.csv", index=False, encoding="utf-8-sig")

    print(f"Resumen trials guardado en: {RUTA_SALIDA / 'resumen_trials_caminata.csv'}")
    print(f"Errores guardados en: {RUTA_SALIDA / 'errores_procesamiento.csv'}")

    # =====================================================
    # PROMEDIO POR TRIAL
    # =====================================================
    if not df_ciclos.empty:
        g_trial = df_ciclos.groupby(
            ["subject_id", "trial_num", "direction", "lado", "phase"]
        )[["x_norm", "z_norm"]]

        prom_trial = g_trial.mean().rename(columns={"x_norm": "x_mean", "z_norm": "z_mean"})
        prom_trial["x_std"] = g_trial.std()["x_norm"]
        prom_trial["z_std"] = g_trial.std()["z_norm"]
        prom_trial = prom_trial.reset_index()

        prom_trial.to_csv(RUTA_SALIDA / "promedio_por_trial.csv", index=False, encoding="utf-8-sig")
    else:
        prom_trial = pd.DataFrame()

    # =====================================================
    # PROMEDIO POR SUJETO
    # =====================================================
    if not prom_trial.empty:
        g_subj = prom_trial.groupby(["subject_id", "lado", "phase"])[["x_mean", "z_mean"]]

        prom_sujeto = g_subj.mean().rename(columns={"x_mean": "x_mean", "z_mean": "z_mean"})
        prom_sujeto["x_std"] = g_subj.std()["x_mean"]
        prom_sujeto["z_std"] = g_subj.std()["z_mean"]
        prom_sujeto = prom_sujeto.reset_index()

        prom_sujeto.to_csv(RUTA_SALIDA / "promedio_por_sujeto.csv", index=False, encoding="utf-8-sig")
    else:
        prom_sujeto = pd.DataFrame()

    # =====================================================
    # PROMEDIO GLOBAL
    # =====================================================
    if not prom_sujeto.empty:
        g_global = prom_sujeto.groupby(["lado", "phase"])[["x_mean", "z_mean"]]

        prom_global = g_global.mean().rename(columns={"x_mean": "x_mean", "z_mean": "z_mean"})
        prom_global["x_std"] = g_global.std()["x_mean"]
        prom_global["z_std"] = g_global.std()["z_mean"]
        prom_global = prom_global.reset_index()

        prom_global.to_csv(RUTA_SALIDA / "promedio_global_caminata.csv", index=False, encoding="utf-8-sig")
    else:
        prom_global = pd.DataFrame()

    # =====================================================
    # RESÚMENES
    # =====================================================
    if not df_ciclos.empty:
        n_ciclos_reales = (
            df_ciclos[["subject_id", "trial_num", "direction", "lado", "cycle_id_local"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        n_ciclos_reales = 0

    print("\n=== REPORTE ===")
    print(f"Ciclos totales reales: {n_ciclos_reales}")

    if not df_resumen.empty and "n_ciclos" in df_resumen.columns:
        print("\nCiclos totales por lado:")
        print(df_resumen.groupby("lado")["n_ciclos"].sum())

        print("\nTop 10 trials con más ciclos:")
        print(
            df_resumen.sort_values("n_ciclos", ascending=False)
            .head(10)[["subject_id", "trial_num", "direction", "lado", "archivo", "n_ciclos"]]
            .to_string(index=False)
        )

        # top sujetos
        resumen_sujetos = (
            df_resumen[df_resumen["estado"] == "ok"]
            .groupby(["subject_id", "lado"])
            .agg(
                n_trials_validos=("archivo", "count"),
                n_ciclos=("n_ciclos", "sum"),
                longitud_funcional_mediana=("longitud_funcional", "median")
            )
            .reset_index()
            .sort_values("n_ciclos", ascending=False)
        )

        print("\nTop 10 sujetos con más ciclos:")
        print(resumen_sujetos.head(10).to_string(index=False))

    print(f"\nEvento de realineación usado: {EVENTO_REALINEACION}")
    print(f"\nArchivos de salida en:\n{RUTA_SALIDA}")

    # =====================================================
    # PLOTS
    # =====================================================
    if GENERAR_PLOTS and not prom_global.empty:
        for lado in LADOS:
            subg = prom_global[prom_global["lado"] == lado].sort_values("phase").reset_index(drop=True)
            if subg.empty:
                continue

            plt.figure(figsize=(7, 6))
            plt.plot(subg["x_mean"], subg["z_mean"], label=f"Lado {lado}")
            plt.xlabel("X normalizada")
            plt.ylabel("Z normalizada")
            plt.title(f"Promedio global caminata - lado {lado} ({EVENTO_REALINEACION})")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.plot(subg["phase"], subg["x_mean"], label="X mean")
            plt.plot(subg["phase"], subg["z_mean"], label="Z mean")
            plt.xlabel("Fase")
            plt.ylabel("Posición normalizada")
            plt.title(f"Promedio global por fase - lado {lado} ({EVENTO_REALINEACION})")
            plt.grid(True)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    main()