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

RUTA_SALIDA = RUTA_OPTICAL / "salidas_lado_izquierdo_desde_cero"
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

# lado a procesar
LADO = "l"

# filtros de calidad
MIN_DX_ABS = 0.03
MIN_DZ_ABS = 0.005

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

    return np.median(L_valid)


def preparar_lado(df, lado="l", ref="sacrum"):
    out = df.copy()
    out[f"{lado}_x_rel"] = out[f"{lado}_meta_tars_x"] - out[f"{ref}_x"]
    out[f"{lado}_z_rel"] = out[f"{lado}_meta_tars_z"] - out[f"{ref}_z"]
    return out


def extraer_ciclos_lado(df, lado="l", npts=100,
                        min_frames=15, max_frames=120, min_dist=12):
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
        tnew, xnew, znew = reinterpolar_ciclo(seg["x_rel"].values, seg["z_rel"].values, npts=npts)

        ciclo = pd.DataFrame({
            "phase": tnew,
            "x_rel": xnew,
            "z_rel": znew,
            "frame_start": int(seg["frame_num"].iloc[0]),
            "frame_end": int(seg["frame_num"].iloc[-1]),
            "n_frames": len(seg)
        })
        ciclos.append(ciclo)

    return sub, z_smooth, mins, ciclos


def calidad_ciclo(c):
    dx = c["x_rel"].max() - c["x_rel"].min()
    dz = c["z_rel"].max() - c["z_rel"].min()
    return dx, dz


def calcular_error_cierre(df, xcol="x_mean", zcol="z_mean", phasecol="phase"):
    df = df.sort_values(phasecol).copy()

    x0 = df.iloc[0][xcol]
    z0 = df.iloc[0][zcol]
    x1 = df.iloc[-1][xcol]
    z1 = df.iloc[-1][zcol]

    err = np.sqrt((x1 - x0)**2 + (z1 - z0)**2)

    return {
        "x_inicio": float(x0),
        "z_inicio": float(z0),
        "x_final": float(x1),
        "z_final": float(z1),
        "dx_cierre": float(x1 - x0),
        "dz_cierre": float(z1 - z0),
        "error_cierre": float(err)
    }


def cierre_suave_lineal(df, xcol="x_mean", zcol="z_mean", phasecol="phase"):
    df = df.sort_values(phasecol).copy()

    phase = df[phasecol].to_numpy()
    x = df[xcol].to_numpy()
    z = df[zcol].to_numpy()

    dx = x[-1] - x[0]
    dz = z[-1] - z[0]

    x_corr = x - phase * dx
    z_corr = z - phase * dz

    out = df.copy()
    out["x_cerrado"] = x_corr
    out["z_cerrado"] = z_corr

    return out


def longitud_trayectoria(df, xcol="x_cerrado", zcol="z_cerrado"):
    x = df[xcol].to_numpy()
    z = df[zcol].to_numpy()
    ds = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
    return float(np.sum(ds))


def metricas_trayectoria(df, lado, xcol, zcol, phasecol="phase"):
    df = df.sort_values(phasecol).copy()

    x = df[xcol].to_numpy()
    z = df[zcol].to_numpy()
    phase = df[phasecol].to_numpy()

    err = calcular_error_cierre(df, xcol=xcol, zcol=zcol, phasecol=phasecol)

    return {
        "lado": lado,
        "dx_norm": float(np.max(x) - np.min(x)),
        "dz_norm": float(np.max(z) - np.min(z)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "fase_xmax": float(phase[np.argmax(x)]),
        "fase_xmin": float(phase[np.argmin(x)]),
        "fase_zmax": float(phase[np.argmax(z)]),
        "fase_zmin": float(phase[np.argmin(z)]),
        "longitud_trayectoria": longitud_trayectoria(df, xcol=xcol, zcol=zcol),
        "x_inicio": err["x_inicio"],
        "z_inicio": err["z_inicio"],
        "x_final": err["x_final"],
        "z_final": err["z_final"],
        "dx_cierre": err["dx_cierre"],
        "dz_cierre": err["dz_cierre"],
        "error_cierre": err["error_cierre"],
    }


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

            cols_necesarias = [
                f"{MARCADOR_REF}_x", f"{MARCADOR_REF}_z",
                f"{LADO}_iliac_x", f"{LADO}_iliac_y", f"{LADO}_iliac_z",
                f"{LADO}_stifle_x", f"{LADO}_stifle_y", f"{LADO}_stifle_z",
                f"{LADO}_hock_x", f"{LADO}_hock_y", f"{LADO}_hock_z",
                f"{LADO}_meta_tars_x", f"{LADO}_meta_tars_y", f"{LADO}_meta_tars_z",
            ]

            if not set(cols_necesarias).issubset(df.columns):
                continue

            L = longitud_funcional_pata(df, LADO)

            if pd.isna(L) or L <= 0:
                resumen_trials.append({
                    "subject_id": info["subject_id"],
                    "trial_num": info["trial_num"],
                    "direction": info["direction"],
                    "lado": LADO,
                    "archivo": path_txt.name,
                    "fps": meta["fps"],
                    "n_frames": len(df),
                    "n_ciclos": 0,
                    "longitud_funcional": np.nan,
                    "estado": "sin_longitud_funcional"
                })
                continue

            dfr = preparar_lado(df, lado=LADO, ref=MARCADOR_REF)

            sub, z_smooth, mins, ciclos = extraer_ciclos_lado(
                dfr,
                lado=LADO,
                npts=NPTS,
                min_frames=MIN_FRAMES_CICLO,
                max_frames=MAX_FRAMES_CICLO,
                min_dist=MIN_DIST_MINIMOS
            )

            if not ciclos:
                resumen_trials.append({
                    "subject_id": info["subject_id"],
                    "trial_num": info["trial_num"],
                    "direction": info["direction"],
                    "lado": LADO,
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

                # se mantiene la misma convención que en el pipeline base
                # solo corregimos dirección b en el eje de avance
                if info["direction"] == "b":
                    c2["x_norm"] = -c2["x_norm"]
                    c2["x_rel"] = -c2["x_rel"]

                c2["subject_id"] = info["subject_id"]
                c2["trial_num"] = info["trial_num"]
                c2["direction"] = info["direction"]
                c2["lado"] = LADO
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
                "lado": LADO,
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

    df_resumen.to_csv(RUTA_SALIDA / "resumen_trials_izquierdo.csv", index=False, encoding="utf-8-sig")
    df_errores.to_csv(RUTA_SALIDA / "errores_procesamiento_izquierdo.csv", index=False, encoding="utf-8-sig")

    if not df_ciclos.empty:
        df_ciclos.to_csv(RUTA_SALIDA / "ciclos_izquierdo_normalizados.csv", index=False, encoding="utf-8-sig")

    print(f"Resumen trials guardado en: {RUTA_SALIDA / 'resumen_trials_izquierdo.csv'}")
    print(f"Errores guardados en: {RUTA_SALIDA / 'errores_procesamiento_izquierdo.csv'}")

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

        prom_trial.to_csv(RUTA_SALIDA / "promedio_izquierdo_por_trial.csv", index=False, encoding="utf-8-sig")
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

        prom_sujeto.to_csv(RUTA_SALIDA / "promedio_izquierdo_por_sujeto.csv", index=False, encoding="utf-8-sig")
    else:
        prom_sujeto = pd.DataFrame()

    # =====================================================
    # PROMEDIO GLOBAL DEL IZQUIERDO
    # =====================================================
    if not prom_sujeto.empty:
        g_global = prom_sujeto.groupby(["lado", "phase"])[["x_mean", "z_mean"]]

        prom_global = g_global.mean().rename(columns={"x_mean": "x_mean", "z_mean": "z_mean"})
        prom_global["x_std"] = g_global.std()["x_mean"]
        prom_global["z_std"] = g_global.std()["z_mean"]
        prom_global = prom_global.reset_index()

        prom_global.to_csv(RUTA_SALIDA / "promedio_global_izquierdo.csv", index=False, encoding="utf-8-sig")
    else:
        prom_global = pd.DataFrame()

    # =====================================================
    # CIERRE SUAVE
    # =====================================================
    if prom_global.empty:
        print("\nNo se pudo construir el promedio global izquierdo.")
        return

    prom_global = prom_global.sort_values("phase").reset_index(drop=True)

    err_orig = calcular_error_cierre(prom_global, xcol="x_mean", zcol="z_mean", phasecol="phase")
    prom_cerrado = cierre_suave_lineal(prom_global, xcol="x_mean", zcol="z_mean", phasecol="phase")
    err_corr = calcular_error_cierre(prom_cerrado, xcol="x_cerrado", zcol="z_cerrado", phasecol="phase")

    met_orig = metricas_trayectoria(prom_global, lado=LADO, xcol="x_mean", zcol="z_mean", phasecol="phase")
    met_corr = metricas_trayectoria(prom_cerrado, lado=LADO, xcol="x_cerrado", zcol="z_cerrado", phasecol="phase")

    prom_cerrado.to_csv(RUTA_SALIDA / "promedio_global_izquierdo_cerrado.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([met_orig]).to_csv(RUTA_SALIDA / "metricas_izquierdo_original.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([met_corr]).to_csv(RUTA_SALIDA / "metricas_izquierdo_cerrado.csv", index=False, encoding="utf-8-sig")

    # =====================================================
    # REPORTE
    # =====================================================
    print("\n=== REPORTE LADO IZQUIERDO ===")

    if not df_ciclos.empty:
        n_ciclos_reales = (
            df_ciclos[["subject_id", "trial_num", "direction", "lado", "cycle_id_local"]]
            .drop_duplicates()
            .shape[0]
        )
    else:
        n_ciclos_reales = 0

    print(f"Ciclos totales reales: {n_ciclos_reales}")

    if not df_resumen.empty:
        print("\nTop 10 trials con más ciclos:")
        print(
            df_resumen.sort_values("n_ciclos", ascending=False)
            .head(10)[["subject_id", "trial_num", "direction", "archivo", "n_ciclos"]]
            .to_string(index=False)
        )

    print("\n=== ERROR DE CIERRE ORIGINAL ===")
    print(f"Inicio: ({err_orig['x_inicio']:.6f}, {err_orig['z_inicio']:.6f})")
    print(f"Final : ({err_orig['x_final']:.6f}, {err_orig['z_final']:.6f})")
    print(f"dx cierre: {err_orig['dx_cierre']:.6f}")
    print(f"dz cierre: {err_orig['dz_cierre']:.6f}")
    print(f"Error original: {err_orig['error_cierre']:.6f}")

    print("\n=== ERROR DE CIERRE DESPUÉS DEL AJUSTE ===")
    print(f"Inicio: ({err_corr['x_inicio']:.6f}, {err_corr['z_inicio']:.6f})")
    print(f"Final : ({err_corr['x_final']:.6f}, {err_corr['z_final']:.6f})")
    print(f"dx cierre: {err_corr['dx_cierre']:.6f}")
    print(f"dz cierre: {err_corr['dz_cierre']:.6f}")
    print(f"Error final: {err_corr['error_cierre']:.6f}")

    print(f"\nArchivos guardados en:\n{RUTA_SALIDA}")

    # =====================================================
    # PLOTS
    # =====================================================
    if GENERAR_PLOTS:
        plt.figure(figsize=(7, 6))
        plt.plot(prom_global["x_mean"], prom_global["z_mean"], label="Original", linewidth=2)
        plt.plot(prom_cerrado["x_cerrado"], prom_cerrado["z_cerrado"], label="Cierre suave", linewidth=2)
        plt.scatter([prom_global.iloc[0]["x_mean"]], [prom_global.iloc[0]["z_mean"]], label="Inicio original")
        plt.scatter([prom_global.iloc[-1]["x_mean"]], [prom_global.iloc[-1]["z_mean"]], label="Final original")
        plt.xlabel("X normalizada")
        plt.ylabel("Z normalizada")
        plt.title("Lado izquierdo - curva original vs corregida")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(prom_global["phase"], prom_global["x_mean"], label="X original")
        plt.plot(prom_cerrado["phase"], prom_cerrado["x_cerrado"], label="X cerrado")
        plt.plot(prom_global["phase"], prom_global["z_mean"], label="Z original")
        plt.plot(prom_cerrado["phase"], prom_cerrado["z_cerrado"], label="Z cerrado")
        plt.xlabel("Fase")
        plt.ylabel("Posición normalizada")
        plt.title("Lado izquierdo - corrección por fase")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(7, 6))
        plt.plot(prom_cerrado["x_cerrado"], prom_cerrado["z_cerrado"], label="Curva cerrada", linewidth=2)
        plt.scatter([prom_cerrado.iloc[0]["x_cerrado"]], [prom_cerrado.iloc[0]["z_cerrado"]], label="Inicio cerrado")
        plt.scatter([prom_cerrado.iloc[-1]["x_cerrado"]], [prom_cerrado.iloc[-1]["z_cerrado"]], label="Final cerrado")
        plt.xlabel("X normalizada")
        plt.ylabel("Z normalizada")
        plt.title("Lado izquierdo - verificación del cierre")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()