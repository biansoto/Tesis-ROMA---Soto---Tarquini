from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

moves = {
    "identificacion_perros.py": "scripts/01_procesamiento_marcha/00_identificacion_perros_dataset.py",
    "leer_archivos.py": "scripts/01_procesamiento_marcha/00_inspeccion_archivos.py",
    "1_1_procesamiento_nase.py": "scripts/01_procesamiento_marcha/01_procesamiento_dataset_3ddogs.py",
    "1_procesamiento_base.py": "scripts/01_procesamiento_marcha/02_procesamiento_base_experimental.py",
    "pataizquierda.py": "scripts/01_procesamiento_marcha/03_procesamiento_pata_izquierda.py",
    "trayectorias.py": "scripts/01_procesamiento_marcha/04_graficos_trayectorias.py",

    "curva_objetivo.py": "scripts/02_trayectoria_objetivo/01_seleccion_curva_objetivo.py",
    "modelo_matematico_trayectoria.py": "scripts/02_trayectoria_objetivo/02_ajuste_fourier_trayectoria.py",
    "trayectoria_objetivo_final_fourier.py": "scripts/02_trayectoria_objetivo/03_trayectoria_final_fourier.py",
    "validación_curva_perro.py": "scripts/02_trayectoria_objetivo/04_validacion_curva_perro.py",
    "escalar_trayectoria_cm.py": "archive/legacy_por_revisar/03_escalado_trayectoria_cm_revisar_medidas.py",

    "mecanismo_fourier.py": "scripts/03_modelo_cinematico/01_cinematica_inversa_fourier.py",
    "modelo_cinematico_2_eslabones_figura.py": "scripts/03_modelo_cinematico/02_figura_modelo_dos_eslabones.py",
    "modelo_preliminar_ortesis_alineada.py": "scripts/03_modelo_cinematico/03_modelo_preliminar_ortesis_alineada.py",

    "parametros_zapata.py": "scripts/04_asistencia_y_zapata/01_parametros_zapata.py",
    "zapata_puntual_con_dutty_cycle.py": "scripts/04_asistencia_y_zapata/02_zapata_puntual_con_duty_cycle.py",
    "zapata_extendida_tipo_patin.py": "scripts/04_asistencia_y_zapata/03_zapata_extendida_tipo_patin.py",
    "integrado_zapata_rocker_curva.py": "scripts/04_asistencia_y_zapata/04_integrado_zapata_rocker_curva.py",
    "mecanismo_articular_fisico_amortiguado.py": "scripts/04_asistencia_y_zapata/05_mecanismo_articular_amortiguado.py",

    "figura de arquitectura mecánica final evaluada-adoptada.py": "scripts/05_figuras_tesis/01_figura_arquitectura_mecanica_final.py",
    "1_modelo_cad_2d.py": "scripts/05_figuras_tesis/02_modelo_cad_2d_version_1.py",
    "2_modelo_cad_2d.py": "scripts/05_figuras_tesis/03_modelo_cad_2d_version_2.py",

    "optimizacion_geometrica_jansen.py": "archive/exploratorio_jansen/01_optimizacion_geometrica_jansen.py",
    "optimizar_jansen_reducido.py": "archive/exploratorio_jansen/02_optimizar_jansen_reducido.py",
    "optimizar_jansen_4barras_restringido.py": "archive/exploratorio_jansen/03_optimizar_jansen_4barras_restringido.py",
    "optimizar_jansen_6_barras.py": "archive/exploratorio_jansen/04_optimizar_jansen_6_barras.py",
    "7_comparacionTheoJansen.py": "archive/exploratorio_jansen/05_comparacion_theo_jansen.py",

    "2_validacioncierre.py": "archive/legacy_procesamiento_intermedio/02_validacion_cierre.py",
    "3_cierresuaveporlado.py": "archive/legacy_procesamiento_intermedio/03_cierre_suave_por_lado.py",
    "4_comparacionlados.py": "archive/legacy_procesamiento_intermedio/04_comparacion_lados.py",
    "5_ajusteporfourier.py": "archive/legacy_procesamiento_intermedio/05_ajuste_por_fourier.py",
    "6_ajustecinematicofourier.py": "archive/legacy_procesamiento_intermedio/06_analisis_cinematico_fourier.py",
}

readme = "# ROMA - Código de procesamiento y modelado\n\nRepositorio con scripts de Python utilizados durante el desarrollo del proyecto de tesis ROMA.\n\nROMA es un dispositivo ortésico-exoesquelético pasivo para asistencia parcial de marcha en miembros posteriores caninos.\n\n## Estructura\n\n- scripts/01_procesamiento_marcha\n- scripts/02_trayectoria_objetivo\n- scripts/03_modelo_cinematico\n- scripts/04_asistencia_y_zapata\n- scripts/05_figuras_tesis\n- archive/\n- data/\n- outputs/\n\n## Flujo principal\n\n1. Procesamiento de registros de marcha.\n2. Selección de trayectoria objetivo.\n3. Ajuste de trayectoria mediante Fourier.\n4. Resolución de cinemática inversa planar.\n5. Simulación preliminar de asistencia pasiva y zapata rocker.\n6. Generación de figuras para la tesis.\n\n## Nota\n\nAlgunos datos originales no se incluyen por tamaño, licencia o privacidad. Puede ser necesario ajustar rutas antes de ejecutar los scripts en otra computadora.\n\n## Requisitos\n\nPython 3.10 o superior.\n\nInstalar dependencias con:\n\npip install -r requirements.txt\n\n## Autoras\n\nBianca Soto Acosta y Martina Tarquini.\n"

requirements = "numpy\npandas\nmatplotlib\nscipy\nopenpyxl\n"

gitignore = "__pycache__/\n*.py[cod]\n.ipynb_checkpoints/\nvenv/\n.env/\n.venv/\n.DS_Store\nThumbs.db\n*.tmp\n*.log\ndata/raw/\n*.mp4\n*.avi\n*.mov\n*.zip\n"

def write_file(relative_path, content):
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def move_file(src, dst):
    src_path = ROOT / src
    dst_path = ROOT / dst

    if not src_path.exists():
        print(f"NO ENCONTRADO: {src}")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        print(f"YA EXISTE, no se pisa: {dst}")
        return

    shutil.move(str(src_path), str(dst_path))
    print(f"MOVIDO: {src} -> {dst}")

def main():
    write_file("README.md", readme)
    write_file("requirements.txt", requirements)
    write_file(".gitignore", gitignore)

    write_file("scripts/README.md", "Scripts organizados según la etapa metodológica de la tesis.\n")
    write_file("data/README.md", "Carpeta prevista para datos de entrada y datos procesados. Los datos crudos pueden no estar incluidos.\n")
    write_file("outputs/README.md", "Carpeta prevista para figuras, métricas y tablas generadas por los scripts.\n")
    write_file("archive/README.md", "Scripts exploratorios, alternativas descartadas o versiones intermedias.\n")

    for folder in [
        "data/raw",
        "data/processed",
        "outputs/figuras",
        "outputs/metricas",
        "outputs/tablas",
    ]:
        write_file(f"{folder}/.gitkeep", "")

    for src, dst in moves.items():
        move_file(src, dst)

    print("\nListo. Ahora corré: git status")

if __name__ == "__main__":
    main()