from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

moves = {
    # 01 - procesamiento de marcha
    "identificacion_perros.py": "scripts/01_procesamiento_marcha/00_identificacion_perros_dataset.py",
    "leer_archivos.py": "scripts/01_procesamiento_marcha/00_inspeccion_archivos.py",
    "1_1_procesamiento_nase.py": "scripts/01_procesamiento_marcha/01_procesamiento_dataset_3ddogs.py",
    "1_procesamiento_base.py": "scripts/01_procesamiento_marcha/02_procesamiento_base_experimental.py",
    "pataizquierda.py": "scripts/01_procesamiento_marcha/03_procesamiento_pata_izquierda.py",
    "trayectorias.py": "scripts/01_procesamiento_marcha/04_graficos_trayectorias.py",

    # 02 - trayectoria objetivo
    "curva_objetivo.py": "scripts/02_trayectoria_objetivo/01_seleccion_curva_objetivo.py",
    "modelo_matematico_trayectoria.py": "scripts/02_trayectoria_objetivo/02_ajuste_fourier_trayectoria.py",
    "trayectoria_objetivo_final_fourier.py": "scripts/02_trayectoria_objetivo/03_trayectoria_final_fourier.py",
    "validación_curva_perro.py": "scripts/02_trayectoria_objetivo/04_validacion_curva_perro.py",

    # OJO: este script tenía medidas viejas, por eso queda archivado para revisión.
    "escalar_trayectoria_cm.py": "archive/legacy_por_revisar/03_escalado_trayectoria_cm_revisar_medidas.py",

    # 03 - modelo cinemático
    "mecanismo_fourier.py": "scripts/03_modelo_cinematico/01_cinematica_inversa_fourier.py",
    "modelo_cinematico_2_eslabones_figura.py": "scripts/03_modelo_cinematico/02_figura_modelo_dos_eslabones.py",
    "modelo_preliminar_ortesis_alineada.py": "scripts/03_modelo_cinematico/03_modelo_preliminar_ortesis_alineada.py",

    # 04 - asistencia y zapata
    "parametros_zapata.py": "scripts/04_asistencia_y_zapata/01_parametros_zapata.py",
    "zapata_puntual_con_dutty_cycle.py": "scripts/04_asistencia_y_zapata/02_zapata_puntual_con_duty_cycle.py",
    "zapata_extendida_tipo_patin.py": "scripts/04_asistencia_y_zapata/03_zapata_extendida_tipo_patin.py",
    "integrado_zapata_rocker_curva.py": "scripts/04_asistencia_y_zapata/04_integrado_zapata_rocker_curva.py",
    "mecanismo_articular_fisico_amortiguado.py": "scripts/04_asistencia_y_zapata/05_mecanismo_articular_amortiguado.py",

    # 05 - figuras tesis / CAD conceptual
    "figura de arquitectura mecánica final evaluada-adoptada.py": "scripts/05_figuras_tesis/01_figura_arquitectura_mecanica_final.py",
    "1_modelo_cad_2d.py": "scripts/05_figuras_tesis/02_modelo_cad_2d_version_1.py",
    "2_modelo_cad_2d.py": "scripts/05_figuras_tesis/03_modelo_cad_2d_version_2.py",

    # Exploratorio / descartado
    "optimizacion_geometrica_jansen.py": "archive/exploratorio_jansen/01_optimizacion_geometrica_jansen.py",
    "optimizar_jansen_reducido.py": "archive/exploratorio_jansen/02_optimizar_jansen_reducido.py",
    "optimizar_jansen_4barras_restringido.py": "archive/exploratorio_jansen/03_optimizar_jansen_4barras_restringido.py",
    "optimizar_jansen_6_barras.py": "archive/exploratorio_jansen/04_optimizar_jansen_6_barras.py",
    "7_comparacionTheoJansen.py": "archive/exploratorio_jansen/05_comparacion_theo_jansen.py",

    # Intermedios / legacy
    "2_validacioncierre.py": "archive/legacy_procesamiento_intermedio/02_validacion_cierre.py",
    "3_cierresuaveporlado.py": "archive/legacy_procesamiento_intermedio/03_cierre_suave_por_lado.py",
    "4_comparacionlados.py": "archive/legacy_procesamiento_intermedio/04_comparacion_lados.py",
    "5_ajusteporfourier.py": "archive/legacy_procesamiento_intermedio/05_ajuste_por_fourier.py",
    "6_ajustecinematicofourier.py": "archive/legacy_procesamiento_intermedio/06_analisis_cinematico_fourier.py",
}

readme = """# ROMA - Código de procesamiento y modelado

Este repositorio contiene los scripts de Python utilizados durante el desarrollo del Proyecto Final de Carrera **ROMA: Rehabilitación Osteomuscular y Marcha Asistida**.

ROMA es un dispositivo ortésico-exoesquelético pasivo para asistencia parcial de marcha en miembros posteriores caninos.

## Estructura del repositorio

```text
scripts/
├── 01_procesamiento_marcha/
├── 02_trayectoria_objetivo/
├── 03_modelo_cinematico/
├── 04_asistencia_y_zapata/
└── 05_figuras_tesis/

archive/
├── exploratorio_jansen/
├── legacy_por_revisar/
└── legacy_procesamiento_intermedio/

data/
outputs/
