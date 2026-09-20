# ROMA - Código de procesamiento y modelado

Repositorio con scripts de Python utilizados durante el desarrollo del proyecto de tesis ROMA.

ROMA es un dispositivo ortésico-exoesquelético pasivo para asistencia parcial de marcha en miembros posteriores caninos.

## Estructura

- scripts/01_procesamiento_marcha
- scripts/02_trayectoria_objetivo
- scripts/03_modelo_cinematico
- scripts/04_asistencia_y_zapata
- scripts/05_figuras_tesis
- archive/
- data/
- outputs/

## Flujo principal

1. Procesamiento de registros de marcha.
2. Selección de trayectoria objetivo.
3. Ajuste de trayectoria mediante Fourier.
4. Resolución de cinemática inversa planar.
5. Simulación preliminar de asistencia pasiva y zapata rocker.
6. Generación de figuras para la tesis.

## Nota

Algunos datos originales no se incluyen por tamaño, licencia o privacidad. Puede ser necesario ajustar rutas antes de ejecutar los scripts en otra computadora.

## Requisitos

Python 3.10 o superior.

Instalar dependencias con:

pip install -r requirements.txt

## Autoras

Bianca Soto Acosta y Martina Tarquini.
