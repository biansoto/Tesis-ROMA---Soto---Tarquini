from pathlib import Path

RUTA = Path(r"C:\Users\bianc\Desktop\bioingenieria\tesis\procesamiento de trayectorias\procesamiento_dataset\salidas_caminata_global")

for f in RUTA.glob("*.csv"):
    print(f.name)