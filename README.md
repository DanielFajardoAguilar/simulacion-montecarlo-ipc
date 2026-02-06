# Python – Simulación Monte Carlo con CSV histórico

Script en Python que toma una serie histórica desde un archivo CSV, calcula rendimientos logarítmicos, estima media y volatilidad diaria/anual y genera simulaciones Monte Carlo usando un modelo de Movimiento Browniano Geométrico (GBM). Incluye un resumen estadístico de los valores terminales y una gráfica con trayectorias simuladas.

## Para qué sirve
- Estimar rendimientos y volatilidad histórica a partir de precios.
- Generar escenarios de precio a un horizonte definido (simulación estocástica).
- Obtener percentiles y métricas rápidas para análisis exploratorio.

## Requisitos
- Python 3.9+ (recomendado)
- Paquetes:
  - pandas
  - numpy
  - matplotlib

Instalación:
pip install pandas numpy matplotlib

## Estructura del repositorio
- `Montecarlo.py` script principal (CLI)
- `IPC_MXX.csv` archivo CSV con la serie histórica

## Formato esperado del CSV
El CSV debe tener al menos:
- una columna de fecha (por ejemplo: Date)
- una columna de precio (por ejemplo: Adjusted o Close)

Ejemplo de ejecución:
python "Montecarlo.py" --csv "IPC_MXX.csv" --date_col Date --price_col Adjusted --days 180 --sims 300 --seed 42

## Parámetros
- `--csv` ruta al archivo CSV
- `--date_col` nombre de la columna de fecha
- `--price_col` nombre de la columna de precio
- `--days` horizonte a simular (días)
- `--sims` número de trayectorias
- `--seed` semilla para reproducibilidad

## Salida
1) Resumen histórico:
- número de observaciones y rango de fechas
- último precio
- media y volatilidad diaria (log)
- aproximación anual (252 días)

2) Resumen Monte Carlo:
- mínimo, máximo, media y percentiles del precio terminal
- gráfica con trayectorias simuladas
