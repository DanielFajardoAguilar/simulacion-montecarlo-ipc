"""
Simulación Monte Carlo de un activo usando datos históricos desde un CSV.
Modelo: Geometric Brownian Motion (GBM) con log-rendimientos.

Requisitos:
  pip install pandas numpy matplotlib

Ejemplos:
  python mc_csv_demo.py --csv "IPC_MXX_yahoo.csv" --days 180 --sims 300
  python mc_csv_demo.py --csv "IPC_MXX_yahoo.csv" --price_col Adjusted --days 252 --sims 500
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_close_from_csv(csv_path: str, date_col: str = "Date", price_col: str = "Adjusted") -> pd.Series:
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")
    close = df.set_index(date_col)[price_col].astype(float)


    if date_col not in df.columns:
        raise ValueError(f"No existe la columna de fecha '{date_col}' en el CSV.")

    # Parseo robusto de fechas 
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Elegir columna de precio: Adjusted si existe, si no Close
    if price_col not in df.columns:
        if price_col.lower() == "adjusted" and "Close" in df.columns:
            price_col = "Close"
        else:
            raise ValueError(
                f"No existe la columna de precio '{price_col}'. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    close = df.set_index(date_col)[price_col].copy()
    close.name = "Close"

    if len(close) < 30:
        raise ValueError("Muy pocos datos para estimar volatilidad con sentido (mínimo ~30).")

    # Filtro por si hubiera ceros/negativos 
    close = close[close > 0]
    if len(close) < 30:
        raise ValueError("Tras filtrar precios no válidos quedaron muy pocos datos.")
    return close


def log_returns(close: pd.Series) -> pd.Series:
    r = np.log(close / close.shift(1)).dropna()
    r.name = "log_return"
    return r


def simulate_gbm_paths(s0: float, mu: float, sigma: float, days: int, sims: int, seed: int = 42) -> np.ndarray:
    """
    GBM discreto con dt=1 día:
      S_{t+1} = S_t * exp((mu - 0.5*sigma^2) + sigma*Z_t)
    mu y sigma en escala diaria.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((sims, days))

    drift = mu - 0.5 * sigma**2
    increments = drift + sigma * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((sims, 1)), log_paths])

    return s0 * np.exp(log_paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Ruta al CSV (ej. IPC_MXX.csv)")
    ap.add_argument("--date_col", type=str, default="Date", help="Nombre de la columna fecha en el CSV")
    ap.add_argument("--price_col", type=str, default="Adjusted", help="Columna de precio (Adjusted o Close)")
    ap.add_argument("--days", type=int, default=180, help="Días a simular")
    ap.add_argument("--sims", type=int, default=200, help="Número de trayectorias")
    ap.add_argument("--seed", type=int, default=42, help="Semilla RNG")
    args = ap.parse_args()

    # 1) Datos
    close = load_close_from_csv(args.csv, date_col=args.date_col, price_col=args.price_col)
    r = log_returns(close)

    # 2) Parámetros (históricos)
    mu_d = float(r.mean())
    sigma_d = float(r.std(ddof=1))
    mu_a = mu_d * 252
    sigma_a = sigma_d * np.sqrt(252)

    s0 = float(close.iloc[-1])

    # 3) Simulación
    paths = simulate_gbm_paths(s0, mu_d, sigma_d, args.days, args.sims, args.seed)
    terminal = paths[:, -1]

    # 4) Resumen
    q01, q05, q50, q95, q99 = np.quantile(terminal, [0.01, 0.05, 0.50, 0.95, 0.99])

    print("\n====================")
    print("RESUMEN HISTÓRICO")
    print("====================")
    print(f"CSV: {args.csv}")
    print(f"Precio usado: {args.price_col}")
    print(f"Observaciones: {len(close)} | Rango: {close.index.min().date()} a {close.index.max().date()}")
    print(f"Último precio: {s0:,.2f}")
    print(f"Mu diario (log): {mu_d:.6f}")
    print(f"Vol diaria:      {sigma_d:.6f}")
    print(f"Mu anual aprox:  {mu_a:.4f}")
    print(f"Vol anual:       {sigma_a:.4f}")

    print("\n====================")
    print("RESUMEN MONTE CARLO")
    print("====================")
    print(f"Horizonte: {args.days} días | Simulaciones: {args.sims}")
    print(f"Min:   {terminal.min():,.2f}")
    print(f"P01:   {q01:,.2f}")
    print(f"P05:   {q05:,.2f}")
    print(f"Med:   {q50:,.2f}")
    print(f"P95:   {q95:,.2f}")
    print(f"P99:   {q99:,.2f}")
    print(f"Max:   {terminal.max():,.2f}")
    print(f"Mean:  {terminal.mean():,.2f}")

    # 5) Gráfica
    plt.figure(figsize=(10, 6))
    for i in range(min(args.sims, 100)):
        plt.plot(paths[i, :], lw=1)

    plt.title(f"Simulación Monte Carlo (GBM) - {args.price_col} a {args.days} días")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Precio simulado")
    plt.grid(True, which="both")
    plt.show()


if __name__ == "__main__":
    main()
