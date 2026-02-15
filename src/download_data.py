#!/usr/bin/env python3
"""Download adjusted close prices for configured tickers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config(ROOT / "configs" / "default_config.yaml")
    np.random.seed(int(config["seed"]))

    tickers_path = ROOT / config["data"]["tickers_file"]
    output_path = ROOT / config["data"]["raw_prices_file"]

    tickers_df = pd.read_csv(tickers_path)
    symbols = tickers_df["symbol"].dropna().astype(str).tolist()
    if not symbols:
        raise ValueError("No symbols found in tickers file")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required. Install dependencies with requirements.txt") from exc

    prices = yf.download(
        symbols,
        start=config["download"]["start_date"],
        end=config["download"]["end_date"],
        interval=config["download"]["interval"],
        auto_adjust=True,
        progress=False,
    )

    if "Close" in prices:
        prices = prices["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])

    prices = prices.sort_index().dropna(how="all")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output_path)

    print(f"Saved prices: {output_path}")
    print(f"Rows={len(prices)}, Columns={prices.shape[1]}")


if __name__ == "__main__":
    main()
