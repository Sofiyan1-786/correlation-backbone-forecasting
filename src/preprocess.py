#!/usr/bin/env python3
"""Preprocess downloaded prices into cleaned log-return series."""

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

    prices_path = ROOT / config["data"]["raw_prices_file"]
    returns_path = ROOT / config["data"]["returns_file"]

    prices = pd.read_parquet(prices_path).sort_index()
    prices = prices.ffill().bfill().dropna(axis=1, how="any")

    log_prices = np.log(prices)
    returns = log_prices.diff().dropna()

    lower = returns.quantile(0.005)
    upper = returns.quantile(0.995)
    returns = returns.clip(lower=lower, upper=upper, axis=1)

    returns_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(returns_path)

    print(f"Saved returns: {returns_path}")
    print(f"Rows={len(returns)}, Columns={returns.shape[1]}")


if __name__ == "__main__":
    main()
