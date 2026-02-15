#!/usr/bin/env python3
"""Build a thresholded correlation network from preprocessed returns."""

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

    returns = pd.read_parquet(ROOT / config["data"]["returns_file"])
    topology_ratio = float(config["split"]["topology_ratio"])
    threshold = float(config["network"]["threshold"])

    topology_end = max(2, int(len(returns) * topology_ratio))
    topo_returns = returns.iloc[:topology_end]

    corr = topo_returns.corr().fillna(0.0)
    adjacency = (np.abs(corr.values) > threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    corr_path = ROOT / config["outputs"]["correlation_file"]
    adj_path = ROOT / config["outputs"]["adjacency_file"]
    corr_path.parent.mkdir(parents=True, exist_ok=True)

    corr.to_csv(corr_path)
    np.save(adj_path, adjacency)

    print(f"Saved correlation matrix: {corr_path}")
    print(f"Saved adjacency matrix: {adj_path}")
    print(f"Assets={adjacency.shape[0]}, Threshold={threshold:.3f}")


if __name__ == "__main__":
    main()
