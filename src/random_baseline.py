#!/usr/bin/env python3
"""Create a seeded random baseline with the same K as MDS."""

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

    corr = pd.read_csv(ROOT / config["outputs"]["correlation_file"], index_col=0)
    mds_nodes = pd.read_csv(ROOT / config["outputs"]["mds_nodes_file"])

    n = corr.shape[0]
    k = int(len(mds_nodes))

    rng = np.random.default_rng(int(config["seed"]))
    random_nodes = np.sort(rng.choice(n, size=k, replace=False))

    out_df = pd.DataFrame(
        {
            "node_index": random_nodes,
            "symbol": [corr.index[i] for i in random_nodes],
        }
    )

    out_path = ROOT / config["outputs"]["random_nodes_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved random baseline nodes: {out_path}")
    print(f"K={k} sampled with seed={config['seed']}")


if __name__ == "__main__":
    main()
