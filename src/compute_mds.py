#!/usr/bin/env python3
"""Compute a greedy minimum dominating set approximation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def greedy_dominating_set(adjacency: np.ndarray) -> list[int]:
    n = adjacency.shape[0]
    neighborhoods = [set(np.where(adjacency[i] == 1)[0]) | {i} for i in range(n)]

    uncovered = set(range(n))
    selected: list[int] = []

    while uncovered:
        best_node = max(range(n), key=lambda i: len(neighborhoods[i] & uncovered))
        selected.append(best_node)
        uncovered -= neighborhoods[best_node]

    return sorted(set(selected))


def main() -> None:
    config = load_config(ROOT / "configs" / "default_config.yaml")

    corr = pd.read_csv(ROOT / config["outputs"]["correlation_file"], index_col=0)
    adjacency = np.load(ROOT / config["outputs"]["adjacency_file"])

    if adjacency.shape[0] != corr.shape[0]:
        raise ValueError("Adjacency size does not match correlation matrix")

    mds_nodes = greedy_dominating_set(adjacency)
    symbols = corr.index.to_list()

    out_df = pd.DataFrame(
        {
            "node_index": mds_nodes,
            "symbol": [symbols[i] for i in mds_nodes],
        }
    )

    out_path = ROOT / config["outputs"]["mds_nodes_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    compression = len(mds_nodes) / len(symbols)
    print(f"Saved MDS nodes: {out_path}")
    print(f"K={len(mds_nodes)} / N={len(symbols)} (compression={compression:.2%})")


if __name__ == "__main__":
    main()
