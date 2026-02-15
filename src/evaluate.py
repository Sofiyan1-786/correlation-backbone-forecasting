#!/usr/bin/env python3
"""Evaluate model predictions and build quarterly Sharpe comparison tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float("nan")
    std = np.std(returns, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * np.mean(returns) / std)


def main() -> None:
    config = load_config(ROOT / "configs" / "default_config.yaml")
    pred_path = ROOT / config["outputs"]["predictions_file"]
    results_path = ROOT / config["outputs"]["results_file"]

    pred = pd.read_csv(pred_path, parse_dates=["date"]).set_index("date")

    strat = pd.DataFrame(index=pred.index)
    strat["Full"] = (pred["pred_full"] > 0).astype(float) * pred["actual"]
    strat["MDS"] = (pred["pred_mds"] > 0).astype(float) * pred["actual"]
    strat["Random"] = (pred["pred_random"] > 0).astype(float) * pred["actual"]

    rows: list[dict[str, float | str | int]] = []
    for period, block in strat.groupby(strat.index.to_period("Q")):
        if len(block) < 20:
            continue
        rows.append(
            {
                "Period": str(period),
                "Full_Sharpe": annualized_sharpe(block["Full"].to_numpy()),
                "MDS_Sharpe": annualized_sharpe(block["MDS"].to_numpy()),
                "Random_Sharpe": annualized_sharpe(block["Random"].to_numpy()),
                "N_obs": int(len(block)),
            }
        )

    if not rows:
        rows.append(
            {
                "Period": "FULL_SAMPLE",
                "Full_Sharpe": annualized_sharpe(strat["Full"].to_numpy()),
                "MDS_Sharpe": annualized_sharpe(strat["MDS"].to_numpy()),
                "Random_Sharpe": annualized_sharpe(strat["Random"].to_numpy()),
                "N_obs": int(len(strat)),
            }
        )

    out_df = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(results_path, index=False)

    print(f"Saved evaluation results: {results_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
