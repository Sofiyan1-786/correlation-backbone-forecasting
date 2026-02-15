#!/usr/bin/env python3
"""Run paired statistical tests for MDS vs random Sharpe results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_std(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def main() -> None:
    config = load_config(ROOT / "configs" / "default_config.yaml")

    results_path = ROOT / config["outputs"]["results_file"]
    stats_path = ROOT / config["outputs"]["stats_file"]

    df = pd.read_csv(results_path)
    required = {"MDS_Sharpe", "Random_Sharpe"}
    if not required.issubset(df.columns):
        raise ValueError(f"Results file must contain columns: {sorted(required)}")

    x = df["MDS_Sharpe"].astype(float).to_numpy()
    y = df["Random_Sharpe"].astype(float).to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    d = x - y

    n = int(len(d))
    if n == 0:
        raise ValueError("No valid paired observations found")

    t_res = ttest_rel(x, y, nan_policy="omit")
    try:
        w_res = wilcoxon(d)
        w_stat = float(w_res.statistic)
        p_w = float(w_res.pvalue)
    except ValueError:
        w_stat = float("nan")
        p_w = float("nan")

    sign_k = int((d > 0).sum())
    sign_p = float(binomtest(sign_k, n, 0.5, alternative="greater").pvalue)

    summary = {
        "n": n,
        "mds_mean": float(np.mean(x)),
        "mds_std": sample_std(x),
        "random_mean": float(np.mean(y)),
        "random_std": sample_std(y),
        "diff_mean_mds_minus_random": float(np.mean(d)),
        "diff_std_mds_minus_random": sample_std(d),
        "t_stat": float(t_res.statistic),
        "p_value_ttest": float(t_res.pvalue),
        "w_stat": w_stat,
        "p_value_wilcoxon": p_w,
        "sign_k_positive": sign_k,
        "sign_pvalue_greater": sign_p,
    }

    lines = [
        "Paired Statistical Summary: MDS Sharpe vs Random Sharpe",
        "=" * 60,
        f"n                         : {summary['n']}",
        f"MDS mean ± std            : {summary['mds_mean']:.6f} ± {summary['mds_std']:.6f}",
        f"Random mean ± std         : {summary['random_mean']:.6f} ± {summary['random_std']:.6f}",
        f"Diff mean ± std           : {summary['diff_mean_mds_minus_random']:.6f} ± {summary['diff_std_mds_minus_random']:.6f}",
        f"Paired t-test             : t={summary['t_stat']:.6f}, p={summary['p_value_ttest']:.6f}",
        f"Wilcoxon signed-rank      : W={summary['w_stat']:.6f}, p={summary['p_value_wilcoxon']:.6f}",
        f"Sign test (MDS > Random)  : k={summary['sign_k_positive']}/{summary['n']}, p={summary['sign_pvalue_greater']:.6f}",
    ]

    summary_text = "\n".join(lines)
    print(summary_text)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(summary_text + "\n", encoding="utf-8")

    summary_csv_path = stats_path.with_suffix(".csv")
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    print(f"Saved summary text: {stats_path}")
    print(f"Saved summary csv: {summary_csv_path}")


if __name__ == "__main__":
    main()
