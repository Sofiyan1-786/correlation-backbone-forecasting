#!/usr/bin/env python3
"""Train ST-GNN models for full, MDS, and random universes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class STGNN(nn.Module):
    """Minimal graph-temporal model used for reproducibility checks."""

    def __init__(self, seq_len: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gcn = nn.Linear(seq_len, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.einsum("nm,bms->bns", adj, x)
        h = F.relu(self.gcn(support))
        h = self.norm(h)
        h = self.dropout(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    row_sum = adj.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    return adj / row_sum[:, None]


def build_sequences(features: np.ndarray, target: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(features) - seq_len):
        x.append(features[i : i + seq_len].T)
        y.append(target[i + seq_len])
    return np.array(x), np.array(y)


def prepare_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_train = train_df.mean(axis=1).to_numpy()
    target_test = test_df.mean(axis=1).to_numpy()

    x_train_raw = train_df.iloc[:, indices].to_numpy()
    x_test_raw = test_df.iloc[:, indices].to_numpy()

    mu = x_train_raw.mean(axis=0)
    sigma = x_train_raw.std(axis=0)
    sigma[sigma == 0] = 1.0

    x_train_norm = (x_train_raw - mu) / sigma
    x_test_norm = (x_test_raw - mu) / sigma

    x_train, y_train = build_sequences(x_train_norm, target_train, seq_len)
    x_test, y_test = build_sequences(x_test_norm, target_test, seq_len)
    return x_train, y_train, x_test, y_test


def train_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    adjacency: np.ndarray,
    seq_len: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> np.ndarray:
    model = STGNN(seq_len=seq_len, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    adj_t = torch.as_tensor(normalize_adjacency(adjacency), dtype=torch.float32, device=device)
    train_ds = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for bx, by in train_dl:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            pred = model(bx, adj_t)
            loss = loss_fn(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.as_tensor(x_test, dtype=torch.float32, device=device), adj_t).cpu().numpy()
    return preds


def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float("nan")
    std = np.std(returns, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * np.mean(returns) / std)


def compute_quarterly_results(pred_df: pd.DataFrame) -> pd.DataFrame:
    data = pred_df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")

    strat = pd.DataFrame(index=data.index)
    strat["Full"] = (data["pred_full"] > 0).astype(float) * data["actual"]
    strat["MDS"] = (data["pred_mds"] > 0).astype(float) * data["actual"]
    strat["Random"] = (data["pred_random"] > 0).astype(float) * data["actual"]

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

    return pd.DataFrame(rows)


def main() -> None:
    config = load_config(ROOT / "configs" / "default_config.yaml")
    seed = int(config["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    returns = pd.read_parquet(ROOT / config["data"]["returns_file"])
    corr = pd.read_csv(ROOT / config["outputs"]["correlation_file"], index_col=0)
    adjacency = np.load(ROOT / config["outputs"]["adjacency_file"])
    mds_nodes = pd.read_csv(ROOT / config["outputs"]["mds_nodes_file"])["node_index"].to_numpy(dtype=int)

    random_nodes_path = ROOT / config["outputs"]["random_nodes_file"]
    if random_nodes_path.exists():
        random_nodes = pd.read_csv(random_nodes_path)["node_index"].to_numpy(dtype=int)
    else:
        rng = np.random.default_rng(seed)
        random_nodes = np.sort(rng.choice(len(corr), size=len(mds_nodes), replace=False))
        pd.DataFrame({"node_index": random_nodes, "symbol": [corr.index[i] for i in random_nodes]}).to_csv(
            random_nodes_path,
            index=False,
        )

    if returns.shape[1] != adjacency.shape[0]:
        raise ValueError("Returns columns and adjacency dimensions do not match")

    train_end = max(50, int(len(returns) * float(config["split"]["train_ratio"])))
    train_df = returns.iloc[:train_end]
    test_df = returns.iloc[train_end:]

    seq_len = int(config["model"]["seq_len"])
    hidden_dim = int(config["model"]["hidden_dim"])
    dropout = float(config["model"]["dropout"])
    epochs = int(config["model"]["epochs"])
    batch_size = int(config["model"]["batch_size"])
    learning_rate = float(config["model"]["learning_rate"])

    full_idx = np.arange(returns.shape[1])
    xtr_full, ytr_full, xte_full, yte_full = prepare_variant(train_df, test_df, full_idx, seq_len)
    xtr_mds, ytr_mds, xte_mds, _ = prepare_variant(train_df, test_df, mds_nodes, seq_len)
    xtr_rand, ytr_rand, xte_rand, _ = prepare_variant(train_df, test_df, random_nodes, seq_len)

    if min(len(xtr_full), len(xtr_mds), len(xtr_rand), len(xte_full)) == 0:
        raise ValueError("Insufficient observations after sequence construction")

    pred_full = train_and_predict(
        xtr_full,
        ytr_full,
        xte_full,
        adjacency,
        seq_len,
        hidden_dim,
        dropout,
        epochs,
        batch_size,
        learning_rate,
        device,
    )
    pred_mds = train_and_predict(
        xtr_mds,
        ytr_mds,
        xte_mds,
        adjacency[np.ix_(mds_nodes, mds_nodes)],
        seq_len,
        hidden_dim,
        dropout,
        epochs,
        batch_size,
        learning_rate,
        device,
    )
    pred_rand = train_and_predict(
        xtr_rand,
        ytr_rand,
        xte_rand,
        adjacency[np.ix_(random_nodes, random_nodes)],
        seq_len,
        hidden_dim,
        dropout,
        epochs,
        batch_size,
        learning_rate,
        device,
    )

    test_dates = test_df.index[seq_len:]
    pred_df = pd.DataFrame(
        {
            "date": test_dates,
            "actual": yte_full,
            "pred_full": pred_full,
            "pred_mds": pred_mds,
            "pred_random": pred_rand,
        }
    )

    pred_path = ROOT / config["outputs"]["predictions_file"]
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)

    results = compute_quarterly_results(pred_df)
    results_path = ROOT / config["outputs"]["results_file"]
    results.to_csv(results_path, index=False)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved evaluation results: {results_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
