# Topology-Aware Financial Forecasting with Minimum Dominating Sets

This repository provides a clean reproducibility pipeline for topology-aware dimensionality reduction in financial time series forecasting. The workflow builds a correlation network, extracts a minimum dominating set (MDS) backbone, trains comparable ST-GNN models, and reports paired statistical tests against a random baseline.

The implementation is intentionally lightweight and reviewer-oriented: it avoids proprietary raw-data redistribution, uses deterministic seeds, and keeps all file paths configuration-driven for easy reruns in a fresh environment.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduction Steps

```bash
python src/download_data.py
python src/preprocess.py
python src/build_correlation_network.py
python src/compute_mds.py
python src/train_stgnn.py
python src/statistical_tests.py
```

## Data Disclaimer

Market data are downloaded from public provider APIs at run time and remain subject to each provider's terms of use. This repository does not redistribute vendor-provided raw price files.

## Citation

If you use this codebase, please cite the associated manuscript:

`Topology-Aware Dimensionality Reduction for Correlated Time Series Forecasting via Minimum Dominating Sets`.
