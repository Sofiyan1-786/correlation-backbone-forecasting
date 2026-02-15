# Topology-Aware Dimensionality Reduction via Minimum Dominating Sets (MDS)

This repository contains code to reproduce experiments for:
"Topology-Aware Dimensionality Reduction for Correlated Time Series Forecasting via Minimum Dominating Sets".

## What is included
- Data acquisition scripts (no raw vendor data redistributed)
- Preprocessing pipeline
- Correlation network construction + threshold selection
- MDS backbone extraction
- ST-GNN training/evaluation scripts
- Config files and fixed random seeds

## Data availability
Raw price series are fetched from original sources subject to their terms of use.
This repository does not redistribute vendor-provided raw data.
Use the provided scripts to reconstruct the dataset locally.

## Quickstart
1. Create environment
2. Run data fetch
3. Run preprocessing
4. Run experiments
5. Reproduce tables/figures
