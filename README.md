# Differentially Private Covariance Estimation Reproduction

This repository contains code and notes for reproducing experiments from the paper on differentially private covariance estimation.

## Implementation Notes

The main experiment script is [dp_covariance.py](d:/trust_worthy/dp_covariance.py). It follows the paper-style data layout `X in R^{d x n}`, uses the paper-scale covariance `C = XX^T`, and reports error as `||C - Ĉ||_F / n`.

For the eigenvector-sampling step inside `KT` and `Algorithm 1`, the repository currently supports two implementations:

- `simple`: a lightweight Monte Carlo approximation that samples many unit-vector candidates and reweights them by the exponential-mechanism score. This version is faster and easier to debug.
- `acg`: a closer-to-paper sampler inspired by Appendix B `Algorithm 2`, using an angular central Gaussian proposal with rejection sampling. This version is slower but more faithful to the intended sampling procedure.

You can choose the sampler at runtime with:

```bash
python .\dp_covariance.py --sampler-mode simple
python .\dp_covariance.py --sampler-mode acg
python .\airfoil_reduce_d_experiment.py --sampler-mode simple
```

## Dataset Info

The shapes below are reported in the paper-style layout `X in R^{d x n}`, where `d` is the feature dimension and `n` is the number of samples after preprocessing.

Paper datsets:

- Wine: `(13, 178)`
- Airfoil: `(6, 1503)`
- Adult: `(105, 48842)`

Different datsets we explored：

- Breast Cancer: `(30, 569)`
- Digits: `(61, 1797)`
- California: `(8, 20640)`
- Volkert: `(147, 58310)`

## Running time

ACG rejection sampler version

| Dataset | Laplace | Gaussian | KT | ITU | Alg1 | Trace | Tail | 
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wine | 0.03s | 0.06s | 2.72s | 2.17s | 2.19s |
| Airfoil | 0.03s | 0.05s | 0.98s | 0.83s | 0.83s |
| Adult | 4.96s | 12.48s | 303.65s | 135.82s | 131.32s |
| Breast Cancer | 0.08s | 0.22s | 45.45s | 17.69s | 19.10s |
| Digits | 0.25s | 0.70s | 89.71s | 38.64s | 44.12s |
| California | 0.07s | 0.18s | 11.83s | 5.14s | 7.75s |
| Volkert | 5.97s | 16.64s | 769.47s | 336.73s | 1758.05s |

Simple version

| Dataset | Laplace | Gaussian | KT | ITU | Alg1 | Trace | Tail | 
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wine | 0.04s | 0.06s | 1.04s | 0.70s | 0.70s |
| Airfoil | 0.03s | 0.05s | 0.30s | 0.24s | 0.23s |
| Adult | 4.55s | 12.24s | 192.94s | 85.92s | 85.09s |
| Breast Cancer | 0.07s | 0.16s | 6.09s | 3.21s | 3.18s |
| Digits | 0.31s | 0.87s | 36.73s | 15.86s | 15.94s |
| California | 0.08s | 0.18s | 0.53s | 0.43s | 0.42s |
| Volkert | 6.87s | 19.60s | 557.49s | 240.44s | 240.25s |
