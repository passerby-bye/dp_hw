# Differentially Private Covariance Estimation Reproduction

This repository reproduces experiments from [Differentially Private Covariance Estimation (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/file/4158f6d19559955bae372bb00f6204e4-Paper.pdf) and extends the benchmark with additional datasets and two algorithms from [Differentially Private Covariance Revisited (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2a5e2f031a8e1c9d0b5a5b2a8a5b2a5e-Abstract-Conference.html).

## Implementation Notes

The main experiment script is [dp_covariance.py](d:/trust_worthy/dp_covariance.py). It follows the paper-style data layout `X in R^{d x n}`, uses the paper-scale covariance `C = XX^T`, and reports error as `||C - Ĉ||_F / n`.

For the eigenvector-sampling step inside `KT` and `Algorithm 1`, the repository currently supports two implementations:

- `simple`: a lightweight Monte Carlo approximation that samples many unit-vector candidates and reweights them by the exponential-mechanism score. This version is faster and easier to debug.
- `acg`: a closer-to-paper sampler inspired by Appendix B `Algorithm 2`, using an angular central Gaussian proposal with rejection sampling. This version is slower but more faithful to the intended sampling procedure.
- 

In addition to the baselines from the original paper (Laplace, Gaussian, KT), we also implement two mechanisms from the 2022 follow-up work, which approach the problem via zero-Concentrated DP (zCDP):

- **Trace-Sensitive (`dp_trace_algo_2022`)**: Applies a Gaussian (Wigner-style) noise matrix directly to the empirical covariance $\frac{1}{n}X^TX$.
- **Tail-Sensitive (`dp_tail_algo_2022`)**: Splits the privacy budget $\rho$ equally between two steps: (1) estimate eigenvector directions using the Trace-Sensitive mechanism with budget $\rho/2$; (2) independently estimate eigenvalues along those directions with the remaining $\rho/2$. This separation reduces error when the covariance spectrum decays quickly.

Both mechanisms are implemented in [`dp_cov/mechanisms.py`](dp_cov/mechanisms.py) using PyTorch and are included in the experiment runner alongside the original baselines.

You can choose the sampler at runtime with:

```bash
python .\dp_covariance.py --sampler-mode simple
python .\dp_covariance.py --sampler-mode acg
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

Trace-Sensitive and Tail-Sensitive (2022) use direct Gaussian noise and do not rely on the eigenvector sampler, so they have no `simple` / `acg` variants.

ACG rejection sampler version (`--sampler-mode acg`)

| Dataset | Laplace | Gaussian | KT | ITU | Alg1 | Trace 2022 | Tail 2022 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wine | 0.03s | 0.06s | 2.72s | 2.17s | 2.19s | 0.42s | 0.40s |
| Airfoil | 0.03s | 0.05s | 0.98s | 0.83s | 0.83s | 0.15s | 0.15s |
| Adult | 4.96s | 12.48s | 303.65s | 135.82s | 131.32s | 22.67s | 22.32s |
| Breast Cancer | 0.08s | 0.22s | 45.45s | 17.69s | 19.10s | 1.89s | 1.83s |
| Digits | 0.25s | 0.70s | 89.71s | 38.64s | 44.12s | 7.71s | 7.33s |
| California | 0.07s | 0.18s | 11.83s | 5.14s | 7.75s | 0.31s | 0.35s |
| Volkert | 5.97s | 16.64s | 769.47s | 336.73s | 1758.05s | 43.48s | 42.42s |

Simple version (`--sampler-mode simple`)

| Dataset | Laplace | Gaussian | KT | ITU | Alg1 | Trace 2022 | Tail 2022 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wine | 0.04s | 0.06s | 1.04s | 0.70s | 0.70s | 0.42s | 0.40s |
| Airfoil | 0.03s | 0.05s | 0.30s | 0.24s | 0.23s | 0.15s | 0.15s |
| Adult | 4.55s | 12.24s | 192.94s | 85.92s | 85.09s | 22.67s | 22.32s |
| Breast Cancer | 0.07s | 0.16s | 6.09s | 3.21s | 3.18s | 1.89s | 1.83s |
| Digits | 0.31s | 0.87s | 36.73s | 15.86s | 15.94s | 7.71s | 7.33s |
| California | 0.08s | 0.18s | 0.53s | 0.43s | 0.42s | 0.31s | 0.35s |
| Volkert | 6.87s | 19.60s | 557.49s | 240.44s | 240.25s | 43.48s | 42.42s|
