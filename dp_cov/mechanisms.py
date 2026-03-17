"""All DP covariance mechanisms: Laplace, Gaussian, KT, Algorithm 1, and 2022 algorithms."""

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import root_scalar

from .core import gram_matrix, _clip_psd_full, _clip_eigenvalues_to_data_bound, _sample_eigenvector


# ---------------------------------------------------------------------------
# PSD post-processing helper
# ---------------------------------------------------------------------------

def _rank_k_psd(C_noisy, k, n):
    """Top-k eigendecomposition with non-negative eigenvalue clipping."""
    d = C_noisy.shape[0]
    eigvals, eigvecs = eigh(C_noisy)
    C_hat = np.zeros((d, d))
    for i in range(d - 1, d - 1 - k, -1):
        eigval = eigvals[i]
        if eigval > 0:
            C_hat += min(eigval, float(n)) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return _clip_eigenvalues_to_data_bound(C_hat, n)


# ---------------------------------------------------------------------------
# Laplace baseline
# ---------------------------------------------------------------------------

def dp_laplace(X, epsilon, k, rng):
    """Laplace mechanism baseline (Sec 4, baseline 'L').

    Paper (Sec 2, Laplace Mechanism): add i.i.d. Lap(Δ_f / ε) noise to each
    output coordinate, where Δ_f is the l1-sensitivity of the query.

    For C = XX^T the l1-sensitivity of any single entry C_{ij} = Σ_k x_ik x_jk
    scales as O(d) when column norms are 1, giving scale = 2d / epsilon.
    The factor 2 comes from the sensitivity being bounded by 2 (removing one
    column changes each entry by at most 2 * 1 * 1 = 2 in absolute value,
    and summing over d rows gives the l1-sensitivity of 2d).
    """
    del k  # Laplace acts on the full matrix; rank truncation is not used
    d, n = X.shape
    C = gram_matrix(X)                                    # C = XX^T  (Sec 2)
    noise = rng.laplace(0.0, (2.0 * d) / epsilon, C.shape)  # Lap(2d / ε) per entry
    noise = (noise + noise.T) / 2                         # symmetrize noise matrix
    return _clip_psd_full(C + noise, n)


# ---------------------------------------------------------------------------
# Gaussian baseline
# ---------------------------------------------------------------------------

def dp_gaussian(X, epsilon, delta, k, rng):
    """Gaussian mechanism baseline (Sec 3.1, baseline 'G').

    Paper (Sec 3.1): the Gaussian mechanism achieves (ε, δ)-DP and has the
    error bound:
        ||C - Ĉ||_F  ≤  O( d^{3/2} * sqrt(log(1/δ)) / ε )

    The noise scale σ = d^{3/2} * sqrt(log(1/δ)) / ε follows from calibrating
    the Gaussian mechanism to the Frobenius (l2) sensitivity of C = XX^T.
    Evaluated in experiments at δ ∈ {1e-16, 1e-10, 1e-3} (Sec 4).
    """
    del k  # Gaussian acts on the full matrix; rank truncation is not used
    _, n = X.shape
    C = gram_matrix(X)                                    # C = XX^T  (Sec 2)
    d = C.shape[0]
    # σ = d^{3/2} * sqrt(log(1/δ)) / ε  (Sec 3.1 Gaussian bound)
    sigma = (d ** 1.5 * np.sqrt(np.log(1.0 / delta))) / epsilon
    noise = rng.standard_normal(C.shape)
    noise = (noise + noise.T) / 2                         # symmetrize noise matrix
    return _clip_psd_full(C + sigma * noise, n)


# ---------------------------------------------------------------------------
# KT baseline
# ---------------------------------------------------------------------------

def dp_kt(X, epsilon, delta, k, rng, sampler_mode=None):
    """Kapralov-Talwar (KT) baseline — pure ε-DP (Sec 4, baseline 'KT').

    Paper (Sec 4 / Corollary 2): KT iterates k rounds of rank-one subtraction.
    Each round splits its budget equally between eigenvector sampling and
    eigenvalue estimation:
        ε_per  = ε / k          (budget per round, by composition, Lemma 2)
        ε_ev   = ε_per / 2      (for exponential-mechanism eigenvector)
        ε_lam  = ε_per / 2      (for Laplace-noisy eigenvalue)

    Unlike Algorithm 1, the residual is updated by *subtracting* the rank-1
    term (C_s ← C_s - λ g g^T) rather than projecting onto the orthogonal
    complement.  This is the key structural difference discussed in Sec 3.2.
    """
    del delta  # KT is pure ε-DP; δ is not used
    if rng is None:
        rng = np.random.default_rng()

    d, n = X.shape
    if k is None:
        k = d  # default: full-rank reconstruction

    C_s = gram_matrix(X).copy()   # C_s: residual matrix, starts at C = XX^T
    C_hat = np.zeros((d, d))

    # Budget split: ε_per = ε/k per round (Lemma 2 composition over k rounds)
    eps_per = epsilon / k
    eps_ev  = eps_per / 2.0   # eigenvector budget (exponential mechanism)
    eps_lam = eps_per / 2.0   # eigenvalue budget  (Laplace mechanism)

    for _ in range(k):
        # Step: sample top eigenvector of residual via exponential mechanism
        # density ∝ exp( (ε_ev/4) * u^T C_s u )  (Alg 1 Step 2a, sensitivity 2)
        g = _sample_eigenvector(C_s, eps_ev, rng, sampler_mode=sampler_mode)

        # Noisy eigenvalue: λ̂ = g^T C_s g + Lap(1/ε_lam)
        # Lap scale = 1/ε_lam because sensitivity of g^T C g is 1 for unit g
        lam = float(g @ C_s @ g) + rng.laplace(0.0, 1.0 / eps_lam)
        lam = max(lam, 0.0)          # clip to non-negative (paper: eigenvalue rounding)

        C_hat += lam * np.outer(g, g)         # accumulate rank-1 term
        C_s = C_s - lam * np.outer(g, g)     # subtract rank-1 term from residual
        C_s = (C_s + C_s.T) / 2.0            # re-symmetrize after subtraction

    # Ĉ = (1/n) * Σ_i λ̂_i g_i g_i^T,  then clip to PSD  (Sec 4)
    return _clip_psd_full(C_hat, n)


# ---------------------------------------------------------------------------
# Algorithm 1: Iterative Eigenvector Sampling
# ---------------------------------------------------------------------------

def _orth_complement(g):
    """Return an orthonormal basis for the subspace orthogonal to unit vector g.

    Paper (Alg 1 Step 2b): after sampling θ̂_i, update P_{i+1} as an
    orthonormal basis of {θ̂_1,...,θ̂_i}⊥.  Done incrementally here: given
    the current direction g (= û_i in the reduced subspace), columns 1..end
    of the full left-singular-vector matrix of g span g⊥.
    """
    # SVD of g as a column vector: U[:,0] = g,  U[:,1:] spans g⊥
    U, _, _ = np.linalg.svd(g.reshape(-1, 1), full_matrices=True)
    return U[:, 1:]   # shape (d-i) × (d-i-1): orthonormal basis of û_i⊥


def _algorithm1_privacy_split(C, epsilon, rng, beta=0.01, adaptive=False):
    """Compute noisy eigenvalues and per-round privacy budgets for Algorithm 1.

    Paper (Alg 1 Step 1 + Corollary 1 / Supplement IT-U):

    Step 1 (shared by both variants):
        ε_0 = ε / 2
        λ̂_i = λ_i(C) + Lap(2 / ε_0)  for i=1,...,d
        Scale 2/ε_0: l1-sensitivity of eigenvalue vector ≤ 2
        (Theorem 1 proof: ||Λ(XX^T)-Λ(X̃X̃^T)||_1 = tr(xx^T) ≤ 1, so ≤ 2).

    Adaptive split — Corollary 1 ('AD'):
        τ  = (2 / ε_0) * log(2d / β)
        ε_i = (ε/2) * sqrt(λ̂_i + τ) / Σ_j sqrt(λ̂_j + τ)
        Total: ε_0 + Σ ε_i = ε/2 + ε/2 = ε  ✓

    Uniform split — Supplement IT-U:
        ε_i = ε / (2d)  for all i
        Total: ε_0 + d*(ε/2d) = ε/2 + ε/2 = ε  ✓
    """
    d = C.shape[0]
    eps0 = epsilon / 2.0                    # ε_0 = ε/2  (Corollary 1)

    eigvals, _ = eigh(C)
    eigvals = eigvals[::-1]                 # descending order: λ_1 ≥ ... ≥ λ_d
    # Alg 1 Step 1: λ̂_i = λ_i(C) + Lap(2/ε_0)
    noisy_eigvals = eigvals + rng.laplace(0.0, 2.0 / eps0, size=d)

    if adaptive:
        # Corollary 1: τ = (2/ε_0)*log(2d/β) = (4/ε)*log(2d/β)
        tau = (2.0 / eps0) * np.log(2.0 * d / beta)
        weights = np.sqrt(np.maximum(noisy_eigvals + tau, 1e-12))
        # ε_i = (ε/2) * w_i / Σ_j w_j,  where w_i = sqrt(λ̂_i + τ)
        eps_rounds = (epsilon / 2.0) * weights / np.sum(weights)
    else:
        # IT-U: uniform ε_i = ε/(2d) for all i  (Supplement)
        eps_rounds = np.full(d, epsilon / (2.0 * d))

    return noisy_eigvals, eps_rounds


def _dp_algorithm1_core(X, epsilon, rng=None, beta=0.01, adaptive=False, rank_k=None, sampler_mode=None):
    """Core of Algorithm 1: Iterative Eigenvector Sampling (Sec 3).

    Full pseudocode from paper:

        Input:  C = XX^T,  params ε_0, ε_1, ..., ε_d

        Step 1. λ̂_i = λ_i(C) + Lap(2/ε_0)  for i=1,...,d   [← _algorithm1_privacy_split]
                C_1 = C,  P_1 = I_d

        Step 2. For i = 1,...,d:
          (a) û_i ~ exp( (ε_i/4) * u^T C_i u ) on S^{d-i}   [← _sample_exponential_mechanism]
              θ̂_i = P_i^T û_i                               (lift to R^d)
          (b) P_{i+1}: orthonormal basis ⊥ to {θ̂_1,...,θ̂_i} [← _orth_complement]
          (c) C_{i+1} = P_{i+1} C P_{i+1}^T  ∈ R^{(d-i)×(d-i)}

        Step 3. Output Ĉ = Σ_i λ̂_i θ̂_i θ̂_i^T / n  (then clip to PSD)

    Privacy (Theorem 1): preserves (Σ_{i=0}^d ε_i)-DP by composition (Lemma 2).
    Utility (Theorem 2): ||C - Ĉ||_F ≤ Õ(sqrt(Σ_i d*λ_i/ε_i + sqrt(d)/ε_0)).
    Rank-k variant (after Theorem 2): stop loop at i=k; distance from best rank-k approx.
    """
    if rng is None:
        rng = np.random.default_rng()

    d, n = X.shape
    # Step 1: C = XX^T  (Sec 2 input convention)
    C = gram_matrix(X)
    # Compute λ̂_i and ε_i for all rounds  (Alg 1 Step 1 + Corollary 1)
    noisy_eigvals, eps_rounds = _algorithm1_privacy_split(
        C, epsilon, rng, beta=beta, adaptive=adaptive
    )
    # Rank-k truncation: run only k rounds (Theorem 2 rank-k variant)
    n_rounds = d if rank_k is None else min(rank_k, d)
    noisy_eigvals = noisy_eigvals[:n_rounds]
    eps_rounds    = eps_rounds[:n_rounds]

    # Step 1 init: P_1 = I_d (columns are the current subspace basis in R^d)
    P_i = np.eye(d)   # P_i ∈ R^{d×(d-i+1)}: orthonormal basis of working subspace
    C_i = C.copy()    # C_i = P_i C P_i^T ∈ R^{(d-i+1)×(d-i+1)}, starts as C
    thetas = []

    for i in range(n_rounds):
        # Step 2a: û_i ~ exp. mechanism on S^{d-i} with budget ε_i
        u_i = _sample_eigenvector(C_i, eps_rounds[i], rng, sampler_mode=sampler_mode)
        # θ̂_i = P_i^T û_i  — lift û_i from subspace coords back to R^d
        theta_i = P_i @ u_i
        thetas.append(theta_i)

        if C_i.shape[0] == 1:
            break  # 1-dim subspace: nothing left to decompose

        # Step 2b: P_{i+1} — append one more orthogonal direction removed
        P_i = P_i @ _orth_complement(u_i)      # P_{i+1} = P_i * orth(û_i)
        # Step 2c: project C into new (d-i-1)-dim subspace
        C_i = P_i.T @ C @ P_i                  # C_{i+1} = P_{i+1} C P_{i+1}^T

    # Step 3: Ĉ = Σ_i λ̂_i θ̂_i θ̂_i^T  (Alg 1 Step 3)
    C_hat = np.zeros((d, d))
    for lam, theta in zip(noisy_eigvals, thetas):
        C_hat += lam * np.outer(theta, theta)
    # Project to PSD and clip eigenvalues to [0, n] (Sec. 4 post-processing).
    return _clip_psd_full(C_hat, n)


def dp_algorithm1_uniform(X, epsilon, delta, k, rng=None, beta=0.01, sampler_mode=None):
    """Appendix IT-U variant: uniform privacy splitting for Algorithm 1."""
    del delta, k
    return _dp_algorithm1_core(X, epsilon, rng=rng, beta=beta, adaptive=False, sampler_mode=sampler_mode)


def dp_algorithm1_adaptive(X, epsilon, delta, k, rng=None, beta=0.01, sampler_mode=None):
    """Appendix AD variant: adaptive privacy splitting following Corollary 1."""
    del delta
    return _dp_algorithm1_core(
        X, epsilon, rng=rng, beta=beta, adaptive=True, rank_k=k, sampler_mode=sampler_mode
    )


def dp_algorithm1_strict(X, epsilon, delta, k, rng=None, sampler_mode=None):
    """Backwards-compatible alias for the appendix IT-U/full-reconstruction version."""
    return dp_algorithm1_uniform(X, epsilon, delta, k, rng=rng, sampler_mode=sampler_mode)


def dp_algorithm1(X, epsilon, delta, k, rng=None, sampler_mode=None):
    """Main Algorithm 1 entry point used in experiments: appendix AD variant."""
    return dp_algorithm1_adaptive(X, epsilon, delta, k, rng=rng, sampler_mode=sampler_mode)


def dp_algorithm1_rank_k(X, epsilon, delta, k, rng=None, sampler_mode=None):
    """Rank-k post-processing wrapper around the main Algorithm 1 variant."""
    return dp_algorithm1(X, epsilon, delta, k, rng=rng, sampler_mode=sampler_mode)


# ---------------------------------------------------------------------------
# 2022 algorithms: zCDP Gaussian covariance mechanisms
# ---------------------------------------------------------------------------

def _convert_symm_mat(ZZ, d):
    """Fill a symmetric d×d torch tensor from a flat upper-triangle vector."""
    S = torch.empty([d, d])
    k = 0
    for i in range(d):
        for j in range(i, d):
            S[i, j] = ZZ[0, k]
            k += 1
    for i in range(d):
        for j in range(i + 1, d):
            S[j, i] = S[i, j]
    return S


def _get_gauss_wigner_matrix(d):
    """Sample a symmetric Gaussian (Wigner-style) noise matrix of size d×d."""
    Z = torch.normal(0, 1, size=(1, int(d * (d + 1) / 2)))
    return _convert_symm_mat(Z, d)


def _get_gauss_noise_vector(d):
    """Sample a d-dimensional Gaussian noise vector."""
    return torch.normal(0, 1, size=(d,))


def _get_rho(eps, delta):
    """Convert (ε, δ)-DP to ρ-zCDP via the standard conversion formula.

    Solves: ρ + 2·sqrt(ρ·log(1/δ)) = ε  for ρ > 0.
    Falls back to the Gaussian-mechanism approximation ρ ≈ ε²/(2·log(1.25/δ))
    if the root-finder fails.
    """
    if delta <= 0:
        return eps

    def _eq(x):
        return x + 2 * np.sqrt(x * np.log(1.0 / delta)) - eps

    try:
        return root_scalar(_eq, bracket=[0, eps]).root
    except Exception:
        return eps ** 2 / (2 * np.log(1.25 / delta))


def GaussCov_2022(X_torch, n, d, rho, delta=0.0, r=1.0, b_fleig=True):
    """2022 Gaussian covariance mechanism (zCDP).

    Adds scaled Wigner noise to the empirical covariance 1/n * X^T X.
    The noise scale follows from the zCDP sensitivity of the covariance query:
        sens = sqrt(2) * r² / n
        σ    = sens / sqrt(2·ρ)

    Args:
        X_torch: torch tensor of shape (n, d) — rows are samples.
        n, d:    number of samples and features.
        rho:     zCDP privacy parameter.
        r:       bound on per-sample norm (default 1.0 for unit-normalized data).
        b_fleig: if True, project output to PSD and clip eigenvalues to [0, r²].
    """
    cov = torch.mm(X_torch.t(), X_torch) / n
    W = _get_gauss_wigner_matrix(d)
    sens = np.sqrt(2) * r * r / n
    cov_tilde = cov + (sens / np.sqrt(2 * rho)) * W
    if b_fleig:
        D, U = torch.linalg.eigh(cov_tilde)
        D = torch.clamp(D, 0, r * r)
        cov_tilde = torch.mm(U, torch.mm(torch.diag(D), U.t()))
    return cov_tilde


def SeparateCov_2022(X_torch, n, d, rho, r=1.0, b_fleig=True):
    """2022 Separate direction–eigenvalue covariance mechanism (zCDP).

    Splits the privacy budget ρ equally:
      - Half budget (ρ/2) estimates the eigenvector directions via GaussCov_2022.
      - Remaining half (ρ/2) estimates eigenvalues on those directions with
        independent Gaussian noise (sens = r²·sqrt(2)/n).

    This mirrors the paper's Trace-Sensitive / Tail-Sensitive separation idea.

    Args:
        X_torch: torch tensor of shape (n, d) — rows are samples.
        n, d:    number of samples and features.
        rho:     total zCDP privacy budget.
        r:       per-sample norm bound (default 1.0).
        b_fleig: if True, clip estimated eigenvalues to [0, r²].
    """
    cov = torch.mm(X_torch.t(), X_torch) / n
    rho0 = 0.5 * rho

    # Step 1: estimate eigenvector directions with half the budget
    cov_gauss = GaussCov_2022(X_torch, n, d, rho0, r=r, b_fleig=False)
    Ug, _, _ = torch.linalg.svd(cov_gauss)

    # Step 2: project true covariance onto estimated directions, add noise
    D = torch.diag(torch.mm(Ug.t(), torch.mm(cov, Ug)))
    Z = _get_gauss_noise_vector(d)
    sens = r * r * np.sqrt(2) / n
    D_tilde = D + (sens / np.sqrt(rho0)) * Z
    if b_fleig:
        D_tilde = torch.clamp(D_tilde, 0, r * r)

    return torch.mm(Ug, torch.mm(torch.diag(D_tilde), Ug.t()))


def dp_gaussCov_algo_2022(X_dn, epsilon, delta, k, rng=None):
    """
    Converts (ε, δ)-DP to ρ-zCDP, applies GaussCov_2022, and rescales
    the output back to the XX^T scale used by the rest of the experiments.

    Args:
        X_dn: numpy array of shape (d, n) — paper convention, columns are samples.
    """
    d, n = X_dn.shape
    X_torch = torch.from_numpy(X_dn.T).float()   # (n, d)
    rho = _get_rho(epsilon, delta)
    cov_tilde = GaussCov_2022(X_torch, n, d, rho, delta=delta, r=1.0)
    return cov_tilde.detach().numpy() * n          # rescale to XX^T format


def dp_trace_algo_2022(X_dn, epsilon, delta, k, rng=None):
    """2022 Trace-Sensitive algorithm wrapper (SeparateCov_2022).

    Converts (ε, δ)-DP to ρ-zCDP, applies SeparateCov_2022, and rescales
    the output back to the XX^T scale used by the rest of the experiments.

    Args:
        X_dn: numpy array of shape (d, n) — paper convention, columns are samples.
    """
    d, n = X_dn.shape
    X_torch = torch.from_numpy(X_dn.T).float()   # (n, d)
    rho = _get_rho(epsilon, delta)
    cov_tilde = SeparateCov_2022(X_torch, n, d, rho, r=1.0)
    return cov_tilde.detach().numpy() * n          # rescale to XX^T format
