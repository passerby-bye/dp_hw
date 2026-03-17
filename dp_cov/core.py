"""Core math utilities: covariance, evaluation metrics, and eigenvector samplers."""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Global sampler mode
# ---------------------------------------------------------------------------

SAMPLER_MODE = "auto"


def set_sampler_mode(mode):
    """Set the eigenvector sampler used by KT and Algorithm 1."""
    global SAMPLER_MODE
    if mode not in {"simple", "acg", "auto"}:
        raise ValueError(f"Unsupported sampler mode: {mode}")
    SAMPLER_MODE = mode


# ---------------------------------------------------------------------------
# Covariance matrices
# ---------------------------------------------------------------------------

def true_covariance(X):
    """Paper-scale covariance used as the reference for error evaluation."""
    return X @ X.T


def gram_matrix(X):
    """Unnormalized covariance matrix used internally by the DP algorithms."""
    return X @ X.T


# ---------------------------------------------------------------------------
# Evaluation metric and PSD clipping
# ---------------------------------------------------------------------------

def frobenius_error(C_true, C_hat, n):
    # Paper (Sec 4): normalized Frobenius distance  ||C_hat - C||_F / n
    return np.linalg.norm(C_true - C_hat, "fro") / n


def _clip_eigenvalues_to_data_bound(C_hat, n):
    """Clip eigenvalues to [0, n] as in the paper's post-processing step."""
    eigvals, eigvecs = eigh(C_hat)
    eigvals = np.clip(eigvals, 0.0, float(n))
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _clip_psd_full(C_hat, n):
    """Project a covariance estimate onto the PSD cone and clip eigenvalues to [0, n]."""
    # Symmetrize first to absorb any floating-point asymmetry
    C_hat = (C_hat + C_hat.T) / 2.0
    eigvals, eigvecs = eigh(C_hat)
    eigvals = np.clip(eigvals, 0.0, float(n))
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ---------------------------------------------------------------------------
# Eigenvector samplers (exponential mechanism)
# ---------------------------------------------------------------------------

def _sample_exponential_mechanism(C, epsilon, rng, num_candidates=500):
    """
    Lightweight approximation of the exponential mechanism on the sphere.

    Paper (Sec 2 / Alg 1 Step 2a): the exact sampler draws u from the density
        f_{C_i}(u)  ∝  exp( (epsilon_i / 4) * u^T C_i u )   on S^{d-1}
    which is a Bingham distribution.  The factor 4 in the denominator comes
    from the exponential-mechanism sensitivity: for unit-norm columns,
        |g(XX^T, θ) - g(X̃X̃^T, θ)| = |θ^T(xx^T - x̃x̃^T)θ| ≤ 2,
    so Lemma 4 gives density ∝ exp( eps * g / (2*Δ_g) ) = exp( eps*u^TCu / 4 ).

    The exact sampler is Algorithm 2 (Kent et al. rejection sampler, Sec 3.2).
    This implementation replaces it with a Monte-Carlo softmax over
    `num_candidates` uniform unit vectors — cheaper but approximate.
    """
    d = C.shape[0]
    # Draw uniform random unit vectors on S^{d-1}
    candidates = rng.standard_normal((num_candidates, d))
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    # Utility score: g(C, u) = u^T C u  (Alg 1 Step 2a)
    scores = np.einsum("ij,jk,ik->i", candidates, C, candidates)
    # Weights ∝ exp( (epsilon_i / 4) * u^T C u )  — exact density up to normalizer
    logits = (epsilon / 4.0) * scores
    logits -= np.max(logits)   # numerical stability
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return candidates[rng.choice(num_candidates, p=weights)]


# def _solve_acg_b(eigvals_a):
#     """Solve the saddle-point equation for Algorithm 2's ACG proposal."""
#     lower = max(1e-12, -2.0 * float(np.min(eigvals_a)) + 1e-12)

#     def f(val):
#         return np.sum(1.0 / (val + 2.0 * eigvals_a)) - 1.0

#     upper = max(lower * 2.0, 1.0)
#     for _ in range(80):
#         if f(upper) <= 0:
#             break
#         upper *= 2.0
#     else:
#         raise RuntimeError("Failed to bracket Algorithm 2 saddle-point parameter b.")

#     return brentq(f, lower, upper)


# def _sample_exponential_mechanism_acg(C, epsilon, rng, num_candidates=500, max_attempts=2000):
#     """
#     Algorithm 2-style rejection sampler with an ACG proposal.

#     The extracted supplement formula for b is numerically inconsistent on this
#     problem family. We therefore use the standard saddle-point condition
#     sum_i 1 / (b + 2 lambda_i(A)) = 1 to obtain a valid positive proposal
#     parameter b
#     """
#     d = C.shape[0]
#     if d == 1:
#         return np.array([1.0])

#     try:
#         eigvals_c = eigh(C, eigvals_only=True)
#         lam_min = float(eigvals_c[0])
#         A = -(epsilon / 4.0) * C + (epsilon / 4.0) * lam_min * np.eye(d)
#         eigvals_a = eigh(A, eigvals_only=True)
#         b = _solve_acg_b(eigvals_a)
#         # if _SAMPLER_DIAG:
#         #     residual = float(np.sum(1.0 / (b + 2.0 * eigvals_a)) - 1.0)
#         #     # print(f"    [acg] d={d}  b={b:.6f}  residual={residual:.2e}"
#         #     #       f"  eigvals_a=[{eigvals_a[0]:.3f}..{eigvals_a[-1]:.3f}]")
#         Omega = np.eye(d) + (2.0 / b) * A
#         Omega = (Omega + Omega.T) / 2.0
#         chol = np.linalg.cholesky(Omega)
#         log_m = -(d - b) / 2.0 + (d / 2.0) * np.log(d / b)

#         best_u = None
#         best_log_accept = -np.inf
#         for _ in range(max_attempts):
#             z = np.linalg.solve(chol, rng.standard_normal(d))
#             norm_z = np.linalg.norm(z)
#             if norm_z == 0:
#                 continue
#             u = z / norm_z
#             quad_a = float(u @ A @ u)
#             quad_o = float(u @ Omega @ u)
#             log_accept = -quad_a - log_m - (d / 2.0) * np.log(max(quad_o, 1e-300))
#             if log_accept > best_log_accept:
#                 best_log_accept = log_accept
#                 best_u = u
#             if np.log(rng.uniform()) <= min(0.0, log_accept):
#                 return u
#         # if _SAMPLER_DIAG:
#         #     print(f"    [acg] d={d}  exhausted {max_attempts} attempts,"
#         #           f" returning best (log_accept={best_log_accept:.3f})")
#         if best_u is not None:
#             return best_u
#     except (np.linalg.LinAlgError, RuntimeError, FloatingPointError) as e:
#         if _SAMPLER_DIAG:
#             print(f"    [acg] d={C.shape[0]}  exception: {e}")

#     return _sample_exponential_mechanism(C, epsilon, rng, num_candidates=num_candidates)
def _solve_acg_b(eigvals_a):
    """Solve the saddle-point equation for Algorithm 2's ACG proposal."""
    lower = max(1e-12, -2.0 * float(np.min(eigvals_a)) + 1e-12)

    def f(val):
        return np.sum(1.0 / (val + 2.0 * eigvals_a)) - 1.0

    upper = max(lower * 2.0, 1.0)
    for _ in range(80):
        if f(upper) <= 0:
            break
        upper *= 2.0
    else:
        raise RuntimeError("Failed to bracket Algorithm 2 saddle-point parameter b.")

    for _ in range(100):
        mid = 0.5 * (lower + upper)
        if f(mid) > 0:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


def _sample_exponential_mechanism_acg(C, epsilon, rng, num_candidates=500, max_attempts=2000):
    """
    Algorithm 2-style rejection sampler with an ACG proposal.

    The extracted supplement formula for b is numerically inconsistent on this
    problem family. We therefore use the standard saddle-point condition
    sum_i 1 / (b + 2 lambda_i(A)) = 1 to obtain a valid positive proposal
    parameter, and fall back to the original Monte-Carlo approximation if the
    rejection sampler becomes numerically unstable.
    """
    d = C.shape[0]
    if d == 1:
        return np.array([1.0])

    try:
        eigvals_c = eigh(C, eigvals_only=True)
        lam_min = float(eigvals_c[0])
        A = -(epsilon / 4.0) * C + (epsilon / 4.0) * lam_min * np.eye(d)
        eigvals_a = eigh(A, eigvals_only=True)
        b = _solve_acg_b(eigvals_a)
        Omega = np.eye(d) + (2.0 / b) * A
        Omega = (Omega + Omega.T) / 2.0
        chol = np.linalg.cholesky(Omega)
        log_m = -(d - b) / 2.0 + (d / 2.0) * np.log(d / b)

        for _ in range(max_attempts):
            z = np.linalg.solve(chol, rng.standard_normal(d))
            norm_z = np.linalg.norm(z)
            if norm_z == 0:
                continue
            u = z / norm_z
            quad_a = float(u @ A @ u)
            quad_o = float(u @ Omega @ u)
            log_accept = -quad_a - log_m - (d / 2.0) * np.log(max(quad_o, 1e-300))
            if np.log(rng.uniform()) <= min(0.0, log_accept):
                return u
    except (np.linalg.LinAlgError, RuntimeError, FloatingPointError):
        pass

    return _sample_exponential_mechanism(C, epsilon, rng, num_candidates=num_candidates)


_SAMPLER_DIAG = False  # set to True to print eigenvector alignment diagnostics

def _sample_eigenvector(C, epsilon, rng, sampler_mode=None):
    """Dispatch between the simple and ACG-style samplers."""
    mode = SAMPLER_MODE if sampler_mode is None else sampler_mode
    if mode == "simple":
        u = _sample_exponential_mechanism(C, epsilon, rng)
    elif mode in {"acg", "auto"}:
        u = _sample_exponential_mechanism_acg(C, epsilon, rng)
    else:
        raise ValueError(f"Unsupported sampler mode: {mode}")

    if _SAMPLER_DIAG and C.shape[0] > 1:
        eigvals, eigvecs = eigh(C)
        true_top = eigvecs[:, -1]
        alignment = abs(float(u @ true_top))
        # print(f"  [diag] d={C.shape[0]:3d}  eps={epsilon:.4f}"
        #       f"  |u·v1|={alignment:.4f}"
        #       f"  lam1={eigvals[-1]:.4f}")

    return u
