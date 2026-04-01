#!/usr/bin/env python3
"""
Proof Tightening: Explicit Constants and Width-Scaling Verification
====================================================================
Tasks 1A-1E: Compute g_aa and g_aaa explicitly, verify against finite differences
Tasks 2A-2E: Solve for W*(0), degenerate perturbation theory, width scaling
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh, solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, warnings
warnings.filterwarnings('ignore')

def h(z, lam):    return (1-lam)*z + lam*np.tanh(z)
def hp(z, lam):   s = np.tanh(z); return (1-lam) + lam*(1-s*s)
def hpp(z, lam):  s = np.tanh(z); return lam*(-2*s*(1-s*s))
def hppp(z, lam): s = np.tanh(z); return lam*(-2*(1-s*s)*(1-3*s*s))

T0 = time.time()
print("="*70)
print("  PROOF TIGHTENING: EXPLICIT CONSTANTS + WIDTH SCALING")
print("="*70)

# ── Fixed experimental setup ────────────────────────────────
np.random.seed(77)
d_fix, N_fix = 5, 500
alpha_reg = 4e-3
X = np.random.randn(N_fix, d_fix)
wl  = np.array([1., .5, -.3, .2, -.1])
wnl = np.array([.3, -.7, .5, -.2, .4])
y_data = X @ wl + 0.4*np.tanh(1.5 * X @ wnl) + 0.05*np.random.randn(N_fix)
Sigma = (X.T @ X) / N_fix
gamma = (X.T @ y_data) / N_fix

print(f"\n  Data: d={d_fix}, N={N_fix}, α={alpha_reg}")
print(f"  Σ eigenvalues: {np.sort(np.linalg.eigvalsh(Sigma))}")
print(f"  ‖Σ⁻¹γ‖ = {np.linalg.norm(np.linalg.solve(Sigma, gamma)):.4f}")

# ══════════════════════════════════════════════════════════════
#  TASK 2A: Solve Sylvester equation for W*(0) explicitly
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TASK 2A: Explicit W*(0) from Sylvester equation")
print("="*70)

def compute_W0_theory(m, v, Sigma, gamma, alpha):
    """Solve (vv^T ⊗ Σ + αI) vec(W) = v ⊗ γ exactly."""
    vsq = np.sum(v**2)
    Sig_inv_gamma = np.linalg.solve(Sigma, gamma)
    # W* = v · γ^T Σ^{-1} / (||v||^2 + α/λ_k(Σ)) — but this is per-eigenmode
    # Exact: W*_j = v_j / (||v||^2 + α/λ_k) · Σ^{-1}γ for each eigendirection
    # Since Σ^{-1}γ is a single vector, W_j = c_j · Σ^{-1}γ where
    # c_j = v_j / (||v||^2 + α · (e_j^T (I_m) e_j) ... )
    # Actually simpler: the system decouples row by row only if we use
    # the full Kronecker structure.
    # (vv^T ⊗ Σ + αI_md) vec(W) = v ⊗ γ
    # In block form for row j: Σ Σ_k v_j v_k w_k + α w_j = v_j γ
    # Summing v_j times row j: Σ (||v||^2 Σ + α I) (Σ_k v_k w_k / ||v||^2 ... )
    # Let u = Σ_j v_j w_j (the "effective weight"). Then:
    # v_j v^T W Σ + α w_j = v_j γ^T ⟹ w_j = (v_j/α)(γ^T - u^T Σ)
    # where u = v^T W = Σ_j v_j w_j
    # Substituting: u = Σ_j v_j · (v_j/α)(γ - Σ u) = (||v||^2/α)(γ - Σ u)
    # So: α u = ||v||^2 (γ - Σ u)
    # (α I + ||v||^2 Σ) u = ||v||^2 γ
    # u = ||v||^2 (α I + ||v||^2 Σ)^{-1} γ
    
    M = alpha * np.eye(d_fix) + vsq * Sigma
    u_eff = vsq * np.linalg.solve(M, gamma)
    
    # Then w_j = (v_j / α)(γ - Σ u_eff)
    residual = gamma - Sigma @ u_eff
    W_theory = np.outer(v, residual) / alpha
    
    return W_theory, u_eff

def loss_fn(wf, lam, m, v):
    W = wf.reshape(m, d_fix); Z = X @ W.T
    return 0.5*np.mean((h(Z, lam) @ v - y_data)**2) + 0.5*alpha_reg*np.sum(wf**2)

def grad_fn(wf, lam, m, v):
    W = wf.reshape(m, d_fix); Z = X @ W.T
    r = h(Z, lam) @ v - y_data
    return ((1/N_fix)*((r[:,None]*hp(Z,lam)*v[None,:]).T @ X) + alpha_reg*W).ravel()

def hess_fd(wf, lam, m, v, eps=1e-5):
    g0 = grad_fn(wf, lam, m, v); n = len(wf)
    H = np.empty((n, n))
    for i in range(n):
        wp = wf.copy(); wp[i] += eps
        H[i] = (grad_fn(wp, lam, m, v) - g0) / eps
    return 0.5*(H + H.T)

# Verify for m=10
m_test = 10
v_test = np.linspace(0.5, 1.5, m_test)
W0_thy, u_eff = compute_W0_theory(m_test, v_test, Sigma, gamma, alpha_reg)

# Numerical solution at λ=0
W0_init = np.random.randn(m_test * d_fix) * 0.01
res0 = minimize(lambda w: loss_fn(w, 0., m_test, v_test),
                W0_init, jac=lambda w: grad_fn(w, 0., m_test, v_test),
                method='L-BFGS-B', options={'maxiter': 8000, 'gtol': 1e-15})
W0_num = res0.x.reshape(m_test, d_fix)

err_W0 = np.linalg.norm(W0_num - W0_thy) / np.linalg.norm(W0_num)
print(f"  m={m_test}: ‖W*_num - W*_thy‖/‖W*_num‖ = {err_W0:.6e}")
print(f"  ‖w_j*‖ range: [{np.min(np.linalg.norm(W0_thy, axis=1)):.5f}, {np.max(np.linalg.norm(W0_thy, axis=1)):.5f}]")
print(f"  v_j/‖v‖² · ‖Σ⁻¹γ‖ range: [{np.min(v_test)/np.sum(v_test**2)*np.linalg.norm(np.linalg.solve(Sigma, gamma)):.5f}, {np.max(v_test)/np.sum(v_test**2)*np.linalg.norm(np.linalg.solve(Sigma, gamma)):.5f}]")

# ══════════════════════════════════════════════════════════════
#  TASK 2B: Degenerate perturbation theory for λ₁'(0)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TASK 2B: Degenerate perturbation theory")
print("="*70)

def compute_lam1_prime_theory(m, v, W0, Sigma_data, X_data, y_d, alpha):
    """
    Compute λ₁'(0) via degenerate perturbation theory.
    
    At λ=0, the eigenvalue α has multiplicity (m-1)d on the subspace {v^T δW = 0}.
    The perturbation dH/dλ|_{λ=0} restricted to this subspace determines λ₁'(0)
    as the smallest eigenvalue of the restricted perturbation matrix.
    """
    n_par = m * d_fix
    
    # Build orthonormal basis for the flat subspace {v^T δW = 0}
    # This is the nullspace of the md × d matrix (v ⊗ I_d) in row space
    # Easier: the constraint is Σ_j v_j δw_j = 0 in R^d
    # A basis: for each pair (j, k) with j < m, take δw_j = e_k / ‖v‖,
    # δw_{j'} = -v_j/(v_{j'} ‖v‖) e_k for some j', with all others 0.
    # Simpler: use QR on the constraint.
    
    # The constraint matrix is v^T ⊗ I_d acting on vec(δW),
    # i.e., C = kron(v.T, I_d) which is d × md
    C = np.kron(v.reshape(1, -1), np.eye(d_fix))  # d × md
    # Nullspace of C
    _, S_c, Vt_c = np.linalg.svd(C, full_matrices=True)
    # Columns of Vt_c.T corresponding to zero singular values
    rank_C = np.sum(S_c > 1e-12)
    Q_flat = Vt_c[rank_C:].T  # md × (m-1)d, orthonormal basis for flat subspace
    
    assert Q_flat.shape == (n_par, (m-1)*d_fix), f"Flat subspace wrong dim: {Q_flat.shape}"
    
    # Compute dH/dλ|_{λ=0} numerically (finite difference in λ)
    wf = W0.ravel()
    dlam = 1e-5
    H0 = hess_fd(wf, 0., m, v)
    H_eps = hess_fd(wf, dlam, m, v)
    dH = (H_eps - H0) / dlam
    
    # Restrict dH to flat subspace
    dH_restricted = Q_flat.T @ dH @ Q_flat  # (m-1)d × (m-1)d
    
    # Eigenvalues of restricted perturbation
    eigs_restricted = np.sort(np.linalg.eigvalsh(dH_restricted))
    lam1_prime_theory = eigs_restricted[0]
    
    # Also compute full-Hessian numerical derivative for comparison
    eigs_0 = np.sort(eigh(H0, eigvals_only=True))
    eigs_eps = np.sort(eigh(H_eps, eigvals_only=True))
    lam1_prime_numerical = (eigs_eps[0] - eigs_0[0]) / dlam
    
    return lam1_prime_theory, lam1_prime_numerical, eigs_restricted, Q_flat

lp_thy, lp_num, eigs_rest, Q_flat = compute_lam1_prime_theory(
    m_test, v_test, W0_thy, Sigma, X, y_data, alpha_reg)

print(f"  m={m_test}:")
print(f"    λ₁'(0) from degenerate pert. theory: {lp_thy:.8f}")
print(f"    λ₁'(0) from full Hessian FD:         {lp_num:.8f}")
print(f"    Relative error: {abs(lp_thy - lp_num) / abs(lp_num):.4e}")
print(f"    Smallest 5 restricted eigenvalues: {eigs_rest[:5]}")

# ══════════════════════════════════════════════════════════════
#  TASK 2B (continued): Analytical formula for dH/dλ
# ══════════════════════════════════════════════════════════════
print("\n  --- Analytical dH/dλ computation ---")

def compute_dH_analytical(m, v, W0, X_data, y_d, alpha):
    """
    Compute dH/dλ|_{λ=0} analytically.
    
    For tanh: σ'(z) - 1 = -z² + O(z⁴)
    
    GN component: dH^GN_jk/dλ = v_j v_k E[(σ'(w_j^T x)-1) xx^T] 
                                + v_j v_k E[(σ'(w_k^T x)-1) xx^T]
    For tanh: = -v_j v_k (E[(w_j^T x)² xx^T] + E[(w_k^T x)² xx^T])
    
    Residual component: dH^res/dλ = E[r · σ''(w_j^T x) xx^T] ≈ 0 at good fit
    """
    N = len(X_data)
    n_par = m * d_fix
    dH = np.zeros((n_par, n_par))
    
    # Pre-compute z_j = w_j^T x and weighted moments
    Z = X_data @ W0.T  # N × m
    
    # Residual at λ=0
    r = Z @ v - y_d  # N
    
    for j in range(m):
        for k in range(m):
            # GN contribution: -v_j v_k (E[z_j² xx^T] + E[z_k² xx^T])
            Mjj = (1/N) * (X_data * (Z[:, j]**2)[:, None]).T @ X_data  # d×d
            Mkk = (1/N) * (X_data * (Z[:, k]**2)[:, None]).T @ X_data  # d×d
            block_jk = -v[j] * v[k] * (Mjj + Mkk)
            
            # Residual contribution: v_j E[r · σ''(z_j) x x^T] δ_{jk}
            # For tanh, σ''(z) = -2 tanh(z)(1-tanh²(z)) ≈ -2z near 0
            # At λ=0, σ''(z_j) ≈ -2 z_j
            if j == k:
                Mres = (1/N) * (X_data * (r * (-2 * Z[:, j]))[:, None]).T @ X_data
                block_jk += v[j] * Mres
            
            dH[j*d_fix:(j+1)*d_fix, k*d_fix:(k+1)*d_fix] = block_jk
    
    dH = 0.5 * (dH + dH.T)  # symmetrise
    return dH

dH_analytical = compute_dH_analytical(m_test, v_test, W0_thy, X, y_data, alpha_reg)

# Compare with FD
wf0 = W0_thy.ravel()
dlam = 1e-5
dH_fd = (hess_fd(wf0, dlam, m_test, v_test) - hess_fd(wf0, 0., m_test, v_test)) / dlam

err_dH = np.linalg.norm(dH_analytical - dH_fd) / np.linalg.norm(dH_fd)
print(f"  ‖dH_analytical - dH_fd‖/‖dH_fd‖ = {err_dH:.4e}")

# Restrict analytical dH to flat subspace
dH_rest_analytical = Q_flat.T @ dH_analytical @ Q_flat
eigs_rest_analytical = np.sort(np.linalg.eigvalsh(dH_rest_analytical))
print(f"  λ₁'(0) from analytical restricted dH: {eigs_rest_analytical[0]:.8f}")
print(f"  vs FD restricted: {eigs_rest[0]:.8f}")
print(f"  Relative error: {abs(eigs_rest_analytical[0] - eigs_rest[0])/abs(eigs_rest[0]):.4e}")

# ══════════════════════════════════════════════════════════════
#  TASK 2C: Derive constant K in λ₁'(0) = K/m
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TASK 2C: Width scaling constant K")
print("="*70)

widths = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
scaling_data = []

for m in widths:
    v = np.linspace(0.5, 1.5, m)
    W0, _ = compute_W0_theory(m, v, Sigma, gamma, alpha_reg)
    
    # Verify W0 is correct
    wf = W0.ravel()
    gnorm = np.linalg.norm(grad_fn(wf, 0., m, v))
    
    # Compute λ₁'(0) analytically
    dH_a = compute_dH_analytical(m, v, W0, X, y_data, alpha_reg)
    
    # Build flat subspace basis
    n_par = m * d_fix
    C = np.kron(v.reshape(1,-1), np.eye(d_fix))
    _, S_c, Vt_c = np.linalg.svd(C, full_matrices=True)
    rank_C = np.sum(S_c > 1e-12)
    Q = Vt_c[rank_C:].T
    
    dH_r = Q.T @ dH_a @ Q
    eigs_r = np.sort(np.linalg.eigvalsh(dH_r))
    lp = eigs_r[0]
    
    # Theoretical λ*
    lam_star_thy = -alpha_reg / lp if lp < 0 else None
    
    # Also compute ||w_j*|| stats
    wj_norms = np.linalg.norm(W0, axis=1)
    vsq = np.sum(v**2)
    
    scaling_data.append({
        'm': m, 'vsq': vsq, 'lp': lp,
        'lam_star': lam_star_thy,
        'max_wj': np.max(wj_norms),
        'gnorm': gnorm,
    })
    
    ls_tag = f"{lam_star_thy:.4f}" if lam_star_thy and lam_star_thy < 10 else (f"{lam_star_thy:.1f}" if lam_star_thy else "---")
    print(f"  m={m:3d}  ‖v‖²={vsq:.2f}  λ₁'(0)={lp:.8f}  λ*_thy={ls_tag:>8s}  max‖w_j‖={np.max(wj_norms):.5f}  ‖∇‖={gnorm:.1e}")

# Fit K: λ₁'(0) = K/m
ms = np.array([d['m'] for d in scaling_data], dtype=float)
lps = np.array([d['lp'] for d in scaling_data])
# Fit lp = K/m + K2/m^2
A_fit = np.column_stack([1/ms, 1/ms**2])
(K_fit, K2_fit), _, _, _ = np.linalg.lstsq(A_fit, lps, rcond=None)
print(f"\n  Fit: λ₁'(0) = {K_fit:.6f}/m + {K2_fit:.4f}/m²")
print(f"  → K = {K_fit:.6f}")
print(f"  → λ* = α/|K/m| = α·m/|K| = {alpha_reg}/|{K_fit:.6f}|·m = {alpha_reg/abs(K_fit):.4f}·m")

# Fit λ* = A·m for those with crossings
lam_stars = [(d['m'], d['lam_star']) for d in scaling_data if d['lam_star'] and 0 < d['lam_star'] < 20]
if len(lam_stars) >= 2:
    ms_ls = np.array([x[0] for x in lam_stars], dtype=float)
    ls_ls = np.array([x[1] for x in lam_stars])
    slope = np.polyfit(ms_ls, ls_ls, 1)
    print(f"  Direct fit λ* = {slope[0]:.4f}·m + {slope[1]:.2f}")
    print(f"  Theory predicts slope = α/|K| = {alpha_reg/abs(K_fit):.4f}")

# ══════════════════════════════════════════════════════════════
#  TASK 1A-1B: Compute g_aa explicitly
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TASK 1A-1B: Explicit g_aa computation")
print("="*70)

m = m_test
v = v_test

# First: continue branch to λ* numerically
n_lam = 401
lam_arr = np.linspace(0, 1, n_lam)
W_cur = W0_thy.ravel().copy()

all_eigs_track = []
W_all = []
for i, lam in enumerate(lam_arr):
    lc = float(lam)
    res = minimize(lambda w, _l=lc: loss_fn(w, _l, m, v),
                   W_cur, jac=lambda w, _l=lc: grad_fn(w, _l, m, v),
                   method='L-BFGS-B', options={'maxiter': 3000, 'gtol': 1e-14})
    W_cur = res.x.copy()
    eigs = np.sort(eigh(hess_fd(W_cur, lam, m, v), eigvals_only=True))
    all_eigs_track.append(eigs)
    W_all.append(W_cur.copy())

all_eigs_track = np.array(all_eigs_track)

# Find λ*
e1 = all_eigs_track[:, 0]
for i in range(1, n_lam):
    if e1[i-1] > 0 and e1[i] <= 0:
        frac = e1[i-1] / (e1[i-1] - e1[i])
        lam_star = lam_arr[i-1] + frac*(lam_arr[i] - lam_arr[i-1])
        cross_idx = i
        break

print(f"  λ* = {lam_star:.6f}")

# Re-optimise at λ*
res_star = minimize(lambda w: loss_fn(w, lam_star, m, v),
                    W_all[cross_idx], jac=lambda w: grad_fn(w, lam_star, m, v),
                    method='L-BFGS-B', options={'maxiter': 8000, 'gtol': 1e-15})
W_star = res_star.x
H_star = hess_fd(W_star, lam_star, m, v)
evals_star, evecs_star = eigh(H_star)
v0 = evecs_star[:, 0]  # kernel direction

print(f"  eig₁ = {evals_star[0]:.8f}, eig₂ = {evals_star[1]:.8f}")
print(f"  |eig₁/eig₂| = {abs(evals_star[0]/evals_star[1]):.6f}")

# ── Numerical g_aa via reduced equation ──────────────────────
a_range = np.linspace(-0.5, 0.5, 201)
phi_star = np.array([loss_fn(W_star + a*v0, lam_star, m, v) for a in a_range])
phi_star -= phi_star[100]  # center

# Fit: phi(a) = c1*a + c2*a^2 + c3*a^3 + c4*a^4
Amat = np.column_stack([a_range, a_range**2, a_range**3, a_range**4])
c1_n, c2_n, c3_n, c4_n = np.linalg.lstsq(Amat, phi_star, rcond=None)[0]

g_aa_num = 2 * c3_n  # g(a) = φ'(a), so g_aa = φ'''(0) = 6c3... 
# Wait: φ(a) = c1 a + c2 a² + c3 a³ + c4 a⁴
# g(a) = φ'(a) = c1 + 2c2 a + 3c3 a² + 4c4 a³
# g_a(0) = c1 (should be ≈ 0 since W_star is critical)
# g_aa(0) = 2·(2c2)... no.
# g(a) = ∂_a φ(a) = c1 + 2c2 a + 3c3 a² + 4c4 a³
# g_a(0,0) = c1 ≈ 0 (branch straightened)  
# g_{aa}(0,0) = d/da[g(a)]|_{a=0} = 2c2... 
# But wait — g(0,μ)=0 for all μ means c1=0 on the trivial branch.
# Actually c1 ≈ 0 because W_star is a critical point.
# g_aa = ∂²g/∂a² = ∂³φ/∂a³ = 6c3
# g_aaa = ∂³g/∂a³ = ∂⁴φ/∂a⁴ = 24c4

g_aa_num = 6 * c3_n
g_aaa_num = 24 * c4_n
print(f"\n  Numerical reduced coefficients:")
print(f"    c1 = {c1_n:.8f} (should ≈ 0)")
print(f"    c2 = {c2_n:.8f}")
print(f"    c3 = {c3_n:.8f}")
print(f"    c4 = {c4_n:.8f}")
print(f"    g_aa  = 6·c3 = {g_aa_num:.8f}")
print(f"    g_aaa = 24·c4 = {g_aaa_num:.8f}")
print(f"    |g_aa/g_aaa| = {abs(g_aa_num/g_aaa_num):.6f}")

# ── Analytical g_aa: D³L[v0, v0, v0] ────────────────────────
print("\n  --- Analytical g_aa ---")

W_star_mat = W_star.reshape(m, d_fix)
Z_star = X @ W_star_mat.T  # N × m
r_star = h(Z_star, lam_star) @ v - y_data  # N

# v0 reshaped as m × d blocks
v0_blocks = v0.reshape(m, d_fix)  # v_{0,j} for j=1..m

# g_aa = D³L[v0,v0,v0] = third directional derivative of loss
# Compute by finite differences for verification
eps_fd = 1e-4
g_plus = grad_fn(W_star + eps_fd * v0, lam_star, m, v)
g_minus = grad_fn(W_star - eps_fd * v0, lam_star, m, v)
g_center = grad_fn(W_star, lam_star, m, v)
# D²L[v0] · v0 = (g(W+εv0) - g(W-εv0))/(2ε) · v0  — this gives g_a at ε
# We need D³L[v0,v0,v0] = d/dε [v0^T H(W+εv0) v0]|_{ε=0}
# = (v0^T H(W+εv0) v0 - v0^T H(W-εv0) v0) / (2ε)

H_plus = hess_fd(W_star + eps_fd * v0, lam_star, m, v)
H_minus = hess_fd(W_star - eps_fd * v0, lam_star, m, v)
g_aa_fd3 = (v0 @ H_plus @ v0 - v0 @ H_minus @ v0) / (2 * eps_fd)
print(f"  g_aa from D³L FD: {g_aa_fd3:.8f}")
print(f"  g_aa from polynomial fit: {g_aa_num:.8f}")
print(f"  Relative difference: {abs(g_aa_fd3 - g_aa_num)/max(abs(g_aa_num), 1e-15):.4f}")

# Now compute analytically using the explicit formula
# g_aa = Σ_j [terms from GN + residual third derivatives]
# For tanh at λ*: h''(z, λ*) = λ* · σ''(z) = λ*(-2 tanh(z)(1-tanh²(z)))
# ≈ -2λ* z for small z

# The dominant GN contribution:
# D³L^GN[v0,v0,v0] = Σ_j v_j · E[ h''(w_j^T x, λ*) · (v_{0,j}^T x)² · (Σ_k v_k h'(w_k^T x, λ*) (v_{0,k}^T x)) ]
#                   + cross terms from differentiating the other h' factor

g_aa_analytical = 0.0
for j in range(m):
    zj = Z_star[:, j]  # w_j^T x, N-vector
    hjpp = hpp(zj, lam_star)  # h''(w_j^T x, λ*), N-vector
    vjx = X @ v0_blocks[j]    # v_{0,j}^T x, N-vector
    
    # Sum over k for the Jacobian factor
    Jac_v0 = sum(v[k] * hp(Z_star[:, k], lam_star) * (X @ v0_blocks[k])
                 for k in range(m))  # Σ_k v_k h'(w_k^T x) (v_{0,k}^T x)
    
    # Term 1: from differentiating h' in the j-th factor
    # v_j · E[r · h''(z_j) · (v_{0,j}^T x) · (Σ_k v_k h'(z_k) (v_{0,k}^T x)) ]... 
    # Actually: D³(½E[(f-y)²])[v0,v0,v0] has the structure:
    # = E[Df[v0] · D²f[v0,v0]] + E[(f-y) · D³f[v0,v0,v0]]
    # where Df[v0] = Σ_j v_j h'(z_j)(v_{0,j}^T x)
    # D²f[v0,v0] = Σ_j v_j h''(z_j)(v_{0,j}^T x)²  
    # D³f[v0,v0,v0] = Σ_j v_j h'''(z_j)(v_{0,j}^T x)³

# Cleaner: use f = Σ_j v_j h(w_j^T x, λ)
# D_W f [δW] = Σ_j v_j h'(w_j^T x)(δw_j^T x)
# D²_W f [δW, δW] = Σ_j v_j h''(w_j^T x)(δw_j^T x)²
# D³_W f [δW, δW, δW] = Σ_j v_j h'''(w_j^T x)(δw_j^T x)³

# Then D³(½‖f-y‖²)[v0,v0,v0] = E[Df[v0] · D²f[v0,v0]] + E[r · D³f[v0,v0,v0]]
# (from the Leibniz rule on D³(f·(f-y)))
# Wait, more carefully:
# L = ½E[(f-y)²]
# DL[v0] = E[(f-y) Df[v0]]
# D²L[v0,v0] = E[Df[v0]² + (f-y)D²f[v0,v0]]
# D³L[v0,v0,v0] = E[3 Df[v0] D²f[v0,v0] + (f-y) D³f[v0,v0,v0]]
# (the factor 3 comes from symmetry of the third derivative)

Df_v0 = sum(v[j] * hp(Z_star[:,j], lam_star) * (X @ v0_blocks[j]) for j in range(m))  # N
D2f_v0 = sum(v[j] * hpp(Z_star[:,j], lam_star) * (X @ v0_blocks[j])**2 for j in range(m))  # N
D3f_v0 = sum(v[j] * hppp(Z_star[:,j], lam_star) * (X @ v0_blocks[j])**3 for j in range(m))  # N

g_aa_analytical = 3 * np.mean(Df_v0 * D2f_v0) + np.mean(r_star * D3f_v0)
# regularization contributes 0 to third derivative (it's quadratic)

print(f"\n  Analytical g_aa = {g_aa_analytical:.8f}")
print(f"  Numerical g_aa  = {g_aa_fd3:.8f}")
print(f"  Relative error: {abs(g_aa_analytical - g_aa_fd3)/max(abs(g_aa_fd3),1e-15):.4e}")

# Decompose contributions
term_GN = 3 * np.mean(Df_v0 * D2f_v0)
term_res = np.mean(r_star * D3f_v0)
print(f"  GN contribution (3·E[Df·D²f]): {term_GN:.8f}")
print(f"  Residual contribution (E[r·D³f]): {term_res:.8f}")

# ── g_aaa: D⁴L[v0,v0,v0,v0] ────────────────────────────────
print("\n  --- Analytical g_aaa ---")
# D⁴L[v0^4] = E[3(D²f[v0,v0])² + 4Df[v0]D³f[v0,v0,v0] + (f-y)D⁴f[v0^4]]
# + regularization: α · D⁴(½‖W‖²) = 0
# D⁴f[v0^4] = Σ_j v_j h''''(z_j)(v_{0,j}^T x)⁴

# h''''(z,λ) = λ σ''''(z)
# For tanh: σ''''(z) = d/dz[-2(1-s²)(1-3s²)] where s=tanh(z)
# = -2[-2s(1-s²)(1-3s²) + (1-s²)(-6s(1-s²))]
# At z=0: σ''''(0) = -2[0 + (-6·0)] ... let me compute numerically
z_test = np.array([0.0])
eps_h = 1e-4
hpppp_0 = (hppp(z_test+eps_h, lam_star) - hppp(z_test-eps_h, lam_star)) / (2*eps_h)

# Actually for the fourth derivative we need h'''' which is messy.
# Better to compute D⁴f numerically:
def D4f_v0_vals(W_s, v0_dir, lam_s, m_val, v_vec, eps=1e-3):
    """D⁴f[v0,v0,v0,v0] via 4th-order FD of f along v0."""
    def f_eval(a):
        W = (W_s + a*v0_dir).reshape(m_val, d_fix)
        Z = X @ W.T
        return h(Z, lam_s) @ v_vec  # N-vector
    # 4th derivative: (f(2ε)-4f(ε)+6f(0)-4f(-ε)+f(-2ε))/ε⁴
    return (f_eval(2*eps) - 4*f_eval(eps) + 6*f_eval(0) - 4*f_eval(-eps) + f_eval(-2*eps)) / eps**4

D4f = D4f_v0_vals(W_star, v0, lam_star, m, v)

g_aaa_analytical = (3 * np.mean(D2f_v0**2) 
                    + 4 * np.mean(Df_v0 * D3f_v0)
                    + np.mean(r_star * D4f))

# FD verification
eps_g = 5e-4
H_2p = hess_fd(W_star + 2*eps_g*v0, lam_star, m, v)
H_p  = hess_fd(W_star + eps_g*v0, lam_star, m, v)
H_0  = hess_fd(W_star, lam_star, m, v)
H_m  = hess_fd(W_star - eps_g*v0, lam_star, m, v)
H_2m = hess_fd(W_star - 2*eps_g*v0, lam_star, m, v)
# D⁴L[v0^4] = v0^T [d²H/da²] v0 = v0^T (H(+ε)-2H(0)+H(-ε))/ε² v0
D2H_v0 = (v0 @ H_p @ v0 - 2*v0 @ H_0 @ v0 + v0 @ H_m @ v0) / eps_g**2
g_aaa_fd = D2H_v0

print(f"  Analytical g_aaa = {g_aaa_analytical:.8f}")
print(f"  FD g_aaa (D²(v0·H·v0)):  {g_aaa_fd:.8f}")
print(f"  Polynomial fit g_aaa = 24·c4 = {g_aaa_num:.8f}")
print(f"  Rel error (analytical vs FD): {abs(g_aaa_analytical - g_aaa_fd)/max(abs(g_aaa_fd),1e-15):.4e}")

# Decompose
t1 = 3 * np.mean(D2f_v0**2)
t2 = 4 * np.mean(Df_v0 * D3f_v0)
t3 = np.mean(r_star * D4f)
print(f"  Contributions: 3E[(D²f)²]={t1:.8f}, 4E[Df·D³f]={t2:.8f}, E[r·D⁴f]={t3:.8f}")

# ══════════════════════════════════════════════════════════════
#  TASK 1D: Explicit ratio with all constants
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TASK 1D: Explicit ratio theorem")
print("="*70)

ratio = abs(g_aa_analytical / g_aaa_analytical)
print(f"  |g_aa/g_aaa| = {ratio:.6f}")
print(f"  |g_aa| = {abs(g_aa_analytical):.8f}")
print(f"  |g_aaa| = {abs(g_aaa_analytical):.8f}")

# Bound: |g_aa| ≤ 3 · |E[Df·D²f]| + |E[r·D³f]|
# The GN term: E[Df·D²f] involves h''(z_j) which has factor ~ z_j for tanh
# Key: max|z_j| = max|w_j^T x| 
max_preact = np.max(np.abs(Z_star))
mean_preact_sq = np.mean(Z_star**2)
print(f"\n  Pre-activation statistics at λ*:")
print(f"    max|w_j^T x| = {max_preact:.4f}")
print(f"    mean(w_j^T x)² = {mean_preact_sq:.6f}")
print(f"    λ* = {lam_star:.4f}")
print(f"    max‖w_j*‖ = {np.max(np.linalg.norm(W_star_mat, axis=1)):.5f}")

# Analytical bound on |g_aa|:
# |3E[Df·D²f]| ≤ 3 · E[|Df|·|D²f|]
# |D²f[v0,v0]| = |Σ_j v_j h''(z_j)(v_{0,j}^T x)²|
#              ≤ Σ_j |v_j| · |h''(z_j)| · (v_{0,j}^T x)²
#              ≤ Σ_j |v_j| · 2λ*|z_j| · (v_{0,j}^T x)²  (using |h''(z)| ≤ 2λ*|z| for small z)
# So |g_aa| ≲ 6λ* · E[|Df| · max_j|z_j| · Σ_j|v_j|(v_{0,j}^T x)²]

# For g_aaa, the dominant term 3E[(D²f)²] involves h''² ~ (λ*z)² which is ALSO small
# The key term is 4E[Df·D³f] where D³f = Σ_j v_j h'''(z_j)(v_{0,j}^T x)³
# h'''(z,λ) = λσ'''(z), σ'''(0)=-2, so h'''(0,λ*)=-2λ*
# This does NOT vanish at z=0!

# So: |g_aaa| ≥ |4E[Df·D³f]| - |3E[(D²f)²]| - |E[r·D⁴f]|
# And 4E[Df·D³f] has leading contribution 4·(-2λ*)·E[Df · Σ_j v_j (v_{0,j}^T x)³]

print(f"\n  Theoretical bound decomposition:")
print(f"    GN bound on |g_aa|: 3|E[Df·D²f]| = {3*abs(np.mean(Df_v0*D2f_v0)):.8f}")
print(f"    Residual bound:     |E[r·D³f]| = {abs(np.mean(r_star*D3f_v0)):.8f}")
print(f"    Leading |g_aaa| term: 4|E[Df·D³f]| = {4*abs(np.mean(Df_v0*D3f_v0)):.8f}")
print(f"    Suppressed term: 3E[(D²f)²] = {3*np.mean(D2f_v0**2):.8f}")

# ══════════════════════════════════════════════════════════════
#  TASK 2D: Full width-scaling verification (5 tests)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TASK 2D: Width-scaling verification (5 tests)")
print("="*70)

# For each m, compare theory vs numerics
test_widths = [5, 8, 10, 15, 20, 30]
verification_table = []

for m in test_widths:
    v = np.linspace(0.5, 1.5, m)
    n_par = m * d_fix
    
    # Theory
    W0_t, _ = compute_W0_theory(m, v, Sigma, gamma, alpha_reg)
    dH_t = compute_dH_analytical(m, v, W0_t, X, y_data, alpha_reg)
    C = np.kron(v.reshape(1,-1), np.eye(d_fix))
    _, S_c, Vt_c = np.linalg.svd(C, full_matrices=True)
    Q = Vt_c[np.sum(S_c > 1e-12):].T
    dH_r = Q.T @ dH_t @ Q
    lp_theory = np.sort(np.linalg.eigvalsh(dH_r))[0]
    ls_theory = -alpha_reg / lp_theory if lp_theory < 0 else None
    
    # Numerical: continue branch
    W_c = W0_t.ravel().copy()
    n_l = 201
    la = np.linspace(0, 1, n_l)
    eig_track = []
    for lam in la:
        lc = float(lam)
        res = minimize(lambda w, _l=lc: loss_fn(w, _l, m, v),
                       W_c, jac=lambda w, _l=lc: grad_fn(w, _l, m, v),
                       method='L-BFGS-B', options={'maxiter': 2000, 'gtol': 1e-13})
        W_c = res.x.copy()
        if n_par <= 250:
            eigs = np.sort(eigh(hess_fd(W_c, lam, m, v), eigvals_only=True))
            eig_track.append(eigs[0])
        else:
            from scipy.sparse.linalg import LinearOperator, eigsh
            def hv(vv, _w=W_c, _l=lc):
                eps=1e-5
                return (grad_fn(_w+eps*vv,_l,m,v)-grad_fn(_w-eps*vv,_l,m,v))/(2*eps)
            Hop = LinearOperator((n_par,n_par), matvec=hv)
            try:
                eig_track.append(eigsh(Hop,k=1,which='SA',tol=1e-6)[0][0])
            except:
                eig_track.append(np.nan)
    
    eig_track = np.array(eig_track)
    
    # Numerical λ₁'(0)
    lp_num = (eig_track[1] - eig_track[0]) / (la[1] - la[0])
    
    # Numerical λ*
    ls_num = None
    for i in range(1, len(la)):
        if eig_track[i-1] > 0 and eig_track[i] <= 0:
            frac = eig_track[i-1] / (eig_track[i-1] - eig_track[i])
            ls_num = la[i-1] + frac*(la[i]-la[i-1])
            break
    
    # λ₁(1) for overcritical case
    lam1_at_1 = eig_track[-1]
    lam1_at_1_pred = alpha_reg + lp_theory * 1.0  # first-order prediction
    
    row = {
        'm': m, 'lp_thy': lp_theory, 'lp_num': lp_num,
        'ls_thy': ls_theory, 'ls_num': ls_num,
        'lam1_0': eig_track[0], 'lam1_1': lam1_at_1,
        'lam1_1_pred': lam1_at_1_pred,
    }
    verification_table.append(row)
    
    ls_t = f"{ls_theory:.4f}" if ls_theory and ls_theory < 10 else "---"
    ls_n = f"{ls_num:.4f}" if ls_num else "---"
    print(f"  m={m:3d}  λ₁'(0): thy={lp_theory:.6f} num={lp_num:.6f} err={abs(lp_theory-lp_num)/max(abs(lp_num),1e-15):.3f}"
          f"  λ*: thy={ls_t:>7s} num={ls_n:>7s}")

# ══════════════════════════════════════════════════════════════
#  TASK 2E: Second-order correction
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TASK 2E: Second-order correction to λ*")
print("="*70)

# For m=10, compute λ₁''(0) from FD of eigenvalue
m = m_test; v = v_test
W0_flat = W0_thy.ravel()

# λ₁(λ) at several small λ values
lam_probe = np.linspace(0, 0.3, 31)
eig1_probe = []
W_p = W0_flat.copy()
for lam in lam_probe:
    lc = float(lam)
    res = minimize(lambda w, _l=lc: loss_fn(w, _l, m, v),
                   W_p, jac=lambda w, _l=lc: grad_fn(w, _l, m, v),
                   method='L-BFGS-B', options={'maxiter': 3000, 'gtol': 1e-14})
    W_p = res.x.copy()
    eigs = np.sort(eigh(hess_fd(W_p, lam, m, v), eigvals_only=True))
    eig1_probe.append(eigs[0])

eig1_probe = np.array(eig1_probe)

# Fit λ₁(λ) = a + bλ + cλ²
A2 = np.column_stack([np.ones_like(lam_probe), lam_probe, lam_probe**2])
abc = np.linalg.lstsq(A2, eig1_probe, rcond=None)[0]
lam1_0_fit, lam1p_fit, lam1pp_fit = abc

print(f"  Fit: λ₁(λ) = {lam1_0_fit:.6f} + {lam1p_fit:.6f}·λ + {lam1pp_fit:.6f}·λ²")
print(f"  λ₁(0) = {lam1_0_fit:.6f} (should = α = {alpha_reg})")
print(f"  λ₁'(0) = {lam1p_fit:.6f}")
print(f"  λ₁''(0) = {2*lam1pp_fit:.6f}")

# First-order λ*
ls_1st = -lam1_0_fit / lam1p_fit
# Second-order λ*: solve a + bλ + cλ² = 0
disc = lam1p_fit**2 - 4*lam1pp_fit*lam1_0_fit
if disc >= 0:
    ls_2nd_a = (-lam1p_fit - np.sqrt(disc)) / (2*lam1pp_fit)
    ls_2nd_b = (-lam1p_fit + np.sqrt(disc)) / (2*lam1pp_fit)
    ls_2nd = min(x for x in [ls_2nd_a, ls_2nd_b] if x > 0)
else:
    ls_2nd = None

print(f"\n  λ* (1st order): {ls_1st:.6f}")
print(f"  λ* (2nd order): {ls_2nd:.6f}" if ls_2nd else "  λ* (2nd order): complex roots")
print(f"  λ* (numerical): {lam_star:.6f}")
print(f"  1st-order error: {abs(ls_1st - lam_star)/lam_star:.4f}")
if ls_2nd:
    print(f"  2nd-order error: {abs(ls_2nd - lam_star)/lam_star:.4f}")

# ══════════════════════════════════════════════════════════════
#  FIGURE: Comprehensive verification
# ══════════════════════════════════════════════════════════════
print("\n[Plotting]...")

fig = plt.figure(figsize=(24, 18))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Proof Tightening: Explicit Constants and Width-Scaling Verification', fontsize=15, y=0.99)

# Panel (0,0): W*(0) theory vs numerical
ax = fig.add_subplot(gs[0, 0])
ax.scatter(W0_thy.ravel(), W0_num.ravel(), s=8, alpha=0.7)
lim = max(abs(W0_thy).max(), abs(W0_num).max()) * 1.2
ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
ax.set_xlabel('W*(0) theory'); ax.set_ylabel('W*(0) numerical')
ax.set_title(f'Task 2A: W*(0) (err={err_W0:.1e})'); ax.grid(alpha=.25)
ax.set_aspect('equal')

# Panel (0,1): λ₁(λ) fit and predictions
ax = fig.add_subplot(gs[0, 1])
ax.plot(lam_probe, eig1_probe, 'bo-', ms=4, lw=1.5, label='numerical')
lam_fine = np.linspace(0, 0.3, 100)
ax.plot(lam_fine, abc[0] + abc[1]*lam_fine + abc[2]*lam_fine**2, 'r--', lw=1.5,
        label=f'quadratic fit')
ax.axhline(0, color='k', ls='--', alpha=.3)
if ls_1st < 1.5:
    ax.axvline(ls_1st, color='green', ls=':', label=f'λ* 1st={ls_1st:.3f}')
if ls_2nd and ls_2nd < 1.5:
    ax.axvline(ls_2nd, color='orange', ls=':', label=f'λ* 2nd={ls_2nd:.3f}')
ax.axvline(lam_star, color='blue', ls=':', alpha=.5, label=f'λ* num={lam_star:.3f}')
ax.set_xlabel('λ'); ax.set_ylabel('λ₁(λ)')
ax.set_title('Task 2E: Eigenvalue trajectory + corrections'); ax.legend(fontsize=8); ax.grid(alpha=.25)

# Panel (0,2): g_aa components
ax = fig.add_subplot(gs[0, 2])
labels = ['g_aa\n(analytical)', 'g_aa\n(FD)', '3E[Df·D²f]\n(GN)', 'E[r·D³f]\n(residual)']
vals = [g_aa_analytical, g_aa_fd3, term_GN, term_res]
colors = ['blue', 'red', 'green', 'orange']
ax.bar(range(len(vals)), vals, color=colors, alpha=0.7)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Value'); ax.set_title('Task 1A: g_aa decomposition')
ax.axhline(0, color='k', lw=0.5); ax.grid(alpha=.25, axis='y')

# Panel (0,3): g_aaa components
ax = fig.add_subplot(gs[0, 3])
labels = ['g_aaa\n(analytical)', 'g_aaa\n(FD)', '3E[(D²f)²]', '4E[Df·D³f]', 'E[r·D⁴f]']
vals = [g_aaa_analytical, g_aaa_fd, t1, t2, t3]
colors = ['blue', 'red', 'green', 'orange', 'purple']
ax.bar(range(len(vals)), vals, color=colors, alpha=0.7)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('Value'); ax.set_title('Task 1C: g_aaa decomposition')
ax.axhline(0, color='k', lw=0.5); ax.grid(alpha=.25, axis='y')

# Panel (1,0): λ₁'(0) scaling
ax = fig.add_subplot(gs[1, 0])
ms_all = np.array([d['m'] for d in scaling_data], dtype=float)
lps_all = np.array([d['lp'] for d in scaling_data])
ax.plot(ms_all, lps_all, 'bo-', ms=8, lw=2, label="analytical λ₁'(0)")
m_fit = np.linspace(3, 100, 200)
ax.plot(m_fit, K_fit/m_fit + K2_fit/m_fit**2, 'r--', lw=1.5,
        label=f'fit: {K_fit:.5f}/m + {K2_fit:.3f}/m²')
ax.set_xlabel('m'); ax.set_ylabel("λ₁'(0)")
ax.set_title(f"Task 2C: λ₁'(0) scaling (K={K_fit:.5f})"); ax.legend(fontsize=9); ax.grid(alpha=.25)

# Panel (1,1): λ₁'(0) · m vs m (should be constant = K)
ax = fig.add_subplot(gs[1, 1])
ax.plot(ms_all, lps_all * ms_all, 'bo-', ms=8, lw=2)
ax.axhline(K_fit, color='r', ls='--', lw=1.5, label=f'K = {K_fit:.5f}')
ax.set_xlabel('m'); ax.set_ylabel("λ₁'(0) · m")
ax.set_title("λ₁'(0)·m should → K"); ax.legend(); ax.grid(alpha=.25)

# Panel (1,2): λ* vs m (theory and numerical)
ax = fig.add_subplot(gs[1, 2])
vt = [(d['m'], d['ls_thy']) for d in verification_table if d['ls_thy'] and d['ls_thy'] < 5]
vn = [(d['m'], d['ls_num']) for d in verification_table if d['ls_num']]
if vt:
    ax.plot([x[0] for x in vt], [x[1] for x in vt], 'rs-', ms=9, lw=2, label='theory (1st order)')
if vn:
    ax.plot([x[0] for x in vn], [x[1] for x in vn], 'bo-', ms=9, lw=2, label='numerical')
ax.axhline(1, color='k', ls=':', alpha=.4, label='λ=1 boundary')
ax.set_xlabel('m'); ax.set_ylabel('λ*')
ax.set_title('Task 2D: λ* theory vs numerical'); ax.legend(); ax.grid(alpha=.25)

# Panel (1,3): λ₁(0) verification
ax = fig.add_subplot(gs[1, 3])
lam1_0s = [d['lam1_0'] for d in verification_table]
ax.plot([d['m'] for d in verification_table], lam1_0s, 'go-', ms=9, lw=2)
ax.axhline(alpha_reg, color='r', ls='--', lw=1.5, label=f'α = {alpha_reg}')
ax.set_xlabel('m'); ax.set_ylabel('λ₁(0)')
ax.set_title('Test 1: λ₁(0) = α for all m'); ax.legend(); ax.grid(alpha=.25)

# Panel (2,0): λ₁'(0) theory vs numerical
ax = fig.add_subplot(gs[2, 0])
for d in verification_table:
    ax.plot(d['lp_thy'], d['lp_num'], 'ko', ms=10)
    ax.annotate(f"m={d['m']}", (d['lp_thy'], d['lp_num']), fontsize=9,
                xytext=(5, 5), textcoords='offset points')
rng = [min(d['lp_thy'] for d in verification_table), max(d['lp_thy'] for d in verification_table)]
ax.plot(rng, rng, 'r--', lw=1.5)
ax.set_xlabel("λ₁'(0) theory"); ax.set_ylabel("λ₁'(0) numerical")
ax.set_title("Test 2: λ₁'(0) theory vs numerical"); ax.grid(alpha=.25)

# Panel (2,1): Reduced potential at λ*
ax = fig.add_subplot(gs[2, 1])
ax.plot(a_range, phi_star, 'b-', lw=2, label='actual φ(a)')
fit_vals = Amat @ np.array([c1_n, c2_n, c3_n, c4_n])
ax.plot(a_range, fit_vals, 'r--', lw=1.5, label=f'polynomial fit')
ax.set_xlabel('a'); ax.set_ylabel('φ(a)−φ(0)')
ax.set_title(f'Reduced potential (|g_aa/g_aaa|={ratio:.3f})'); ax.legend(); ax.grid(alpha=.25)

# Panel (2,2): Overcritical λ₁(1) verification
ax = fig.add_subplot(gs[2, 2])
over = [(d['m'], d['lam1_1'], d['lam1_1_pred']) for d in verification_table if d['ls_num'] is None]
if over:
    ms_o = [x[0] for x in over]
    ax.plot(ms_o, [x[1] for x in over], 'bo-', ms=9, lw=2, label='λ₁(1) numerical')
    ax.plot(ms_o, [x[2] for x in over], 'rs--', ms=8, lw=1.5, label='λ₁(1) = α+λ₁\'·1')
    ax.axhline(0, color='k', ls='--', alpha=.3)
    ax.set_xlabel('m'); ax.set_ylabel('λ₁(1)')
    ax.set_title('Test 3: λ₁(1) for m > m*'); ax.legend(); ax.grid(alpha=.25)

# Panel (2,3): Summary
ax = fig.add_subplot(gs[2, 3])
ax.axis('off')
summary = (
    f"SUMMARY OF EXPLICIT CONSTANTS\n"
    f"{'='*40}\n\n"
    f"g_aa = {g_aa_analytical:.6f}\n"
    f"  = 3E[Df·D²f] + E[r·D³f]\n"
    f"  = {term_GN:.6f} + {term_res:.6f}\n\n"
    f"g_aaa = {g_aaa_analytical:.6f}\n"
    f"  = 3E[(D²f)²]+4E[Df·D³f]+E[r·D⁴f]\n"
    f"  = {t1:.6f}+{t2:.6f}+{t3:.6f}\n\n"
    f"|g_aa/g_aaa| = {ratio:.4f}\n\n"
    f"Width scaling: λ₁'(0) = {K_fit:.5f}/m\n"
    f"  → λ* = {alpha_reg:.4f}·m/{abs(K_fit):.5f}\n"
    f"  → λ* = {alpha_reg/abs(K_fit):.4f}·m\n\n"
    f"2nd-order: λ₁''(0) = {2*lam1pp_fit:.5f}\n"
    f"  1st-order λ*: {ls_1st:.4f} (err {abs(ls_1st-lam_star)/lam_star:.1%})\n"
    f"  2nd-order λ*: {ls_2nd:.4f} (err {abs(ls_2nd-lam_star)/lam_star:.1%})" if ls_2nd else ""
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('/home/claude/fig_proof_tightening.png', dpi=160, bbox_inches='tight')
plt.close()
print("  → fig_proof_tightening.png")

print(f"\n{'='*70}")
print(f"  ALL TASKS COMPLETE ({time.time()-T0:.0f}s)")
print(f"{'='*70}")
