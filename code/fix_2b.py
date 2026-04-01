#!/usr/bin/env python3
"""
Task 2B Fix: Exact analytical dH/dλ (no Taylor approximation)
==============================================================
Replace σ'(z)-1 ≈ -z² with exact -tanh²(z)
Replace σ''(z) ≈ -2z with exact -2tanh(z)(1-tanh²(z))
Then re-run the full verification pipeline.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
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
print("  TASK 2B FIX: EXACT ANALYTICAL dH/dλ")
print("="*70)

np.random.seed(77)
d, N = 5, 500
alpha_reg = 4e-3
X = np.random.randn(N, d)
wl  = np.array([1., .5, -.3, .2, -.1])
wnl = np.array([.3, -.7, .5, -.2, .4])
y_data = X @ wl + 0.4*np.tanh(1.5 * X @ wnl) + 0.05*np.random.randn(N)
Sigma = (X.T @ X) / N
gamma = (X.T @ y_data) / N

def loss_fn(wf, lam, m, v):
    W = wf.reshape(m, d); Z = X @ W.T
    return 0.5*np.mean((h(Z, lam) @ v - y_data)**2) + 0.5*alpha_reg*np.sum(wf**2)

def grad_fn(wf, lam, m, v):
    W = wf.reshape(m, d); Z = X @ W.T
    r = h(Z, lam) @ v - y_data
    return ((1/N)*((r[:,None]*hp(Z,lam)*v[None,:]).T @ X) + alpha_reg*W).ravel()

def hess_fd(wf, lam, m, v, eps=1e-5):
    g0 = grad_fn(wf, lam, m, v); n = len(wf)
    H = np.empty((n, n))
    for i in range(n):
        wp = wf.copy(); wp[i] += eps
        H[i] = (grad_fn(wp, lam, m, v) - g0) / eps
    return 0.5*(H + H.T)

def compute_W0_theory(m, v):
    vsq = np.sum(v**2)
    M = alpha_reg * np.eye(d) + vsq * Sigma
    u_eff = vsq * np.linalg.solve(M, gamma)
    residual = gamma - Sigma @ u_eff
    return np.outer(v, residual) / alpha_reg, u_eff

# ── Three versions of dH/dλ ─────────────────────────────────
def compute_dH_taylor(m, v, W0, label="Taylor"):
    """OLD: uses σ'(z)-1 ≈ -z², σ''(z) ≈ -2z"""
    Z = X @ W0.T
    r = Z @ v - y_data
    n_par = m * d
    dH = np.zeros((n_par, n_par))
    for j in range(m):
        for k in range(m):
            # Taylor: σ'(z)-1 ≈ -z²
            Mjj = (1/N) * (X * (Z[:,j]**2)[:,None]).T @ X
            Mkk = (1/N) * (X * (Z[:,k]**2)[:,None]).T @ X
            block = -v[j]*v[k] * (Mjj + Mkk)
            if j == k:
                # Taylor: σ''(z) ≈ -2z
                Mres = (1/N) * (X * (r * (-2*Z[:,j]))[:,None]).T @ X
                block += v[j] * Mres
            dH[j*d:(j+1)*d, k*d:(k+1)*d] = block
    return 0.5*(dH + dH.T)

def compute_dH_exact(m, v, W0, label="Exact"):
    """NEW: uses exact σ'(z)-1 = -tanh²(z), σ''(z) = -2tanh(z)(1-tanh²(z))"""
    Z = X @ W0.T
    r = Z @ v - y_data
    n_par = m * d
    dH = np.zeros((n_par, n_par))
    for j in range(m):
        for k in range(m):
            # Exact: σ'(z)-1 = sech²(z)-1 = -tanh²(z)
            sp_m1_j = -np.tanh(Z[:,j])**2  # σ'(z_j) - 1
            sp_m1_k = -np.tanh(Z[:,k])**2  # σ'(z_k) - 1
            Mjj = (1/N) * (X * sp_m1_j[:,None]).T @ X
            Mkk = (1/N) * (X * sp_m1_k[:,None]).T @ X
            block = v[j]*v[k] * (Mjj + Mkk)
            if j == k:
                # Exact: σ''(z) = -2tanh(z)(1-tanh²(z))
                s = np.tanh(Z[:,j])
                spp = -2*s*(1-s*s)
                Mres = (1/N) * (X * (r * spp)[:,None]).T @ X
                block += v[j] * Mres
            dH[j*d:(j+1)*d, k*d:(k+1)*d] = block
    return 0.5*(dH + dH.T)

def compute_dH_fd(m, v, W0, dlam=1e-5):
    """FD ground truth"""
    wf = W0.ravel()
    H0 = hess_fd(wf, 0., m, v)
    H1 = hess_fd(wf, dlam, m, v)
    return (H1 - H0) / dlam

def get_flat_basis(m, v):
    n_par = m * d
    C = np.kron(v.reshape(1,-1), np.eye(d))
    _, S_c, Vt_c = np.linalg.svd(C, full_matrices=True)
    rank_C = np.sum(S_c > 1e-12)
    return Vt_c[rank_C:].T  # n_par × (m-1)d

# ══════════════════════════════════════════════════════════════
#  Step 1-3: Compare all three versions for m=10
# ══════════════════════════════════════════════════════════════
print("\n[Step 1-3] Comparing Taylor vs Exact vs FD for m=10")
m = 10
v = np.linspace(0.5, 1.5, m)
W0, _ = compute_W0_theory(m, v)

dH_tay = compute_dH_taylor(m, v, W0)
dH_exa = compute_dH_exact(m, v, W0)
dH_ref = compute_dH_fd(m, v, W0)

err_tay = np.linalg.norm(dH_tay - dH_ref) / np.linalg.norm(dH_ref)
err_exa = np.linalg.norm(dH_exa - dH_ref) / np.linalg.norm(dH_ref)
print(f"  ‖dH_taylor - dH_fd‖ / ‖dH_fd‖ = {err_tay:.4e}  (OLD: 17% error)")
print(f"  ‖dH_exact  - dH_fd‖ / ‖dH_fd‖ = {err_exa:.4e}  (NEW: should be <1%)")

# Restricted eigenvalues
Q = get_flat_basis(m, v)
for label, dH in [("Taylor", dH_tay), ("Exact", dH_exa), ("FD", dH_ref)]:
    dH_r = Q.T @ dH @ Q
    eigs_r = np.sort(np.linalg.eigvalsh(dH_r))
    lp = eigs_r[0]
    ls = -alpha_reg / lp if lp < 0 else None
    ls_tag = f"{ls:.5f}" if ls else "---"
    print(f"  {label:8s}: λ₁'(0) = {lp:.8f}  → λ* = {ls_tag}")

# ══════════════════════════════════════════════════════════════
#  Step 3: Full width sweep with exact formula
# ══════════════════════════════════════════════════════════════
print("\n[Step 3] Full width sweep: exact analytical vs FD vs numerical")
widths = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
results = []

for m in widths:
    v = np.linspace(0.5, 1.5, m)
    W0, _ = compute_W0_theory(m, v)
    Q = get_flat_basis(m, v)
    
    # Exact analytical
    dH_e = compute_dH_exact(m, v, W0)
    dH_r_e = Q.T @ dH_e @ Q
    lp_exact = np.sort(np.linalg.eigvalsh(dH_r_e))[0]
    
    # Taylor
    dH_t = compute_dH_taylor(m, v, W0)
    dH_r_t = Q.T @ dH_t @ Q
    lp_taylor = np.sort(np.linalg.eigvalsh(dH_r_t))[0]
    
    # FD ground truth
    dH_f = compute_dH_fd(m, v, W0)
    dH_r_f = Q.T @ dH_f @ Q
    lp_fd = np.sort(np.linalg.eigvalsh(dH_r_f))[0]
    
    # Numerical λ* from continuation
    W_c = W0.ravel().copy()
    n_l = 201
    la = np.linspace(0, 1, n_l)
    eig_track = []
    for lam in la:
        lc = float(lam)
        res = minimize(lambda w, _l=lc: loss_fn(w, _l, m, v),
                       W_c, jac=lambda w, _l=lc: grad_fn(w, _l, m, v),
                       method='L-BFGS-B', options={'maxiter': 2000, 'gtol': 1e-13})
        W_c = res.x.copy()
        n_par = m * d
        if n_par <= 250:
            eigs = np.sort(eigh(hess_fd(W_c, lam, m, v), eigvals_only=True))
            eig_track.append(eigs[0])
        else:
            from scipy.sparse.linalg import LinearOperator, eigsh
            def hv(vv, _w=W_c, _l=lc):
                eps=1e-5
                return (grad_fn(_w+eps*vv,_l,m,v)-grad_fn(_w-eps*vv,_l,m,v))/(2*eps)
            try:
                eig_track.append(eigsh(LinearOperator((n_par,n_par),matvec=hv),k=1,which='SA',tol=1e-6)[0][0])
            except:
                eig_track.append(np.nan)
    eig_track = np.array(eig_track)
    
    ls_num = None
    for i in range(1, len(la)):
        if eig_track[i-1] > 0 and eig_track[i] <= 0:
            frac = eig_track[i-1]/(eig_track[i-1]-eig_track[i])
            ls_num = la[i-1] + frac*(la[i]-la[i-1])
            break
    
    ls_exact = -alpha_reg/lp_exact if lp_exact < 0 else None
    ls_taylor = -alpha_reg/lp_taylor if lp_taylor < 0 else None
    
    err_e = abs(lp_exact - lp_fd)/abs(lp_fd) if abs(lp_fd) > 1e-15 else 0
    err_t = abs(lp_taylor - lp_fd)/abs(lp_fd) if abs(lp_fd) > 1e-15 else 0
    
    results.append({
        'm': m, 'lp_exact': lp_exact, 'lp_taylor': lp_taylor, 'lp_fd': lp_fd,
        'ls_exact': ls_exact, 'ls_taylor': ls_taylor, 'ls_num': ls_num,
        'err_exact': err_e, 'err_taylor': err_t, 'eig_track': eig_track, 'lam_arr': la,
    })
    
    def fmt(x): return f"{x:.4f}" if x and x < 10 else "---"
    print(f"  m={m:3d}  λ₁'(0): exact={lp_exact:.7f} taylor={lp_taylor:.7f} fd={lp_fd:.7f}"
          f"  err: {err_e:.3f}/{err_t:.3f}"
          f"  λ*: ex={fmt(ls_exact):>7s} ty={fmt(ls_taylor):>7s} num={fmt(ls_num):>7s}")

# ── Fit K from exact values ──────────────────────────────────
ms = np.array([r['m'] for r in results], dtype=float)
lps_exact = np.array([r['lp_exact'] for r in results])
lps_taylor = np.array([r['lp_taylor'] for r in results])
lps_fd = np.array([r['lp_fd'] for r in results])

A_fit = np.column_stack([1/ms, 1/ms**2])
(K_exact, K2_exact) = np.linalg.lstsq(A_fit, lps_exact, rcond=None)[0]
(K_taylor, K2_taylor) = np.linalg.lstsq(A_fit, lps_taylor, rcond=None)[0]
(K_fd, K2_fd) = np.linalg.lstsq(A_fit, lps_fd, rcond=None)[0]

print(f"\n  Width constant K (from λ₁'(0) = K/m + K₂/m²):")
print(f"    Exact:  K = {K_exact:.6f}   → λ* = {alpha_reg/abs(K_exact):.4f}·m")
print(f"    Taylor: K = {K_taylor:.6f}  → λ* = {alpha_reg/abs(K_taylor):.4f}·m")
print(f"    FD:     K = {K_fd:.6f}   → λ* = {alpha_reg/abs(K_fd):.4f}·m")

# ── Verify λ* predictions ───────────────────────────────────
print(f"\n  λ* comparison (exact analytical vs numerical):")
for r in results:
    if r['ls_num']:
        err = abs(r['ls_exact'] - r['ls_num'])/r['ls_num'] if r['ls_exact'] else float('inf')
        err_t = abs(r['ls_taylor'] - r['ls_num'])/r['ls_num'] if r['ls_taylor'] else float('inf')
        print(f"    m={r['m']:3d}  λ*_num={r['ls_num']:.4f}  λ*_exact={r['ls_exact']:.4f} (err {err:.1%})"
              f"  λ*_taylor={r['ls_taylor']:.4f} (err {err_t:.1%})" if r['ls_taylor'] else
              f"    m={r['m']:3d}  λ*_num={r['ls_num']:.4f}  λ*_exact={r['ls_exact']:.4f} (err {err:.1%})")

# ══════════════════════════════════════════════════════════════
#  Step 4: Quantify Taylor error
# ══════════════════════════════════════════════════════════════
print(f"\n[Step 4] Taylor approximation error analysis")
m = 10; v = np.linspace(0.5, 1.5, m)
W0, _ = compute_W0_theory(m, v)
Z0 = X @ W0.T
max_preact = np.max(np.abs(Z0))
rms_preact = np.sqrt(np.mean(Z0**2))

# Pointwise comparison: -tanh²(z) vs -z²
z_test = Z0.ravel()
exact_vals = -np.tanh(z_test)**2
taylor_vals = -z_test**2
pw_err = np.mean(np.abs(exact_vals - taylor_vals)) / np.mean(np.abs(exact_vals))
max_pw_err = np.max(np.abs(exact_vals - taylor_vals)) / np.max(np.abs(exact_vals))

print(f"  Pre-activations: max|z| = {max_preact:.4f}, rms = {rms_preact:.4f}")
print(f"  σ'(z)-1: mean|exact-taylor|/mean|exact| = {pw_err:.4f}")
print(f"  σ'(z)-1: max|exact-taylor|/max|exact| = {max_pw_err:.4f}")
print(f"  Taylor error ≈ z⁴/3, at z={max_preact:.3f}: {max_preact**4/3:.6f}")

# ══════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════
print("\n[Plotting]...")

fig = plt.figure(figsize=(24, 18))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Task 2B Fix: Exact vs Taylor Analytical dH/dλ', fontsize=15, y=0.99)

# (0,0): λ₁'(0) three versions vs m
ax = fig.add_subplot(gs[0, 0])
ax.plot(ms, lps_exact, 'go-', ms=8, lw=2, label='Exact analytical')
ax.plot(ms, lps_taylor, 'rs--', ms=7, lw=1.5, label='Taylor approx')
ax.plot(ms, lps_fd, 'b^:', ms=7, lw=1.5, label='FD ground truth')
ax.set_xlabel('m'); ax.set_ylabel("λ₁'(0)")
ax.set_title("λ₁'(0): three methods"); ax.legend(fontsize=9); ax.grid(alpha=.25)

# (0,1): Relative error of exact vs taylor
ax = fig.add_subplot(gs[0, 1])
errs_e = [r['err_exact'] for r in results]
errs_t = [r['err_taylor'] for r in results]
ax.semilogy(ms, errs_e, 'go-', ms=8, lw=2, label='Exact vs FD')
ax.semilogy(ms, errs_t, 'rs-', ms=8, lw=2, label='Taylor vs FD')
ax.axhline(0.01, color='k', ls='--', alpha=.3, label='1% threshold')
ax.set_xlabel('m'); ax.set_ylabel('Relative error')
ax.set_title("Relative error in λ₁'(0)"); ax.legend(fontsize=9); ax.grid(alpha=.25)

# (0,2): λ* predictions vs numerical
ax = fig.add_subplot(gs[0, 2])
for r in results:
    if r['ls_num']:
        ax.plot(r['ls_num'], r['ls_exact'], 'go', ms=10, zorder=5)
        ax.plot(r['ls_num'], r['ls_taylor'], 'rs', ms=8, zorder=4)
        ax.annotate(f"m={r['m']}", (r['ls_num'], r['ls_exact']),
                    fontsize=8, xytext=(3, 5), textcoords='offset points')
rng = [0.3, 1.1]
ax.plot(rng, rng, 'k--', lw=1, label='y=x')
ax.set_xlabel('λ* numerical'); ax.set_ylabel('λ* predicted')
ax.set_title('λ*: Exact (○) vs Taylor (□) vs numerical')
ax.legend(fontsize=9); ax.grid(alpha=.25)

# (0,3): Pointwise σ'(z)-1 comparison
ax = fig.add_subplot(gs[0, 3])
z_plot = np.linspace(-0.5, 0.5, 200)
ax.plot(z_plot, -np.tanh(z_plot)**2, 'g-', lw=2.5, label='exact: −tanh²(z)')
ax.plot(z_plot, -z_plot**2, 'r--', lw=2, label='Taylor: −z²')
ax.fill_between(z_plot, -np.tanh(z_plot)**2, -z_plot**2, alpha=0.15, color='red')
ax.axvspan(-max_preact, max_preact, alpha=0.08, color='blue', label=f'pre-act range ±{max_preact:.2f}')
ax.set_xlabel('z'); ax.set_ylabel("σ'(z)−1")
ax.set_title('Taylor approximation quality'); ax.legend(fontsize=8); ax.grid(alpha=.25)

# (1,0): spectral softening curves for all m
ax = fig.add_subplot(gs[1, 0:2])
cmap = plt.cm.viridis
for i, r in enumerate(results):
    c = cmap(i/(len(results)-1))
    ax.plot(r['lam_arr'], r['eig_track'], lw=1.5, color=c, label=f"m={r['m']}")
    if r['ls_num']:
        ax.plot(r['ls_num'], 0, 'o', color=c, ms=7, zorder=5)
    if r['ls_exact'] and r['ls_exact'] < 1.5:
        ax.plot(r['ls_exact'], 0, 's', color=c, ms=5, zorder=4, alpha=0.5)
ax.axhline(0, color='k', ls='--', alpha=.4)
ax.set_xlabel('λ'); ax.set_ylabel('Smallest eigenvalue')
ax.set_title('Spectral softening (○=num, □=exact theory)'); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25)

# (1,2): λ* vs m — all three predictions
ax = fig.add_subplot(gs[1, 2])
ms_num = [r['m'] for r in results if r['ls_num']]
ls_num = [r['ls_num'] for r in results if r['ls_num']]
ms_exa = [r['m'] for r in results if r['ls_exact'] and r['ls_exact'] < 5]
ls_exa = [r['ls_exact'] for r in results if r['ls_exact'] and r['ls_exact'] < 5]
ms_tay = [r['m'] for r in results if r['ls_taylor'] and r['ls_taylor'] < 5]
ls_tay = [r['ls_taylor'] for r in results if r['ls_taylor'] and r['ls_taylor'] < 5]
ax.plot(ms_num, ls_num, 'bo-', ms=9, lw=2, label='numerical', zorder=5)
ax.plot(ms_exa, ls_exa, 'gs--', ms=8, lw=1.5, label='exact analytical', zorder=4)
ax.plot(ms_tay, ls_tay, 'r^:', ms=7, lw=1.5, label='Taylor', zorder=3)
m_fit = np.linspace(3, 25, 100)
ax.plot(m_fit, alpha_reg*m_fit/abs(K_exact), 'g:', lw=1, alpha=.5)
ax.axhline(1, color='k', ls=':', alpha=.3)
ax.set_xlabel('m'); ax.set_ylabel('λ*')
ax.set_title(f'λ* vs m (exact: {alpha_reg/abs(K_exact):.4f}·m)'); ax.legend(fontsize=9); ax.grid(alpha=.25)

# (1,3): K·m product constancy
ax = fig.add_subplot(gs[1, 3])
ax.plot(ms, lps_exact*ms, 'go-', ms=8, lw=2, label='exact')
ax.plot(ms, lps_fd*ms, 'b^:', ms=7, lw=1.5, label='FD')
ax.axhline(K_exact, color='green', ls='--', lw=1.5, label=f'K_exact={K_exact:.5f}')
ax.axhline(K_fd, color='blue', ls='--', lw=1.5, alpha=.5, label=f'K_fd={K_fd:.5f}')
ax.set_xlabel('m'); ax.set_ylabel("λ₁'(0)·m")
ax.set_title('Constancy of K = λ₁\'(0)·m'); ax.legend(fontsize=8); ax.grid(alpha=.25)

# (2,0-1): Error convergence with m
ax = fig.add_subplot(gs[2, 0:2])
ax.plot(ms, errs_e, 'go-', ms=9, lw=2.5, label='|exact − FD| / |FD|')
ax.plot(ms, errs_t, 'rs-', ms=9, lw=2.5, label='|Taylor − FD| / |FD|')
ax.axhline(0.01, color='k', ls='--', alpha=.3)
ax.set_xlabel('m', fontsize=12); ax.set_ylabel('Relative error', fontsize=12)
ax.set_title('Convergence: exact formula eliminates 17% Taylor error', fontsize=13)
ax.legend(fontsize=11); ax.grid(alpha=.25)
ax.set_ylim(bottom=0)

# (2,2-3): Summary text
ax = fig.add_subplot(gs[2, 2:4])
ax.axis('off')

# Compute mean errors
mean_err_e = np.mean(errs_e)
mean_err_t = np.mean(errs_t)
improvement = mean_err_t / max(mean_err_e, 1e-15)

summary = (
    f"TASK 2B FIX SUMMARY\n"
    f"{'='*50}\n\n"
    f"OLD (Taylor: σ'(z)-1 ≈ -z²):\n"
    f"  Mean error in λ₁'(0): {mean_err_t:.1%}\n"
    f"  K_taylor = {K_taylor:.6f}\n"
    f"  → λ* = {alpha_reg/abs(K_taylor):.4f}·m\n\n"
    f"NEW (Exact: σ'(z)-1 = -tanh²(z)):\n"
    f"  Mean error in λ₁'(0): {mean_err_e:.1%}\n"
    f"  K_exact = {K_exact:.6f}\n"
    f"  → λ* = {alpha_reg/abs(K_exact):.4f}·m\n\n"
    f"FD REFERENCE:\n"
    f"  K_fd = {K_fd:.6f}\n"
    f"  → λ* = {alpha_reg/abs(K_fd):.4f}·m\n\n"
    f"IMPROVEMENT: {improvement:.0f}× reduction in error\n\n"
    f"λ* PREDICTIONS (exact vs numerical):\n"
)
for r in results:
    if r['ls_num'] and r['ls_exact']:
        err = abs(r['ls_exact']-r['ls_num'])/r['ls_num']
        summary += f"  m={r['m']:3d}: λ*_exact={r['ls_exact']:.3f} vs λ*_num={r['ls_num']:.3f} ({err:.1%})\n"

ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('/home/claude/fig_2b_fix.png', dpi=160, bbox_inches='tight')
plt.close()
print("  → fig_2b_fix.png")

print(f"\n{'='*70}")
print(f"  DONE ({time.time()-T0:.0f}s)")
print(f"{'='*70}")
