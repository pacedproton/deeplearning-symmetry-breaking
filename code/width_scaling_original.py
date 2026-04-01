#!/usr/bin/env python3
"""
Phase 4.3: Width-Scaling Experiment
====================================
Track λ*(m) as hidden-layer width m varies.
Also compute the theoretical prediction for λ* from the eigenvalue-derivative formula.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, warnings
warnings.filterwarnings('ignore')

def h(z, lam):  return (1-lam)*z + lam*np.tanh(z)
def hp(z, lam):
    s = np.tanh(z)
    return (1-lam) + lam*(1-s*s)

np.random.seed(77)
d, Ns = 5, 500
reg = 4e-3

X = np.random.randn(Ns, d)
wl  = np.array([1., .5, -.3, .2, -.1])
wnl = np.array([.3, -.7, .5, -.2, .4])
y = X@wl + 0.4*np.tanh(1.5*X@wnl) + 0.05*np.random.randn(Ns)

# data moments needed for theoretical prediction
Sigma = (X.T @ X) / Ns  # empirical covariance

print("="*70)
print("  WIDTH-SCALING EXPERIMENT: λ*(m) for m = 5..100")
print("="*70)
T0 = time.time()

widths = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
results = {}

for m in widths:
    t0 = time.time()
    v = np.linspace(0.5, 1.5, m)
    n_par = m * d

    def loss(wf, lam):
        W = wf.reshape(m, d); Z = X @ W.T
        return 0.5*np.mean((h(Z, lam) @ v - y)**2) + 0.5*reg*np.sum(wf**2)

    def grad(wf, lam):
        W = wf.reshape(m, d); Z = X @ W.T
        r = h(Z, lam) @ v - y
        return ((1/Ns)*((r[:,None]*hp(Z,lam)*v[None,:]).T @ X) + reg*W).ravel()

    def hess_fd(wf, lam, eps=1e-5):
        g0 = grad(wf, lam); n = len(wf)
        H = np.empty((n, n))
        for i in range(n):
            wp = wf.copy(); wp[i] += eps
            H[i] = (grad(wp, lam) - g0) / eps
        return 0.5*(H + H.T)

    # solve at λ=0
    W0 = np.random.randn(n_par) * 0.01
    def obj0(w): return loss(w, 0.)
    def jac0(w): return grad(w, 0.)
    res0 = minimize(obj0, W0, jac=jac0, method='L-BFGS-B',
                    options={'maxiter': 8000, 'gtol': 1e-14})
    W_cur = res0.x.copy()

    # --- Theoretical prediction for λ* ---
    # At λ=0: H = diag(v_j^2) ⊗ Σ + reg·I  (Gauss-Newton approx at minimum)
    # The eigenvalue perturbation dλ₁/dλ at λ=0 is computed from the
    # Hessian derivative.
    #
    # For tanh: σ''(0)=0, σ'''(0)=-2
    # dH/dλ|_{λ=0} has diagonal blocks: -2·v_j² · E[(w_j^T x)² xx^T]
    # (from the Gauss-Newton component; residual term is small at minimum)
    #
    # Eigenvalue at λ=0: λ₁(0) ≈ min_j(v_j²) · λ_min(Σ) + reg
    # Eigenvalue derivative: λ₁'(0) ≈ -2 · Σ_j v_j² · u₀^T E[(w_j^T x)² xx^T] u₀

    W0_mat = W_cur.reshape(m, d)
    H0 = hess_fd(W_cur, 0.)
    evals0, evecs0 = eigh(H0)
    lam1_0 = evals0[0]
    u0 = evecs0[:, 0]

    # compute dH/dλ at λ=0 by finite difference in λ
    dlam = 1e-4
    H_eps = hess_fd(W_cur, dlam)
    dH_dlam = (H_eps - H0) / dlam
    lam1_prime = u0 @ dH_dlam @ u0  # first-order eigenvalue perturbation

    if lam1_prime < 0:
        lam_star_theory = -lam1_0 / lam1_prime
    else:
        lam_star_theory = None

    # --- Numerical sweep to find actual λ* ---
    n_lam = min(201, max(101, 401 // max(1, m // 5)))
    lam_arr = np.linspace(0, 1, n_lam)
    eig_min_arr = []
    W_track = W_cur.copy()

    for il, lam in enumerate(lam_arr):
        lc = float(lam)
        def obj_l(w, _l=lc): return loss(w, _l)
        def jac_l(w, _l=lc): return grad(w, _l)
        res = minimize(obj_l, W_track, jac=jac_l, method='L-BFGS-B',
                       options={'maxiter': 2000, 'gtol': 1e-13})
        W_track = res.x.copy()

        # for small m, full Hessian; for large m, just Lanczos-style (use 6 smallest)
        if n_par <= 200:
            eigs = np.sort(eigh(hess_fd(W_track, lam), eigvals_only=True))
            eig_min_arr.append(eigs[0])
        else:
            # use finite-diff Hessian-vector product + partial eigendecomp
            from scipy.sparse.linalg import LinearOperator, eigsh
            def hv(v_vec, _w=W_track, _l=lc):
                eps = 1e-5
                gp = grad(_w + eps*v_vec, _l)
                gm = grad(_w - eps*v_vec, _l)
                return (gp - gm) / (2*eps)
            Hop = LinearOperator((n_par, n_par), matvec=hv)
            try:
                eigs_small = eigsh(Hop, k=3, which='SA', tol=1e-6)[0]
                eig_min_arr.append(np.min(eigs_small))
            except:
                eig_min_arr.append(np.nan)

    eig_min_arr = np.array(eig_min_arr)

    # find crossing
    lam_star_num = None
    for i in range(1, len(lam_arr)):
        if eig_min_arr[i-1] > 0 and eig_min_arr[i] <= 0:
            if not np.isnan(eig_min_arr[i-1]) and not np.isnan(eig_min_arr[i]):
                frac = eig_min_arr[i-1] / (eig_min_arr[i-1] - eig_min_arr[i])
                lam_star_num = lam_arr[i-1] + frac*(lam_arr[i] - lam_arr[i-1])
                break

    # also check: ratio of eigenvalues at crossing (simple kernel?)
    kernel_ratio = None
    if lam_star_num is not None and n_par <= 200:
        idx_cross = np.argmin(np.abs(eig_min_arr))
        lc = float(lam_arr[idx_cross])
        def obj_c(w, _l=lc): return loss(w, _l)
        def jac_c(w, _l=lc): return grad(w, _l)
        res_c = minimize(obj_c, W_track, jac=jac_c, method='L-BFGS-B',
                         options={'maxiter': 3000, 'gtol': 1e-14})
        eigs_c = np.sort(eigh(hess_fd(res_c.x, lam_arr[idx_cross]), eigvals_only=True))
        if abs(eigs_c[1]) > 1e-10:
            kernel_ratio = abs(eigs_c[0] / eigs_c[1])

    dt = time.time() - t0
    results[m] = {
        'lam_star_num': lam_star_num,
        'lam_star_theory': lam_star_theory,
        'lam1_0': lam1_0,
        'lam1_prime': lam1_prime,
        'kernel_ratio': kernel_ratio,
        'eig_min': eig_min_arr,
        'lam_arr': lam_arr,
        'n_par': n_par,
        'time': dt,
    }
    tag_n = f"{lam_star_num:.4f}" if lam_star_num else "none"
    tag_t = f"{lam_star_theory:.4f}" if lam_star_theory else "none"
    kr_tag = f"{kernel_ratio:.4f}" if kernel_ratio else "---"
    print(f"  m={m:3d}  n_par={n_par:4d}  λ₁(0)={lam1_0:.5f}  λ₁'(0)={lam1_prime:.5f}"
          f"  λ*_num={tag_n}  λ*_thy={tag_t}  |e₁/e₂|={kr_tag}  ({dt:.1f}s)")

print(f"\n  Total time: {time.time()-T0:.0f}s")

# ══════════════════════════════════════════════════════════════
#  ANALYSIS: scaling of λ*(m)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SCALING ANALYSIS")
print("="*70)

ms_valid = []
ls_num = []
ls_thy = []
l1_0s = []
l1_ps = []
for m in widths:
    r = results[m]
    if r['lam_star_num'] is not None:
        ms_valid.append(m)
        ls_num.append(r['lam_star_num'])
        l1_0s.append(r['lam1_0'])
        l1_ps.append(r['lam1_prime'])
    if r['lam_star_theory'] is not None:
        ls_thy.append(r['lam_star_theory'])
    else:
        ls_thy.append(np.nan)

ms_valid = np.array(ms_valid, dtype=float)
ls_num = np.array(ls_num)
l1_0s = np.array(l1_0s)
l1_ps = np.array(l1_ps)

# fit λ*(m) ~ a + b/m or λ*(m) ~ a + b*m^(-p)
if len(ms_valid) >= 4:
    # try λ* = a + b/m
    A = np.column_stack([np.ones(len(ms_valid)), 1.0/ms_valid])
    (a_fit, b_fit), res_fit, _, _ = np.linalg.lstsq(A, ls_num, rcond=None)
    print(f"  Fit λ*(m) = {a_fit:.4f} + {b_fit:.2f}/m")
    print(f"  → λ*(∞) ≈ {a_fit:.4f}")

    # try log-log for power law: λ*(m) - λ*(∞) ~ m^(-p)
    if a_fit < min(ls_num):
        resid = ls_num - a_fit
        logm = np.log(ms_valid)
        logr = np.log(np.abs(resid))
        p_fit = np.polyfit(logm, logr, 1)[0]
        print(f"  Power law: λ*(m) - λ*(∞) ~ m^{p_fit:.2f}")

# scaling of λ₁(0) and λ₁'(0) separately
if len(ms_valid) >= 3:
    p1 = np.polyfit(np.log(ms_valid), np.log(l1_0s), 1)[0]
    p2 = np.polyfit(np.log(ms_valid), np.log(np.abs(l1_ps)), 1)[0]
    print(f"  λ₁(0) ~ m^{p1:.2f}")
    print(f"  |λ₁'(0)| ~ m^{p2:.2f}")
    print(f"  → λ* = λ₁(0)/|λ₁'(0)| ~ m^{p1-p2:.2f}")

# ══════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════
print("\n[Plotting]...")

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
fig.suptitle('Width-Scaling Analysis: λ*(m) for Activation Homotopy Bifurcation', fontsize=15, y=0.98)

# Panel (0,0): eigenvalue curves for all m
ax = axes[0,0]
cmap = plt.cm.viridis
for i, m in enumerate(widths):
    r = results[m]
    c = cmap(i / (len(widths)-1))
    ax.plot(r['lam_arr'], r['eig_min'], lw=1.5, color=c, label=f'm={m}')
    if r['lam_star_num'] is not None:
        ax.plot(r['lam_star_num'], 0, 'o', color=c, ms=7, zorder=5)
ax.axhline(0, color='k', ls='--', alpha=.4)
ax.set_xlabel('λ', fontsize=12); ax.set_ylabel('Smallest eigenvalue', fontsize=12)
ax.set_title('Spectral softening curves', fontsize=13)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25)

# Panel (0,1): λ*(m) — numerical and theoretical
ax = axes[0,1]
all_ms = np.array(widths, dtype=float)
all_num = np.array([results[m]['lam_star_num'] if results[m]['lam_star_num'] else np.nan for m in widths])
all_thy = np.array([results[m]['lam_star_theory'] if results[m]['lam_star_theory'] else np.nan for m in widths])
mask_n = ~np.isnan(all_num)
mask_t = ~np.isnan(all_thy)
ax.plot(all_ms[mask_n], all_num[mask_n], 'bo-', ms=9, lw=2, label='λ* numerical', zorder=5)
ax.plot(all_ms[mask_t], all_thy[mask_t], 'rs--', ms=8, lw=1.5, label='λ* theory (1st order)', zorder=4)
if len(ms_valid) >= 4:
    m_plot = np.linspace(min(widths), max(widths), 200)
    ax.plot(m_plot, a_fit + b_fit/m_plot, 'g:', lw=2,
            label=f'fit: {a_fit:.3f} + {b_fit:.1f}/m')
    ax.axhline(a_fit, color='green', ls='--', alpha=.4)
ax.set_xlabel('m (hidden units)', fontsize=12); ax.set_ylabel('λ*', fontsize=12)
ax.set_title('Bifurcation point vs width', fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=.25)

# Panel (0,2): λ₁(0) and |λ₁'(0)| vs m (log-log)
ax = axes[0,2]
all_l10 = np.array([results[m]['lam1_0'] for m in widths])
all_l1p = np.array([abs(results[m]['lam1_prime']) for m in widths])
ax.loglog(all_ms, all_l10, 'bo-', ms=8, lw=2, label=f'λ₁(0) ~ m^{{{p1:.2f}}}')
ax.loglog(all_ms, all_l1p, 'rs-', ms=8, lw=2, label=f"|λ₁'(0)| ~ m^{{{p2:.2f}}}")
ax.set_xlabel('m', fontsize=12); ax.set_ylabel('Value', fontsize=12)
ax.set_title('Eigenvalue and derivative scaling', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=.25, which='both')

# Panel (1,0): λ* vs 1/m
ax = axes[1,0]
ax.plot(1.0/all_ms[mask_n], all_num[mask_n], 'bo-', ms=9, lw=2, label='numerical')
if len(ms_valid) >= 4:
    inv_m = np.linspace(0, 1.0/min(widths), 100)
    ax.plot(inv_m, a_fit + b_fit*inv_m, 'g--', lw=2, label=f'linear fit')
ax.set_xlabel('1/m', fontsize=12); ax.set_ylabel('λ*', fontsize=12)
ax.set_title('λ* vs 1/m (test for λ*→const as m→∞)', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=.25)

# Panel (1,1): kernel ratio at λ*
ax = axes[1,1]
krs = [(m, results[m]['kernel_ratio']) for m in widths if results[m]['kernel_ratio'] is not None]
if krs:
    kr_m, kr_v = zip(*krs)
    ax.semilogy(kr_m, kr_v, 'go-', ms=9, lw=2)
    ax.axhline(0.1, color='r', ls='--', alpha=.4, label='threshold 0.1')
    ax.set_xlabel('m', fontsize=12); ax.set_ylabel('|eig₁/eig₂| at λ*', fontsize=12)
    ax.set_title('Simple-kernel test vs width', fontsize=13)
    ax.legend(fontsize=10); ax.grid(alpha=.25)

# Panel (1,2): theory vs numerical scatter
ax = axes[1,2]
mask_both = mask_n & mask_t
if mask_both.any():
    ax.plot(all_thy[mask_both], all_num[mask_both], 'ko', ms=10, zorder=5)
    lo = min(all_thy[mask_both].min(), all_num[mask_both].min()) * 0.9
    hi = max(all_thy[mask_both].max(), all_num[mask_both].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y=x')
    # label each point
    for mm, tn, tt in zip(all_ms[mask_both], all_num[mask_both], all_thy[mask_both]):
        ax.annotate(f'm={int(mm)}', xy=(tt, tn), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('λ* (1st-order theory)', fontsize=12)
ax.set_ylabel('λ* (numerical)', fontsize=12)
ax.set_title('Theory vs numerical prediction', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=.25)
ax.set_aspect('equal')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/claude/fig_width_scaling.png', dpi=160, bbox_inches='tight')
plt.close()
print("  → fig_width_scaling.png")

# ── Final summary ────────────────────────────────────────────
print(f"\n{'='*70}")
print("  WIDTH-SCALING RESULTS")
print(f"{'='*70}")
hdr = "  {:>5s}  {:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
    "m", "n_par", "lam1(0)", "lam1'(0)", "lam*_num", "lam*_thy", "|e1/e2|")
print(hdr)
print("  " + "-"*75)
for m in widths:
    r = results[m]
    n = f"{r['lam_star_num']:.5f}" if r['lam_star_num'] else "---"
    t = f"{r['lam_star_theory']:.5f}" if r['lam_star_theory'] else "---"
    k = f"{r['kernel_ratio']:.5f}" if r['kernel_ratio'] else "---"
    print(f"  {m:5d}  {r['n_par']:6d}  {r['lam1_0']:10.6f}  {r['lam1_prime']:10.6f}  {n:>10s}  {t:>10s}  {k:>10s}")

if len(ms_valid) >= 4:
    print(f"\n  Scaling fit: λ*(m) = {a_fit:.4f} + {b_fit:.2f}/m")
    print(f"  → λ*(∞) ≈ {a_fit:.4f} (finite nonlinearity threshold)")
    print(f"  → λ₁(0) ~ m^{p1:.2f},  |λ₁'(0)| ~ m^{p2:.2f}")
    print(f"  → λ* ~ m^{p1-p2:.2f}")

print(f"\n  Total time: {time.time()-T0:.0f}s")
