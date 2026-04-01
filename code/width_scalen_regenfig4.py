#!/usr/bin/env python3
"""
Regenerate fig_width_scaling.png with correct exact-analytical λ* predictions.
Uses the exact formula σ'(z)-1 = -tanh²(z) from fix_2b.py.
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
def hp(z, lam): s = np.tanh(z); return (1-lam) + lam*(1-s*s)

np.random.seed(77)
d, Ns = 5, 500
alpha_reg = 4e-3
X = np.random.randn(Ns, d)
wl  = np.array([1., .5, -.3, .2, -.1])
wnl = np.array([.3, -.7, .5, -.2, .4])
y = X@wl + 0.4*np.tanh(1.5*X@wnl) + 0.05*np.random.randn(Ns)
Sigma = (X.T @ X) / Ns
gamma = (X.T @ y) / Ns

def loss_fn(wf, lam, m, v):
    W = wf.reshape(m, d); Z = X @ W.T
    return 0.5*np.mean((h(Z,lam)@v - y)**2) + 0.5*alpha_reg*np.sum(wf**2)

def grad_fn(wf, lam, m, v):
    W = wf.reshape(m, d); Z = X @ W.T
    r = h(Z,lam)@v - y
    return ((1/Ns)*((r[:,None]*hp(Z,lam)*v[None,:]).T @ X) + alpha_reg*W).ravel()

def hess_fd(wf, lam, m, v, eps=1e-5):
    g0 = grad_fn(wf, lam, m, v); n = len(wf)
    H = np.empty((n,n))
    for i in range(n):
        wp = wf.copy(); wp[i] += eps
        H[i] = (grad_fn(wp, lam, m, v) - g0)/eps
    return 0.5*(H+H.T)

def compute_W0(m, v):
    vsq = np.sum(v**2)
    M = alpha_reg*np.eye(d) + vsq*Sigma
    u_eff = vsq*np.linalg.solve(M, gamma)
    return np.outer(v, (gamma - Sigma@u_eff)) / alpha_reg

def compute_lp_exact(m, v, W0):
    """Exact analytical λ₁'(0) using -tanh²(z) instead of -z²"""
    Z = X @ W0.T
    r = Z @ v - y
    n_par = m*d
    dH = np.zeros((n_par, n_par))
    for j in range(m):
        for k in range(m):
            sp_j = -np.tanh(Z[:,j])**2
            sp_k = -np.tanh(Z[:,k])**2
            Mj = (1/Ns)*(X * sp_j[:,None]).T @ X
            Mk = (1/Ns)*(X * sp_k[:,None]).T @ X
            block = v[j]*v[k]*(Mj+Mk)
            if j == k:
                s = np.tanh(Z[:,j])
                spp = -2*s*(1-s*s)
                Mres = (1/Ns)*(X*(r*spp)[:,None]).T @ X
                block += v[j]*Mres
            dH[j*d:(j+1)*d, k*d:(k+1)*d] = block
    dH = 0.5*(dH+dH.T)
    C = np.kron(v.reshape(1,-1), np.eye(d))
    _, S_c, Vt_c = np.linalg.svd(C, full_matrices=True)
    Q = Vt_c[np.sum(S_c > 1e-12):].T
    dH_r = Q.T @ dH @ Q
    return np.sort(np.linalg.eigvalsh(dH_r))[0]

print("="*60)
print("  REGENERATING fig_width_scaling.png (exact analytical)")
print("="*60)
T0 = time.time()

widths = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
results = {}

for m in widths:
    v = np.linspace(0.5, 1.5, m)
    n_par = m*d
    W0 = compute_W0(m, v)
    
    lp_exact = compute_lp_exact(m, v, W0)
    ls_exact = -alpha_reg/lp_exact if lp_exact < 0 else None
    
    W_c = W0.ravel().copy()
    n_l = 201
    la = np.linspace(0, 1, n_l)
    eig_track = []
    for lam in la:
        lc = float(lam)
        res = minimize(lambda w, _l=lc: loss_fn(w, _l, m, v),
                       W_c, jac=lambda w, _l=lc: grad_fn(w, _l, m, v),
                       method='L-BFGS-B', options={'maxiter':2000,'gtol':1e-13})
        W_c = res.x.copy()
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
            ls_num = la[i-1]+frac*(la[i]-la[i-1])
            break
    
    kr = None
    if ls_num is not None and n_par <= 200:
        idx_c = np.argmin(np.abs(eig_track))
        lc = float(la[idx_c])
        res_c = minimize(lambda w, _l=lc: loss_fn(w,_l,m,v),
                         W_c, jac=lambda w, _l=lc: grad_fn(w,_l,m,v),
                         method='L-BFGS-B', options={'maxiter':3000,'gtol':1e-14})
        eigs_c = np.sort(eigh(hess_fd(res_c.x, la[idx_c], m, v), eigvals_only=True))
        if abs(eigs_c[1]) > 1e-10:
            kr = abs(eigs_c[0]/eigs_c[1])
    
    results[m] = {'lp': lp_exact, 'ls_exact': ls_exact, 'ls_num': ls_num,
                  'lam1_0': eig_track[0], 'eig_track': eig_track, 'lam_arr': la,
                  'kr': kr}
    
    def fmt(x): return f"{x:.4f}" if x and x < 10 else "---"
    print(f"  m={m:3d}  λ₁'(0)={lp_exact:.7f}  λ*_exact={fmt(ls_exact):>7s}  λ*_num={fmt(ls_num):>7s}")

ms = np.array(widths, dtype=float)
lps = np.array([results[m]['lp'] for m in widths])
A_fit = np.column_stack([1/ms, 1/ms**2])
(K, K2) = np.linalg.lstsq(A_fit, lps, rcond=None)[0]
print(f"\n  K = {K:.6f},  λ* ≈ {alpha_reg/abs(K):.4f}·m")

# ── FIGURE ───────────────────────────────────────────────────
print("\n[Plotting]...")
fig, axes = plt.subplots(2, 3, figsize=(22, 13))
fig.suptitle('Width-Scaling Analysis: λ*(m) for Activation Homotopy Bifurcation', fontsize=15, y=0.98)

cmap = plt.cm.viridis
ax = axes[0,0]
for i, m in enumerate(widths):
    r = results[m]
    c = cmap(i/(len(widths)-1))
    et = r['eig_track']
    la_arr = r['lam_arr']
    
    # Find crossing index (eigenvalue goes from positive to negative)
    cross_i = None
    for j in range(1, len(et)):
        if et[j-1] > 0 and et[j] <= 0:
            cross_i = j
            break
    
    if cross_i is not None:
        # Solid line up to and slightly past crossing
        ax.plot(la_arr[:cross_i+2], et[:cross_i+2], lw=2, color=c, label=f'm={m}')
        # Faded dashed line after crossing
        ax.plot(la_arr[cross_i+1:], et[cross_i+1:], lw=0.7, color=c, alpha=0.25, ls='--')
        # Crossing marker
        ax.plot(r['ls_num'], 0, 'o', color=c, ms=7, zorder=5)
    else:
        # No crossing — full solid line
        ax.plot(la_arr, et, lw=1.5, color=c, label=f'm={m}')
    
    if r['ls_exact'] and r['ls_exact'] < 1.5:
        ax.plot(r['ls_exact'], 0, 's', color=c, ms=5, mew=1.5, mfc='none', zorder=4)

ax.axhline(0, color='k', ls='--', alpha=.4)
ax.set_xlabel('λ', fontsize=12); ax.set_ylabel('Smallest eigenvalue', fontsize=12)
ax.set_title('Spectral softening (solid = pre-bifurcation)', fontsize=13)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25)

ax = axes[0,1]
all_num = [(m, results[m]['ls_num']) for m in widths if results[m]['ls_num']]
all_exa = [(m, results[m]['ls_exact']) for m in widths if results[m]['ls_exact'] and results[m]['ls_exact'] < 5]
if all_num:
    ax.plot([x[0] for x in all_num], [x[1] for x in all_num], 'bo-', ms=9, lw=2, label='λ* numerical', zorder=5)
if all_exa:
    ax.plot([x[0] for x in all_exa], [x[1] for x in all_exa], 'gs--', ms=8, lw=1.5, label='λ* exact analytical', zorder=4)
m_fit = np.linspace(3, max(widths), 200)
ax.plot(m_fit, alpha_reg*m_fit/abs(K), 'g:', lw=1.5, alpha=.6, label=f'fit: {alpha_reg/abs(K):.4f}·m')
ax.axhline(1, color='k', ls=':', alpha=.3, label='λ=1 boundary')
ax.set_xlabel('m (hidden units)', fontsize=12); ax.set_ylabel('λ*', fontsize=12)
ax.set_title('Bifurcation point vs width', fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=.25)

ax = axes[0,2]
ax.loglog(ms, np.abs(lps), 'gs-', ms=9, lw=2.5, label="|λ₁'(0)| (exact analytical)", zorder=5)
p2 = np.polyfit(np.log(ms), np.log(np.abs(lps)), 1)[0]
m_fit_log = np.logspace(np.log10(3), np.log10(100), 50)
fit_line = np.exp(np.polyval(np.polyfit(np.log(ms), np.log(np.abs(lps)), 1), np.log(m_fit_log)))
ax.loglog(m_fit_log, fit_line, 'g:', lw=1.5, alpha=0.6, label=f'fit: ~ m$^{{{p2:.2f}}}$')
ax.loglog(m_fit_log, abs(K)/m_fit_log, 'r--', lw=1, alpha=0.4, label=f'K/m = {abs(K):.4f}/m')
ax.set_xlabel('m', fontsize=12); ax.set_ylabel("|λ₁'(0)|", fontsize=12)
ax.set_title(f"Eigenvalue softening rate (slope = {p2:.2f})", fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=.25, which='both')

ax = axes[1,0]
if all_num:
    ms_n = np.array([x[0] for x in all_num], dtype=float)
    ls_n = np.array([x[1] for x in all_num])
    ax.plot(1/ms_n, ls_n, 'bo-', ms=9, lw=2, label='numerical', zorder=5)
if all_exa:
    ms_e = np.array([x[0] for x in all_exa], dtype=float)
    ls_e = np.array([x[1] for x in all_exa])
    ax.plot(1/ms_e, ls_e, 'gs--', ms=8, lw=1.5, label='exact analytical', zorder=4)
    # Linear fit to exact analytical: λ* = a + b·(1/m)
    fit_inv = np.polyfit(1/ms_e, ls_e, 1)
    inv_plot = np.linspace(0, max(1/ms_e)*1.1, 100)
    ax.plot(inv_plot, np.polyval(fit_inv, inv_plot), 'g:', lw=1.5,
            label=f'fit: {fit_inv[1]:.2f} {fit_inv[0]:+.1f}/m')
ax.set_xlabel('1/m', fontsize=12); ax.set_ylabel('λ*', fontsize=12)
ax.set_title('λ* vs 1/m (linearised)', fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=.25)

ax = axes[1,1]
krs = [(m, results[m]['kr']) for m in widths if results[m]['kr'] is not None]
if krs:
    ax.semilogy([x[0] for x in krs], [x[1] for x in krs], 'go-', ms=9, lw=2)
    ax.axhline(0.1, color='r', ls='--', alpha=.4, label='threshold 0.1')
ax.set_xlabel('m', fontsize=12); ax.set_ylabel('|eig₁/eig₂| at λ*', fontsize=12)
ax.set_title('Simple-kernel test vs width', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=.25)

ax = axes[1,2]
both = [(m, results[m]['ls_exact'], results[m]['ls_num'])
        for m in widths if results[m]['ls_exact'] and results[m]['ls_num']]
if both:
    for m_val, le, ln in both:
        ax.plot(le, ln, 'ko', ms=10, zorder=5)
        ax.annotate(f'm={int(m_val)}', (le, ln), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')
    lo = min(min(x[1] for x in both), min(x[2] for x in both)) * 0.9
    hi = max(max(x[1] for x in both), max(x[2] for x in both)) * 1.1
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
else:
    ax.text(0.5, 0.5, 'No overlapping\ntheory + numerical\nλ* values', transform=ax.transAxes,
            ha='center', va='center', fontsize=12, color='gray')
ax.set_xlabel('λ* (exact analytical)', fontsize=12)
ax.set_ylabel('λ* (numerical)', fontsize=12)
ax.set_title('Theory vs numerical prediction', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=.25)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('/home/claude/fig_width_scaling.png', dpi=160, bbox_inches='tight')
plt.close()
print(f"  → fig_width_scaling.png")
print(f"\n  Done ({time.time()-T0:.0f}s)")
