#!/usr/bin/env python3
"""
Diagnose jumps in spectral softening curves.
For each width, track: loss, gradient norm, 5 smallest eigenvalues,
and detect discontinuities.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
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

print("="*65)
print("  DIAGNOSING SPECTRAL SOFTENING JUMPS")
print("="*65)

test_widths = [10, 20, 50, 100]
n_lam = 201
la = np.linspace(0, 1, n_lam)

fig = plt.figure(figsize=(24, 5*len(test_widths)))
gs = GridSpec(len(test_widths), 4, figure=fig, hspace=0.4, wspace=0.35)

for wi, m in enumerate(test_widths):
    v = np.linspace(0.5, 1.5, m)
    n_par = m * d
    W0 = compute_W0(m, v)
    W_cur = W0.ravel().copy()
    
    losses, gnorms, eig_sets, w_dists = [], [], [], []
    W_prev = W_cur.copy()
    
    print(f"\n  m={m} (n_par={n_par}):")
    
    for i, lam in enumerate(la):
        lc = float(lam)
        res = minimize(lambda w, _l=lc: loss_fn(w, _l, m, v),
                       W_cur, jac=lambda w, _l=lc: grad_fn(w, _l, m, v),
                       method='L-BFGS-B', options={'maxiter': 3000, 'gtol': 1e-13})
        W_cur = res.x.copy()
        
        losses.append(res.fun)
        gnorms.append(np.linalg.norm(res.jac))
        w_dists.append(np.linalg.norm(W_cur - W_prev))
        W_prev = W_cur.copy()
        
        if n_par <= 500:
            eigs = np.sort(eigh(hess_fd(W_cur, lam, m, v), eigvals_only=True))
            eig_sets.append(eigs[:6])
        else:
            from scipy.sparse.linalg import LinearOperator, eigsh
            def hv(vv, _w=W_cur, _l=lc):
                eps=1e-5
                return (grad_fn(_w+eps*vv,_l,m,v)-grad_fn(_w-eps*vv,_l,m,v))/(2*eps)
            try:
                evals = eigsh(LinearOperator((n_par,n_par),matvec=hv),k=6,which='SA',tol=1e-6)[0]
                eig_sets.append(np.sort(evals))
            except:
                eig_sets.append(np.full(6, np.nan))
    
    losses = np.array(losses)
    gnorms = np.array(gnorms)
    eig_sets = np.array(eig_sets)
    w_dists = np.array(w_dists)
    
    # Detect jumps: large changes in eigenvalue between consecutive λ steps
    e1 = eig_sets[:, 0]
    de1 = np.abs(np.diff(e1))
    median_de = np.median(de1[de1 > 0])
    jump_idx = np.where(de1 > 10 * median_de)[0]
    
    print(f"    Loss range: [{losses.min():.6f}, {losses.max():.6f}]")
    print(f"    Grad norm range: [{gnorms.min():.1e}, {gnorms.max():.1e}]")
    print(f"    eig₁ range: [{e1.min():.6f}, {e1.max():.6f}]")
    print(f"    Median |Δeig₁|: {median_de:.6f}")
    print(f"    Jump indices (>10× median): {jump_idx.tolist()}")
    for ji in jump_idx:
        print(f"      λ={la[ji]:.3f}→{la[ji+1]:.3f}: eig₁ {e1[ji]:.5f}→{e1[ji+1]:.5f}"
              f"  Δeig={de1[ji]:.5f}  ΔW={w_dists[ji+1]:.5f}"
              f"  loss {losses[ji]:.6f}→{losses[ji+1]:.6f}"
              f"  ‖∇‖={gnorms[ji+1]:.1e}")
    
    # Panel 1: 5 smallest eigenvalues
    ax = fig.add_subplot(gs[wi, 0])
    colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b']
    for k in range(min(6, eig_sets.shape[1])):
        ax.plot(la, eig_sets[:, k], lw=1.5 if k < 2 else 0.8, color=colors[k],
                label=f'eig {k+1}')
    ax.axhline(0, color='k', ls='--', alpha=.3)
    for ji in jump_idx:
        ax.axvline(la[ji], color='red', ls=':', alpha=.5, lw=1)
    ax.set_xlabel('λ'); ax.set_ylabel('Eigenvalue')
    ax.set_title(f'm={m}: 6 smallest eigenvalues')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
    
    # Panel 2: loss along branch
    ax = fig.add_subplot(gs[wi, 1])
    ax.plot(la, losses, 'b-', lw=1.5)
    for ji in jump_idx:
        ax.axvline(la[ji], color='red', ls=':', alpha=.5, lw=1)
    ax.set_xlabel('λ'); ax.set_ylabel('Loss')
    ax.set_title(f'm={m}: loss along branch'); ax.grid(alpha=.2)
    
    # Panel 3: gradient norm (convergence quality)
    ax = fig.add_subplot(gs[wi, 2])
    ax.semilogy(la, gnorms, 'g-', lw=1)
    for ji in jump_idx:
        ax.axvline(la[ji], color='red', ls=':', alpha=.5, lw=1)
    ax.set_xlabel('λ'); ax.set_ylabel('‖∇L‖')
    ax.set_title(f'm={m}: gradient norm'); ax.grid(alpha=.2)
    
    # Panel 4: step size ‖W(λ) - W(λ-Δλ)‖
    ax = fig.add_subplot(gs[wi, 3])
    ax.semilogy(la[1:], w_dists[1:], 'm-', lw=1)
    for ji in jump_idx:
        ax.axvline(la[ji], color='red', ls=':', alpha=.5, lw=1)
    ax.set_xlabel('λ'); ax.set_ylabel('‖ΔW‖')
    ax.set_title(f'm={m}: continuation step size'); ax.grid(alpha=.2)

fig.suptitle('Spectral softening diagnostics: eigenvalues, loss, convergence, step size',
             fontsize=15, y=1.0)
plt.savefig('/home/claude/fig_jump_diagnosis.png', dpi=140, bbox_inches='tight')
plt.close()
print("\n  → fig_jump_diagnosis.png")
