#!/usr/bin/env python3
"""
Bifurcation Validation — Phase 4.1b: Refined Analysis
======================================================
Key insight from 4.1a: with v=[1,1], S₂ symmetry makes the transverse
eigenvalue exactly 0 at λ=0. The diagonal is a saddle from the start.

This script:
  (A) Confirms S₂ zero-eigenvalue mechanism and tracks what happens
  (B) Breaks S₂ with asymmetric v to find genuine interior λ* > 0
  (C) Systematic scan over v-asymmetry parameter
  (D) Higher-dim refined analysis with near-critical reduced potentials
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, warnings
warnings.filterwarnings('ignore')

def h(z, lam):  return (1-lam)*z + lam*np.tanh(z)
def hp(z, lam):
    s = np.tanh(z)
    return (1-lam) + lam*(1-s*s)

np.random.seed(42)
N = 2000
xd = np.random.randn(N)
yd = np.tanh(1.5*xd) + 0.02*np.random.randn(N)

def loss2(w, lam, v):
    p = v[0]*h(w[0]*xd, lam) + v[1]*h(w[1]*xd, lam)
    return 0.5*np.mean((p - yd)**2)

def grad2(w, lam, v):
    z1, z2 = w[0]*xd, w[1]*xd
    r = v[0]*h(z1,lam) + v[1]*h(z2,lam) - yd
    return np.array([np.mean(r*v[0]*hp(z1,lam)*xd),
                     np.mean(r*v[1]*hp(z2,lam)*xd)])

def hess2(w, lam, v, eps=1e-5):
    g0 = grad2(w, lam, v)
    H = np.zeros((2,2))
    for i in range(2):
        wp = w.copy(); wp[i] += eps
        H[i] = (grad2(wp, lam, v) - g0) / eps
    return 0.5*(H + H.T)

print("="*65)
print("  REFINED BIFURCATION ANALYSIS")
print("="*65)
T0 = time.time()

# ═══════════════════════════════════════════════════════════
#  (A) S₂ symmetric case: understand the zero eigenvalue
# ═══════════════════════════════════════════════════════════
print("\n[A] S₂ symmetric case v=[1,1] — anatomy of the zero eigenvalue")
v_sym = np.array([1.0, 1.0])
n_lam = 501
lam_arr = np.linspace(0, 1, n_lam)

# Track diagonal solution
c0 = np.mean(xd*yd)/np.mean(xd**2)
w_d = c0/2
rec_w_sym, rec_eigs_sym = [], []
for lam in lam_arr:
    def fl(ws, _l=float(lam)): return loss2(np.array([ws,ws]), _l, v_sym)
    res = minimize_scalar(fl, bounds=(w_d-1, w_d+1), method='bounded',
                          options={'xatol':1e-14})
    w_d = res.x
    ws = np.array([w_d, w_d])
    eigs = np.sort(eigh(hess2(ws, lam, v_sym), eigvals_only=True))
    rec_w_sym.append(w_d)
    rec_eigs_sym.append(eigs)

rec_eigs_sym = np.array(rec_eigs_sym)
rec_w_sym = np.array(rec_w_sym)

# Also find the off-diagonal minima for several λ values
print("  Finding off-diagonal minima...")
off_diag_data = []
for lam in np.linspace(0, 1, 51):
    lc = float(lam)
    found = {}
    for w0 in [np.array([2.0, -1.0]), np.array([-1.0, 2.0]),
               np.array([1.5, 0.0]), np.array([0.0, 1.5]),
               np.array([1.0, -0.5]), np.array([-0.5, 1.0])]:
        def obj(w, _l=lc): return loss2(w, _l, v_sym)
        def jac(w, _l=lc): return grad2(w, _l, v_sym)
        res = minimize(obj, w0, jac=jac, method='L-BFGS-B',
                       options={'maxiter':2000, 'gtol':1e-11})
        if np.linalg.norm(res.jac) < 1e-6:
            key = (round(res.x[0],2), round(res.x[1],2))
            if key not in found and abs(res.x[0]-res.x[1]) > 0.05:
                found[key] = res.x.copy()
                eigs = eigh(hess2(res.x, lam, v_sym), eigvals_only=True)
                off_diag_data.append((lam, res.x[0], res.x[1], res.fun,
                                      int(np.sum(eigs < -1e-7))))

off_diag = np.array(off_diag_data) if off_diag_data else None
if off_diag is not None:
    n_min = np.sum(off_diag[:,4] == 0)
    print(f"  Found {len(off_diag)} off-diagonal critical points ({n_min} minima)")
else:
    print("  No off-diagonal critical points found")

print(f"  Transverse eigenvalue: λ=0 → {rec_eigs_sym[0,0]:.6e},  λ=1 → {rec_eigs_sym[-1,0]:.6e}")
print(f"  Longitudinal eigenvalue: λ=0 → {rec_eigs_sym[0,1]:.6e},  λ=1 → {rec_eigs_sym[-1,1]:.6e}")

# ═══════════════════════════════════════════════════════════
#  (B) Break S₂: scan v = [1, α] for α ∈ (0, 1)
# ═══════════════════════════════════════════════════════════
print("\n[B] Breaking S₂: scanning v = [1, α]")
alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
lam_fine = np.linspace(0, 1, 401)

alpha_results = {}
for alpha in alphas:
    v = np.array([1.0, alpha])
    # find initial critical point at λ=0: linear problem
    # loss = 0.5 E[(v1*w1*x + v2*w2*x - y)^2]
    # gradient = 0 gives (v1*w1 + v2*w2) = E[xy]/E[x^2]
    c0 = np.mean(xd*yd)/np.mean(xd**2)
    if abs(v[0] + v[1]) > 1e-10:
        # start on "equal contribution" line
        w_init = c0 / (v[0] + v[1])
        w_cur = np.array([w_init, w_init])
    else:
        w_cur = np.array([c0/v[0], 0.0])

    eig_track = []
    w_track = []
    for lam in lam_fine:
        lc = float(lam)
        def obj(w, _l=lc, _v=v): return loss2(w, _l, _v)
        def jac(w, _l=lc, _v=v): return grad2(w, _l, _v)
        res = minimize(obj, w_cur, jac=jac, method='L-BFGS-B',
                       options={'maxiter':2000, 'ftol':1e-15, 'gtol':1e-12})
        w_cur = res.x.copy()
        eigs = np.sort(eigh(hess2(w_cur, lam, v), eigvals_only=True))
        eig_track.append(eigs)
        w_track.append(w_cur.copy())

    eig_track = np.array(eig_track)
    w_track = np.array(w_track)

    # detect crossing
    e_min = eig_track[:, 0]
    crossing = None
    for i in range(1, len(lam_fine)):
        if e_min[i-1] > 0 and e_min[i] <= 0:
            frac = e_min[i-1] / (e_min[i-1] - e_min[i])
            crossing = lam_fine[i-1] + frac*(lam_fine[i]-lam_fine[i-1])
            break

    alpha_results[alpha] = {
        'eigs': eig_track, 'w': w_track, 'crossing': crossing,
        'e_min_0': e_min[0], 'e_min_end': e_min[-1]
    }
    tag = f"λ*={crossing:.4f}" if crossing else "no crossing"
    print(f"  α={alpha:.2f}  eig_min(0)={e_min[0]:.5f}  eig_min(1)={e_min[-1]:.5f}  {tag}")

# ═══════════════════════════════════════════════════════════
#  (C) Detailed analysis at best crossing
# ═══════════════════════════════════════════════════════════
# Find the alpha with the cleanest interior crossing
best_alpha = None
for alpha in alphas:
    cr = alpha_results[alpha]['crossing']
    if cr is not None and 0.05 < cr < 0.95:
        best_alpha = alpha
        break

if best_alpha is None:
    # fall back to alpha with smallest positive eig_min(0)
    best_alpha = max([a for a in alphas if alpha_results[a]['e_min_0'] > 0],
                     key=lambda a: -alpha_results[a]['e_min_0'], default=0.5)

print(f"\n[C] Detailed analysis at α = {best_alpha}")
v_best = np.array([1.0, best_alpha])
ar = alpha_results[best_alpha]
lam_star_best = ar['crossing']
if lam_star_best is None:
    lam_star_best = lam_fine[np.argmin(np.abs(ar['eigs'][:,0]))]
print(f"  λ* = {lam_star_best:.4f}")

# reduced potential at several μ values
idx_star = np.argmin(np.abs(lam_fine - lam_star_best))
w_star = ar['w'][idx_star]

s_range = np.linspace(-2.5, 2.5, 501)
# For asymmetric v, the "transverse" direction is NOT simply (1,-1)
# — use the actual eigenvector
H_star = hess2(w_star, lam_star_best, v_best)
evals_star, evecs_star = eigh(H_star)
v0_dir = evecs_star[:, 0]  # softest direction

phi_slices_best = {}
mus = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
for mu in mus:
    lt = np.clip(lam_star_best + mu, 0, 1)
    idx_l = np.argmin(np.abs(lam_fine - lt))
    wc = ar['w'][idx_l]
    phi = np.array([loss2(wc + s*v0_dir, lt, v_best) for s in s_range])
    phi -= phi[len(s_range)//2]
    phi_slices_best[mu] = phi

# polynomial fit at λ*
phi_star = np.array([loss2(w_star + s*v0_dir, lam_star_best, v_best) for s in s_range])
phi_star -= phi_star[len(s_range)//2]
Amat = np.column_stack([s_range**2, s_range**3, s_range**4])
c2b, c3b, c4b = np.linalg.lstsq(Amat, phi_star, rcond=None)[0]
print(f"  Reduced coefficients: c₂={c2b:.6f}  c₃={c3b:.6f}  c₄={c4b:.6f}")
print(f"  |c₃/c₄| = {abs(c3b)/max(abs(c4b),1e-15):.4f}")

# bifurcation diagram for this alpha
print("  Building bifurcation diagram...")
bif_best = []
n_bif = 61
lam_bif = np.linspace(0, 1, n_bif)
seeds = [np.array([a, b]) for a in np.linspace(-3, 3, 7)
         for b in np.linspace(-3, 3, 7)]
prev = []
for il, lam in enumerate(lam_bif):
    lc = float(lam)
    def obj(w, _l=lc): return loss2(w, _l, v_best)
    def jac(w, _l=lc): return grad2(w, _l, v_best)
    starts = list(prev) + (seeds if il % 5 == 0 else [])
    found = {}
    for w0 in starts:
        res = minimize(obj, w0, jac=jac, method='L-BFGS-B',
                       options={'maxiter':500, 'gtol':1e-9})
        if np.linalg.norm(res.jac) < 1e-5:
            key = (round(res.x[0],2), round(res.x[1],2))
            if key not in found:
                found[key] = res.x.copy()
                eigs = eigh(hess2(res.x, lam, v_best), eigvals_only=True)
                mi = int(np.sum(eigs < -1e-7))
                bif_best.append((lam, res.x[0], res.x[1], res.fun, mi))
    prev = list(found.values())

bif_best = np.array(bif_best)

# ═══════════════════════════════════════════════════════════
#  (D) Higher-dim with sharper analysis
# ═══════════════════════════════════════════════════════════
print("\n[D] Higher-dimensional (d=5, m=10) — sharper analysis")
np.random.seed(77)
d, m, Ns = 5, 10, 500
reg = 4e-3
v_hd = np.linspace(0.5, 1.5, m)
X = np.random.randn(Ns, d)
wl  = np.array([1., .5, -.3, .2, -.1])
wnl = np.array([.3, -.7, .5, -.2, .4])
y_hd = X@wl + 0.4*np.tanh(1.5*X@wnl) + 0.05*np.random.randn(Ns)
n_par = m*d

def loss_hd(wf, lam):
    W = wf.reshape(m,d); Z = X@W.T
    return 0.5*np.mean((h(Z,lam)@v_hd - y_hd)**2) + 0.5*reg*np.sum(wf**2)

def grad_hd(wf, lam):
    W = wf.reshape(m,d); Z = X@W.T
    r = h(Z,lam)@v_hd - y_hd
    return ((1/Ns)*((r[:,None]*hp(Z,lam)*v_hd[None,:]).T @ X) + reg*W).ravel()

def hess_hd(wf, lam, eps=1e-5):
    g0 = grad_hd(wf, lam); n = len(wf)
    H = np.empty((n,n))
    for i in range(n):
        wp = wf.copy(); wp[i] += eps
        H[i] = (grad_hd(wp,lam)-g0)/eps
    return 0.5*(H+H.T)

n_lam3 = 201
lam3 = np.linspace(0, 1, n_lam3)
W_cur = np.random.randn(n_par)*0.01
def obj0(w): return loss_hd(w, 0.)
def jac0(w): return grad_hd(w, 0.)
W_cur = minimize(obj0, W_cur, jac=jac0, method='L-BFGS-B',
                 options={'maxiter':5000,'gtol':1e-13}).x

hd_eigs, hd_loss, hd_W = [], [], []
for i, lam in enumerate(lam3):
    lc = float(lam)
    def obj_l(w, _l=lc): return loss_hd(w, _l)
    def jac_l(w, _l=lc): return grad_hd(w, _l)
    res = minimize(obj_l, W_cur, jac=jac_l, method='L-BFGS-B',
                   options={'maxiter':2000,'gtol':1e-13})
    W_cur = res.x.copy()
    eigs = np.sort(eigh(hess_hd(W_cur, lam), eigvals_only=True))
    hd_eigs.append(eigs); hd_loss.append(res.fun); hd_W.append(W_cur.copy())
    if i % 40 == 0:
        print(f"  λ={lam:.3f}  loss={res.fun:.6f}  eig_min={eigs[0]:.5f}")

hd_eigs = np.array(hd_eigs); hd_loss = np.array(hd_loss)

# find eigenvalue crossings in higher dim
hd_crossings = []
for i in range(1, n_lam3):
    if hd_eigs[i-1,0] > 1e-5 and hd_eigs[i,0] < -1e-5:
        frac = hd_eigs[i-1,0]/(hd_eigs[i-1,0]-hd_eigs[i,0])
        hd_crossings.append(lam3[i-1]+frac*(lam3[i]-lam3[i-1]))
    elif hd_eigs[i-1,0]*hd_eigs[i,0] < 0:
        frac = abs(hd_eigs[i-1,0])/(abs(hd_eigs[i-1,0])+abs(hd_eigs[i,0]))
        hd_crossings.append(lam3[i-1]+frac*(lam3[i]-lam3[i-1]))
print(f"  HD eigenvalue crossings: {hd_crossings}")

# detailed reduced potential at softest point
soft_idx = np.argmin(hd_eigs[:,0])
lam_soft = lam3[soft_idx]
W_soft = hd_W[soft_idx]
evals_hd, evecs_hd = eigh(hess_hd(W_soft, lam_soft))
v0_hd = evecs_hd[:,0]
v1_hd = evecs_hd[:,1]

a_range_hd = np.linspace(-1.5, 1.5, 201)
phi_hd_slices = {}
for mu in [-0.08, -0.04, 0.0, 0.04, 0.08]:
    lt = np.clip(lam_soft+mu, 0, 1)
    phi = np.array([loss_hd(W_soft + a*v0_hd, lt) for a in a_range_hd])
    phi -= phi[len(a_range_hd)//2]
    phi_hd_slices[mu] = phi

phi_hd_star = phi_hd_slices[0.0]
Ahd = np.column_stack([a_range_hd**2, a_range_hd**3, a_range_hd**4])
c2h, c3h, c4h = np.linalg.lstsq(Ahd, phi_hd_star, rcond=None)[0]
print(f"  Softest at λ={lam_soft:.4f}, eig={hd_eigs[soft_idx,0]:.6f}")
print(f"  Coefficients: c₂={c2h:.6f}  c₃={c3h:.6f}  c₄={c4h:.6f}")
print(f"  |c₃/c₄|={abs(c3h)/max(abs(c4h),1e-15):.3f}")

# loss jump detection
dloss = np.diff(hd_loss)
jump_idx = np.argmax(np.abs(dloss))
print(f"  Largest loss change: Δλ ∈ [{lam3[jump_idx]:.3f}, {lam3[jump_idx+1]:.3f}], ΔL={dloss[jump_idx]:.6f}")

print(f"\n  Total time: {time.time()-T0:.0f}s")

# ═══════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════
print("\n[Plotting]...")

# ─── FIGURE 1: S₂ anatomy ───────────────────────────────────
fig = plt.figure(figsize=(24, 18))
gs = GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.38)
fig.suptitle('Bifurcation in Activation Homotopy — Complete Analysis', fontsize=16, y=0.99)

# Row 0: Landscapes at 4 λ values (symmetric case)
for pi, ls in enumerate([0.0, 0.3, 0.7, 1.0]):
    ax = fig.add_subplot(gs[0, pi])
    gr = np.linspace(-3.5, 3.5, 140)
    G1, G2 = np.meshgrid(gr, gr)
    z1 = G1[:,:,None]*xd[None,None,:]
    z2 = G2[:,:,None]*xd[None,None,:]
    pred = h(z1,ls) + h(z2,ls)
    Lgrid = 0.5*np.mean((pred - yd[None,None,:])**2, axis=2)
    Lp = np.log10(Lgrid - Lgrid.min() + 1e-8)
    vmin, vmax = np.percentile(Lp, [2, 80])
    cf = ax.contourf(G1, G2, Lp, levels=np.linspace(vmin, vmax, 28), cmap='inferno')
    ax.contour(G1, G2, Lp, levels=8, colors='w', linewidths=.2, alpha=.4)
    ax.plot(gr, gr, 'c--', lw=.8, alpha=.6)
    ax.set_xlabel('w₁'); ax.set_ylabel('w₂')
    ax.set_title(f'λ = {ls}  (v=[1,1])', fontsize=11)
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, pad=.02, shrink=.85)

# Row 1, left: Eigenvalue tracking for multiple α
ax = fig.add_subplot(gs[1, 0:2])
cmap_a = plt.cm.viridis
for ia, alpha in enumerate([0.0, 0.3, 0.5, 0.7, 0.9, 1.0]):
    col = cmap_a(ia/5)
    ar = alpha_results[alpha]
    ax.plot(lam_fine, ar['eigs'][:,0], lw=1.8, color=col, label=f'α={alpha:.1f}')
    if ar['crossing'] is not None:
        ax.plot(ar['crossing'], 0, 'o', color=col, ms=8, zorder=5)
ax.axhline(0, color='k', ls='--', alpha=.4)
ax.set_xlabel('λ', fontsize=12); ax.set_ylabel('Smallest eigenvalue', fontsize=12)
ax.set_title('Smallest Hessian eigenvalue for v=[1,α]', fontsize=13)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=.25)

# Row 1, right: Bifurcation diagram for best alpha
ax = fig.add_subplot(gs[1, 2:4])
for mi_val, color, label in [(0,'blue','min'), (1,'orange','saddle'), (2,'red','index-2')]:
    mask = bif_best[:,4] == mi_val
    if mask.any():
        ax.scatter(bif_best[mask,0], bif_best[mask,1]-bif_best[mask,2],
                   c=color, s=10, alpha=.7, label=label, zorder=3)
ax.axhline(0, color='k', ls='--', alpha=.4)
if lam_star_best:
    ax.axvline(lam_star_best, color='green', ls=':', lw=1.5, alpha=.6)
ax.set_xlabel('λ', fontsize=12); ax.set_ylabel('w₁ − w₂', fontsize=12)
ax.set_title(f'Bifurcation diagram  v=[1, {best_alpha}]', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=.25)

# Row 2, col 0: Reduced potential for best alpha
ax = fig.add_subplot(gs[2, 0])
cmap_mu = plt.cm.coolwarm
for i, mu in enumerate(mus):
    col = cmap_mu(i/(len(mus)-1))
    ax.plot(s_range, phi_slices_best[mu], lw=1.5, color=col, label=f'μ={mu:+.02f}')
ax.set_xlabel('a (center direction)', fontsize=11)
ax.set_ylabel('φ(a)−φ(0)', fontsize=11)
ax.set_title(f'Reduced potential  v=[1,{best_alpha}]', fontsize=12)
ax.legend(fontsize=8); ax.grid(alpha=.25)
ax.set_xlim(-2, 2)

# Row 2, col 1: Polynomial fit
ax = fig.add_subplot(gs[2, 1])
ax.plot(s_range, phi_star, 'b-', lw=2, label='actual')
fit_b = Amat @ np.array([c2b, c3b, c4b])
ax.plot(s_range, fit_b, 'r--', lw=1.5,
        label=f'{c2b:.4f}a² + {c3b:.4f}a³ + {c4b:.4f}a⁴')
ax.set_xlabel('a'); ax.set_ylabel('φ(a)−φ(0)')
nf = 'pitchfork' if abs(c3b)/max(abs(c4b),1e-15) < 0.5 else 'transcritical'
ax.set_title(f'Fit at λ* ({nf})', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=.25); ax.set_xlim(-2, 2)

# Row 2, col 2: HD eigenvalues
ax = fig.add_subplot(gs[2, 2])
for k in range(min(6, n_par)):
    ax.plot(lam3, hd_eigs[:,k], lw=1.2)
ax.axhline(0, color='k', ls='--', alpha=.4)
for cr in hd_crossings:
    ax.axvline(cr, color='green', ls=':', lw=1.5, alpha=.6)
ax.set_xlabel('λ'); ax.set_ylabel('Eigenvalue')
ax.set_title('d=5, m=10: smallest eigenvalues', fontsize=12); ax.grid(alpha=.25)

# Row 2, col 3: HD reduced potential
ax = fig.add_subplot(gs[2, 3])
for mu, phi in phi_hd_slices.items():
    ax.plot(a_range_hd, phi, lw=1.5, label=f'μ={mu:+.02f}')
ax.set_xlabel('a'); ax.set_ylabel('φ(a)−φ(0)')
ax.set_title(f'HD reduced potential (λ*={lam_soft:.3f})', fontsize=12)
ax.legend(fontsize=8); ax.grid(alpha=.25)

plt.savefig('/home/claude/fig_refined_master.png', dpi=160, bbox_inches='tight')
plt.close()
print("  → fig_refined_master.png")

# ─── FIGURE 2: λ* vs α phase diagram ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Phase Diagram: Bifurcation Point vs Symmetry Breaking', fontsize=14)

ax = axes[0]
alphas_plot = sorted(alpha_results.keys())
lam_stars = [alpha_results[a]['crossing'] for a in alphas_plot]
for a, ls in zip(alphas_plot, lam_stars):
    if ls is not None:
        ax.plot(a, ls, 'bo', ms=10, zorder=5)
    else:
        ax.plot(a, 0, 'rx', ms=10, mew=2, zorder=5)
ax.set_xlabel('α  (v = [1, α])', fontsize=12)
ax.set_ylabel('λ*  (bifurcation point)', fontsize=12)
ax.set_title('Bifurcation parameter vs symmetry breaking', fontsize=13)
ax.grid(alpha=.25)

ax = axes[1]
for a in [0.0, 0.3, 0.7, 1.0]:
    ar = alpha_results[a]
    ax.plot(lam_fine, ar['eigs'][:,0], lw=2, label=f'α={a}')
ax.axhline(0, color='k', ls='--', alpha=.4)
ax.set_xlabel('λ'); ax.set_ylabel('Smallest eigenvalue')
ax.set_title('Spectral softening by α'); ax.legend(); ax.grid(alpha=.25)

# HD loss and spectrum
ax = axes[2]
ax.plot(lam3, hd_loss, 'b-', lw=2, label='loss')
ax2 = ax.twinx()
ax2.plot(lam3, hd_eigs[:,0], 'r-', lw=2, label='eig_min')
ax2.axhline(0, color='r', ls='--', alpha=.3)
ax.set_xlabel('λ'); ax.set_ylabel('Loss', color='b'); ax2.set_ylabel('Min eigenvalue', color='r')
ax.set_title('d=5, m=10: Loss and spectral softening')
ax.grid(alpha=.25)

plt.tight_layout(rect=[0,0,1,.94])
plt.savefig('/home/claude/fig_phase_diagram.png', dpi=160, bbox_inches='tight')
plt.close()
print("  → fig_phase_diagram.png")

# ─── FIGURE 3: Contour evolution for best alpha ─────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
fig.suptitle(f'Loss landscape evolution  v=[1, {best_alpha}]', fontsize=14)
if lam_star_best:
    deltas = [-0.2, 0, 0.2, 0.5]
    tags = ['before', 'at λ*', 'after', 'well after']
else:
    deltas = [0, 0.25, 0.5, 0.75]
    tags = ['start', 'early', 'mid', 'late']

for pi, (dl, tag) in enumerate(zip(deltas, tags)):
    lt = np.clip((lam_star_best or 0) + dl, 0, 1)
    idx = np.argmin(np.abs(lam_fine - lt))
    wc = ar['w'][idx] if 'w' in dir(ar) else alpha_results[best_alpha]['w'][idx]
    span = 3.0
    gr = np.linspace(-span, span, 140)  # absolute coords
    G1, G2 = np.meshgrid(gr, gr)
    z1 = G1[:,:,None]*xd[None,None,:]
    z2 = G2[:,:,None]*xd[None,None,:]
    pred = v_best[0]*h(z1,lt) + v_best[1]*h(z2,lt)
    Lgrid = 0.5*np.mean((pred - yd[None,None,:])**2, axis=2)
    Lgrid -= Lgrid.min()

    ax = axes[pi]
    vmax = np.percentile(Lgrid, 65)
    cf = ax.contourf(G1, G2, Lgrid, levels=np.linspace(0, max(vmax,1e-6), 28), cmap='viridis')
    ax.contour(G1, G2, Lgrid, levels=10, colors='w', linewidths=.2, alpha=.4)
    ax.set_xlabel('w₁'); ax.set_ylabel('w₂')
    ax.set_title(f'λ = {lt:.2f} ({tag})', fontsize=11)
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, pad=.02, shrink=.9)

plt.tight_layout(rect=[0,0,1,.94])
plt.savefig('/home/claude/fig_contour_evolution.png', dpi=160, bbox_inches='tight')
plt.close()
print("  → fig_contour_evolution.png")

print(f"\n{'='*65}")
print(f"  ALL DONE  ({time.time()-T0:.0f}s)")
print(f"{'='*65}")
