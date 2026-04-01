# Test Status

Last updated: 2026-03-30

## Summary
- Total theorems verified: 6
- Passing: 6
- Failing: 0

## By Module

### lean/BilinearFlatness.lean — Proposition 6.3
| Theorem/Lemma | Status | Notes |
|---|---|---|
| `kronecker_vecMulVec_apply` | ✅ PASS | Helper: pointwise Kronecker formula |
| `mulVec_injective_of_det_ne_zero` | ✅ PASS | Helper: det≠0 → injective |
| `vProjection` (def) | ✅ PASS | Linear map: w ↦ v⊤W |
| `kronecker_factor` | ✅ PASS | Kron action factors through vProjection |
| `kronecker_ker_eq_vProj_ker` | ✅ PASS | ker(Kron) = ker(vProj) when v≠0, S invertible |
| `bilinear_hessian_ker_dim` | ✅ PASS | **Prop 6.3**: ker dim = (m-1)*d |

### lean/RegularizedHessian.lean — Theorem 6.4(i)
| Theorem/Lemma | Status | Notes |
|---|---|---|
| `kronecker_quadform` | ✅ PASS | w⊤Kw = (vProjw)⊤S(vProjw) |
| `posDef_of_dotProduct_pos'` | ✅ PASS | Bridge: dotProduct positivity → PosDef |
| `smul_one_quadform'` | ✅ PASS | w⊤(α·I)w = α‖w‖² |
| `dotProduct_pos_of_ne_zero'` | ✅ PASS | w≠0 → 0 < ⟨w,w⟩ |
| `symm_of_posDef'` | ✅ PASS | PosDef → IsSymm |
| `kronecker_isSymm'` | ✅ PASS | (vv⊤)⊗S symmetric when S symmetric |
| `regularized_hessian_posDef` | ✅ PASS | **Thm 6.4(i)**: H_{0,α} ≻ 0 for α > 0 |
| `regularized_min_eigenvalue_mult` | ✅ PASS | **Thm 6.4(i)**: eigenvalue α has mult (m-1)*d |

## Pending
- Theorem 6.8 (Faà di Bruno formulas for g_aa, g_aaa) — Layer 1, not started
