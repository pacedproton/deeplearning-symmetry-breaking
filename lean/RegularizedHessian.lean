/-
  RegularizedHessian.lean
  Formal verification of Theorem 6.4(i) from:
  "Critical Transitions in Neural Network Landscapes: Spectral Splitting,
   Normal Forms, and Width-Dependent Thresholds under Activation Deformation"

  Statement: H_{0,α} = (vv⊤)⊗Σ + αI is positive definite for α > 0 and Σ ≻ 0.
  The eigenvalue α has multiplicity (m-1)*d (the flat directions of the unregularized Hessian).

  Also re-proves Proposition 6.3 (BilinearFlatness) inline, so this file is self-contained.

  Verified in Lean 4.29.0-rc8 + Mathlib (see lean/lean-toolchain).
-/
import Mathlib

-- ===========================================================
-- §1. Helper lemmas (from BilinearFlatness, reproduced here)
-- ===========================================================

private lemma mulVec_injective_of_det_ne_zero {d : ℕ} (S : Matrix (Fin d) (Fin d) ℝ)
    (h : S.det ≠ 0) : Function.Injective S.mulVec := by
  have hinv := Matrix.nonsing_inv_mul S (isUnit_iff_ne_zero.mpr h)
  intro x y hxy
  calc x = (S⁻¹ * S).mulVec x := by rw [hinv, Matrix.one_mulVec]
    _ = S⁻¹.mulVec (S.mulVec x) := (Matrix.mulVec_mulVec _ _ _).symm
    _ = S⁻¹.mulVec (S.mulVec y) := by rw [hxy]
    _ = (S⁻¹ * S).mulVec y := Matrix.mulVec_mulVec _ _ _
    _ = y := by rw [hinv, Matrix.one_mulVec]

def vProjection (m d : ℕ) (v : Fin m → ℝ) : (Fin m × Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ) where
  toFun w l := ∑ j : Fin m, v j * w (j, l)
  map_add' w₁ w₂ := by ext l; simp [Finset.sum_add_distrib, mul_add]
  map_smul' c w := by
    ext l
    simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    simp [Finset.mul_sum, mul_left_comm]

private lemma kronecker_factor (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (w : Fin m × Fin d → ℝ) (i : Fin m) (k : Fin d) :
    (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S).mulVec w (i, k) =
    v i * S.mulVec (vProjection m d v w) k := by
  simp only [Matrix.mulVec, dotProduct, Matrix.kroneckerMap_apply, Matrix.vecMulVec,
             Matrix.of_apply, Fintype.sum_prod_type, vProjection, LinearMap.coe_mk, AddHom.coe_mk]
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  congr 1; ext l; congr 1; ext j; ring

lemma kronecker_ker_eq_vProj_ker (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (hv : v ≠ 0) (hS : S.det ≠ 0) :
    LinearMap.ker (Matrix.toLin' (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S)) =
    LinearMap.ker (vProjection m d v) := by
  have ⟨i₀, hi₀⟩ : ∃ i, v i ≠ 0 := by by_contra h; push_neg at h; exact hv (funext h)
  ext w; simp only [LinearMap.mem_ker, Matrix.toLin'_apply]
  constructor
  · intro hKw
    have hSvp : S.mulVec (vProjection m d v w) = 0 := by
      ext k; have h1 := congr_fun hKw (i₀, k); rw [kronecker_factor] at h1
      simp only [Pi.zero_apply] at h1
      exact mul_left_cancel₀ hi₀ (h1.trans (mul_zero _).symm)
    exact mulVec_injective_of_det_ne_zero S hS (hSvp.trans (Matrix.mulVec_zero S).symm)
  · intro hvp; ext ⟨i, k⟩; rw [kronecker_factor]; simp [hvp]

/-- **Proposition 6.3 (Bilinear Flatness)**
    ker((vv⊤)⊗Σ) has dimension (m-1)*d when v ≠ 0 and Σ ≻ 0. -/
theorem bilinear_hessian_ker_dim (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (hv : v ≠ 0) (hS : S.PosDef) :
    Module.finrank ℝ
      (LinearMap.ker (Matrix.toLin' (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S))) =
    (m - 1) * d := by
  rw [kronecker_ker_eq_vProj_ker m d v S hv hS.det_pos.ne']
  have hsurj : Function.Surjective (vProjection m d v) := by
    intro f
    have ⟨i₀, hi₀⟩ : ∃ i, v i ≠ 0 := by by_contra h; push_neg at h; exact hv (funext h)
    refine ⟨fun ⟨i, l⟩ => if i = i₀ then f l / v i₀ else 0, ?_⟩
    ext l; simp only [vProjection, LinearMap.coe_mk, AddHom.coe_mk]
    rw [Finset.sum_eq_single i₀]
    · simp only [ite_true]; field_simp
    · intro j _ hjne; simp [hjne]
    · simp
  have hrank := LinearMap.finrank_range_add_finrank_ker (vProjection m d v)
  rw [LinearMap.range_eq_top.mpr hsurj] at hrank; simp at hrank
  have htotal : Module.finrank ℝ (Fin m × Fin d → ℝ) = m * d := by simp [Fintype.card_prod]
  have hm : 0 < m := by
    by_contra hm; push_neg at hm; interval_cases m; exact hv (Subsingleton.elim _ _)
  have hkey : (m - 1) * d = m * d - d := by
    cases m with | zero => omega | succ n => simp; rw [Nat.add_mul]; omega
  omega

-- ===========================================================
-- §2. Helper lemmas for Theorem 6.4(i)
-- ===========================================================

/-- Quadratic form of (vv⊤)⊗S factors through the v-projection. -/
private lemma kronecker_quadform (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (w : Fin m × Fin d → ℝ) :
    dotProduct w ((Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S).mulVec w) =
    dotProduct (fun l => ∑ j : Fin m, v j * w (j, l))
               (S.mulVec (fun l => ∑ j : Fin m, v j * w (j, l))) := by
  have hKw : (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S).mulVec w =
      fun p : Fin m × Fin d => v p.1 * S.mulVec (fun l => ∑ j : Fin m, v j * w (j, l)) p.2 := by
    ext ⟨i, k⟩
    simp only [Matrix.mulVec, dotProduct, Matrix.kroneckerMap_apply, Matrix.vecMulVec,
               Matrix.of_apply, Fintype.sum_prod_type]
    simp_rw [Finset.mul_sum]; rw [Finset.sum_comm]
    congr 1; ext l; congr 1; ext j; ring
  rw [hKw]; simp only [dotProduct, Matrix.mulVec, Fintype.sum_prod_type]
  rw [Finset.sum_comm]; apply Finset.sum_congr rfl; intro k _
  rw [Finset.sum_mul]; apply Finset.sum_congr rfl; intro i _; ring

/-- Build a PosDef certificate from plain dotProduct positivity. -/
private lemma posDef_of_dotProduct_pos' {n : Type*} [Fintype n]
    (A : Matrix n n ℝ) (hsymm : A.IsSymm)
    (hpos : ∀ x : n → ℝ, x ≠ 0 → 0 < dotProduct x (A.mulVec x)) : A.PosDef := by
  constructor
  · exact Matrix.isHermitian_iff_isSelfAdjoint.mpr hsymm
  · intro xf hxf
    have hx : Finsupp.equivFunOnFinite xf ≠ 0 := by
      intro h; apply hxf; ext i; have := congr_fun h i
      simp [Finsupp.equivFunOnFinite] at this; exact this
    have hq := hpos (Finsupp.equivFunOnFinite xf) hx
    suffices h : xf.sum (fun i xi => xf.sum (fun j xj => star xi * A i j * xj)) =
        dotProduct (Finsupp.equivFunOnFinite xf) (A.mulVec (Finsupp.equivFunOnFinite xf)) by
      linarith [h ▸ hq]
    simp only [dotProduct, Matrix.mulVec, Finsupp.equivFunOnFinite_apply]
    rw [Finsupp.sum_fintype _ _ (by intro i; simp)]
    apply Finset.sum_congr rfl; intro i _
    rw [Finsupp.sum_fintype _ _ (by intro j; simp)]
    simp only [star_trivial]; rw [Finset.mul_sum]; congr 1; ext j; ring

/-- Quadratic form of α•I is α‖w‖². -/
private lemma smul_one_quadform' {n : Type*} [Fintype n] [DecidableEq n] (α : ℝ) (w : n → ℝ) :
    dotProduct w ((α • (1 : Matrix n n ℝ)).mulVec w) = α * dotProduct w w := by
  simp only [Matrix.smul_mulVec, Matrix.one_mulVec, dotProduct, Pi.smul_apply, smul_eq_mul]
  simp_rw [show ∀ x : n, w x * (α * w x) = α * (w x * w x) from fun x => by ring]
  rw [← Finset.mul_sum]

/-- A nonzero function has positive self-inner-product. -/
private lemma dotProduct_pos_of_ne_zero' {n : Type*} [Fintype n] (w : n → ℝ) (hw : w ≠ 0) :
    0 < dotProduct w w := by
  obtain ⟨i, hi⟩ : ∃ i, w i ≠ 0 := Function.ne_iff.mp hw
  exact lt_of_lt_of_le (mul_self_pos.mpr hi)
    (Finset.single_le_sum (fun j _ => mul_self_nonneg (w j)) (Finset.mem_univ i))

/-- Extract IsSymm from PosDef. -/
private lemma symm_of_posDef' {n : ℕ} (S : Matrix (Fin n) (Fin n) ℝ) (hS : S.PosDef) :
    S.IsSymm := by
  have h : S.IsHermitian := hS.1; ext i j
  have := congr_fun (congr_fun h i) j
  simp [Matrix.conjTranspose] at this; simp only [Matrix.transpose_apply]; exact this

/-- (vv⊤)⊗S is symmetric when S is symmetric. -/
private lemma kronecker_isSymm' (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (hS : S.IsSymm) : (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S).IsSymm := by
  ext ⟨i, k⟩ ⟨j, l⟩
  simp [Matrix.kroneckerMap_apply, Matrix.vecMulVec, Matrix.of_apply, Matrix.transpose_apply]
  rw [show S l k = S k l from (congr_fun (congr_fun hS l) k).symm]; ring

-- ===========================================================
-- §3. Main theorem: Theorem 6.4(i)
-- ===========================================================

/-- **Theorem 6.4(i) — Regularized Hessian is Positive Definite**

    For a two-layer network at λ = 0 with regularization α > 0:
    H_{0,α} = (vv⊤)⊗Σ + αI ≻ 0.

    Proof: w⊤H_{0,α}w = w⊤((vv⊤)⊗Σ)w + α‖w‖²
                       ≥ 0 + α‖w‖² > 0 for w ≠ 0.
    The first term is nonneg because the Kronecker quadratic form factors as
    (vProj w)⊤ Σ (vProj w) ≥ 0 (by Σ ≻ 0). -/
theorem regularized_hessian_posDef (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (α : ℝ) (hα : 0 < α) (hS : S.PosDef) :
    (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S +
     α • (1 : Matrix (Fin m × Fin d) (Fin m × Fin d) ℝ)).PosDef := by
  set K := Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S
  have hKsymm : K.IsSymm := kronecker_isSymm' m d v S (symm_of_posDef' S hS)
  have hsum_symm : (K + α • (1 : Matrix (Fin m × Fin d) (Fin m × Fin d) ℝ)).IsSymm := by
    simp only [Matrix.IsSymm, Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_one]
    rw [hKsymm]
  apply posDef_of_dotProduct_pos' _ hsum_symm
  intro w hw
  have hKnn : 0 ≤ dotProduct w (K.mulVec w) := by
    rw [kronecker_quadform]
    by_cases hvp : (fun l => ∑ j : Fin m, v j * w (j, l)) = 0
    · simp [hvp]
    · exact le_of_lt ((hS.re_dotProduct_pos hvp).trans_eq (by simp [RCLike.re_to_real]))
  have hIpos : 0 < dotProduct w ((α • (1 : Matrix (Fin m × Fin d) (Fin m × Fin d) ℝ)).mulVec w) :=
    by rw [smul_one_quadform']; exact mul_pos hα (dotProduct_pos_of_ne_zero' w hw)
  linarith [show dotProduct w ((K + α • (1 : Matrix _ _ ℝ)).mulVec w) =
      dotProduct w (K.mulVec w) + dotProduct w ((α • (1 : Matrix _ _ ℝ)).mulVec w) by
    simp [Matrix.add_mulVec, dotProduct_add]]

-- ===========================================================
-- §4. Eigenvalue multiplicity: Theorem 6.4(i) continued
-- ===========================================================

/-- **Theorem 6.4(i) — Eigenvalue α has multiplicity (m-1)*d**

    The kernel of (H_{0,α} - αI) as a linear map equals ker((vv⊤)⊗Σ),
    which has dimension (m-1)*d.

    Proof: H_{0,α} - αI = (vv⊤)⊗Σ + αI - αI = (vv⊤)⊗Σ.
    The kernel dimension is (m-1)*d by Proposition 6.3. -/
theorem regularized_min_eigenvalue_mult (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (α : ℝ) (hv : v ≠ 0) (hS : S.PosDef) :
    Module.finrank ℝ
      (LinearMap.ker (Matrix.toLin'
        (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S +
         α • (1 : Matrix (Fin m × Fin d) (Fin m × Fin d) ℝ)) -
       α • LinearMap.id)) =
    (m - 1) * d := by
  -- Rewrite the linear map: toLin'(K + α•1) - α•id = toLin'(K)
  have hkey : Matrix.toLin' (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S +
      α • (1 : Matrix (Fin m × Fin d) (Fin m × Fin d) ℝ)) - α • LinearMap.id =
      Matrix.toLin' (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S) := by
    apply LinearMap.ext; intro w; ext p
    simp only [LinearMap.sub_apply, LinearMap.smul_apply, LinearMap.id_apply,
               Matrix.toLin'_apply, Matrix.add_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec,
               Pi.sub_apply, Pi.smul_apply, Pi.add_apply, smul_eq_mul]
    ring
  rw [hkey]
  exact bilinear_hessian_ker_dim m d v S hv hS
