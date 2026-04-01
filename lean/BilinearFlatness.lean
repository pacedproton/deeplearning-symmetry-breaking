/-
  BilinearFlatness.lean
  Formal verification of Proposition 6.3 from:
  "Critical Transitions in Neural Network Landscapes: Spectral Splitting,
   Normal Forms, and Width-Dependent Thresholds under Activation Deformation"

  Statement: For a two-layer network L(W, 0) = ½ 𝔼[(v⊤Wx - y)²],
  the Hessian w.r.t. vec(W) is H₀ = (vv⊤) ⊗ Σ.
  When v ≠ 0 and Σ ≻ 0, this has kernel dimension (m-1)*d.

  Verified in Lean 4.29.0-rc8 + Mathlib (see lean/lean-toolchain).
-/
import Mathlib

-- ===========================================================
-- §1. Helper lemmas
-- ===========================================================

/-- Pointwise formula for the Kronecker product action (vv⊤ ⊗ S)w.
    The (i,k)-entry equals v_i · ∑_j v_j · (Sw)_j^k. -/
private lemma kronecker_vecMulVec_apply (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (w : Fin m × Fin d → ℝ) (i : Fin m) (k : Fin d) :
    (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S).mulVec w (i, k) =
    v i * ∑ j : Fin m, v j * ∑ l : Fin d, S k l * w (j, l) := by
  simp only [Matrix.mulVec, Matrix.kroneckerMap_apply, Matrix.vecMulVec, dotProduct,
             Fintype.sum_prod_type, Matrix.of_apply, Finset.mul_sum]
  congr 1; ext j; congr 1; ext l; ring

/-- A matrix with nonzero determinant has injective mulVec.
    Proof: premultiply by the nonsing inverse. -/
private lemma mulVec_injective_of_det_ne_zero {d : ℕ} (S : Matrix (Fin d) (Fin d) ℝ)
    (h : S.det ≠ 0) : Function.Injective S.mulVec := by
  have hinv := Matrix.nonsing_inv_mul S (isUnit_iff_ne_zero.mpr h)
  intro x y hxy
  calc x = (S⁻¹ * S).mulVec x := by rw [hinv, Matrix.one_mulVec]
    _ = S⁻¹.mulVec (S.mulVec x) := (Matrix.mulVec_mulVec _ _ _).symm
    _ = S⁻¹.mulVec (S.mulVec y) := by rw [hxy]
    _ = (S⁻¹ * S).mulVec y := Matrix.mulVec_mulVec _ _ _
    _ = y := by rw [hinv, Matrix.one_mulVec]

-- ===========================================================
-- §2. The v-projection linear map
-- ===========================================================

/-- The v-projection: w ↦ (l ↦ ∑_j v_j · w(j, l)).
    Encodes the condition v⊤W = 0 that defines the flat directions.
    When v ≠ 0, this map is surjective onto Fin d → ℝ. -/
def vProjection (m d : ℕ) (v : Fin m → ℝ) : (Fin m × Fin d → ℝ) →ₗ[ℝ] (Fin d → ℝ) where
  toFun w l := ∑ j : Fin m, v j * w (j, l)
  map_add' w₁ w₂ := by ext l; simp [Finset.sum_add_distrib, mul_add]
  map_smul' c w := by
    ext l
    simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
    simp [Finset.mul_sum, mul_left_comm]

-- ===========================================================
-- §3. Key factorization: Kron action = v_i · (S · vProj(w))_k
-- ===========================================================

/-- The Kronecker action (vv⊤ ⊗ S)w at (i,k) factors as
    v_i · (S · vProjection(w))_k. -/
private lemma kronecker_factor (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (w : Fin m × Fin d → ℝ) (i : Fin m) (k : Fin d) :
    (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S).mulVec w (i, k) =
    v i * S.mulVec (vProjection m d v w) k := by
  rw [kronecker_vecMulVec_apply]
  simp only [Matrix.mulVec, dotProduct, vProjection, LinearMap.coe_mk, AddHom.coe_mk]
  congr 1
  -- Swap the order of summation
  have lhs_eq : ∑ j : Fin m, v j * ∑ l : Fin d, S k l * w (j, l) =
      ∑ j : Fin m, ∑ l : Fin d, v j * (S k l * w (j, l)) := by
    congr 1; ext j; rw [Finset.mul_sum]
  have rhs_eq : ∑ l : Fin d, S k l * ∑ j : Fin m, v j * w (j, l) =
      ∑ l : Fin d, ∑ j : Fin m, S k l * (v j * w (j, l)) := by
    congr 1; ext l; rw [Finset.mul_sum]
  rw [lhs_eq, rhs_eq, Finset.sum_comm]
  congr 1; ext l; congr 1; ext j; ring

-- ===========================================================
-- §4. Kernel equality
-- ===========================================================

/-- The kernel of (vv⊤ ⊗ S) as a linear map equals the kernel of vProjection,
    when v ≠ 0 and S is invertible. -/
lemma kronecker_ker_eq_vProj_ker (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (hv : v ≠ 0) (hS : S.det ≠ 0) :
    LinearMap.ker (Matrix.toLin' (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S)) =
    LinearMap.ker (vProjection m d v) := by
  have ⟨i₀, hi₀⟩ : ∃ i, v i ≠ 0 := by
    by_contra h; push_neg at h; exact hv (funext h)
  ext w; simp only [LinearMap.mem_ker, Matrix.toLin'_apply]
  constructor
  · -- Kw = 0 → vProj(w) = 0
    intro hKw
    have hSvp : S.mulVec (vProjection m d v w) = 0 := by
      ext k
      have h1 := congr_fun hKw (i₀, k)
      rw [kronecker_factor] at h1
      simp only [Pi.zero_apply] at h1
      exact mul_left_cancel₀ hi₀ (h1.trans (mul_zero _).symm)
    exact mulVec_injective_of_det_ne_zero S hS (hSvp.trans (Matrix.mulVec_zero S).symm)
  · -- vProj(w) = 0 → Kw = 0
    intro hvp
    ext ⟨i, k⟩
    rw [kronecker_factor]
    simp [hvp]

-- ===========================================================
-- §5. Main theorem: Proposition 6.3
-- ===========================================================

/-- **Proposition 6.3 (Bilinear Flatness)** — paper §6.2.

    For a two-layer network at λ = 0, the Hessian of L(W, 0) w.r.t. vec(W) is
    H₀ = (vv⊤) ⊗ Σ (Kronecker product).  When v ≠ 0 and Σ ≻ 0, this has:
    - rank d (the "useful" subspace aligned with v ⊗ eₖ for k = 1,...,d)
    - kernel dimension (m-1)*d (the overparameterized flat directions {δW : v⊤δW = 0})

    Proof strategy:
    1. Show ker((vv⊤) ⊗ Σ) = ker(vProjection) via the factored action formula.
    2. vProjection is surjective (v ≠ 0) so by rank-nullity finrank ker = md - d = (m-1)d. -/
theorem bilinear_hessian_ker_dim (m d : ℕ) (v : Fin m → ℝ) (S : Matrix (Fin d) (Fin d) ℝ)
    (hv : v ≠ 0) (hS : S.PosDef) :
    Module.finrank ℝ
      (LinearMap.ker (Matrix.toLin' (Matrix.kroneckerMap (· * ·) (Matrix.vecMulVec v v) S))) =
    (m - 1) * d := by
  -- Step 1: Rewrite ker(Kron) as ker(vProj)
  rw [kronecker_ker_eq_vProj_ker m d v S hv hS.det_pos.ne']
  -- Step 2: vProjection is surjective when v ≠ 0
  have hsurj : Function.Surjective (vProjection m d v) := by
    intro f
    have ⟨i₀, hi₀⟩ : ∃ i, v i ≠ 0 := by
      by_contra h; push_neg at h; exact hv (funext h)
    -- Preimage: set w(i,l) = f(l)/v(i₀) if i = i₀, else 0
    refine ⟨fun ⟨i, l⟩ => if i = i₀ then f l / v i₀ else 0, ?_⟩
    ext l
    simp only [vProjection, LinearMap.coe_mk, AddHom.coe_mk]
    rw [Finset.sum_eq_single i₀]
    · simp only [ite_true]; field_simp
    · intro j _ hjne; simp [hjne]
    · simp
  -- Step 3: Rank-nullity: finrank ker = (m*d) - d = (m-1)*d
  have hrank := LinearMap.finrank_range_add_finrank_ker (vProjection m d v)
  rw [LinearMap.range_eq_top.mpr hsurj] at hrank
  simp at hrank
  have htotal : Module.finrank ℝ (Fin m × Fin d → ℝ) = m * d := by simp [Fintype.card_prod]
  have hm : 0 < m := by
    by_contra hm; push_neg at hm; interval_cases m; exact hv (Subsingleton.elim _ _)
  have hkey : (m - 1) * d = m * d - d := by
    cases m with | zero => omega | succ n => simp; rw [Nat.add_mul]; omega
  omega
