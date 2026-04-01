import Lake
open Lake DSL

package «NeuralLandscapesLean» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «NeuralLandscapesLean» where
  roots := #[`BilinearFlatness]
