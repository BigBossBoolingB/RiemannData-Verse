import Lake
open Lake DSL

package verification_engine {
  -- add package configuration options here
}

lean_lib VerificationEngine {
  -- add library configuration options here
}

@[default_target]
lean_exe verification_engine {
  root := `Main
}

-- Ensure you have a `Main.lean` file in the root of `verification_engine`
-- or adjust the `root` field above if your main executable Lean file is named differently
-- or located elsewhere.

-- To add dependencies, for example Mathlib:
-- require mathlib from git "https://github.com/leanprover-community/mathlib4.git"

-- After adding dependencies, run `lake update` and then `lake build`.
-- This `lakefile.lean` is a basic starting point.
-- You may need to add more configurations for your specific project needs,
-- especially when dealing with complex dependencies or build options.
