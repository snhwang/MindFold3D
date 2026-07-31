# prototypes/

Exploratory scripts kept for provenance. Nothing in this directory is
imported by the production pipeline. Each script sits on top of the
production modules (`cognitive_mapping`, `skeleton_generation`,
`shape_generation`, `topology_extras`) and can be re-run standalone from
the repo root.

These scripts drove two workstreams that landed in production:

1. **Circuit-rank vs cubical β₁ characterization** — the honest framing of
   feature #6 that appears in the v6 manuscript. Established that on the
   1-voxel-wide shapes the graph-cyclic generator produces, circuit rank
   μ = E − V + C equals cubical β₁, and characterized where the two
   diverge (thick / 2×2 face patches).
2. **Hole-marked loop templates** — the construction-plus-validation
   approach that became `HoleMarkedLoopSkeleton` in production and Study 2
   in the manuscript. Reserved-hole voxels are physically excluded from
   fill, guaranteeing cubical β₁ = target on emitted shapes.

## Files

### Circuit rank ↔ β₁ characterization

- **`cross_validate_corpus.py`** — Sweeps all archetypes × all difficulty
  tiers × multiple seeds on a 7³ grid. For each shape, records both the
  graph metrics (V, E, C, circuit rank) and the cubical metrics (χ, β₀,
  β₂, cubical β₁), plus an agreement flag and a thin-structure flag
  (whether the shape is 1-voxel-wide). Emits a JSON summary and a
  per-shape CSV. This is the empirical basis for the manuscript's
  supplementary appendix on circuit rank vs cubical β₁.

- **`investigate_divergence.py`** — Small targeted script that constructs
  specific voxel configurations and reports the two metrics side by
  side. Used to nail down why circuit rank and cubical β₁ diverge on
  "thin" shapes containing 2×2 face patches (small graph cycles bounded
  by voxel faces are counted by circuit rank but not by cubical β₁).

- **`smoke_test_larger_grids.py`** — Sweeps grid sizes {7, 9, 11, 13}³ ×
  `target_loops` ∈ {1, 2, 3, 4} × several voxel counts. Answers "how big
  does the grid need to be before `LoopSkeleton` produces shapes with
  μ ≈ β₁ ≈ target_loops (actual through-holes rather than 2×2 face-patch
  cycles)?". Motivated the 11³ grid used by the hole tiers.

### Hole-preserving / hole-marked generation

- **`prototype_hole_preserving_fill.py`** — First cut at the idea.
  Subclasses `LoopSkeleton`, identifies the interior voxel of each ring
  as "protected" (must remain empty to preserve β₁), and rejects any
  fill candidate that would fill it. Works for the fixed 8-voxel ring
  template used by the existing `_build_single_loop`, but doesn't
  generalize to variable ring sizes or multiple planes.

- **`prototype_hole_marked_loops.py`** — Prototype v2. Generalizes the
  "protected voxels" idea into a template that declares two sets:
  `skeleton` (filled — part of the shape) and `hole` (reserved — cannot
  be filled). Adds axis-aligned rotation (rings in xy, xz, or yz),
  variable rectangular ring sizing, random translation within grid
  bounds, and templates for β₁ ∈ {1, 2, 3, 4}. Fill treats hole voxels
  as forbidden, so tube capping is topologically impossible.

- **`prototype_hole_marked_loops_v3.py`** — Prototype v3. Adds
  accidental-hole closure: after the fill phase completes, scans for
  empty voxels enclosed by shape voxels that are *not* part of any
  reserved hole. Fills them to bring β₁ back to target. This is the
  design that landed in production as `HoleMarkedLoopSkeleton` and
  underlies Study 2's 100% β₁ fidelity result.

## Relationship to production code

The production home for the hole-tier work is
`skeleton_generation.HoleMarkedLoopSkeleton`, wired into the pipeline via
`shape_generation` and registered in the difficulty-tier configuration
alongside the graph-cyclic archetypes. The prototypes remain useful for:

- Re-running the empirical cross-validation that backs the appendix.
- Reproducing the μ-vs-β₁ divergence examples if a reviewer asks.
- Sanity-checking future generator changes against a smaller, known-good
  reference implementation.
