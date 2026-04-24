# Cardano's Trigonometric Eigenvalue Formula (3×3 Symmetric Case)

Used in `shape_generation.py` to extract PCA eigenvalues from the voxel covariance matrix without external linear algebra dependencies.

## Setup

Given the 3×3 symmetric covariance matrix **Σ**, eigenvalues are roots of the characteristic polynomial:

    λ³ - p·λ² + q·λ - r = 0

where:
- `p = tr(Σ)` (sum of diagonal elements)
- `q` = sum of 2×2 principal minors of Σ
- `r = det(Σ)`

## Cardano's Trigonometric Solution

    θ = (1/3) · arccos( (2p³ - 9pq + 27r) / (2·(p² - 3q)^(3/2)) )

    λ_k = (p + 2·√(p² - 3q) · cos(θ - 2πk/3)) / 3,   k = 0, 1, 2

Sorting λ₀ ≥ λ₁ ≥ λ₂ gives λ₁ ≥ λ₂ ≥ λ₃ ≥ 0.

## Anisotropy Index

    Anisotropy = 1 - λ₃ / (λ₁ + ε)

Range: 0 (isotropic) → 1 (maximally anisotropic). ε is a small constant to avoid division by zero for degenerate (planar or linear) shapes.

## Notes

- Numerically stable for symmetric positive-semidefinite matrices (guaranteed by covariance construction).
- All six unique elements of Σ are computed in a single pass over voxels.
- Completes in under 1 ms per shape.
- Avoids NumPy or any external linear algebra dependency.
