"""
Multi-Objective Shape Generation and Feature Analysis Pipeline for MindFold 3D.

Copyright (c) 2026 Scott N. Hwang, Parviz Safadel. All rights reserved.
Patent pending. See docs/PATENT_SPECIFICATION.md for claims.

This module implements the Shape Feature Analysis Pipeline and Optimization-Based
Shape Generator, core novel components of the MindFold 3D invention (Claims 1, 2, 12, 13).

Key inventive methods:
  - analyze_shape_features():        Real-time feature vector computation (Claim 1d)
  - _get_pca_eigenvalues():          PCA eigenvalue decomposition for shape characterization
  - _calculate_anisotropy_index():   PCA-based anisotropy: 1-(lambda3/lambda1) (Claim 12)
  - _calculate_shape_form_index():   PCA-based form: (2*lambda2-lambda1-lambda3)/(lambda1-lambda3) (Claim 12)
  - _calculate_circuit_rank():       Circuit rank μ = |E|−|V|+C of the voxel 6-adjacency graph (Claim 13)
  - _calculate_betti_1():            β₁ of the shape (measured as circuit rank for graph-cyclic shapes;
                                     equals cubical β₁ on 1-voxel-wide skeletons, cross-validated)
  - generate_mirror_reflection():    Chirality detection via mirror analysis (Claim 9)
  - generate_shape_advanced():       Multi-objective scored shape growth (Claim 1c)
  - generate_part_permuted_distractor(): Configural binding distractors (Claim 7)
"""
# shape_generation.py
import random
from typing import List, Tuple, Dict, Any, Optional, Set, Literal
from collections import deque
import math
from shape_features import ShapeFeatureSet


# --- Helper functions for calculating features of a voxel list --- 

def _get_neighbors(voxel: Tuple[int, int, int], grid_size: Tuple[int, int, int], include_diagonal=False) -> List[Tuple[int, int, int]]:
    """Get valid face-adjacent (and optionally diagonal) neighbors of a voxel."""
    x, y, z = voxel
    neighbors = []
    
    base_directions = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0),
        (0, -1, 0), (0, 0, 1), (0, 0, -1)
    ]
    diagonal_directions = []
    if include_diagonal:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if i == 0 and j == 0 and k == 0: continue
                    if (i != 0 and j != 0 and k ==0) or \
                       (i != 0 and k != 0 and j ==0) or \
                       (j != 0 and k != 0 and i ==0) or \
                       (i !=0 and j!=0 and k!=0): # for 2D and 3D diagonals
                         diagonal_directions.append((i,j,k))
    
    all_directions = base_directions + (diagonal_directions if include_diagonal else [])

    for dx, dy, dz in all_directions:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and 0 <= nz < grid_size[2]:
            neighbors.append((nx, ny, nz))
    return neighbors

def _calculate_surface_area(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> int:
    if not voxels_set:
        return 0
    surface_area = 0
    for voxel in voxels_set:
        exposed_faces = 6
        for neighbor in _get_neighbors(voxel, grid_size, include_diagonal=False):
            if neighbor in voxels_set:
                exposed_faces -= 1
        surface_area += exposed_faces
    return surface_area

def _calculate_symmetry_score(voxels_set: Set[Tuple[int, int, int]]) -> float:
    """Bilateral symmetry: max voxel overlap when reflected across any axis-aligned midplane.

    For each of the 3 axes, reflects the shape across the midplane of its bounding box
    and measures what fraction of voxels coincide with the original.
    Returns the best (maximum) overlap across all 3 axes.

    0.0 = fully asymmetric on all axes
    1.0 = perfectly bilaterally symmetric on at least one axis
    """
    if len(voxels_set) < 2:
        return 1.0
    voxels = list(voxels_set)
    coords = [
        [v[0] for v in voxels],
        [v[1] for v in voxels],
        [v[2] for v in voxels],
    ]
    best = 0.0
    for axis in range(3):
        mid = (min(coords[axis]) + max(coords[axis]))  # = 2 * center, kept as int sum
        reflected = set()
        for v in voxels:
            rv = list(v)
            rv[axis] = mid - v[axis]
            reflected.add(tuple(rv))
        overlap = len(voxels_set & reflected) / len(voxels_set)
        if overlap > best:
            best = overlap
    return best


def _calculate_compactness_score(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> float:
    """Mean neighbor count per voxel, normalized to [0, 1] by dividing by 6
    (max possible neighbors)."""
    if not voxels_set or len(voxels_set) < 2:
        return 0.5 # Neutral for single voxel or empty
    
    total_neighbors = 0
    for voxel in voxels_set:
        for neighbor in _get_neighbors(voxel, grid_size):
            if neighbor in voxels_set:
                total_neighbors +=1
    avg_neighbors = total_neighbors / len(voxels_set)  # mean neighbor count (0-6)
    return min(1.0, avg_neighbors / 6.0)  # normalize to [0, 1]

def _count_components(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> int:
    if not voxels_set:
        return 0
    
    visited = set()
    components = 0
    
    for voxel in voxels_set:
        if voxel not in visited:
            components += 1
            q = deque([voxel])
            visited.add(voxel)
            while q:
                curr = q.popleft()
                for neighbor in _get_neighbors(curr, grid_size):
                    if neighbor in voxels_set and neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
    return components

def _calculate_branching_factor(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> int:
    """Approximate branching: count voxels whose in-shape face-neighbor degree
    is >= 3 (i.e. true branch points). Endpoints (degree 1) are not considered
    and no ratio or sum is computed.
    """
    if len(voxels_set) < 3:
        return 0 # Not enough voxels to really branch
    
    branch_points = 0
    for voxel in voxels_set:
        neighbor_count = 0
        for neighbor in _get_neighbors(voxel, grid_size):
            if neighbor in voxels_set:
                neighbor_count += 1
        if neighbor_count >= 3: # A voxel connected to 3 or more other voxels
            branch_points += 1
    return branch_points

def _calculate_circuit_rank(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> int:
    """Circuit rank μ of the voxel 6-adjacency graph: μ = |E| − |V| + C.

    This is the first Betti number of the *adjacency graph* (a graph
    invariant): the number of independent closed paths through the shape's
    face-adjacency structure, equivalently the number of edges that must be
    removed to reduce the graph to a spanning forest.

    It is NOT the cubical β₁ of the shape as a solid. The two diverge in both
    directions on real generator output: every 2×2 face patch contributes a
    graph cycle with no volumetric hole (μ > β₁ on thick shapes), and
    edge-adjacent voxel arrangements can enclose a tunnel the face-adjacency
    graph cannot see (β₁ > μ). Use `_calculate_betti_1` (cubical, via
    topology_extras.cubical_betti_1) for the topological hole count.

    Parameters
    ----------
    voxels_set : set of (x, y, z) tuples
        The occupied voxels.
    grid_size : (nx, ny, nz)
        Bounding-box dimensions used for neighbour indexing.

    Returns
    -------
    int
        The circuit rank μ ≥ 0.

    Notes
    -----
    Why circuit rank as the generator's control quantity for graph-cyclic
    tiers?

    1. It is computable in O(|V|) time from the adjacency graph, whereas
       full cubical β₁ (β₀ + β₂ − χ) requires building the complement's
       cavity structure — too slow for the per-candidate scoring loop.
    2. It is the property the skeleton-first construction directly controls:
       LoopSkeleton injects graph cycles, TreeSkeleton's skeleton is acyclic,
       and the fill phase caps μ against the target.

    For tiers that must guarantee actual volumetric holes, use the hole-tier
    construction mode (HoleMarkedLoopSkeleton), which targets cubical β₁
    directly via reserved-hole templates plus per-shape validation.

    See Also
    --------
    topology_extras.cubical_betti_1 : Full cubical β₁ computation via Euler
        characteristic, provided as a validation reference and for the
        paper's methodological supplement.

    References
    ----------
    Diestel, R. (2024). *Graph Theory* (6th ed.), §1.9 (circuit rank /
    cyclomatic number).  Kong, T. Y., & Rosenfeld, A. (1989). Digital
    topology: Introduction and survey. *Computer Vision, Graphics, and Image
    Processing*, 48(3), 357–393.  (Background on adjacency structures for
    voxel objects; note that graph cycles and cubical holes are distinct
    invariants — see topology_extras for the cubical computation.)

    This implementation supports Claim 13 of the MindFold 3D provisional
    patent (topology-preserving voxel fill by circuit rank).
    """
    if len(voxels_set) < 4:
        return 0  # Minimum 4 voxels for a cycle in 6-connectivity
    edge_count = sum(
        1 for v in voxels_set for n in _get_neighbors(v, grid_size) if n in voxels_set
    ) // 2  # Each edge counted twice
    return max(0, edge_count - len(voxels_set) + _count_components(voxels_set, grid_size))


def _calculate_betti_1(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> int:
    """First Betti number β₁ of the shape as a cubical complex.

    This is the feature-level entry point that the analysis pipeline calls
    when it fills the `betti_1` field of `ShapeFeatureSet`. It computes the
    true cubical β₁ via the Euler-characteristic method
    (topology_extras.cubical_betti_1).

    β₁ is NOT the circuit rank μ of the adjacency graph: μ > β₁ on shapes
    with 2-thick face patches (each 2×2 patch adds a graph cycle with no
    volumetric hole), and β₁ > μ is possible via edge-adjacency tunnels the
    face-adjacency graph cannot see. Use `_calculate_circuit_rank` for the
    graph invariant the generator's fill/scoring loops control.
    """
    # Lazy import: topology_extras imports _count_components from this module.
    from topology_extras import cubical_betti_1
    return cubical_betti_1(voxels_set)


# Backward-compat alias. The old name honestly describes the computation
# (circuit rank / cycle count of the adjacency graph); the new names
# separate the *feature* (β₁) from the *computation* (circuit rank).
_calculate_cycle_count = _calculate_circuit_rank


def _calculate_planarity_score(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> float:
    """Calculates planarity as the max proportion of voxels lying on any single X, Y, or Z plane.
    Returns a score between 0 and 1. 
    1.0 means all voxels are on a single plane (perfectly flat).
    Lower scores mean more 3D spread.
    """
    if not voxels_set or len(voxels_set) < 2:
        return 1.0 # Single voxel or empty set is considered planar

    max_voxels_in_one_plane = 0
    total_voxels = len(voxels_set)

    # Check X-planes (yz planes at different x values)
    x_counts: Dict[int, int] = {}
    for v in voxels_set: x_counts[v[0]] = x_counts.get(v[0], 0) + 1
    if x_counts: max_voxels_in_one_plane = max(max_voxels_in_one_plane, max(x_counts.values()))

    # Check Y-planes (xz planes at different y values)
    y_counts: Dict[int, int] = {}
    for v in voxels_set: y_counts[v[1]] = y_counts.get(v[1], 0) + 1
    if y_counts: max_voxels_in_one_plane = max(max_voxels_in_one_plane, max(y_counts.values()))

    # Check Z-planes (xy planes at different z values)
    z_counts: Dict[int, int] = {}
    for v in voxels_set: z_counts[v[2]] = z_counts.get(v[2], 0) + 1
    if z_counts: max_voxels_in_one_plane = max(max_voxels_in_one_plane, max(z_counts.values()))
    
    return max_voxels_in_one_plane / total_voxels if total_voxels > 0 else 1.0

# --- New Analysis Helper Functions ---
def _get_bounding_box(voxels_set: Set[Tuple[int, int, int]]) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    if not voxels_set:
        return None
    min_x = min(v[0] for v in voxels_set)
    max_x = max(v[0] for v in voxels_set)
    min_y = min(v[1] for v in voxels_set)
    max_y = max(v[1] for v in voxels_set)
    min_z = min(v[2] for v in voxels_set)
    max_z = max(v[2] for v in voxels_set)
    return ((min_x, min_y, min_z), (max_x, max_y, max_z))

def _calculate_bounding_box_dimensions(voxels_set: Set[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    bbox = _get_bounding_box(voxels_set)
    if not bbox:
        return (0,0,0) # Default for empty set
    return (bbox[1][0] - bbox[0][0] + 1, 
            bbox[1][1] - bbox[0][1] + 1, 
            bbox[1][2] - bbox[0][2] + 1)

def _calculate_bounding_box_ratio(voxels_set: Set[Tuple[int, int, int]]) -> float:
    dims = _calculate_bounding_box_dimensions(voxels_set)
    if not dims or min(dims) == 0: # if any dimension is 0 (e.g. empty or single point/line/plane)
        if len(voxels_set) <= 1: return 1.0 # Single point is 1:1:1 ratio
        # For a line or plane, a true ratio might be infinite or ill-defined depending on perspective.
        # Let's return a large number if it's a line/plane, or 1.0 if it's just very small.
        non_zero_dims = [d for d in dims if d > 0]
        if not non_zero_dims: return 1.0 # Should not happen if voxels_set is not empty
        if len(non_zero_dims) < 2 : return 100.0 # Effectively a line or point in terms of ratio among >0 dims
        return max(non_zero_dims) / min(non_zero_dims) if min(non_zero_dims) > 0 else 100.0
    return max(dims) / min(dims)

def _calculate_center_of_mass(voxels_set: Set[Tuple[int, int, int]]) -> Tuple[float, float, float]:
    if not voxels_set:
        return (0.0, 0.0, 0.0) # Default for empty set
    sum_x = sum(v[0] for v in voxels_set)
    sum_y = sum(v[1] for v in voxels_set)
    sum_z = sum(v[2] for v in voxels_set)
    n = len(voxels_set)
    return (sum_x / n, sum_y / n, sum_z / n)

def _calculate_center_of_mass_offset(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> Tuple[float, float, float]:
    com = _calculate_center_of_mass(voxels_set)
    # if not com: # com will always return a tuple now
    #     return (0.0,0.0,0.0)
    grid_center = ((grid_size[0] -1) / 2.0, (grid_size[1]-1) / 2.0, (grid_size[2]-1) / 2.0)
    return (com[0] - grid_center[0], com[1] - grid_center[1], com[2] - grid_center[2])

def _calculate_dominant_axis(voxels_set: Set[Tuple[int, int, int]]) -> Literal['x', 'y', 'z', 'balanced']:
    dims = _calculate_bounding_box_dimensions(voxels_set)
    # if not dims: # dims will always return a tuple now
    #     return 'balanced'
    dx, dy, dz = dims
    if dx == 0 and dy == 0 and dz == 0 and not voxels_set: # Explicitly check for empty case after dims=(0,0,0)
        return 'balanced'
    if dx > dy and dx > dz: return 'x'
    if dy > dx and dy > dz: return 'y'
    if dz > dx and dz > dy: return 'z'
    # Handle cases with two equal dominant axes (e.g., flat square)
    if dx == dy and dx > dz : return 'x' # or 'y', could be 'xy_balanced' in a richer model
    if dx == dz and dx > dy : return 'x' # or 'z'
    if dy == dz and dy > dx : return 'y' # or 'z'
    return 'balanced' # Default fallback

def _get_component_sizes(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> List[int]:
    if not voxels_set: return []
    visited = set()
    component_sizes = []
    for voxel in voxels_set:
        if voxel not in visited:
            current_component_size = 0
            q = deque([voxel])
            visited.add(voxel)
            current_component_size += 1
            while q:
                curr = q.popleft()
                for neighbor in _get_neighbors(curr, grid_size):
                    if neighbor in voxels_set and neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
                        current_component_size += 1
            component_sizes.append(current_component_size)
    return component_sizes

def _calculate_largest_component_ratio(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> float:
    if not voxels_set: return 0.0
    component_sizes = _get_component_sizes(voxels_set, grid_size)
    if not component_sizes: return 0.0
    return max(component_sizes) / len(voxels_set) if len(voxels_set) > 0 else 0.0

def _get_pca_eigenvalues(voxels_set: Set[Tuple[int, int, int]]) -> Optional[Tuple[float, float, float]]:
    """Compute PCA eigenvalues of voxel coordinates using Cardano's formula.

    Builds the 3×3 covariance matrix in a single pass, then extracts eigenvalues
    via the closed-form trigonometric solution for symmetric 3×3 matrices.
    No external dependencies (no NumPy).

    Returns sorted (descending) eigenvalues λ1 ≥ λ2 ≥ λ3 ≥ 0, or None if < 3 voxels.
    """
    n = len(voxels_set)
    if n < 3:
        return None

    # Compute centroid
    sx = sy = sz = 0.0
    for x, y, z in voxels_set:
        sx += x; sy += y; sz += z
    cx, cy, cz = sx / n, sy / n, sz / n

    # Build upper-triangle of 3×3 covariance matrix in one pass
    c00 = c01 = c02 = c11 = c12 = c22 = 0.0
    for x, y, z in voxels_set:
        dx, dy, dz = x - cx, y - cy, z - cz
        c00 += dx * dx
        c01 += dx * dy
        c02 += dx * dz
        c11 += dy * dy
        c12 += dy * dz
        c22 += dz * dz
    c00 /= n; c01 /= n; c02 /= n
    c11 /= n; c12 /= n; c22 /= n

    # Cardano's trigonometric solution for eigenvalues of a 3×3 symmetric matrix.
    # Characteristic polynomial: λ³ - p·λ² + q·λ - r = 0
    p = c00 + c11 + c22  # trace
    q = c00*c11 + c00*c22 + c11*c22 - c01*c01 - c02*c02 - c12*c12  # sum of 2×2 minors
    r = (c00*c11*c22 + 2*c01*c02*c12
         - c00*c12*c12 - c11*c02*c02 - c22*c01*c01)  # determinant

    disc = p*p - 3*q  # discriminant term
    if disc < 1e-15:
        # Near-isotropic: all eigenvalues approximately equal
        v = max(p / 3, 0.0)
        return (v, v, v)

    sqrt_disc = math.sqrt(disc)
    # Argument for arccos, clamped to [-1, 1] for numerical safety
    arg = (2*p*p*p - 9*p*q + 27*r) / (2 * disc * sqrt_disc)
    arg = max(-1.0, min(1.0, arg))
    theta = math.acos(arg) / 3

    # Three roots via trigonometric form
    two_sqrt = 2 * sqrt_disc / 3
    base = p / 3
    l0 = base + two_sqrt * math.cos(theta)
    l1 = base + two_sqrt * math.cos(theta - 2*math.pi/3)
    l2 = base + two_sqrt * math.cos(theta - 4*math.pi/3)

    # Sort descending, clamp non-negative
    eigvals = sorted([max(l0, 0.0), max(l1, 0.0), max(l2, 0.0)], reverse=True)
    return (eigvals[0], eigvals[1], eigvals[2])

# Modified to take eigenvalues as input
def _calculate_anisotropy_index(eigvals: Optional[Tuple[float, float, float]]) -> float:
    """
    Calculates anisotropy index from PCA eigenvalues (λ1 ≥ λ2 ≥ λ3).
    AI = 0.0 for isotropic, approaches 1.0 for elongated/flat shapes.
    """
    if eigvals is None or len(eigvals) != 3:
        return 0.0 # Default to isotropic if no valid eigenvalues

    lambda1, lambda2, lambda3 = eigvals

    if lambda1 < 1e-9: # If largest eigenvalue is effectively zero
        return 0.0  # Degenerate shape
    
    anisotropy = 1.0 - (lambda3 / (lambda1 + 1e-9)) # Epsilon for safety
    return max(0.0, min(anisotropy, 1.0)) # Clamp to [0,1]

def _calculate_shape_form_index(eigvals: Optional[Tuple[float, float, float]]) -> float:
    """Calculates shape form: -1 (prolate/rod) to +1 (oblate/pancake).
    Assumes eigvals are sorted: λ1 ≥ λ2 ≥ λ3.
    Returns 0.0 if isotropic or eigenvalues are invalid.
    """
    if eigvals is None or len(eigvals) != 3:
        return 0.0

    lambda1, lambda2, lambda3 = eigvals

    # Check for isotropy (denominator λ1 - λ3 would be close to 0)
    # Also, if anisotropy_index itself is very low, form is not well-defined.
    # We can use a threshold on (lambda1 - lambda3) or on anisotropy_index if calculated first.
    denominator = lambda1 - lambda3
    if denominator < 1e-9: # Effectively isotropic or too flat/thin for this formula
        return 0.0 

    # Formula: (2 * λ2 - λ1 - λ3) / (λ1 - λ3)
    form_index = (2 * lambda2 - lambda1 - lambda3) / denominator
    return max(-1.0, min(form_index, 1.0)) # Clamp to [-1, 1]

# --- Master Analysis Function --- 
def analyze_shape_features(voxels_set: Set[Tuple[int, int, int]], 
                           grid_size_tuple: Tuple[int, int, int], 
                           input_sfs: Optional[ShapeFeatureSet] = None) -> ShapeFeatureSet:
    """Calculates all (or as many as implemented) features for a given shape.
       For non-geometric or very complex features, it uses values from input_sfs if provided.
    """
    actual_features = {}

    # Directly calculated geometric features
    actual_features["voxel_count"] = len(voxels_set)
    actual_features["grid_size"] = list(grid_size_tuple) # Ensure it's a list for Pydantic/JSON
    actual_features["surface_area"] = _calculate_surface_area(voxels_set, grid_size_tuple)
    actual_features["symmetry_score"] = _calculate_symmetry_score(voxels_set)
    actual_features["compactness_score"] = _calculate_compactness_score(voxels_set, grid_size_tuple)
    actual_features["branching_factor"] = _calculate_branching_factor(voxels_set, grid_size_tuple)
    actual_features["circuit_rank"] = _calculate_circuit_rank(voxels_set, grid_size_tuple)
    actual_features["betti_1"] = _calculate_betti_1(voxels_set, grid_size_tuple)
    actual_features["number_of_components"] = _count_components(voxels_set, grid_size_tuple)
    actual_features["planarity_score"] = _calculate_planarity_score(voxels_set, grid_size_tuple)
    actual_features["bounding_box_ratio"] = _calculate_bounding_box_ratio(voxels_set)
    actual_features["center_of_mass_offset"] = _calculate_center_of_mass_offset(voxels_set, grid_size_tuple)
    actual_features["dominant_axis"] = _calculate_dominant_axis(voxels_set)
    actual_features["largest_component_ratio"] = _calculate_largest_component_ratio(voxels_set, grid_size_tuple)
    
    # PCA-based features
    pca_eigenvalues = _get_pca_eigenvalues(voxels_set)
    actual_features["anisotropy_index"] = _calculate_anisotropy_index(pca_eigenvalues)
    actual_features["shape_form_index"] = _calculate_shape_form_index(pca_eigenvalues)

    # Chirality detection
    voxels_list = [list(v) for v in voxels_set]
    mirror_result = generate_mirror_reflection(voxels_list, grid_size_tuple)
    actual_features["is_chiral"] = mirror_result is not None

    # Carry over task-level descriptors (not computed from geometry)
    if input_sfs:
        for key in ("distractor_similarity", "mental_operation_type"):
            if hasattr(input_sfs, key):
                value = getattr(input_sfs, key)
                if value is not None:
                    actual_features[key] = value
    
    # Create the ShapeFeatureSet instance. Pydantic will use defaults for any missing fields.
    # Need to filter actual_features to only include valid field names for ShapeFeatureSet
    valid_sfs_keys = ShapeFeatureSet.model_fields.keys()
    filtered_actual_features = {k: v for k, v in actual_features.items() if k in valid_sfs_keys and v is not None}
    
    # Ensure critical fields have defaults if somehow missed and not in input_sfs
    # (Pydantic model should handle this with its defaults, but good for robustness)
    if 'grid_size' not in filtered_actual_features: # Should always be there
        filtered_actual_features['grid_size'] = list(grid_size_tuple)
    if 'voxel_count' not in filtered_actual_features: # Should always be there
        filtered_actual_features['voxel_count'] = len(voxels_set)

    # Some features from input_sfs might have been calculated, so use calculated ones primarily.
    # The filtered_actual_features already prefers calculated values if they were set.

    return ShapeFeatureSet(**filtered_actual_features)


def generate_mirror_reflection(voxels: List[List[int]], grid_size: Tuple[int, int, int],
                                axis: str = None) -> Optional[List[List[int]]]:
    """Generate a mirror reflection of a voxel shape.

    Args:
        voxels: List of [x, y, z] voxel coordinates.
        grid_size: (gx, gy, gz) grid dimensions.
        axis: 'x', 'y', or 'z'. If None, picks randomly.

    Returns:
        List of reflected [x, y, z] coordinates, or None if the shape is
        achiral (mirror is identical to original) on all axes.
    """
    axes_to_try = [axis] if axis else random.sample(['x', 'y', 'z'], 3)
    gx, gy, gz = grid_size
    original_set = frozenset(tuple(v) for v in voxels)

    for ax in axes_to_try:
        reflected = []
        for v in voxels:
            x, y, z = v[0], v[1], v[2]
            if ax == 'x':
                reflected.append([gx - 1 - x, y, z])
            elif ax == 'y':
                reflected.append([x, gy - 1 - y, z])
            else:
                reflected.append([x, y, gz - 1 - z])

        reflected_set = frozenset(tuple(v) for v in reflected)
        if reflected_set != original_set:
            return reflected

    return None  # Shape is achiral on all axes


# --- Part-Permuted Distractor Generation (Skill 7: Configural Binding) ---

def _find_branch_points(voxels_set: Set[Tuple[int, int, int]],
                        grid_size: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """Return voxels with degree >= 3 (branch/junction points)."""
    branch_points = set()
    for voxel in voxels_set:
        neighbor_count = sum(1 for n in _get_neighbors(voxel, grid_size)
                            if n in voxels_set)
        if neighbor_count >= 3:
            branch_points.add(voxel)
    return branch_points


def _flood_fill_component(start: Tuple[int, int, int],
                          voxels_set: Set[Tuple[int, int, int]],
                          grid_size: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """BFS flood-fill returning all connected voxels from start within voxels_set."""
    component = set()
    q = deque([start])
    component.add(start)
    while q:
        curr = q.popleft()
        for neighbor in _get_neighbors(curr, grid_size):
            if neighbor in voxels_set and neighbor not in component:
                component.add(neighbor)
                q.append(neighbor)
    return component


def _decompose_shape(voxels_set: Set[Tuple[int, int, int]],
                     grid_size: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
    """Decompose a shape into segments by cutting at branch points.

    Returns None if the shape has no branch points or fewer than 2 segments.
    Otherwise returns:
        {
            "branch_points": set of branch point voxels,
            "segments": [
                {
                    "voxels": set of voxel tuples in this segment,
                    "attachments": [(branch_point, direction_from_bp), ...]
                },
                ...
            ]
        }
    """
    branch_points = _find_branch_points(voxels_set, grid_size)
    if not branch_points:
        return None

    # Remove branch points to isolate segments
    non_branch = voxels_set - branch_points
    if not non_branch:
        return None  # All voxels are high-degree (compact blob)

    # Find connected components among non-branch voxels
    visited: Set[Tuple[int, int, int]] = set()
    segments = []
    for voxel in non_branch:
        if voxel not in visited:
            component = _flood_fill_component(voxel, non_branch, grid_size)
            visited |= component

            # Find attachment info: which branch points is this segment adjacent to?
            attachments = []
            for bp in branch_points:
                for seg_voxel in component:
                    diff = (seg_voxel[0] - bp[0],
                            seg_voxel[1] - bp[1],
                            seg_voxel[2] - bp[2])
                    # Check if face-adjacent (Manhattan distance 1)
                    if abs(diff[0]) + abs(diff[1]) + abs(diff[2]) == 1:
                        attachments.append((bp, diff))
                        break  # One attachment per branch point per segment

            segments.append({
                "voxels": component,
                "attachments": attachments,
            })

    if len(segments) < 2:
        return None

    return {
        "branch_points": branch_points,
        "segments": segments,
    }


def _get_rotation_matrix(from_dir: Tuple[int, int, int],
                         to_dir: Tuple[int, int, int]) -> Optional[List[List[int]]]:
    """Get a 3x3 integer rotation matrix mapping from_dir to to_dir.

    Both from_dir and to_dir must be unit face directions like (1,0,0), (0,-1,0), etc.
    Returns None if from_dir == to_dir (identity, no rotation needed).
    """
    if from_dir == to_dir:
        return None  # Identity

    fx, fy, fz = from_dir
    tx, ty, tz = to_dir

    # If opposite directions, rotate 180 degrees around any perpendicular axis
    if (fx + tx, fy + ty, fz + tz) == (0, 0, 0):
        # Find a perpendicular axis
        if fx != 0:  # from is along x, rotate 180 around y
            return [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
        elif fy != 0:  # from is along y, rotate 180 around x
            return [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        else:  # from is along z, rotate 180 around x
            return [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

    # For 90-degree rotations, compute via cross product
    # cross = from x to gives rotation axis, then build rotation matrix
    cx = fy * tz - fz * ty
    cy = fz * tx - fx * tz
    cz = fx * ty - fy * tx

    # The rotation matrix for 90 degrees around axis (cx, cy, cz):
    # R = I * cos(90) + (1-cos(90)) * axis_outer + sin(90) * skew(axis)
    # For 90 degrees: cos=0, sin=1, so R = axis_outer + skew(axis)
    # But since our axes are unit vectors along grid axes, we can construct directly:
    # R maps from_dir -> to_dir and axis -> axis (axis is perpendicular to both)
    # Build R column by column: R @ from = to, R @ axis = axis, R @ (axis x from) = to x axis

    # Simpler: directly construct the matrix
    # Column 1: where (1,0,0) maps to, etc.
    # We know R @ from_dir = to_dir and R @ cross = cross
    # Third basis vector: from x cross (or to x cross)
    # from_dir, cross, third form an orthonormal basis
    third_from = (fy * cz - fz * cy, fz * cx - fx * cz, fx * cy - fy * cx)
    third_to = (ty * cz - tz * cy, tz * cx - tx * cz, tx * cy - ty * cx)

    # R maps: from_dir -> to_dir, cross -> cross, third_from -> third_to
    # Build R = [to | cross | third_to] @ [from | cross | third_from]^-1
    # Since the basis vectors are orthonormal and integer, the inverse is the transpose
    F = [[fx, cx, third_from[0]],
         [fy, cy, third_from[1]],
         [fz, cz, third_from[2]]]
    T = [[tx, cx, third_to[0]],
         [ty, cy, third_to[1]],
         [tz, cz, third_to[2]]]

    # R = T @ F^T (since F is orthogonal, F^-1 = F^T)
    R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            R[i][j] = sum(T[i][k] * F[j][k] for k in range(3))

    return R


def _apply_rotation(voxels: Set[Tuple[int, int, int]],
                    rotation_matrix: List[List[int]],
                    pivot: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
    """Rotate voxels around a pivot point using an integer rotation matrix."""
    R = rotation_matrix
    px, py, pz = pivot
    result = set()
    for vx, vy, vz in voxels:
        # Translate to origin
        dx, dy, dz = vx - px, vy - py, vz - pz
        # Rotate
        rx = R[0][0] * dx + R[0][1] * dy + R[0][2] * dz
        ry = R[1][0] * dx + R[1][1] * dy + R[1][2] * dz
        rz = R[2][0] * dx + R[2][1] * dy + R[2][2] * dz
        # Translate back
        result.add((round(rx + px), round(ry + py), round(rz + pz)))
    return result


def _voxels_in_bounds(voxels: Set[Tuple[int, int, int]],
                      grid_size: Tuple[int, int, int]) -> bool:
    """Check all voxels are within grid bounds."""
    gx, gy, gz = grid_size
    return all(0 <= x < gx and 0 <= y < gy and 0 <= z < gz
               for x, y, z in voxels)


def _generate_cube_rotations() -> List[List[List[int]]]:
    """Return the 24 proper rotation matrices of the cube."""
    def compose(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Rx = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    Ry = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    Rz = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

    def as_key(m):
        return tuple(tuple(row) for row in m)

    seen = {as_key(I): I}
    frontier = [I]
    while frontier:
        nxt = []
        for m in frontier:
            for gen in (Rx, Ry, Rz):
                new_m = compose(gen, m)
                key = as_key(new_m)
                if key not in seen:
                    seen[key] = new_m
                    nxt.append(new_m)
        frontier = nxt
    return list(seen.values())


_CUBE_ROTATIONS = _generate_cube_rotations()


def canonical_voxel_form(voxels) -> Tuple[Tuple[int, int, int], ...]:
    """Rotation-invariant canonical form of a voxel set.

    Enumerates the 24 proper rotations of the cube; for each rotation,
    translates the result so the min corner is at the origin and sorts
    the voxels lexicographically. Returns the lex-smallest such tuple.
    Two voxel sets that are rotations of each other produce the same
    canonical form, so this is suitable for de-duplicating shapes that
    would appear visually identical after random rotation at display time.
    """
    pts = [tuple(v) for v in voxels]
    if not pts:
        return tuple()
    best = None
    for R in _CUBE_ROTATIONS:
        rotated = [
            (R[0][0] * x + R[0][1] * y + R[0][2] * z,
             R[1][0] * x + R[1][1] * y + R[1][2] * z,
             R[2][0] * x + R[2][1] * y + R[2][2] * z)
            for (x, y, z) in pts
        ]
        mn_x = min(v[0] for v in rotated)
        mn_y = min(v[1] for v in rotated)
        mn_z = min(v[2] for v in rotated)
        translated = tuple(sorted((v[0] - mn_x, v[1] - mn_y, v[2] - mn_z) for v in rotated))
        if best is None or translated < best:
            best = translated
    return best


def keep_largest_component(voxels, grid_size: Tuple[int, int, int]):
    """Return a voxel set containing only the largest connected component.

    The shape generation pipeline enforces single-component output in several
    places, but the enforcement is not airtight: the skeleton fallback accepts
    the last attempt even if validation failed, and voxel mutations can split
    a shape at a cut vertex. This helper is a defensive post-step: if a shape
    has multiple components, discard everything except the largest. Accepts
    either a set of tuples or a list of lists; returns a set of tuples.
    """
    if isinstance(voxels, list):
        vset: Set[Tuple[int, int, int]] = {tuple(v) for v in voxels}
    else:
        vset = set(voxels)
    if not vset or _count_components(vset, grid_size) <= 1:
        return vset

    seen: Set[Tuple[int, int, int]] = set()
    largest: Set[Tuple[int, int, int]] = set()
    for v in list(vset):
        if v in seen:
            continue
        comp = _flood_fill_component(v, vset, grid_size)
        seen |= comp
        if len(comp) > len(largest):
            largest = comp
    return largest


def nudge_voxels_until_unique(voxels: List[List[int]],
                              seen_canonicals: Set,
                              grid_size: Tuple[int, int, int],
                              max_iterations: int = 5000) -> Optional[List[List[int]]]:
    """Find a connectivity-preserving mutation of *voxels* whose canonical
    form is not in *seen_canonicals*.

    Implementation: breadth-first search over 1-voxel add/remove mutations.
    Each frontier shape expands into all of its valid neighbors (add a voxel
    adjacent to an existing one, OR remove a voxel whose removal does not
    change the component count). The first neighbor whose canonical form is
    not in *seen_canonicals* and has not been visited is returned. Since
    *seen_canonicals* contains only the at-most-three already-accepted shapes
    for the current trial and the canonical-form neighborhood graph is large
    (millions of reachable forms within a few BFS layers on any reasonable
    grid), success on or before the third layer is essentially guaranteed.

    The BFS is bounded by *max_iterations* nodes expanded as a defensive cap;
    in practice the function returns within a handful of expansions. Returns
    None only if the search exhausts the entire connected sub-graph reachable
    from the start within the budget without finding a non-blocked form, an
    outcome that does not occur on grids large enough to host meaningful
    shapes.

    Used as the guaranteed last-ditch distractor fallback after the upstream
    tier-aware generator's 10 attempts (with archetype swap) all collide.
    """
    import random as _random
    from collections import deque

    start_set: Set[Tuple[int, int, int]] = {tuple(v) for v in voxels}
    if not start_set:
        return None
    gx, gy, gz = grid_size
    # Preserve the component count of the starting shape so that removals
    # don't introduce orphan voxels and adds don't merge separate components.
    target_n_components = _count_components(start_set, grid_size)

    DIRECTIONS = ((1, 0, 0), (-1, 0, 0),
                  (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1))

    def neighbors(vset: Set[Tuple[int, int, int]]):
        """Yield every connectivity-preserving 1-voxel mutation of *vset*.
        Order is randomized so successive callers get different solutions."""
        # ADD candidates: in-bounds empty cells adjacent to an existing voxel
        adds: Set[Tuple[int, int, int]] = set()
        for v in vset:
            for dx, dy, dz in DIRECTIONS:
                cand = (v[0] + dx, v[1] + dy, v[2] + dz)
                if (0 <= cand[0] < gx and 0 <= cand[1] < gy and 0 <= cand[2] < gz
                        and cand not in vset):
                    adds.add(cand)
        adds_list = list(adds)
        _random.shuffle(adds_list)
        for cand in adds_list:
            new_set = vset | {cand}
            if _count_components(new_set, grid_size) == target_n_components:
                yield new_set

        # REMOVE candidates: any voxel whose removal does not change the
        # component count. Empty / underflow shapes are skipped to avoid
        # collapsing the distractor below a useful size.
        if len(vset) > 1:
            rems = list(vset)
            _random.shuffle(rems)
            for cand in rems:
                trial = vset - {cand}
                if not trial:
                    continue
                if _count_components(trial, grid_size) == target_n_components:
                    yield trial

    # BFS over canonical forms. Visiting by canonical form ensures we don't
    # waste expansions on rotational copies of states we've already seen.
    start_canon = canonical_voxel_form(list(start_set))
    visited: Set = {start_canon}
    frontier: deque = deque([start_set])
    expansions = 0

    while frontier and expansions < max_iterations:
        current = frontier.popleft()
        expansions += 1
        for nxt in neighbors(current):
            canon = canonical_voxel_form(list(nxt))
            if canon in visited:
                continue
            visited.add(canon)
            if canon not in seen_canonicals:
                return [list(v) for v in nxt]
            frontier.append(nxt)

    return None


def generate_part_permuted_distractor(
    voxels: List[List[int]],
    grid_size: Tuple[int, int, int],
    max_attempts: int = 20,
) -> Optional[List[List[int]]]:
    """Generate a part-permuted distractor: same structural segments, different arrangement.

    Decomposes the shape at branch points, then permutes which direction each
    segment extends from its branch point.

    Args:
        voxels: List of [x, y, z] voxel coordinates (target shape).
        grid_size: (gx, gy, gz) grid dimensions.
        max_attempts: Max permutations to try before giving up.

    Returns:
        List of [x, y, z] coordinates for the permuted shape, or None if
        the shape cannot be decomposed or no valid permutation was found.
    """
    from itertools import permutations

    voxels_set = set(tuple(v) for v in voxels)
    original_canonical = frozenset(voxels_set)

    decomposition = _decompose_shape(voxels_set, grid_size)
    if decomposition is None:
        return None

    branch_points = decomposition["branch_points"]
    segments = decomposition["segments"]

    # Find the branch point with the most attached segments
    bp_segment_map: Dict[Tuple[int, int, int], List[int]] = {}
    for seg_idx, seg in enumerate(segments):
        for bp, direction in seg["attachments"]:
            if bp not in bp_segment_map:
                bp_segment_map[bp] = []
            bp_segment_map[bp].append(seg_idx)

    # Pick the branch point with the most segments
    best_bp = max(bp_segment_map.keys(), key=lambda bp: len(bp_segment_map[bp]))
    attached_seg_indices = bp_segment_map[best_bp]

    if len(attached_seg_indices) < 2:
        return None  # Need at least 2 segments to permute

    # Collect the segments and their attachment directions from best_bp
    seg_infos = []
    for seg_idx in attached_seg_indices:
        seg = segments[seg_idx]
        # Find the attachment direction for this specific branch point
        for bp, direction in seg["attachments"]:
            if bp == best_bp:
                seg_infos.append({
                    "idx": seg_idx,
                    "voxels": seg["voxels"],
                    "direction": direction,
                })
                break

    original_directions = [s["direction"] for s in seg_infos]

    # Segments NOT attached to best_bp stay in place
    static_seg_indices = set(range(len(segments))) - set(attached_seg_indices)
    static_voxels = set()
    for idx in static_seg_indices:
        static_voxels |= segments[idx]["voxels"]

    # Try permutations of the directions
    attempts = 0
    for perm in permutations(range(len(seg_infos))):
        if attempts >= max_attempts:
            break
        attempts += 1

        # Skip identity permutation
        if all(perm[i] == i for i in range(len(perm))):
            continue

        # Build new voxel set: branch points + static segments + permuted segments
        new_voxels = set(branch_points) | static_voxels
        valid = True

        for i, target_slot in enumerate(perm):
            seg = seg_infos[i]
            target_dir = seg_infos[target_slot]["direction"]
            source_dir = seg["direction"]

            if source_dir == target_dir:
                # No rotation needed, keep segment as-is
                new_segment_voxels = seg["voxels"]
            else:
                R = _get_rotation_matrix(source_dir, target_dir)
                if R is None:
                    new_segment_voxels = seg["voxels"]
                else:
                    new_segment_voxels = _apply_rotation(seg["voxels"], R, best_bp)

            # Check bounds
            if not _voxels_in_bounds(new_segment_voxels, grid_size):
                valid = False
                break

            # Check overlap with already-placed voxels
            if new_segment_voxels & new_voxels:
                valid = False
                break

            new_voxels |= new_segment_voxels

        if not valid:
            continue

        # Check it's actually different from the original
        if frozenset(new_voxels) == original_canonical:
            continue

        # Check connectivity
        if _count_components(new_voxels, grid_size) != 1:
            continue

        # Success — return the permuted shape
        return [[x, y, z] for x, y, z in new_voxels]

    return None  # No valid permutation found


def generate_shape_from_features(features: ShapeFeatureSet) -> dict:
    """
    This is the original simple generator. Kept for compatibility or simple use cases.
    """
    voxels_list: List[Tuple[int, int, int]] = []
    grid_size = features.grid_size or (7, 7, 7)
    max_voxels = features.voxel_count
    
    current = (grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2)
    voxels_list.append(current)
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    branching_factor_target = features.branching_factor if features.branching_factor is not None else 0
    
    voxels_set = set(voxels_list)

    while len(voxels_set) < max_voxels:
        base_candidates = list(voxels_set)
        if not base_candidates:
            break # Should not happen if started with one voxel

        # Simple branching influence: pick voxels with fewer connections if low branching
        if branching_factor_target < 2 and len(voxels_set) > 1:
            base_candidates.sort(key=lambda v: sum(1 for d in directions if tuple(v[i]+d[i] for i in range(3)) in voxels_set))
            base = base_candidates[0]
        else:
            base = random.choice(base_candidates)
            
        random.shuffle(directions)
        added_voxel = False
        for dx, dy, dz in directions:
            new_voxel = (base[0] + dx, base[1] + dy, base[2] + dz)
            if new_voxel not in voxels_set and all(0 <= new_voxel[i] < grid_size[i] for i in range(3)):
                voxels_set.add(new_voxel)
                added_voxel = True
                break
        if not added_voxel:
            # If no valid neighbor found from current base, try another random voxel from existing set
            # This prevents getting stuck easily if the chosen base is fully surrounded by other voxels or grid boundaries
            if len(voxels_set) < max_voxels and len(voxels_set) > 0:
                non_stuck_base = random.choice(list(voxels_set))
                for dx, dy, dz in directions:
                    new_voxel = (non_stuck_base[0] + dx, non_stuck_base[1] + dy, non_stuck_base[2] + dz)
                    if new_voxel not in voxels_set and all(0 <= new_voxel[i] < grid_size[i] for i in range(3)):
                        voxels_set.add(new_voxel)
                        break # Add one and restart loop
            else:
                 break # Truly stuck or max_voxels reached

    # Return shape object
    analyzed_sfs = analyze_shape_features(voxels_set, grid_size, input_sfs=features)

    return {
        "voxels": list(voxels_set),
        "grid_size": list(grid_size),
        "features": analyzed_sfs.to_dict()
    }


def generate_shape_advanced(target_features: ShapeFeatureSet, max_iterations=2000) -> Dict[str, Any]:
    """
    LEGACY greedy generator. The active production path is generate_shape_skeleton()
    in skeleton_generation.py (the "skeleton-first" method described in the paper).
    Retained for reference, comparison studies, and fidelity benchmarking.
    Not wired into the game UI.

    Generates a shape attempting to meet targets in ShapeFeatureSet.
    Focuses on: voxel_count, grid_size, branching_factor, compactness_score, number_of_components.
    """
    grid_size = target_features.grid_size
    target_voxel_count = target_features.voxel_count

    # --- Target values from the ShapeFeatureSet --- 
    # (using .get() with a default if the feature might not be set by cognitive_mapping)
    target_bf = target_features.branching_factor if target_features.branching_factor is not None else 1.0 # Target for branching points
    # Target compactness: 0 (sparse) to 1 (dense). Use provided or default to medium.
    target_compactness = target_features.compactness_score if target_features.compactness_score is not None else 0.5
    target_components = target_features.number_of_components if target_features.number_of_components is not None else 1
    target_planarity = target_features.planarity_score if target_features.planarity_score is not None else 0.7 # Default towards somewhat planar
    target_bbox_ratio = target_features.bounding_box_ratio if target_features.bounding_box_ratio is not None else 2.0 # Default towards somewhat elongated
    target_dominant_axis = target_features.dominant_axis if target_features.dominant_axis is not None else 'balanced'
    target_anisotropy_index = target_features.anisotropy_index if target_features.anisotropy_index is not None else 0.3 # Default to slightly anisotropic/interesting
    target_shape_form_index = target_features.shape_form_index if target_features.shape_form_index is not None else 0.0 # Default to intermediate form
    # Graph-cycle target: prefer the explicit circuit_rank target; fall back to
    # betti_1 for older callers that only set the topological target.
    if target_features.circuit_rank is not None:
        target_circuit_rank = target_features.circuit_rank
    elif target_features.betti_1 is not None:
        target_circuit_rank = target_features.betti_1
    else:
        target_circuit_rank = 0

    voxels_set: Set[Tuple[int, int, int]] = set()
    
    # Start with a seed voxel (or multiple if target_components > 1 initially)
    # For simplicity, start with one component at the center, then try to split if needed (harder).
    # Or, if target_components > 1, seed multiple points.
    if target_components > 1 and target_voxel_count >= target_components:
        for i in range(target_components):
             # Try to place seeds somewhat apart if possible
            tries = 0
            while tries < 10:
                seed = (random.randint(0, grid_size[0]-1), 
                        random.randint(0, grid_size[1]-1), 
                        random.randint(0, grid_size[2]-1))
                if seed not in voxels_set: # Basic check for overlap with other seeds
                    # More robust: check min distance from other seeds
                    voxels_set.add(seed)
                    break
                tries +=1
            if len(voxels_set) <= i: # Failed to place a distinct seed
                 # Fallback for safety, could lead to fewer components than desired
                 voxels_set.add((grid_size[0]//2 + i, grid_size[1]//2, grid_size[2]//2 % grid_size[2])) 
    else:
        voxels_set.add((grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2))
    
    if not voxels_set: # Ensure at least one voxel if all seeding failed
        voxels_set.add((grid_size[0] // 2, grid_size[1] // 2, grid_size[2] // 2))


    # Iterative refinement/growth
    # The core idea: explore potential next voxels and pick the one that best matches targets.
    for iteration in range(max_iterations):
        if len(voxels_set) >= target_voxel_count:
            current_components = _count_components(voxels_set, grid_size)
            if current_components == target_components: # Or is acceptable
                break # Target met
            # If component count is wrong, may need more complex ops than just adding

        candidate_voxels: List[Tuple[Tuple[int, int, int], float]] = [] # (voxel, score)

        # Consider adding voxels
        potential_add_sites = set()
        for voxel in voxels_set:
            for neighbor in _get_neighbors(voxel, grid_size):
                if neighbor not in voxels_set:
                    potential_add_sites.add(neighbor)
        
        if not potential_add_sites and len(voxels_set) < target_voxel_count:
             # No place to add, shape is likely filling its space or is fragmented oddly
             # Try to find any valid empty spot if desperate, or break
            all_grid_points = set((x,y,z) for x in range(grid_size[0]) for y in range(grid_size[1]) for z in range(grid_size[2]))
            empty_points = list(all_grid_points - voxels_set)
            if empty_points and len(voxels_set) > 0: # Add anywhere if stuck and space available
                voxels_set.add(random.choice(empty_points))
                continue
            else:
                break 

        for cand_add in potential_add_sites:
            temp_voxels = voxels_set.copy()
            temp_voxels.add(cand_add)
            score = 0

            # Score component count (strong preference for target)
            current_num_comp = _count_components(temp_voxels, grid_size)
            if current_num_comp == target_components:
                score += 1000
            else:
                # Penalize if it moves away from target (e.g. connects components when we want separate)
                score -= abs(current_num_comp - target_components) * 500 
            
            # Score compactness (how close to target_compactness)
            c_compactness = _calculate_compactness_score(temp_voxels, grid_size)
            score -= abs(c_compactness - target_compactness) * 100 # Lower is better

            # Score branching factor
            c_bf = _calculate_branching_factor(temp_voxels, grid_size)
            score -= abs(c_bf - target_bf) * 50 # Lower is better
            
            # Score planarity
            c_planarity = _calculate_planarity_score(temp_voxels, grid_size)
            # Score difference from target planarity (1.0 is perfectly flat)
            score -= abs(c_planarity - target_planarity) * 75 # Weight for planarity, adjust as needed
            
            # Score bounding_box_ratio - REMOVED from active scoring
            # c_bbox_ratio = _calculate_bounding_box_ratio(temp_voxels)
            # score -= abs(c_bbox_ratio - target_bbox_ratio) * 40 # Weight for bbox_ratio
            
            # Score dominant_axis - REMOVED from active scoring
            # current_dominant_axis_score_weight = 10 
            # if len(temp_voxels) > 2: 
            #     current_dims = _calculate_bounding_box_dimensions(temp_voxels)
            #     dx, dy, dz = current_dims
            #     max_dim_for_norm = max(1, dx, dy, dz)
            #     norm_dx, norm_dy, norm_dz = dx/max_dim_for_norm, dy/max_dim_for_norm, dz/max_dim_for_norm
            #     axis_score = 0
            #     if target_dominant_axis == 'x':
            #         if dx > dy and dx > dz: axis_score += (current_dominant_axis_score_weight * 0.75)
            #         if dy > dx or dz > dx: axis_score -= (current_dominant_axis_score_weight * 0.5)
            #         if dy + dz > 0: axis_score += (norm_dx - (norm_dy + norm_dz)/2) * current_dominant_axis_score_weight 
            #         else: axis_score += norm_dx * (current_dominant_axis_score_weight * 0.5)
            #     elif target_dominant_axis == 'y':
            #         if dy > dx and dy > dz: axis_score += (current_dominant_axis_score_weight * 0.75)
            #         if dx > dy or dz > dy: axis_score -= (current_dominant_axis_score_weight * 0.5)
            #         if dx + dz > 0: axis_score += (norm_dy - (norm_dx + norm_dz)/2) * current_dominant_axis_score_weight
            #         else: axis_score += norm_dy * (current_dominant_axis_score_weight * 0.5)
            #     elif target_dominant_axis == 'z':
            #         if dz > dx and dz > dy: axis_score += (current_dominant_axis_score_weight * 0.75)
            #         if dx > dz or dy > dz: axis_score -= (current_dominant_axis_score_weight * 0.5)
            #         if dx + dy > 0: axis_score += (norm_dz - (norm_dx + norm_dy)/2) * current_dominant_axis_score_weight
            #         else: axis_score += norm_dz * (current_dominant_axis_score_weight * 0.5)
            #     elif target_dominant_axis == 'balanced':
            #         if dx > 0 and dy > 0 and dz > 0:
            #             dim_mean = (norm_dx + norm_dy + norm_dz) / 3.0
            #             variance = ((norm_dx - dim_mean)**2 + (norm_dy - dim_mean)**2 + (norm_dz - dim_mean)**2) / 3.0
            #             axis_score -= variance * (current_dominant_axis_score_weight * 4) 
            #         else: 
            #             axis_score -= (current_dominant_axis_score_weight * 0.5)
            #     score += axis_score
            
            # Score anisotropy_index
            c_pca_eigenvalues = _get_pca_eigenvalues(temp_voxels)
            c_anisotropy = _calculate_anisotropy_index(c_pca_eigenvalues)
            score -= abs(c_anisotropy - target_anisotropy_index) * 150 

            # Score shape_form_index, especially if the shape is meant to be anisotropic
            if target_anisotropy_index > 0.3: # Only strongly consider form if anisotropy is targeted
                c_shape_form = _calculate_shape_form_index(c_pca_eigenvalues)
                # Penalize deviation from target form. Weight might be less than anisotropy itself.
                score -= abs(c_shape_form - target_shape_form_index) * 75 

            # Score graph-cycle structure as circuit rank μ (the quantity the
            # generator controls; cheap O(|V|) — cubical β₁ is measured
            # post-generation, not in this scoring loop).
            c_circuit_rank = _calculate_circuit_rank(temp_voxels, grid_size)
            score -= abs(c_circuit_rank - target_circuit_rank) * 75

            # Small bonus for just getting closer to voxel count
            score += (len(temp_voxels) / target_voxel_count) * 10

            candidate_voxels.append((cand_add, score))

        if not candidate_voxels:
            if len(voxels_set) < target_voxel_count:
                # This case means no potential_add_sites were found or scored, very stuck
                # Try to add a random voxel not connected if below target count and stuck
                all_grid_points = set((x,y,z) for x in range(grid_size[0]) for y in range(grid_size[1]) for z in range(grid_size[2]))
                available_random_points = list(all_grid_points - voxels_set)
                if available_random_points:
                    voxels_set.add(random.choice(available_random_points))
                    continue # Try next iteration
            break # No good candidates, or truly stuck

        candidate_voxels.sort(key=lambda x: x[1], reverse=True) # Best score first
        
        # Add the best candidate
        if candidate_voxels: # Ensure there is at least one candidate
            best_cand_voxel, best_score = candidate_voxels[0]
            voxels_set.add(best_cand_voxel)
        else:
            # If somehow no candidates are left, break
            break

    # Post-generation cleanup/adjustment (optional)
    # If too many voxels, remove some (e.g., those that least impact desired features)
    while len(voxels_set) > target_voxel_count:
        if not voxels_set: break
        # Simple removal: random voxel that doesn't disconnect if components=1
        removable_voxels = list(voxels_set)
        random.shuffle(removable_voxels)
        removed = False
        for rv in removable_voxels:
            temp_voxels = voxels_set.copy()
            temp_voxels.remove(rv)
            if not temp_voxels or _count_components(temp_voxels, grid_size) == target_components:
                voxels_set.remove(rv)
                removed = True
                break
        if not removed and voxels_set: # Fallback: just remove any voxel
             voxels_set.pop()

    # --- Prepare output --- 
    # The returned 'features' dict should ideally be the *actual* calculated features of the final shape.
    # For now, we return the input target_features, but add the actual voxel_count.
    analyzed_sfs = analyze_shape_features(voxels_set, grid_size, input_sfs=target_features)

    print(f"Advanced generation: Target voxels: {target_voxel_count}, Actual: {analyzed_sfs.voxel_count}")
    print(f"  Target compactness: {target_compactness:.2f}, Actual: {analyzed_sfs.compactness_score:.2f}")
    print(f"  Target branching: {target_bf}, Actual: {analyzed_sfs.branching_factor}")
    print(f"  Target components: {target_components}, Actual: {analyzed_sfs.number_of_components}")
    print(f"  Target planarity: {target_planarity:.2f}, Actual: {analyzed_sfs.planarity_score:.2f}")
    print(f"  Target BBox Ratio: {target_bbox_ratio:.2f}, Actual: {analyzed_sfs.bounding_box_ratio:.2f}")
    print(f"  Target Dominant Axis: {target_dominant_axis}, Actual: {analyzed_sfs.dominant_axis}")
    print(f"  Target Anisotropy Index: {target_anisotropy_index:.2f}, Actual: {analyzed_sfs.anisotropy_index:.2f}")
    print(f"  Target Shape Form Index: {target_shape_form_index:.2f}, Actual: {analyzed_sfs.shape_form_index:.2f}")
    print(f"  Target circuit rank: {target_circuit_rank}, Actual: {analyzed_sfs.circuit_rank} (measured β₁: {analyzed_sfs.betti_1})")

    return {
        "voxels": list(voxels_set),
        "grid_size": list(grid_size),
        "features": analyzed_sfs.to_dict()
    }

def generate_shape_heuristic_v1(features: ShapeFeatureSet) -> dict:
    """
    Generate a voxel-based shape using interpretable features, based on user's suggestion.
    Directly influences: voxel_count, planarity_score, branching_factor, compactness_score.
    - planarity_score (0.0 = mostly flat, higher = more 3D spread)
    - branching_factor (0 = more linear, higher = more arms/branches)
    - compactness_score (0.0 = sparse/less connected, 1.0 = more compact/connected)
    """
    # Extract values from ShapeFeatureSet, providing defaults if not set
    target_voxel_count = features.voxel_count
    planarity = features.planarity_score if features.planarity_score is not None else 0.5
    branching = features.branching_factor if features.branching_factor is not None else 1
    compactness = features.compactness_score if features.compactness_score is not None else 0.5
    grid_size_tuple = features.grid_size # Should always be present in ShapeFeatureSet

    center = [grid_size_tuple[0] // 2, grid_size_tuple[1] // 2, grid_size_tuple[2] // 2]
    
    # Use a set for efficient addition and checking
    voxels_set: Set[Tuple[int,int,int]] = set()
    if target_voxel_count > 0: # Ensure we add at least one if count > 0
        voxels_set.add(tuple(center))

    # Convert voxels_set to list when random.choice is needed and it's guaranteed not empty
    
    def get_local_neighbors(v: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        x, y, z = v
        return [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]

    for _ in range(target_voxel_count - len(voxels_set)): # Iterate for remaining voxels
        if not voxels_set: # Should not happen if target_voxel_count > 0
            break

        current_voxels_list = list(voxels_set) # Get a list copy for random.choice

        if random.random() < compactness and len(current_voxels_list) > 0:
            # Prefer attachment to existing voxels (dense growth)
            # Pick any existing voxel as base
            base = random.choice(current_voxels_list)
        elif len(current_voxels_list) > 0:
            # Occasionally use an early voxel for sparse growth (e.g. the first one)
            base = current_voxels_list[0] 
        else: # Should be unreachable if loop condition is correct
            continue

        possible_next_positions = get_local_neighbors(base)
        random.shuffle(possible_next_positions)
        
        added_this_step = False
        for pos_tuple in possible_next_positions:
            # Ensure pos is a tuple for set operations
            
            if pos_tuple not in voxels_set and all(0 <= pos_tuple[i] < grid_size_tuple[i] for i in range(3)):
                final_pos = list(pos_tuple) # Make it a list to modify z

                # Adjust z based on planarity
                # planarity = 0 means very flat (stick to center z)
                # planarity = 1 means full 3D freedom
                
                # Determine allowed z range based on planarity
                # Max deviation from center z: (grid_size_z / 2 - 0.5) * planarity
                # Example: grid_size_z=7, center_z=3. Max_dev = (3.5-0.5)*planarity = 3*planarity
                # If planarity=0, z_dev=0 (center_z only)
                # If planarity=0.5, z_dev=1.5 (center_z, center_z +/- 1)
                # If planarity=1, z_dev=3 (center_z +/- 3, i.e. full range 0 to 6)

                z_deviation_range = int(round((grid_size_tuple[2] / 2.0) * planarity))
                min_z_allowed = max(0, center[2] - z_deviation_range)
                max_z_allowed = min(grid_size_tuple[2] - 1, center[2] + z_deviation_range)
                
                # If the proposed z is outside this allowed range, try to clamp it or pick one within range.
                # For simplicity here, we'll just say if the *original* z of the neighbor is in range, it's fine.
                # A more direct planarity influence would be to pick Z from the allowed range.
                if not (min_z_allowed <= final_pos[2] <= max_z_allowed):
                    # If the neighbor's natural Z is too far, we might skip it or adjust it.
                    # For this version, let's adjust it to be within the allowed planar slice
                    # This is a strong planarity enforcement
                    if planarity < 0.33: # Highly planar
                         final_pos[2] = center[2]
                    elif planarity < 0.66: # Moderately planar
                         final_pos[2] = random.choice([max(0,center[2]-1), center[2], min(grid_size_tuple[2]-1, center[2]+1)])
                    # else: full 3D, no change to final_pos[2] from pos_tuple[2]
                
                # Ensure the (potentially modified) position is still valid and not occupied
                final_pos_tuple = tuple(final_pos)
                if final_pos_tuple not in voxels_set and all(0 <= final_pos_tuple[i] < grid_size_tuple[i] for i in range(3)):
                    voxels_set.add(final_pos_tuple)
                    added_this_step = True
                    break 
        
        if not added_this_step and len(voxels_set) < target_voxel_count and len(voxels_set) > 0:
            # Could not add by attaching to 'base'. Try picking another base or a random valid empty spot.
            # For simplicity, if stuck, just break or add one more randomly if space.
            all_grid_points = set((x,y,z) for x in range(grid_size_tuple[0]) for y in range(grid_size_tuple[1]) for z in range(grid_size_tuple[2]))
            empty_points = list(all_grid_points - voxels_set)
            if empty_points:
                voxels_set.add(random.choice(empty_points))


    # Apply branching: add "arms" from existing voxels
    # This is a post-processing step to ensure branching factor is met, can increase voxel count
    # A more integrated way would be to bias a growth rule towards branching during main loop.
    current_voxels_list_for_branching = list(voxels_set)
    if branching > 0 and len(current_voxels_list_for_branching) > 0:
        for _ in range(int(round(branching))): # Number of "arms" to try and add
            if len(voxels_set) >= target_voxel_count + int(round(branching * 2)): # Cap total voxels
                break
            
            base_for_branch = random.choice(current_voxels_list_for_branching) # Pick any existing voxel
            
            # Try to add a short arm (1-2 voxels)
            arm_length = random.choice([1,2])
            current_arm_tip = base_for_branch
            
            for _ in range(arm_length):
                branch_neighbors = get_local_neighbors(current_arm_tip)
                random.shuffle(branch_neighbors)
                added_to_arm = False
                for neighbor_pos in branch_neighbors:
                    if neighbor_pos not in voxels_set and all(0 <= neighbor_pos[i] < grid_size_tuple[i] for i in range(3)):
                        voxels_set.add(neighbor_pos)
                        current_arm_tip = neighbor_pos # Extend arm from new tip
                        added_to_arm = True
                        break # Added one segment to arm
                if not added_to_arm:
                    break # Cannot extend this arm further


    # Ensure voxel_count doesn't exceed target too much due to branching
    while len(voxels_set) > target_voxel_count + branching: # Allow some overshoot from branching
        if not voxels_set: break
        voxels_set.pop() # Simple removal of random voxel (sets are unordered)

    # Calculate actual features (simple version)
    final_voxel_count = len(voxels_set)
    # The input planarity, branching, compactness are what drove the generation.
    # Actual calculated values might differ, especially with this heuristic approach.
    # For now, report back the inputs as "features_used"
    
    calculated_compactness = _calculate_compactness_score(voxels_set, grid_size_tuple)
    calculated_branching = _calculate_branching_factor(voxels_set, grid_size_tuple)
    # A simple planarity actual measurement: ratio of voxels in the most populated Z-plane
    actual_planarity_score = 0.0
    if voxels_set:
        z_counts = {}
        for v_x,v_y,v_z in voxels_set:
            z_counts[v_z] = z_counts.get(v_z, 0) + 1
        if z_counts:
            max_in_plane = max(z_counts.values())
            actual_planarity_score = max_in_plane / len(voxels_set)
            # This definition of planarity is: 1.0 = all in one plane, low = spread out.
            # This is OPPOSITE to the user's definition in the function (0.0 = flat, 1.0 = 3D spread)
            # So we need to invert it if we want to match user's definition for the output
            # User definition: planarity_score (0.0 = flat, 1.0 = 3D spread)
            # Our calculation: max_in_plane / total (1.0 = flat)
            # So, actual_planarity_to_report = 1.0 - actual_planarity_score
            actual_planarity_to_report = 1.0 - actual_planarity_score

    analyzed_sfs = analyze_shape_features(voxels_set, grid_size_tuple, input_sfs=features)

    return {
        "voxels": [list(v) for v in voxels_set], # Convert set of tuples to list of lists
        "grid_size": list(grid_size_tuple),
        "features_used": { # These are the input targets that drove this specific generator
            "target_voxel_count": target_voxel_count,
            "target_planarity_score_input": planarity, # The 0-1 value where 0=flat
            "target_branching_factor_input": branching,
            "target_compactness_score_input": compactness,
            "actual_voxel_count": final_voxel_count,
            "calculated_compactness_score_heuristic": calculated_compactness, # Using the _calculate function
            "calculated_branching_factor_heuristic": calculated_branching, # Using the _calculate function
            "calculated_planarity_score_heuristic": actual_planarity_to_report # 1.0 = 3D spread
        },
        "features": analyzed_sfs.to_dict()
    }

# Example usage (can be called from main.py or another script)
if __name__ == "__main__":
    print("--- Testing Simple Generator ---")
    simple_test_features = ShapeFeatureSet(
        voxel_count=8,
        grid_size=(5,5,5),
        branching_factor=1
    )
    simple_shape = generate_shape_from_features(simple_test_features)
    print(f"Simple shape: {len(simple_shape['voxels'])} voxels. Features: {simple_shape['features']}")

    print("\n--- Testing Advanced Generator ---")
    # Create a target feature set (e.g., from cognitive_mapping.py or manually)
    adv_target_features = ShapeFeatureSet(
        voxel_count=12,
        grid_size=(7, 7, 7),
        compactness_score=0.7,  # Target high compactness
        branching_factor=1,    # Target low branching
        number_of_components=1
        # Other features like symmetry_class, planarity_score etc. are in the SFS
        # but not actively used by this version of generate_shape_advanced's scoring.
    )
    advanced_shape = generate_shape_advanced(adv_target_features, max_iterations=50 * adv_target_features.voxel_count)
    print(f"Advanced shape: {len(advanced_shape['voxels'])} voxels.")
    print(f"Features (targets + some calculated): {advanced_shape['features']}")

    adv_target_features_multi_component = ShapeFeatureSet(
        voxel_count=15,
        grid_size=(10, 10, 10),
        compactness_score=0.3,  # Target low compactness (more spread for multi-component)
        branching_factor=3,    
        number_of_components=3 # Target multiple components
    )
    adv_shape_multi = generate_shape_advanced(adv_target_features_multi_component, max_iterations=50 * adv_target_features_multi_component.voxel_count)
    print(f"Advanced multi-component shape: {len(adv_shape_multi['voxels'])} voxels.")
    print(f"Features (targets + some calculated): {adv_shape_multi['features']}")

    print("\n--- Testing Advanced Generator with Planarity ---")
    adv_target_planar = ShapeFeatureSet(
        voxel_count=12,
        grid_size=(7, 7, 7),
        compactness_score=0.6,
        branching_factor=1,
        number_of_components=1,
        planarity_score=0.9 # Target very flat
    )
    advanced_shape_planar = generate_shape_advanced(adv_target_planar, max_iterations=50 * adv_target_planar.voxel_count)
    print(f"Advanced planar shape: {len(advanced_shape_planar['voxels'])} voxels.")
    print(f"Features (targets + some calculated): {advanced_shape_planar['features']}")

    adv_target_3d = ShapeFeatureSet(
        voxel_count=12,
        grid_size=(7, 7, 7),
        compactness_score=0.4,
        branching_factor=2,
        number_of_components=1,
        planarity_score=0.3 # Target more 3D (less planar)
    )
    advanced_shape_3d = generate_shape_advanced(adv_target_3d, max_iterations=50 * adv_target_3d.voxel_count)
    print(f"Advanced 3D shape: {len(advanced_shape_3d['voxels'])} voxels.")
    print(f"Features (targets + some calculated): {advanced_shape_3d['features']}")

    print("\n--- Testing Heuristic Generator V1 (User Suggested) ---")
    heuristic_targets = ShapeFeatureSet(
        voxel_count=15,
        grid_size=(10,10,10),
        planarity_score=0.1,  # Target very flat (0 = flat, 1 = 3D for this func)
        branching_factor=3,   # Target some branches
        compactness_score=0.8 # Target quite compact
    )
    heuristic_shape = generate_shape_heuristic_v1(heuristic_targets)
    print(f"Heuristic shape: {len(heuristic_shape['voxels'])} voxels.")
    print(f"Features used/calculated: {heuristic_shape['features_used']}")

    heuristic_targets_3d = ShapeFeatureSet(
        voxel_count=15,
        grid_size=(10,10,10),
        planarity_score=0.9,  # Target very 3D
        branching_factor=1,   # Target few branches
        compactness_score=0.3 # Target sparse
    )
    heuristic_shape_3d = generate_shape_heuristic_v1(heuristic_targets_3d)
    print(f"Heuristic 3D shape: {len(heuristic_shape_3d['voxels'])} voxels.")
    print(f"Features used/calculated: {heuristic_shape_3d['features_used']}")
