"""
Multi-Objective Shape Generation and Feature Analysis Pipeline for MindFold 3D.

Copyright (c) 2024-2026 Scott N. Hwang, Parviz Safadel. All rights reserved.
Patent pending. See docs/PATENT_SPECIFICATION.md for claims.

This module implements the Shape Feature Analysis Pipeline and Optimization-Based
Shape Generator, core novel components of the MindFold 3D invention (Claims 1, 2, 12, 13).

Key inventive methods:
  - analyze_shape_features():        Real-time feature vector computation (Claim 1d)
  - _get_pca_eigenvalues():          PCA eigenvalue decomposition for shape characterization
  - _calculate_anisotropy_index():   PCA-based anisotropy: 1-(lambda3/lambda1) (Claim 12)
  - _calculate_shape_form_index():   PCA-based form: (2*lambda2-lambda1-lambda3)/(lambda1-lambda3) (Claim 12)
  - _calculate_cycle_count():        Circuit rank: |E|-|V|+C for topological complexity (Claim 13)
  - generate_mirror_reflection():    Chirality detection via mirror analysis (Claim 9)
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
    """Approximates branching. Counts voxels with >2 neighbors (branch points) 
       and voxels with 1 neighbor (endpoints), considers their ratio or sum.
       A simpler heuristic: count voxels with 3 or more neighbors within the shape.
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

def _calculate_cycle_count(voxels_set: Set[Tuple[int, int, int]], grid_size: Tuple[int, int, int]) -> int:
    """Calculate the number of independent cycles (circuit rank) in the adjacency graph.

    Formula: Cycles(G) = |E| - |V| + C
    where E = edges (face-adjacent pairs), V = voxels, C = connected components.
    Cycles correspond to holes or loops in the shape (Chen, 2005).
    """
    if len(voxels_set) < 4:
        return 0  # Minimum 4 voxels for a cycle in 6-connectivity
    edge_count = sum(
        1 for v in voxels_set for n in _get_neighbors(v, grid_size) if n in voxels_set
    ) // 2  # Each edge counted twice
    return max(0, edge_count - len(voxels_set) + _count_components(voxels_set, grid_size))

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
    actual_features["cycle_count"] = _calculate_cycle_count(voxels_set, grid_size_tuple)
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


def keep_largest_component(voxels, grid_size):
    """Return a voxel set containing only the largest connected component.

    Defensive cleanup for shapes that were intended to be single-component
    but slipped through as orphan + main shape. Accepts list-of-lists or a
    set of tuples; returns a set of tuples.
    """
    if isinstance(voxels, list):
        vset = {tuple(v) for v in voxels}
    else:
        vset = set(voxels)
    if not vset or _count_components(vset, grid_size) <= 1:
        return vset
    seen = set()
    largest = set()
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
                              max_iterations: int = 25) -> Optional[List[List[int]]]:
    """Mutate a voxel set (add/remove one voxel at a time) until its
    canonical form is not already in seen_canonicals.

    Used as a last-ditch distractor fallback when the upstream generator
    keeps producing rotational copies of an already-accepted shape.
    Maintains connectivity opportunistically (adds only place voxels
    adjacent to existing ones; removes may disconnect, which is
    acceptable for a distractor). Returns the mutated list, or None
    if no unique form is found within max_iterations.
    """
    import random as _random
    vset: Set[Tuple[int, int, int]] = {tuple(v) for v in voxels}
    gx, gy, gz = grid_size
    # Preserve the component count of the original shape so that a
    # split-producing removal doesn't turn a single-component shape
    # into a main piece + stray voxel.
    target_n_components = _count_components(vset, grid_size) if vset else 1

    for _ in range(max_iterations):
        do_add = (len(vset) < 5) or (_random.random() < 0.5)

        if do_add:
            placed = False
            for _try in range(10):
                base = _random.choice(list(vset))
                dx, dy, dz = _random.choice([(1, 0, 0), (-1, 0, 0),
                                              (0, 1, 0), (0, -1, 0),
                                              (0, 0, 1), (0, 0, -1)])
                cand = (base[0] + dx, base[1] + dy, base[2] + dz)
                if (0 <= cand[0] < gx and 0 <= cand[1] < gy and 0 <= cand[2] < gz
                        and cand not in vset):
                    vset.add(cand)
                    placed = True
                    break
            if not placed:
                continue
        else:
            # Remove a voxel without increasing the component count. Try
            # several candidates; skip any whose removal would split a
            # component (yielding an orphan / stray voxel).
            if len(vset) <= 3:
                continue
            removed = False
            candidates = list(vset)
            _random.shuffle(candidates)
            for cand in candidates[:6]:
                trial = vset - {cand}
                if _count_components(trial, grid_size) <= target_n_components:
                    vset = trial
                    removed = True
                    break
            if not removed:
                continue

        canon = canonical_voxel_form(list(vset))
        if canon not in seen_canonicals:
            return [list(v) for v in vset]

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



if __name__ == "__main__":
    print("--- Testing Simple Generator ---")
    simple_test_features = ShapeFeatureSet(
        voxel_count=8,
        grid_size=(5, 5, 5),
        branching_factor=1
    )
    simple_shape = generate_shape_from_features(simple_test_features)
    print(f"Simple shape: {len(simple_shape['voxels'])} voxels. Features: {simple_shape['features']}")
