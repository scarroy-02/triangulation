"""
Perturbation algorithm for Whitney triangulation - following the paper exactly.

Algorithm (Case 2 for d=2, n=1):
1. For vertex v_i with d(v_i, M) < 3L/2:
2. Find p ∈ M with d(v_i, p) < 3L/2
3. Consider all τ'_j ⊂ T̃_{i-1} with dim(τ'_j) ≤ d-n-2 such that τ'_j * v_i is in T̃
4. For d=2, n=1: d-n-2 = -1, so only τ' = ∅ (empty set)
5. span(∅, T_p M) = T_p M (the tangent line)
6. Pick ṽ_i such that:
   - d(ṽ_i, T_p M) ≥ ρ_1 * c̃ * L
   - |ṽ_i - v_i| ≤ c̃ * L
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Set
from utils import rho_1, c_tilde


def find_nearest_point_on_M(
    v: np.ndarray,
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    max_iter: int = 20,
    tol: float = 1e-12
) -> np.ndarray:
    """
    Find nearest point p on M = {f = 0} to v using Newton projection.
    """
    p = v.copy()
    for _ in range(max_iter):
        fp = f(p)
        if abs(fp) < tol:
            break
        gp = grad_f(p)
        gp_norm_sq = np.dot(gp, gp)
        if gp_norm_sq < tol:
            break
        # Project onto M: p <- p - f(p) * grad_f(p) / |grad_f(p)|^2
        p = p - fp * gp / gp_norm_sq
    return p


def distance_to_tangent_space(
    v: np.ndarray,
    p: np.ndarray,
    grad_f: Callable[[np.ndarray], np.ndarray]
) -> Tuple[float, np.ndarray]:
    """
    Compute distance from v to T_p M (tangent space at p).
    
    For n=1 (curve in 2D), T_p M is a line through p perpendicular to grad_f(p).
    
    Returns:
        dist: distance from v to T_p M
        normal: unit normal to T_p M (direction to move for perturbation)
    """
    gp = grad_f(p)
    gp_norm = np.linalg.norm(gp)
    
    if gp_norm < 1e-12:
        return 0.0, np.array([1.0, 0.0])  # Fallback
    
    # Normal to M at p (perpendicular to tangent)
    normal = gp / gp_norm
    
    # Distance = |projection of (v - p) onto normal|
    v_minus_p = v - p
    signed_dist = np.dot(v_minus_p, normal)
    dist = abs(signed_dist)
    
    # Return the direction that moves v away from T_p M
    if abs(signed_dist) < 1e-12:
        direction = normal  # Arbitrary choice when v is exactly on T_p M
    else:
        direction = normal if signed_dist > 0 else -normal
    
    return dist, direction


def perturb_vertex_case2(
    v: np.ndarray,
    p: np.ndarray,
    grad_f: Callable[[np.ndarray], np.ndarray],
    required_dist: float,
    max_perturb: float
) -> Tuple[np.ndarray, dict]:
    """
    Perturb vertex v (Case 2) to be far from T_p M.
    
    For d=2, n=1:
    - Only span to consider is span(∅, T_p M) = T_p M
    - Need d(ṽ, T_p M) ≥ required_dist = ρ_1 * c̃ * L
    - Constraint: |ṽ - v| ≤ max_perturb = c̃ * L
    
    Returns:
        v_perturbed: perturbed vertex position
        info: debug information
    """
    dist_to_TpM, direction = distance_to_tangent_space(v, p, grad_f)
    
    info = {
        'p': p.copy(),
        'dist_to_TpM_before': dist_to_TpM,
        'required_dist': required_dist,
        'max_perturb': max_perturb
    }
    
    # Already far enough?
    if dist_to_TpM >= required_dist:
        info['action'] = 'already_far'
        info['perturbed'] = False
        return v.copy(), info
    
    # Need to move v in 'direction' until distance >= required_dist
    move_amount = required_dist - dist_to_TpM
    
    # Check constraint: |ṽ - v| ≤ c̃ * L
    if move_amount > max_perturb:
        info['action'] = 'clamped'
        info['requested_move'] = move_amount
        move_amount = max_perturb
        print(f"Warning: perturbation clamped from {info['requested_move']:.6f} to {max_perturb:.6f}")
    
    v_perturbed = v + direction * move_amount
    
    info['action'] = 'moved'
    info['perturbed'] = True
    info['move_amount'] = move_amount
    info['direction'] = direction.copy()
    info['dist_to_TpM_after'] = distance_to_tangent_space(v_perturbed, p, grad_f)[0]
    
    return v_perturbed, info


def perturb_vertices(
    vertices: Dict[int, np.ndarray],
    triangles: List[Tuple[int, int, int]],
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    L: float,
    d: int = 2,
    n: int = 1
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """
    Perturb vertices according to Whitney's algorithm.
    
    Algorithm:
    - Case 1: d(v_i, M) ≥ 3L/2 → keep unchanged
    - Case 2: d(v_i, M) < 3L/2 → perturb to be far from span(τ', T_p M)
    
    For d=2, n=1: only span is T_p M (tangent line)
    
    Args:
        vertices: Original vertex positions {index: position}
        triangles: Triangle connectivity
        f: Implicit function (M = {f = 0})
        grad_f: Gradient of f
        L: Edge length of triangulation
        d: Ambient dimension
        n: Manifold dimension
    
    Returns:
        perturbed: Perturbed vertex positions
        info: Debug info for each vertex
    """
    # Constants from the paper
    rho = rho_1(d, n)
    c = c_tilde(d)
    
    # Required distance from span(τ', T_p M): equation (19)
    required_dist = rho * c * L
    
    # Maximum perturbation: equation (20) / Lemma 5.6
    max_perturb = c * L
    
    # Threshold for Case 1 vs Case 2
    near_threshold = 3 * L / 2
    
    print(f"Constants for d={d}, n={n}:")
    print(f"  ρ_1 = {rho:.6f}")
    print(f"  c̃ = {c:.6f}")
    print(f"  L = {L:.6f}")
    print(f"  required_dist = ρ_1 * c̃ * L = {required_dist:.6e}")
    print(f"  max_perturb = c̃ * L = {max_perturb:.6e}")
    print(f"  near_threshold = 3L/2 = {near_threshold:.6f}")
    
    perturbed = {}
    info = {}
    
    for idx, v in vertices.items():
        # Estimate distance to M
        f_val = f(v)
        grad_val = grad_f(v)
        grad_norm = np.linalg.norm(grad_val)
        
        if grad_norm < 1e-12:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 0, 'reason': 'grad_zero'}
            continue
        
        # Approximate distance to M
        dist_to_M = abs(f_val) / grad_norm
        
        # Case 1: Far from M
        if dist_to_M >= near_threshold:
            perturbed[idx] = v.copy()
            info[idx] = {
                'case': 1,
                'dist_to_M': dist_to_M,
                'perturbed': False
            }
            continue
        
        # Case 2: Near M (d(v_i, M) < 3L/2)
        # Step 1: Find p ∈ M with d(v_i, p) < 3L/2
        p = find_nearest_point_on_M(v, f, grad_f)
        
        # Verify d(v, p) < 3L/2
        dist_v_to_p = np.linalg.norm(v - p)
        
        # Step 2: For d=2, n=1, only τ' = ∅, so span(∅, T_p M) = T_p M
        # Step 3: Perturb v to be far from T_p M
        v_pert, pert_info = perturb_vertex_case2(v, p, grad_f, required_dist, max_perturb)
        
        perturbed[idx] = v_pert
        info[idx] = {
            'case': 2,
            'dist_to_M': dist_to_M,
            'dist_v_to_p': dist_v_to_p,
            **pert_info
        }
    
    return perturbed, info


def print_perturbation_stats(info: Dict[int, dict]):
    """Print statistics about perturbation."""
    case0 = sum(1 for i in info.values() if i.get('case') == 0)
    case1 = sum(1 for i in info.values() if i.get('case') == 1)
    case2_far = sum(1 for i in info.values() if i.get('case') == 2 and i.get('action') == 'already_far')
    case2_moved = sum(1 for i in info.values() if i.get('case') == 2 and i.get('action') == 'moved')
    case2_clamped = sum(1 for i in info.values() if i.get('case') == 2 and i.get('action') == 'clamped')
    
    print(f"\nPerturbation statistics:")
    print(f"  Case 0 (degenerate): {case0}")
    print(f"  Case 1 (d(v,M) ≥ 3L/2): {case1}")
    print(f"  Case 2 (near M):")
    print(f"    - already far from T_p M: {case2_far}")
    print(f"    - moved: {case2_moved}")
    if case2_clamped > 0:
        print(f"    - clamped: {case2_clamped}")
    
    # Print details of moved vertices
    if case2_moved > 0:
        print(f"\n  Moved vertices details:")
        for idx, inf in info.items():
            if inf.get('case') == 2 and inf.get('action') == 'moved':
                print(f"    v_{idx}: dist_before={inf['dist_to_TpM_before']:.6e}, "
                      f"dist_after={inf['dist_to_TpM_after']:.6e}, "
                      f"move={inf['move_amount']:.6e}")


if __name__ == "__main__":
    from coxeter import generate_coxeter_A2
    
    # Test with a circle
    radius = 1.0
    f = lambda p: p[0]**2 + p[1]**2 - radius**2
    grad_f = lambda p: np.array([2*p[0], 2*p[1]])
    
    # Use L = reach/54
    L = radius / 54
    box_min = np.array([-1.5, -1.5])
    box_max = np.array([1.5, 1.5])
    
    print(f"=== Whitney Perturbation Test ===")
    print(f"Manifold: Circle with radius {radius} (reach = {radius})")
    print(f"L = reach/54 = {L:.6f}\n")
    
    vertices, triangles = generate_coxeter_A2(box_min, box_max, L)
    print(f"Generated T: {len(vertices)} vertices, {len(triangles)} triangles\n")
    
    perturbed, info = perturb_vertices(vertices, triangles, f, grad_f, L)
    print_perturbation_stats(info)