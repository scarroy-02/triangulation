"""
Perturbation algorithm for Whitney triangulation.

For d=2 (ambient), n=1 (manifold dimension):
- We need the 0-skeleton (vertices) to be far from the tangent space T_p M
- d - n - 1 = 0: vertices must be far from M's tangent
- d - n - 2 = -1: no lower-dimensional faces to consider

Algorithm:
1. Case 1: If d(v, M) >= 3L/2, keep v unchanged
2. Case 2: If d(v, M) < 3L/2:
   - Find nearest point p on M
   - Compute tangent T_p M
   - Perturb v so that it is at distance >= rho_1 * c_tilde * L from T_p M
   - Perturbation must be <= c_tilde * L
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from utils import rho_1, c_tilde


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
    
    Args:
        vertices: Original vertex positions
        triangles: Triangle connectivity
        f: Implicit function defining M = {x : f(x) = 0}
        grad_f: Gradient of f
        L: Edge length
        d: Ambient dimension (default 2)
        n: Manifold dimension (default 1)
    
    Returns:
        perturbed: Perturbed vertex positions
        info: Debug info for each vertex
    """
    # Get constants
    rho = rho_1(d, n)
    c = c_tilde(d)
    
    # Required distance from tangent space T_p M
    required_dist = rho * c * L
    
    # Maximum perturbation allowed
    max_perturb = c * L
    
    # Threshold for "near" the manifold (Case 1 vs Case 2)
    near_threshold = 3 * L / 2
    
    perturbed = {}
    info = {}
    
    for idx, v in vertices.items():
        # Compute distance to manifold (approximately)
        # For implicit function f, distance ~ |f(v)| / |grad_f(v)|
        f_val = f(v)
        grad_val = grad_f(v)
        grad_norm = np.linalg.norm(grad_val)
        
        if grad_norm < 1e-12:
            # Gradient too small, can't compute distance reliably
            perturbed[idx] = v.copy()
            info[idx] = {'case': 0, 'reason': 'grad_zero'}
            continue
        
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
        
        # Case 2: Near M - need to check/ensure distance from tangent
        # Find nearest point p on M (approximate by projection)
        # p ≈ v - f(v) * grad_f(v) / |grad_f(v)|^2
        p = v - f_val * grad_val / (grad_norm**2)
        
        # Tangent space T_p M is perpendicular to grad_f(p)
        # For n=1 (curve), T_p M is a line
        grad_at_p = grad_f(p)
        grad_at_p_norm = np.linalg.norm(grad_at_p)
        
        if grad_at_p_norm < 1e-12:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 2, 'reason': 'grad_at_p_zero'}
            continue
        
        # Normal to M at p (points away from M)
        normal = grad_at_p / grad_at_p_norm
        
        # Tangent direction (perpendicular to normal in 2D)
        tangent = np.array([-normal[1], normal[0]])
        
        # Distance from v to the tangent line T_p M
        # T_p M passes through p with direction tangent
        # Distance = |component of (v - p) perpendicular to tangent|
        #          = |component of (v - p) along normal|
        v_minus_p = v - p
        dist_to_tangent = abs(np.dot(v_minus_p, normal))
        
        # Check if v is already far enough from tangent
        if dist_to_tangent >= required_dist:
            perturbed[idx] = v.copy()
            info[idx] = {
                'case': 2,
                'subcase': 'already_far',
                'dist_to_M': dist_to_M,
                'dist_to_tangent': dist_to_tangent,
                'required_dist': required_dist,
                'perturbed': False
            }
            continue
        
        # Need to perturb v away from tangent plane
        # Move v in the normal direction until distance >= required_dist
        
        # Current signed distance (positive if on same side as grad, negative otherwise)
        signed_dist = np.dot(v_minus_p, normal)
        
        # Determine which direction to move
        if abs(signed_dist) < 1e-12:
            # v is almost exactly on the tangent plane, pick a direction
            direction = normal
        else:
            # Move away from tangent in the same direction v is already on
            direction = normal if signed_dist > 0 else -normal
        
        # How much to move
        move_amount = required_dist - dist_to_tangent
        
        # Ensure we don't exceed max perturbation
        if move_amount > max_perturb:
            print(f"Warning: vertex {idx} needs perturbation {move_amount:.6f} > max {max_perturb:.6f}")
            move_amount = max_perturb
        
        # Compute perturbed position
        v_perturbed = v + direction * move_amount
        
        perturbed[idx] = v_perturbed
        info[idx] = {
            'case': 2,
            'subcase': 'moved',
            'dist_to_M': dist_to_M,
            'dist_to_tangent': dist_to_tangent,
            'required_dist': required_dist,
            'move_amount': move_amount,
            'direction': direction,
            'perturbed': True
        }
    
    return perturbed, info


def print_perturbation_stats(info: Dict[int, dict]):
    """Print statistics about perturbation."""
    case1 = sum(1 for i in info.values() if i.get('case') == 1)
    case2_already = sum(1 for i in info.values() if i.get('case') == 2 and i.get('subcase') == 'already_far')
    case2_moved = sum(1 for i in info.values() if i.get('case') == 2 and i.get('subcase') == 'moved')
    other = len(info) - case1 - case2_already - case2_moved
    
    print(f"Perturbation statistics:")
    print(f"  Case 1 (far from M): {case1}")
    print(f"  Case 2 (near M, already far from tangent): {case2_already}")
    print(f"  Case 2 (near M, moved): {case2_moved}")
    if other > 0:
        print(f"  Other: {other}")


if __name__ == "__main__":
    from coxeter import generate_coxeter_A2
    
    # Test with a circle
    radius = 1.0
    f = lambda p: p[0]**2 + p[1]**2 - radius**2
    grad_f = lambda p: np.array([2*p[0], 2*p[1]])
    
    # Generate triangulation
    L = radius / 54  # Use the theoretical bound
    box_min = np.array([-1.5, -1.5])
    box_max = np.array([1.5, 1.5])
    
    print(f"L = {L:.6f}, L/reach = {L/radius:.6f}")
    
    vertices, triangles = generate_coxeter_A2(box_min, box_max, L)
    print(f"Generated {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Perturb
    perturbed, info = perturb_vertices(vertices, triangles, f, grad_f, L)
    print_perturbation_stats(info)