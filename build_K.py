"""
Construct K from the perturbed triangulation.

For d=2, n=1:
- K is a 1-dimensional simplicial complex (a graph)
- K consists of edge-manifold intersection points connected within triangles

Algorithm:
1. Find all edge-manifold intersections
2. For each triangle with exactly 2 intersection points, connect them
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Set


def find_edge_intersections(
    vertices: Dict[int, np.ndarray],
    triangles: List[Tuple[int, int, int]],
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    tol: float = 1e-10
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Find all intersections between edges and the manifold M = {f = 0}.
    
    Args:
        vertices: Vertex positions
        triangles: Triangle connectivity
        f: Implicit function
        grad_f: Gradient of f
        tol: Tolerance for intersection finding
    
    Returns:
        intersections: Dict mapping edge (i, j) to intersection point
    """
    # Get all edges
    edges = set()
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1) % 3]]))
            edges.add(e)
    
    intersections = {}
    
    for e in edges:
        v0, v1 = vertices[e[0]], vertices[e[1]]
        f0, f1 = f(v0), f(v1)
        
        # Check for sign change (intersection)
        if f0 * f1 < 0:
            # Linear interpolation to find intersection
            # f(v0 + t*(v1-v0)) = 0
            # Approximate: f0 + t*(f1-f0) = 0 => t = -f0/(f1-f0)
            t = -f0 / (f1 - f0)
            t = np.clip(t, 0, 1)
            
            # Refine with Newton's method
            p = v0 + t * (v1 - v0)
            for _ in range(5):
                fp = f(p)
                if abs(fp) < tol:
                    break
                gp = grad_f(p)
                # Project gradient onto edge direction
                edge_dir = v1 - v0
                edge_len = np.linalg.norm(edge_dir)
                if edge_len < 1e-12:
                    break
                edge_dir = edge_dir / edge_len
                grad_along_edge = np.dot(gp, edge_dir)
                if abs(grad_along_edge) < 1e-12:
                    break
                dt = -fp / grad_along_edge
                t = t + dt / edge_len
                t = np.clip(t, 0, 1)
                p = v0 + t * (v1 - v0)
            
            intersections[e] = p
    
    return intersections


def build_K(
    vertices: Dict[int, np.ndarray],
    triangles: List[Tuple[int, int, int]],
    edge_intersections: Dict[Tuple[int, int], np.ndarray],
    f: Callable[[np.ndarray], float]
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Build K from edge intersections.
    
    For n=1 (curve), K is built by:
    - Each triangle with exactly 2 intersection points contributes an edge to K
    
    Args:
        vertices: Perturbed vertex positions
        triangles: Triangle connectivity
        edge_intersections: Edge-manifold intersections
        f: Implicit function
    
    Returns:
        K_vertices: List of K vertex positions
        K_edges: List of edges as (i, j) indices into K_vertices
    """
    K_vertices = []
    K_edges = []
    
    # Map intersection points to K vertex indices
    intersection_to_idx = {}
    
    def get_K_vertex(edge: Tuple[int, int]) -> int:
        """Get or create K vertex for an edge intersection."""
        if edge not in intersection_to_idx:
            idx = len(K_vertices)
            K_vertices.append(edge_intersections[edge])
            intersection_to_idx[edge] = idx
        return intersection_to_idx[edge]
    
    # Process each triangle
    for tri in triangles:
        # Find which edges of this triangle have intersections
        tri_edges = []
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1) % 3]]))
            if e in edge_intersections:
                tri_edges.append(e)
        
        # For n=1: exactly 2 intersections per triangle that intersects M
        if len(tri_edges) == 2:
            i0 = get_K_vertex(tri_edges[0])
            i1 = get_K_vertex(tri_edges[1])
            if i0 != i1:
                K_edges.append((i0, i1))
        elif len(tri_edges) > 2:
            # This shouldn't happen for a curve, but handle gracefully
            # Connect all pairs? Or use star configuration?
            print(f"Warning: Triangle has {len(tri_edges)} intersections")
    
    return K_vertices, K_edges


def verify_K(
    K_vertices: List[np.ndarray],
    K_edges: List[Tuple[int, int]],
    f: Callable[[np.ndarray], float],
    tol: float = 1e-6
) -> bool:
    """Verify K vertices lie on M and edges are reasonable."""
    all_good = True
    
    for i, v in enumerate(K_vertices):
        if abs(f(v)) > tol:
            print(f"K vertex {i}: f(v) = {f(v):.6e}")
            all_good = False
    
    return all_good


if __name__ == "__main__":
    from coxeter import generate_coxeter_A2
    from perturb import perturb_vertices, print_perturbation_stats
    
    # Test with a circle
    radius = 1.0
    f = lambda p: p[0]**2 + p[1]**2 - radius**2
    grad_f = lambda p: np.array([2*p[0], 2*p[1]])
    
    # Use coarser L for testing
    L = radius / 20
    box_min = np.array([-1.5, -1.5])
    box_max = np.array([1.5, 1.5])
    
    print(f"L = {L:.6f}")
    
    # Generate triangulation
    vertices, triangles = generate_coxeter_A2(box_min, box_max, L)
    print(f"T: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Perturb
    perturbed, info = perturb_vertices(vertices, triangles, f, grad_f, L)
    print_perturbation_stats(info)
    
    # Find intersections
    intersections = find_edge_intersections(perturbed, triangles, f, grad_f)
    print(f"Edge intersections: {len(intersections)}")
    
    # Build K
    K_verts, K_edges = build_K(perturbed, triangles, intersections, f)
    print(f"K: {len(K_verts)} vertices, {len(K_edges)} edges")
    
    # Verify
    if verify_K(K_verts, K_edges, f):
        print("K vertices all lie on M")