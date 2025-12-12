"""
Whitney triangulation for d=3, n=2 (surface in 3D).

Algorithm:
1. Generate 3D triangulation T (tetrahedra)
2. Perturb vertices → T̃ (vertices far from tangent planes)
3. Find edge-surface intersections
4. Build K: triangulated surface approximating M
5. Subdivide tetrahedra at intersections

For d=3, n=2:
- d-n-1 = 0: vertices must be far from spans
- d-n-2 = -1: only τ' = ∅, so span = T_pM (tangent plane)
- Each tetrahedron typically intersected at 3 or 4 edges
- K faces are triangles (3 intersections) or quads split into 2 triangles (4 intersections)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from coxeter3d import generate_kuhn_triangulation, get_tetrahedron_edges, get_all_edges
from utils import rho_1, c_tilde


# =============================================================================
# Perturbation (same logic as 2D, but with tangent plane instead of line)
# =============================================================================

def find_nearest_point_on_M(
    v: np.ndarray,
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    max_iter: int = 20,
    tol: float = 1e-12
) -> np.ndarray:
    """Find nearest point p on M = {f = 0} to v using Newton projection."""
    p = v.copy()
    for _ in range(max_iter):
        fp = f(p)
        if abs(fp) < tol:
            break
        gp = grad_f(p)
        gp_norm_sq = np.dot(gp, gp)
        if gp_norm_sq < tol:
            break
        p = p - fp * gp / gp_norm_sq
    return p


def distance_to_tangent_plane(
    v: np.ndarray,
    p: np.ndarray,
    grad_f: Callable[[np.ndarray], np.ndarray]
) -> Tuple[float, np.ndarray]:
    """
    Compute distance from v to T_p M (tangent plane at p).
    
    For n=2 (surface in 3D), T_p M is a plane through p perpendicular to grad_f(p).
    
    Returns:
        dist: distance from v to T_p M
        normal: unit normal to T_p M
    """
    gp = grad_f(p)
    gp_norm = np.linalg.norm(gp)
    
    if gp_norm < 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0])
    
    normal = gp / gp_norm
    v_minus_p = v - p
    signed_dist = np.dot(v_minus_p, normal)
    dist = abs(signed_dist)
    
    direction = normal if signed_dist >= 0 else -normal
    return dist, direction


def perturb_vertices_3d(
    vertices: Dict[int, np.ndarray],
    tetrahedra: List[Tuple[int, int, int, int]],
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    L: float,
    required_dist_factor: float = 1.0,
    d: int = 3,
    n: int = 2
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """
    Perturb vertices according to Whitney's algorithm for d=3, n=2.
    """
    rho = rho_1(d, n)
    c = c_tilde(d)
    
    required_dist = rho * c * L * required_dist_factor
    max_perturb = c * L * required_dist_factor
    near_threshold = 3 * L / 2
    
    print(f"3D Perturbation constants (d={d}, n={n}):")
    print(f"  ρ_1 = {rho:.6f}")
    print(f"  c̃ = {c:.6f}")
    print(f"  required_dist = {required_dist:.6e}")
    print(f"  max_perturb = {max_perturb:.6e}")
    
    perturbed = {}
    info = {}
    
    for idx, v in vertices.items():
        f_val = f(v)
        grad_val = grad_f(v)
        grad_norm = np.linalg.norm(grad_val)
        
        if grad_norm < 1e-12:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 0}
            continue
        
        dist_to_M = abs(f_val) / grad_norm
        
        if dist_to_M >= near_threshold:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 1, 'dist_to_M': dist_to_M}
            continue
        
        p = find_nearest_point_on_M(v, f, grad_f)
        dist_to_TpM, direction = distance_to_tangent_plane(v, p, grad_f)
        
        if dist_to_TpM >= required_dist:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 2, 'action': 'already_far'}
            continue
        
        move_amount = min(required_dist - dist_to_TpM, max_perturb)
        v_pert = v + direction * move_amount
        
        perturbed[idx] = v_pert
        info[idx] = {
            'case': 2, 'action': 'moved',
            'v_orig': v.copy(), 'v_pert': v_pert.copy(),
            'move_amount': move_amount
        }
    
    return perturbed, info


# =============================================================================
# Find intersections and build K
# =============================================================================

def find_edge_intersection(v0, v1, f, grad_f, tol=1e-10):
    """Find intersection point on edge (v0, v1) with surface M."""
    f0, f1 = f(v0), f(v1)
    
    if f0 * f1 >= 0:
        return None
    
    t = -f0 / (f1 - f0)
    t = np.clip(t, 0, 1)
    
    p = v0 + t * (v1 - v0)
    edge_dir = v1 - v0
    edge_len = np.linalg.norm(edge_dir)
    
    if edge_len > 1e-12:
        edge_dir = edge_dir / edge_len
        for _ in range(10):
            fp = f(p)
            if abs(fp) < tol:
                break
            gp = grad_f(p)
            grad_along_edge = np.dot(gp, edge_dir)
            if abs(grad_along_edge) < 1e-12:
                break
            dt = -fp / grad_along_edge
            t = np.clip(t + dt / edge_len, 0, 1)
            p = v0 + t * (v1 - v0)
    
    return p


def build_K_surface(
    vertices: Dict[int, np.ndarray],
    tetrahedra: List[Tuple[int, int, int, int]],
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray]
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Build output surface K from edge-surface intersections.
    
    For each tetrahedron:
    - 3 intersections → 1 triangle
    - 4 intersections → 2 triangles (quad split)
    
    Returns:
        K_vertices: List of vertex positions
        K_faces: List of triangles as (i, j, k) indices
    """
    K_vertices = []
    K_faces = []
    edge_to_K_idx = {}  # edge -> K vertex index
    
    def get_K_vertex(edge, intersection_point):
        if edge not in edge_to_K_idx:
            idx = len(K_vertices)
            K_vertices.append(intersection_point.copy())
            edge_to_K_idx[edge] = idx
        return edge_to_K_idx[edge]
    
    for tet in tetrahedra:
        edges = get_tetrahedron_edges(tet)
        
        # Find intersections on each edge
        intersections = []
        for e in edges:
            v0, v1 = vertices[e[0]], vertices[e[1]]
            p = find_edge_intersection(v0, v1, f, grad_f)
            if p is not None:
                k_idx = get_K_vertex(e, p)
                intersections.append((e, k_idx, p))
        
        n_int = len(intersections)
        
        if n_int == 3:
            # One triangle
            i0, i1, i2 = intersections[0][1], intersections[1][1], intersections[2][1]
            K_faces.append((i0, i1, i2))
        
        elif n_int == 4:
            # Quadrilateral - split into 2 triangles
            # Order points around the quad
            points = [(intersections[i][1], intersections[i][2]) for i in range(4)]
            
            # Simple ordering: find centroid and sort by angle
            centroid = np.mean([p[1] for p in points], axis=0)
            
            # Project to plane and sort by angle
            # Use first point as reference
            ref = points[0][1] - centroid
            ref_norm = np.linalg.norm(ref)
            if ref_norm < 1e-12:
                # Fallback: just use as-is
                i0, i1, i2, i3 = [p[0] for p in points]
            else:
                ref = ref / ref_norm
                # Find a perpendicular vector in the plane
                normal = np.cross(points[1][1] - centroid, points[2][1] - centroid)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    normal = np.array([0, 0, 1])
                else:
                    normal = normal / normal_norm
                perp = np.cross(normal, ref)
                
                # Compute angles
                angles = []
                for idx, pos in points:
                    v = pos - centroid
                    angle = np.arctan2(np.dot(v, perp), np.dot(v, ref))
                    angles.append((angle, idx))
                angles.sort()
                i0, i1, i2, i3 = [a[1] for a in angles]
            
            # Two triangles
            K_faces.append((i0, i1, i2))
            K_faces.append((i0, i2, i3))
        
        # n_int == 0, 1, 2, 5, 6 are edge cases (degenerate or doesn't intersect)
    
    return K_vertices, K_faces


# =============================================================================
# Figure 2 subdivision for 3D
# =============================================================================

def subdivide_tetrahedron_figure2(
    tet_verts: List[np.ndarray],
    tet_indices: List[int],
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    new_vertex_start_idx: int
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[Tuple[int, int, int]], int]:
    """
    Subdivide tetrahedron according to Figure 2 construction.
    
    For 3 intersections: creates triangular face, barycenter, and subdivided tets
    For 4 intersections: creates quad face (2 triangles), barycenter, and subdivided tets
    
    Returns:
        new_vertices: List of new vertices (intersection points + barycenter)
        new_tetrahedra: List of new tetrahedra
        K_faces: List of K faces (triangles)
        barycenter_idx: Index of barycenter (or None)
    """
    v0, v1, v2, v3 = tet_verts
    i0, i1, i2, i3 = tet_indices
    
    # Get all 6 edges
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    edge_verts = [
        (v0, v1), (v0, v2), (v0, v3),
        (v1, v2), (v1, v3), (v2, v3)
    ]
    edge_indices = [
        (i0, i1), (i0, i2), (i0, i3),
        (i1, i2), (i1, i3), (i2, i3)
    ]
    local_to_global = {0: i0, 1: i1, 2: i2, 3: i3}
    
    # Find intersections
    intersections = {}  # edge_idx -> intersection point
    for edge_idx, (va, vb) in enumerate(edge_verts):
        p = find_edge_intersection(va, vb, f, grad_f)
        if p is not None:
            intersections[edge_idx] = p
    
    n_int = len(intersections)
    new_vertices = []
    new_tetrahedra = []
    K_faces = []
    barycenter_idx = None
    
    if n_int < 3:
        # No proper intersection - keep original
        new_tetrahedra.append((i0, i1, i2, i3))
        return new_vertices, new_tetrahedra, K_faces, barycenter_idx
    
    # Create new vertices for intersection points
    int_edges = sorted(intersections.keys())
    p_indices = {}
    for edge_idx in int_edges:
        p_idx = new_vertex_start_idx + len(new_vertices)
        new_vertices.append(intersections[edge_idx])
        p_indices[edge_idx] = p_idx
    
    # Compute barycenter of intersection points
    int_points = [intersections[e] for e in int_edges]
    b = np.mean(int_points, axis=0)
    b_idx = new_vertex_start_idx + len(new_vertices)
    new_vertices.append(b)
    barycenter_idx = b_idx
    
    if n_int == 3:
        # One triangular K face
        e0, e1, e2 = int_edges
        K_faces.append((p_indices[e0], p_indices[e1], p_indices[e2]))
        
        # Create tetrahedra: connect barycenter to original vertices and intersection points
        # We create tetrahedra by connecting b to each face of the "boundary"
        
        # The boundary consists of:
        # - 3 triangles connecting each original vertex to 2 intersection points
        # - 1 triangle connecting 3 intersection points (the K face)
        
        # Identify which original vertices are on which side
        # Vertex is "positive" if f(v) > 0, "negative" if f(v) < 0
        signs = [np.sign(f(v)) for v in tet_verts]
        pos_verts = [i for i in range(4) if signs[i] > 0]
        neg_verts = [i for i in range(4) if signs[i] < 0]
        
        # Each original vertex connects to b and to nearby intersection points
        for local_v in range(4):
            # Find edges incident to this vertex that have intersections
            incident_int_edges = []
            for edge_idx, (a, b_local) in enumerate(edges):
                if (a == local_v or b_local == local_v) and edge_idx in intersections:
                    incident_int_edges.append(edge_idx)
            
            if len(incident_int_edges) >= 2:
                # This vertex has 2 incident intersection edges
                # Create tetrahedra: (vertex, p1, p2, barycenter)
                for j in range(len(incident_int_edges)):
                    e_a = incident_int_edges[j]
                    e_b = incident_int_edges[(j+1) % len(incident_int_edges)]
                    new_tetrahedra.append((
                        local_to_global[local_v],
                        p_indices[e_a],
                        p_indices[e_b],
                        b_idx
                    ))
            elif len(incident_int_edges) == 0:
                # This vertex has no incident intersections
                # It connects to all intersection points through the barycenter
                # Create tetrahedra with neighboring vertices
                pass
        
        # Also create tetrahedra for the K face connecting to non-adjacent vertices
        # This is complex - let's use a simpler approach:
        # Just create tetrahedra by connecting b to each triangular face we can form
        
        # Simplified approach: connect barycenter to all possible valid triangles
        all_points = list(range(4))  # original vertex local indices
        int_point_indices = [p_indices[e] for e in int_edges]
        
        # For each original vertex, create tet with b and the 2 nearest intersection points
        for local_v in range(4):
            incident = []
            for edge_idx, (a, b_local) in enumerate(edges):
                if (a == local_v or b_local == local_v) and edge_idx in int_edges:
                    incident.append(p_indices[edge_idx])
            
            if len(incident) == 2:
                new_tetrahedra.append((local_to_global[local_v], incident[0], incident[1], b_idx))
        
        # Connect pairs of original vertices that share an edge without intersection
        for edge_idx, (a, b_local) in enumerate(edges):
            if edge_idx not in int_edges:
                # This edge has no intersection - the two vertices are on the same side
                # Find the intersection points adjacent to both
                a_incident = [e for e in int_edges if a in edges[e]]
                b_incident = [e for e in int_edges if b_local in edges[e]]
                
                # Create tet connecting these two vertices to their shared intersection point and b
                shared = set(a_incident) & set(b_incident)
                if len(shared) >= 1:
                    p = p_indices[list(shared)[0]]
                    new_tetrahedra.append((local_to_global[a], local_to_global[b_local], p, b_idx))
    
    elif n_int == 4:
        # Quadrilateral K face - split into 2 triangles
        e0, e1, e2, e3 = int_edges
        
        # Order the quad vertices properly (by angle around centroid)
        quad_points = [(p_indices[e], intersections[e]) for e in int_edges]
        centroid = np.mean([p[1] for p in quad_points], axis=0)
        
        # Sort by angle
        ref = quad_points[0][1] - centroid
        ref_norm = np.linalg.norm(ref)
        if ref_norm > 1e-12:
            ref = ref / ref_norm
            normal = np.cross(quad_points[1][1] - centroid, quad_points[2][1] - centroid)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                normal = normal / normal_norm
                perp = np.cross(normal, ref)
                
                angles = []
                for idx, pos in quad_points:
                    v = pos - centroid
                    angle = np.arctan2(np.dot(v, perp), np.dot(v, ref))
                    angles.append((angle, idx))
                angles.sort()
                ordered = [a[1] for a in angles]
            else:
                ordered = [p[0] for p in quad_points]
        else:
            ordered = [p[0] for p in quad_points]
        
        # Two triangles for the quad
        K_faces.append((ordered[0], ordered[1], ordered[2]))
        K_faces.append((ordered[0], ordered[2], ordered[3]))
        
        # Create tetrahedra connecting b to faces
        for local_v in range(4):
            incident = []
            for edge_idx, (a, b_local) in enumerate(edges):
                if (a == local_v or b_local == local_v) and edge_idx in int_edges:
                    incident.append(p_indices[edge_idx])
            
            if len(incident) == 2:
                new_tetrahedra.append((local_to_global[local_v], incident[0], incident[1], b_idx))
            elif len(incident) == 3:
                # Three intersection points incident to this vertex
                new_tetrahedra.append((local_to_global[local_v], incident[0], incident[1], b_idx))
                new_tetrahedra.append((local_to_global[local_v], incident[1], incident[2], b_idx))
        
        # Handle edges without intersections
        for edge_idx, (a, b_local) in enumerate(edges):
            if edge_idx not in int_edges:
                # Find common adjacent intersection
                a_edges = [e for e in int_edges if a in edges[e]]
                b_edges = [e for e in int_edges if b_local in edges[e]]
                common = set(a_edges) & set(b_edges)
                
                for c_edge in common:
                    new_tetrahedra.append((
                        local_to_global[a], 
                        local_to_global[b_local], 
                        p_indices[c_edge], 
                        b_idx
                    ))
    
    else:
        # n_int >= 5 shouldn't happen for well-behaved surfaces
        new_tetrahedra.append((i0, i1, i2, i3))
    
    return new_vertices, new_tetrahedra, K_faces, barycenter_idx


def build_subdivided_complex_3d(
    vertices: Dict[int, np.ndarray],
    tetrahedra: List[Tuple[int, int, int, int]],
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray]
):
    """Build complete subdivided 3D complex with K as internal boundary."""
    all_vertices = list(vertices.values())
    vertex_id_to_idx = {vid: idx for idx, vid in enumerate(vertices.keys())}
    
    all_tetrahedra = []
    K_faces = []
    barycenter_indices = []
    
    for tet in tetrahedra:
        tet_verts = [vertices[tet[i]] for i in range(4)]
        tet_indices = [vertex_id_to_idx[tet[i]] for i in range(4)]
        
        new_vert_start = len(all_vertices)
        
        new_verts, new_tets, k_faces, b_idx = subdivide_tetrahedron_figure2(
            tet_verts, tet_indices, f, grad_f, new_vert_start
        )
        
        all_vertices.extend(new_verts)
        all_tetrahedra.extend(new_tets)
        K_faces.extend(k_faces)
        
        if b_idx is not None:
            barycenter_indices.append(b_idx)
    
    return np.array(all_vertices), all_tetrahedra, K_faces, barycenter_indices


# =============================================================================
# Visualization
# =============================================================================

def visualize_whitney_3d(
    f, grad_f, box_min, box_max, L,
    required_dist_factor=1.0,
    output_path=None,
    title="",
    show_tetrahedra=False,
    elev=25, azim=45
):
    """Visualize 3D Whitney algorithm."""
    
    # Generate and perturb
    vertices, tetrahedra = generate_kuhn_triangulation(box_min, box_max, L)
    perturbed, info = perturb_vertices_3d(
        vertices, tetrahedra, f, grad_f, L, required_dist_factor
    )
    
    # Build K surface
    K_vertices, K_faces = build_K_surface(perturbed, tetrahedra, f, grad_f)
    
    n_moved = sum(1 for i in info.values() if i.get('action') == 'moved')
    
    print(f"\n{title}")
    print(f"  T̃: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra")
    print(f"  Perturbed: {n_moved} vertices")
    print(f"  K: {len(K_vertices)} vertices, {len(K_faces)} faces")
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw K faces (green surface)
    if K_faces:
        K_verts_arr = np.array(K_vertices)
        faces_3d = [[K_verts_arr[f[0]], K_verts_arr[f[1]], K_verts_arr[f[2]]] for f in K_faces]
        
        poly = Poly3DCollection(faces_3d, alpha=0.7, linewidths=0.5)
        poly.set_facecolor('#90EE90')  # Light green
        poly.set_edgecolor('#228B22')  # Dark green
        ax.add_collection3d(poly)
    
    # Draw K vertices
    if K_vertices:
        K_arr = np.array(K_vertices)
        ax.scatter(K_arr[:, 0], K_arr[:, 1], K_arr[:, 2], 
                  c='green', s=20, edgecolors='darkgreen')
    
    # Draw perturbed vertices (red)
    moved_verts = []
    for idx, v in vertices.items():
        if info[idx].get('action') == 'moved':
            moved_verts.append(perturbed[idx])
    if moved_verts:
        mv_arr = np.array(moved_verts)
        ax.scatter(mv_arr[:, 0], mv_arr[:, 1], mv_arr[:, 2],
                  c='red', s=50, edgecolors='darkred', label='Perturbed vertices')
    
    # Optionally draw some tetrahedra edges
    if show_tetrahedra:
        edges = get_all_edges(tetrahedra)
        edge_lines = []
        for e in edges:
            v0, v1 = perturbed[e[0]], perturbed[e[1]]
            edge_lines.append([v0, v1])
        lc = Line3DCollection(edge_lines, colors='gray', linewidths=0.3, alpha=0.3)
        ax.add_collection3d(lc)
    
    # Styling
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\nK: {len(K_faces)} triangles | Perturbed: {n_moved} vertices")
    ax.view_init(elev=elev, azim=azim)
    
    # Set equal aspect ratio
    max_range = np.max(box_max - box_min) / 2
    mid = (box_max + box_min) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    else:
        plt.show()
    
    return {
        'vertices': perturbed,
        'tetrahedra': tetrahedra,
        'K_vertices': K_vertices,
        'K_faces': K_faces
    }


# =============================================================================
# Example surfaces
# =============================================================================

def sphere(center=(0, 0, 0), radius=1.0):
    """Sphere centered at (cx, cy, cz) with given radius."""
    cx, cy, cz = center
    f = lambda p: (p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2 - radius**2
    grad_f = lambda p: np.array([2*(p[0]-cx), 2*(p[1]-cy), 2*(p[2]-cz)])
    return f, grad_f, radius


def ellipsoid(a=1.0, b=0.7, c=0.5):
    """Ellipsoid with semi-axes a, b, c."""
    f = lambda p: (p[0]/a)**2 + (p[1]/b)**2 + (p[2]/c)**2 - 1
    grad_f = lambda p: np.array([2*p[0]/a**2, 2*p[1]/b**2, 2*p[2]/c**2])
    reach = min(c**2/a, c**2/b, b**2/a)  # Approximate
    return f, grad_f, reach


def torus(R=1.0, r=0.3):
    """Torus with major radius R and minor radius r."""
    def f(p):
        x, y, z = p
        return (np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2
    
    def grad_f(p):
        x, y, z = p
        rho = np.sqrt(x**2 + y**2)
        if rho < 1e-12:
            return np.array([0.0, 0.0, 2*z])
        factor = 2 * (rho - R) / rho
        return np.array([factor * x, factor * y, 2*z])
    
    return f, grad_f, r


if __name__ == "__main__":
    output_dir = "/mnt/user-data/outputs"
    
    # Example 1: Sphere
    print("=" * 60)
    print("Example 1: Sphere")
    print("=" * 60)
    f, grad_f, reach = sphere(radius=1.0)
    L = 0.25
    visualize_whitney_3d(
        f, grad_f,
        box_min=np.array([-1.3, -1.3, -1.3]),
        box_max=np.array([1.3, 1.3, 1.3]),
        L=L,
        required_dist_factor=30,
        output_path=f"{output_dir}/whitney3d_sphere.png",
        title=f"Whitney 3D: Sphere (L={L})",
        show_tetrahedra=False,
        elev=25, azim=45
    )
    
    # Example 2: Ellipsoid
    print("\n" + "=" * 60)
    print("Example 2: Ellipsoid")
    print("=" * 60)
    f, grad_f, reach = ellipsoid(a=1.2, b=0.8, c=0.5)
    L = 0.2
    visualize_whitney_3d(
        f, grad_f,
        box_min=np.array([-1.5, -1.0, -0.8]),
        box_max=np.array([1.5, 1.0, 0.8]),
        L=L,
        required_dist_factor=30,
        output_path=f"{output_dir}/whitney3d_ellipsoid.png",
        title=f"Whitney 3D: Ellipsoid (L={L})",
        show_tetrahedra=False,
        elev=20, azim=30
    )
    
    # Example 3: Torus
    print("\n" + "=" * 60)
    print("Example 3: Torus")
    print("=" * 60)
    f, grad_f, reach = torus(R=1.0, r=0.4)
    L = 0.2
    visualize_whitney_3d(
        f, grad_f,
        box_min=np.array([-1.6, -1.6, -0.6]),
        box_max=np.array([1.6, 1.6, 0.6]),
        L=L,
        required_dist_factor=25,
        output_path=f"{output_dir}/whitney3d_torus.png",
        title=f"Whitney 3D: Torus (L={L})",
        show_tetrahedra=False,
        elev=30, azim=45
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)