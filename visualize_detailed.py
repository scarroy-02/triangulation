"""
Detailed visualization of Whitney algorithm on a small section.

Shows:
1. Ambient Coxeter triangulation (tetrahedra edges)
2. Manifold M (implicit surface)
3. K surface (Whitney approximation)
4. Vertex perturbations
5. Edge-manifold intersections

This allows verification that the algorithm is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import Dict, List, Tuple, Callable
from coxeter_A3 import generate_coxeter_A3, get_all_edges, get_tetrahedron_edges
from utils import rho_1, c_tilde


def find_nearest_point_on_M(v, f, grad_f, max_iter=20, tol=1e-12):
    """Find nearest point p on M = {f = 0} to v."""
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


def distance_to_tangent_plane(v, p, grad_f):
    """Compute distance from v to tangent plane T_p M."""
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


def perturb_vertices_3d(vertices, tetrahedra, f, grad_f, L, required_dist_factor=1.0):
    """Perturb vertices according to Whitney's algorithm."""
    d, n = 3, 2
    rho = rho_1(d, n)
    c = c_tilde(d)
    required_dist = rho * c * L * required_dist_factor
    max_perturb = c * L * required_dist_factor
    near_threshold = 3 * L / 2
    
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
            'direction': direction.copy(),
            'move_amount': move_amount
        }
    
    return perturbed, info


def find_edge_intersection(v0, v1, f, grad_f, tol=1e-10):
    """Find intersection point on edge with manifold M."""
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


def build_K_surface(vertices, tetrahedra, f, grad_f):
    """Build K surface from edge-manifold intersections."""
    K_vertices = []
    K_faces = []
    edge_to_K_idx = {}
    
    def get_K_vertex(edge, intersection_point):
        if edge not in edge_to_K_idx:
            idx = len(K_vertices)
            K_vertices.append(intersection_point.copy())
            edge_to_K_idx[edge] = idx
        return edge_to_K_idx[edge]
    
    for tet in tetrahedra:
        edges = get_tetrahedron_edges(tet)
        intersections = []
        
        for e in edges:
            v0, v1 = vertices[e[0]], vertices[e[1]]
            p = find_edge_intersection(v0, v1, f, grad_f)
            if p is not None:
                k_idx = get_K_vertex(e, p)
                intersections.append((e, k_idx, p))
        
        n_int = len(intersections)
        
        if n_int == 3:
            i0, i1, i2 = intersections[0][1], intersections[1][1], intersections[2][1]
            K_faces.append((i0, i1, i2))
        elif n_int == 4:
            points = [(intersections[i][1], intersections[i][2]) for i in range(4)]
            centroid = np.mean([p[1] for p in points], axis=0)
            
            ref = points[0][1] - centroid
            ref_norm = np.linalg.norm(ref)
            if ref_norm > 1e-12:
                ref = ref / ref_norm
                normal = np.cross(points[1][1] - centroid, points[2][1] - centroid)
                normal_norm = np.linalg.norm(normal)
                if normal_norm > 1e-12:
                    normal = normal / normal_norm
                    perp = np.cross(normal, ref)
                    angles = []
                    for idx, pos in points:
                        v = pos - centroid
                        angle = np.arctan2(np.dot(v, perp), np.dot(v, ref))
                        angles.append((angle, idx))
                    angles.sort()
                    ordered = [a[1] for a in angles]
                else:
                    ordered = [p[0] for p in points]
            else:
                ordered = [p[0] for p in points]
            
            K_faces.append((ordered[0], ordered[1], ordered[2]))
            K_faces.append((ordered[0], ordered[2], ordered[3]))
    
    return K_vertices, K_faces, edge_to_K_idx


def sample_implicit_surface(f, box_min, box_max, n_points=50):
    """Sample points on the implicit surface for visualization."""
    from skimage import measure
    
    # Create grid
    x = np.linspace(box_min[0], box_max[0], n_points)
    y = np.linspace(box_min[1], box_max[1], n_points)
    z = np.linspace(box_min[2], box_max[2], n_points)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Evaluate function
    F = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            for k in range(n_points):
                F[i,j,k] = f(np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]]))
    
    # Extract isosurface using marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(F, level=0)
        # Scale vertices back to world coordinates
        scale = (box_max - box_min) / (n_points - 1)
        verts = verts * scale + box_min
        return verts, faces
    except:
        return None, None


def visualize_detailed_section(
    f, grad_f,
    box_min, box_max, L,
    required_dist_factor=50.0,
    output_path=None,
    title="",
    elev=25, azim=45,
    show_M=True
):
    """
    Create detailed visualization of a small section.
    """
    print(f"\n{'='*60}")
    print(f"Generating: {title}")
    print(f"Box: {box_min} to {box_max}")
    print(f"L = {L}")
    print(f"{'='*60}")
    
    # Generate Coxeter triangulation
    vertices, tetrahedra = generate_coxeter_A3(box_min, box_max, L)
    print(f"Coxeter Ã₃: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra")
    
    # Perturb vertices
    perturbed, info = perturb_vertices_3d(
        vertices, tetrahedra, f, grad_f, L, required_dist_factor
    )
    n_moved = sum(1 for i in info.values() if i.get('action') == 'moved')
    print(f"Perturbed: {n_moved} vertices")
    
    # Build K surface
    K_vertices, K_faces, edge_intersections = build_K_surface(
        perturbed, tetrahedra, f, grad_f
    )
    print(f"K surface: {len(K_vertices)} vertices, {len(K_faces)} faces")
    print(f"Edge intersections: {len(edge_intersections)}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Draw all tetrahedral edges (gray, thin)
    edges = get_all_edges(tetrahedra)
    edge_lines = []
    for e in edges:
        v0, v1 = perturbed[e[0]], perturbed[e[1]]
        edge_lines.append([v0, v1])
    
    lc = Line3DCollection(edge_lines, colors='gray', linewidths=0.5, alpha=0.4)
    ax.add_collection3d(lc)
    
    # 2. Draw edges that intersect M (blue, thicker)
    intersecting_edge_lines = []
    for e in edge_intersections.keys():
        v0, v1 = perturbed[e[0]], perturbed[e[1]]
        intersecting_edge_lines.append([v0, v1])
    
    if intersecting_edge_lines:
        lc_int = Line3DCollection(intersecting_edge_lines, colors='blue', 
                                   linewidths=1.5, alpha=0.7)
        ax.add_collection3d(lc_int)
    
    # 3. Draw K surface (green, semi-transparent)
    if K_faces:
        K_arr = np.array(K_vertices)
        K_face_verts = [[K_arr[f[0]], K_arr[f[1]], K_arr[f[2]]] for f in K_faces]
        pc = Poly3DCollection(K_face_verts, alpha=0.6, linewidths=1)
        pc.set_facecolor('#90EE90')
        pc.set_edgecolor('#006400')
        ax.add_collection3d(pc)
    
    # 4. Draw intersection points (green dots)
    if K_vertices:
        K_arr = np.array(K_vertices)
        ax.scatter(K_arr[:, 0], K_arr[:, 1], K_arr[:, 2],
                  c='green', s=30, edgecolors='darkgreen', zorder=10)
    
    # 5. Draw actual manifold M using marching cubes (if possible)
    if show_M:
        try:
            M_verts, M_faces = sample_implicit_surface(f, box_min, box_max, n_points=40)
            if M_verts is not None and len(M_faces) > 0:
                M_face_verts = [[M_verts[f[0]], M_verts[f[1]], M_verts[f[2]]] for f in M_faces]
                pc_M = Poly3DCollection(M_face_verts, alpha=0.3, linewidths=0)
                pc_M.set_facecolor('#ADD8E6')
                pc_M.set_edgecolor('none')
                ax.add_collection3d(pc_M)
                print(f"Manifold M: {len(M_verts)} vertices, {len(M_faces)} faces (marching cubes)")
        except Exception as e:
            print(f"Could not render M: {e}")
    
    # 6. Draw original vertices (small gray dots)
    orig_verts = np.array([vertices[i] for i in vertices.keys()])
    ax.scatter(orig_verts[:, 0], orig_verts[:, 1], orig_verts[:, 2],
              c='gray', s=10, alpha=0.3)
    
    # 7. Draw perturbed vertices that moved (red dots with arrows)
    for idx in vertices.keys():
        if info[idx].get('action') == 'moved':
            v_orig = info[idx]['v_orig']
            v_pert = info[idx]['v_pert']
            
            # Original position (white with gray edge)
            ax.scatter([v_orig[0]], [v_orig[1]], [v_orig[2]],
                      c='white', s=80, edgecolors='gray', linewidths=2, zorder=11)
            
            # Perturbed position (red)
            ax.scatter([v_pert[0]], [v_pert[1]], [v_pert[2]],
                      c='red', s=60, edgecolors='darkred', zorder=12)
            
            # Arrow showing perturbation
            ax.quiver(v_orig[0], v_orig[1], v_orig[2],
                     v_pert[0]-v_orig[0], v_pert[1]-v_orig[1], v_pert[2]-v_orig[2],
                     color='red', arrow_length_ratio=0.3, linewidth=2)
    
    # Styling
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\n"
                f"Coxeter Ã₃: {len(tetrahedra)} tets | "
                f"Perturbed: {n_moved} | "
                f"K: {len(K_faces)} faces | "
                f"Intersecting edges: {len(edge_intersections)}")
    ax.view_init(elev=elev, azim=azim)
    
    # Equal aspect
    max_range = np.max(box_max - box_min) / 2
    mid = (box_max + box_min) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=0.5, alpha=0.4, label='T̃ edges (ambient)'),
        Line2D([0], [0], color='blue', linewidth=1.5, label='Edges intersecting M'),
        Patch(facecolor='#90EE90', edgecolor='#006400', alpha=0.6, label='K surface'),
        Patch(facecolor='#ADD8E6', edgecolor='none', alpha=0.3, label='M (actual manifold)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='gray', markersize=10, label='Original vertex'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=8, label='Perturbed vertex'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markeredgecolor='darkgreen', markersize=8, label='K vertex (intersection)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    return {
        'vertices': perturbed,
        'tetrahedra': tetrahedra,
        'K_vertices': K_vertices,
        'K_faces': K_faces,
        'info': info
    }


# Manifold definitions
def sphere(center=(0, 0, 0), radius=1.0):
    cx, cy, cz = center
    f = lambda p: (p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2 - radius**2
    grad_f = lambda p: np.array([2*(p[0]-cx), 2*(p[1]-cy), 2*(p[2]-cz)])
    return f, grad_f, radius


def plane(normal=(0, 0, 1), offset=0.0):
    """Plane: n · x = offset"""
    n = np.array(normal) / np.linalg.norm(normal)
    f = lambda p: np.dot(n, p) - offset
    grad_f = lambda p: n.copy()
    return f, grad_f, float('inf')


if __name__ == "__main__":
    import sys
    
    # Install scikit-image for marching cubes if needed
    try:
        from skimage import measure
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "scikit-image", "--break-system-packages", "-q"])
        from skimage import measure
    
    output_dir = "/mnt/user-data/outputs"
    
    # Example 1: Small section of sphere
    print("\n" + "="*70)
    print("Example 1: Small section of sphere - detailed view")
    print("="*70)
    
    f, grad_f, _ = sphere(center=(0, 0, 0), radius=1.0)
    
    # Small box around top of sphere
    visualize_detailed_section(
        f, grad_f,
        box_min=np.array([-0.4, -0.4, 0.7]),
        box_max=np.array([0.4, 0.4, 1.1]),
        L=0.15,
        required_dist_factor=80,
        output_path=f"{output_dir}/detailed_sphere_top.png",
        title="Sphere (top section): Coxeter Ã₃ + Whitney",
        elev=30, azim=45
    )
    
    # Small box around equator
    visualize_detailed_section(
        f, grad_f,
        box_min=np.array([0.6, -0.4, -0.4]),
        box_max=np.array([1.1, 0.4, 0.4]),
        L=0.15,
        required_dist_factor=80,
        output_path=f"{output_dir}/detailed_sphere_equator.png",
        title="Sphere (equator section): Coxeter Ã₃ + Whitney",
        elev=20, azim=30
    )
    
    # Example 2: Plane (simpler case)
    print("\n" + "="*70)
    print("Example 2: Plane - simple verification")
    print("="*70)
    
    f, grad_f, _ = plane(normal=(1, 1, 1), offset=0.0)
    
    visualize_detailed_section(
        f, grad_f,
        box_min=np.array([-0.5, -0.5, -0.5]),
        box_max=np.array([0.5, 0.5, 0.5]),
        L=0.2,
        required_dist_factor=50,
        output_path=f"{output_dir}/detailed_plane.png",
        title="Plane (x+y+z=0): Coxeter Ã₃ + Whitney",
        elev=25, azim=45
    )
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)