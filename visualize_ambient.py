"""
Visualization of Whitney algorithm showing:
1. Ambient Coxeter Ã₃ triangulation (tetrahedral edges)
2. Manifold M (actual implicit surface)
3. K surface (Whitney PL approximation)
4. Tetrahedra classification (inside/outside/boundary)

Standalone script - no external module dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import Delaunay
from typing import Dict, List, Tuple, Callable


# =============================================================================
# Coxeter Ã₃ triangulation (BCC Delaunay)
# =============================================================================

def generate_bcc_lattice(box_min: np.ndarray, box_max: np.ndarray, L: float):
    """Generate BCC lattice points."""
    margin = L
    n_min = np.floor((box_min - margin) / L).astype(int)
    n_max = np.ceil((box_max + margin) / L).astype(int)
    
    vertices = {}
    idx = 0
    
    # Cubic vertices
    for i in range(n_min[0], n_max[0] + 1):
        for j in range(n_min[1], n_max[1] + 1):
            for k in range(n_min[2], n_max[2] + 1):
                pos = np.array([i, j, k], dtype=float) * L
                vertices[idx] = pos
                idx += 1
    
    # Body-centered vertices
    for i in range(n_min[0], n_max[0] + 1):
        for j in range(n_min[1], n_max[1] + 1):
            for k in range(n_min[2], n_max[2] + 1):
                pos = np.array([i + 0.5, j + 0.5, k + 0.5], dtype=float) * L
                vertices[idx] = pos
                idx += 1
    
    return vertices


def generate_coxeter_A3(box_min: np.ndarray, box_max: np.ndarray, L: float):
    """Generate Coxeter Ã₃ triangulation using BCC Delaunay."""
    vertices = generate_bcc_lattice(box_min, box_max, L)
    
    n_verts = len(vertices)
    points = np.array([vertices[i] for i in range(n_verts)])
    
    delaunay = Delaunay(points)
    
    tetrahedra = []
    margin = L
    for simplex in delaunay.simplices:
        tet_verts = points[simplex]
        centroid = np.mean(tet_verts, axis=0)
        if (np.all(centroid >= box_min - margin) and 
            np.all(centroid <= box_max + margin)):
            tetrahedra.append(tuple(simplex))
    
    return vertices, tetrahedra


def get_tetrahedron_edges(tet):
    """Get all 6 edges of a tetrahedron."""
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edges.append(tuple(sorted([tet[i], tet[j]])))
    return edges


def get_all_edges(tetrahedra):
    """Get all unique edges from tetrahedra."""
    edges = set()
    for tet in tetrahedra:
        for e in get_tetrahedron_edges(tet):
            edges.add(e)
    return edges


# =============================================================================
# Whitney algorithm
# =============================================================================

def find_nearest_point_on_M(v, f, grad_f, max_iter=20, tol=1e-12):
    """Find nearest point on M = {f = 0}."""
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
    """Distance from v to tangent plane T_p M."""
    gp = grad_f(p)
    gp_norm = np.linalg.norm(gp)
    if gp_norm < 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0])
    normal = gp / gp_norm
    v_minus_p = v - p
    signed_dist = np.dot(v_minus_p, normal)
    direction = normal if signed_dist >= 0 else -normal
    return abs(signed_dist), direction


def perturb_vertices_3d(vertices, tetrahedra, f, grad_f, L, required_dist_factor=50.0):
    """Perturb vertices according to Whitney's algorithm."""
    d, n = 3, 2
    t_d = 1 / np.sqrt(1 + 1/d)
    alpha = (1 - t_d) / 2
    c_tilde = (alpha ** (d + 1)) / 2
    rho_1 = (1 / (4 * (d - n))) * (1 - t_d ** 2)
    
    required_dist = rho_1 * c_tilde * L * required_dist_factor
    max_perturb = c_tilde * L * required_dist_factor
    near_threshold = 3 * L / 2
    
    perturbed = {}
    perturbation_info = {}
    
    for idx, v in vertices.items():
        f_val = f(v)
        grad_val = grad_f(v)
        grad_norm = np.linalg.norm(grad_val)
        
        if grad_norm < 1e-12:
            perturbed[idx] = v.copy()
            perturbation_info[idx] = {'moved': False}
            continue
        
        dist_to_M = abs(f_val) / grad_norm
        
        if dist_to_M >= near_threshold:
            perturbed[idx] = v.copy()
            perturbation_info[idx] = {'moved': False}
            continue
        
        p = find_nearest_point_on_M(v, f, grad_f)
        dist_to_TpM, direction = distance_to_tangent_plane(v, p, grad_f)
        
        if dist_to_TpM >= required_dist:
            perturbed[idx] = v.copy()
            perturbation_info[idx] = {'moved': False}
            continue
        
        move_amount = min(required_dist - dist_to_TpM, max_perturb)
        v_pert = v + direction * move_amount
        
        perturbed[idx] = v_pert
        perturbation_info[idx] = {
            'moved': True,
            'original': v.copy(),
            'perturbed': v_pert.copy(),
            'amount': move_amount
        }
    
    return perturbed, perturbation_info


def find_edge_intersection(v0, v1, f, grad_f, tol=1e-10):
    """Find intersection on edge with manifold M."""
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
    
    def get_K_vertex(edge, point):
        if edge not in edge_to_K_idx:
            idx = len(K_vertices)
            K_vertices.append(point.copy())
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
                intersections.append((k_idx, p))
        
        n_int = len(intersections)
        
        if n_int == 3:
            K_faces.append((intersections[0][0], intersections[1][0], intersections[2][0]))
        
        elif n_int == 4:
            points = intersections
            centroid = np.mean([p[1] for p in points], axis=0)
            ref = points[0][1] - centroid
            ref_norm = np.linalg.norm(ref)
            
            if ref_norm > 1e-12:
                ref = ref / ref_norm
                normal = np.cross(points[1][1] - centroid, points[2][1] - centroid)
                n_norm = np.linalg.norm(normal)
                if n_norm > 1e-12:
                    normal = normal / n_norm
                    perp = np.cross(normal, ref)
                    angles = [(np.arctan2(np.dot(p[1] - centroid, perp),
                                         np.dot(p[1] - centroid, ref)), p[0])
                             for p in points]
                    angles.sort()
                    ordered = [a[1] for a in angles]
                    K_faces.append((ordered[0], ordered[1], ordered[2]))
                    K_faces.append((ordered[0], ordered[2], ordered[3]))
    
    return K_vertices, K_faces, edge_to_K_idx


def classify_tetrahedra(vertices, tetrahedra, f):
    """Classify tetrahedra as inside (1), outside (2), or boundary (3)."""
    labels = []
    for tet in tetrahedra:
        signs = [np.sign(f(vertices[tet[i]])) for i in range(4)]
        if all(s <= 0 for s in signs):
            labels.append(1)  # Inside
        elif all(s >= 0 for s in signs):
            labels.append(2)  # Outside
        else:
            labels.append(3)  # Boundary
    return labels


# =============================================================================
# Manifold definitions
# =============================================================================

def sphere(center=(0, 0, 0), radius=1.0):
    cx, cy, cz = center
    f = lambda p: (p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2 - radius**2
    grad_f = lambda p: np.array([2*(p[0]-cx), 2*(p[1]-cy), 2*(p[2]-cz)])
    return f, grad_f


def torus(R=1.0, r=0.4):
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
    return f, grad_f


def plane(normal=(0, 0, 1), offset=0.0):
    n = np.array(normal) / np.linalg.norm(normal)
    f = lambda p: np.dot(n, p) - offset
    grad_f = lambda p: n.copy()
    return f, grad_f


# =============================================================================
# Visualization
# =============================================================================

def visualize_whitney_3d(
    f, grad_f,
    box_min, box_max, L,
    required_dist_factor=50.0,
    output_path=None,
    title="Whitney Triangulation",
    elev=25, azim=45,
    show_ambient=True,
    show_M=True,
    show_K=True,
    show_inside_tets=False,
    show_boundary_tets=True,
    clip_plane=None  # (axis, value, side) e.g., (0, 0.0, 'positive')
):
    """
    Comprehensive visualization of Whitney algorithm.
    
    Parameters:
    -----------
    clip_plane : tuple or None
        If provided, clips the view. Format: (axis, value, side)
        axis: 0=x, 1=y, 2=z
        side: 'positive' or 'negative'
    """
    print(f"\n{'='*60}")
    print(f"Generating visualization: {title}")
    print(f"Box: {box_min} to {box_max}, L={L}")
    print(f"{'='*60}")
    
    # Generate Coxeter triangulation
    vertices, tetrahedra = generate_coxeter_A3(box_min, box_max, L)
    print(f"Coxeter Ã₃: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra")
    
    # Perturb vertices
    perturbed, pert_info = perturb_vertices_3d(
        vertices, tetrahedra, f, grad_f, L, required_dist_factor
    )
    n_moved = sum(1 for info in pert_info.values() if info.get('moved'))
    print(f"Perturbed: {n_moved} vertices")
    
    # Build K surface
    K_vertices, K_faces, edge_intersections = build_K_surface(
        perturbed, tetrahedra, f, grad_f
    )
    K_arr = np.array(K_vertices) if K_vertices else np.array([]).reshape(0, 3)
    print(f"K surface: {len(K_vertices)} vertices, {len(K_faces)} faces")
    
    # Classify tetrahedra
    tet_labels = classify_tetrahedra(perturbed, tetrahedra, f)
    n_inside = sum(1 for l in tet_labels if l == 1)
    n_outside = sum(1 for l in tet_labels if l == 2)
    n_boundary = sum(1 for l in tet_labels if l == 3)
    print(f"Inside: {n_inside}, Outside: {n_outside}, Boundary: {n_boundary}")
    
    # Apply clipping if requested
    def is_visible(centroid):
        if clip_plane is None:
            return True
        axis, value, side = clip_plane
        if side == 'positive':
            return centroid[axis] >= value
        else:
            return centroid[axis] <= value
    
    # Create figure
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get all edges
    all_edges = get_all_edges(tetrahedra)
    
    # 1. Draw ambient triangulation edges (light gray)
    if show_ambient:
        ambient_lines = []
        for e in all_edges:
            v0, v1 = perturbed[e[0]], perturbed[e[1]]
            mid = (v0 + v1) / 2
            if is_visible(mid):
                ambient_lines.append([v0, v1])
        
        if ambient_lines:
            lc = Line3DCollection(ambient_lines, colors='lightgray', 
                                  linewidths=0.3, alpha=0.3)
            ax.add_collection3d(lc)
    
    # 2. Draw inside tetrahedra (light blue faces)
    if show_inside_tets:
        inside_faces = []
        for tet, label in zip(tetrahedra, tet_labels):
            if label == 1:  # Inside
                centroid = np.mean([perturbed[tet[i]] for i in range(4)], axis=0)
                if is_visible(centroid):
                    # All 4 faces
                    face_indices = [
                        [tet[1], tet[2], tet[3]],
                        [tet[0], tet[2], tet[3]],
                        [tet[0], tet[1], tet[3]],
                        [tet[0], tet[1], tet[2]],
                    ]
                    for fi in face_indices:
                        face_verts = [perturbed[i] for i in fi]
                        inside_faces.append(face_verts)
        
        if inside_faces:
            pc = Poly3DCollection(inside_faces, alpha=0.15, linewidths=0.1)
            pc.set_facecolor('#ADD8E6')
            pc.set_edgecolor('#87CEEB')
            ax.add_collection3d(pc)
    
    # 3. Draw boundary tetrahedra edges (darker)
    if show_boundary_tets:
        boundary_edges = set()
        for tet, label in zip(tetrahedra, tet_labels):
            if label == 3:  # Boundary
                centroid = np.mean([perturbed[tet[i]] for i in range(4)], axis=0)
                if is_visible(centroid):
                    for e in get_tetrahedron_edges(tet):
                        boundary_edges.add(e)
        
        boundary_lines = []
        for e in boundary_edges:
            v0, v1 = perturbed[e[0]], perturbed[e[1]]
            boundary_lines.append([v0, v1])
        
        if boundary_lines:
            lc = Line3DCollection(boundary_lines, colors='gray', 
                                  linewidths=0.6, alpha=0.5)
            ax.add_collection3d(lc)
    
    # 4. Draw edges that intersect M (blue)
    intersecting_lines = []
    for e in edge_intersections.keys():
        v0, v1 = perturbed[e[0]], perturbed[e[1]]
        mid = (v0 + v1) / 2
        if is_visible(mid):
            intersecting_lines.append([v0, v1])
    
    if intersecting_lines:
        lc = Line3DCollection(intersecting_lines, colors='blue', 
                              linewidths=1.2, alpha=0.7)
        ax.add_collection3d(lc)
    
    # 5. Draw actual manifold M using marching cubes
    if show_M:
        try:
            from skimage import measure
            
            n_mc = 40
            x = np.linspace(box_min[0], box_max[0], n_mc)
            y = np.linspace(box_min[1], box_max[1], n_mc)
            z = np.linspace(box_min[2], box_max[2], n_mc)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            F = np.zeros_like(X)
            for i in range(n_mc):
                for j in range(n_mc):
                    for k in range(n_mc):
                        F[i,j,k] = f(np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]]))
            
            verts_mc, faces_mc, _, _ = measure.marching_cubes(F, level=0)
            scale = (box_max - box_min) / (n_mc - 1)
            verts_mc = verts_mc * scale + box_min
            
            # Filter by clip plane
            visible_faces = []
            for face in faces_mc:
                centroid = np.mean(verts_mc[face], axis=0)
                if is_visible(centroid):
                    visible_faces.append([verts_mc[face[0]], verts_mc[face[1]], verts_mc[face[2]]])
            
            if visible_faces:
                pc = Poly3DCollection(visible_faces, alpha=0.25, linewidths=0)
                pc.set_facecolor('#FFB6C1')  # Light pink
                pc.set_edgecolor('none')
                ax.add_collection3d(pc)
                print(f"Manifold M: {len(visible_faces)} visible faces (marching cubes)")
        except ImportError:
            print("skimage not available - skipping M visualization")
        except Exception as e:
            print(f"Could not render M: {e}")
    
    # 6. Draw K surface (green)
    if show_K and K_faces:
        visible_K_faces = []
        for face in K_faces:
            centroid = np.mean([K_arr[face[i]] for i in range(3)], axis=0)
            if is_visible(centroid):
                visible_K_faces.append([K_arr[face[0]], K_arr[face[1]], K_arr[face[2]]])
        
        if visible_K_faces:
            pc = Poly3DCollection(visible_K_faces, alpha=0.7, linewidths=0.8)
            pc.set_facecolor('#90EE90')
            pc.set_edgecolor('#228B22')
            ax.add_collection3d(pc)
    
    # 7. Draw K vertices (intersection points)
    if K_vertices:
        visible_K_verts = [v for v in K_arr if is_visible(v)]
        if visible_K_verts:
            visible_K_verts = np.array(visible_K_verts)
            ax.scatter(visible_K_verts[:, 0], visible_K_verts[:, 1], visible_K_verts[:, 2],
                      c='#228B22', s=15, alpha=0.8, zorder=5)
    
    # 8. Draw vertices colored by sign of f
    for idx, v in perturbed.items():
        if is_visible(v):
            f_val = f(v)
            if abs(f_val) < 0.1 * L:  # Near M
                color = '#FFD700'  # Gold
                size = 25
            elif f_val > 0:
                color = '#FF6B6B'  # Red (outside)
                size = 15
            else:
                color = '#4DABF7'  # Blue (inside)
                size = 15
            ax.scatter([v[0]], [v[1]], [v[2]], c=color, s=size, alpha=0.4)
    
    # 9. Draw perturbation arrows
    for idx, info in pert_info.items():
        if info.get('moved'):
            v_orig = info['original']
            v_pert = info['perturbed']
            if is_visible(v_orig) or is_visible(v_pert):
                ax.scatter([v_orig[0]], [v_orig[1]], [v_orig[2]],
                          c='white', s=80, edgecolors='gray', linewidths=2, zorder=10)
                ax.scatter([v_pert[0]], [v_pert[1]], [v_pert[2]],
                          c='red', s=60, edgecolors='darkred', zorder=11)
                ax.quiver(v_orig[0], v_orig[1], v_orig[2],
                         v_pert[0]-v_orig[0], v_pert[1]-v_orig[1], v_pert[2]-v_orig[2],
                         color='red', arrow_length_ratio=0.3, linewidth=2)
    
    # 10. Draw clip plane if specified
    if clip_plane is not None:
        axis, value, side = clip_plane
        axis_names = ['X', 'Y', 'Z']
        
        if axis == 0:  # YZ plane
            yy, zz = np.meshgrid(
                np.linspace(box_min[1], box_max[1], 2),
                np.linspace(box_min[2], box_max[2], 2)
            )
            xx = np.full_like(yy, value)
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='red')
        elif axis == 1:  # XZ plane
            xx, zz = np.meshgrid(
                np.linspace(box_min[0], box_max[0], 2),
                np.linspace(box_min[2], box_max[2], 2)
            )
            yy = np.full_like(xx, value)
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='red')
        else:  # XY plane
            xx, yy = np.meshgrid(
                np.linspace(box_min[0], box_max[0], 2),
                np.linspace(box_min[1], box_max[1], 2)
            )
            zz = np.full_like(xx, value)
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='red')
    
    # Styling
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    clip_info = ""
    if clip_plane:
        axis_names = ['X', 'Y', 'Z']
        clip_info = f" | Clip: {axis_names[clip_plane[0]]}={clip_plane[1]:.1f}"
    
    ax.set_title(f"{title}\n"
                f"Coxeter Ã₃: {len(tetrahedra)} tets | "
                f"K: {len(K_faces)} faces | "
                f"Perturbed: {n_moved}{clip_info}",
                fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    
    # Equal aspect ratio
    max_range = np.max(box_max - box_min) / 2
    mid = (box_max + box_min) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='lightgray', linewidth=0.5, label='Ambient T̃ edges'),
        Line2D([0], [0], color='gray', linewidth=0.8, label='Boundary tet edges'),
        Line2D([0], [0], color='blue', linewidth=1.5, label='Edges crossing M'),
        Patch(facecolor='#FFB6C1', edgecolor='none', alpha=0.3, label='M (actual manifold)'),
        Patch(facecolor='#90EE90', edgecolor='#228B22', alpha=0.7, label='K surface (Whitney)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
               markersize=8, label='Vertex f>0 (outside)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4DABF7',
               markersize=8, label='Vertex f<0 (inside)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=8, label='Perturbed vertex'),
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
        'K_vertices': K_arr,
        'K_faces': K_faces,
        'tet_labels': tet_labels
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import os
    
    output_dir = "whitney_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Sphere - full view
    print("\n" + "="*70)
    print("Example 1: Sphere - full view with cross-section")
    print("="*70)
    
    f, grad_f = sphere(radius=1.0)
    
    # Full sphere with clip
    visualize_whitney_3d(
        f, grad_f,
        box_min=np.array([-1.3, -1.3, -1.3]),
        box_max=np.array([1.3, 1.3, 1.3]),
        L=0.3,
        output_path=f"{output_dir}/sphere_full_clipped.png",
        title="Sphere: Ambient Coxeter Ã₃ + K Surface",
        clip_plane=(0, 0.0, 'positive'),
        elev=20, azim=60,
        show_inside_tets=True
    )
    
    # Example 2: Small section for detailed view
    print("\n" + "="*70)
    print("Example 2: Sphere - small section (detailed)")
    print("="*70)
    
    visualize_whitney_3d(
        f, grad_f,
        box_min=np.array([-0.25, -0.25, 0.8]),
        box_max=np.array([0.25, 0.25, 1.05]),
        L=0.1,
        required_dist_factor=80,
        output_path=f"{output_dir}/sphere_section_detailed.png",
        title="Sphere (top section): Detailed View",
        elev=35, azim=45,
        show_inside_tets=False
    )
    
    # Example 3: Torus with clip
    print("\n" + "="*70)
    print("Example 3: Torus with cross-section")
    print("="*70)
    
    f, grad_f = torus(R=1.0, r=0.4)
    
    visualize_whitney_3d(
        f, grad_f,
        box_min=np.array([-1.6, -1.6, -0.6]),
        box_max=np.array([1.6, 1.6, 0.6]),
        L=0.25,
        output_path=f"{output_dir}/torus_clipped.png",
        title="Torus: Ambient Coxeter Ã₃ + K Surface",
        clip_plane=(1, 0.0, 'positive'),
        elev=25, azim=45,
        show_inside_tets=True
    )
    
    # Example 4: Plane (simple verification case)
    print("\n" + "="*70)
    print("Example 4: Plane z=0 (simple verification)")
    print("="*70)
    
    f, grad_f = plane(normal=(0, 0, 1), offset=0.0)
    
    visualize_whitney_3d(
        f, grad_f,
        box_min=np.array([-0.4, -0.4, -0.3]),
        box_max=np.array([0.4, 0.4, 0.3]),
        L=0.15,
        required_dist_factor=60,
        output_path=f"{output_dir}/plane_z0.png",
        title="Plane z=0: Simple Verification",
        elev=30, azim=30,
        show_M=True
    )
    
    print("\n" + "="*70)
    print("All visualizations complete!")
    print(f"Output directory: {output_dir}/")
    print("="*70)