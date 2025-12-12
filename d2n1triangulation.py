import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from coxeter import generate_coxeter_A2, get_edges
from perturb import find_nearest_point_on_M, distance_to_tangent_space
from utils import rho_1, c_tilde


def perturb_vertices(vertices, triangles, f, grad_f, L, required_dist_factor=1.0, d=2, n=1):
    """Perturb vertices according to Whitney's algorithm."""
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
        dist_to_TpM, direction = distance_to_tangent_space(v, p, grad_f)
        
        if dist_to_TpM >= required_dist:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 2, 'action': 'already_far', 'dist_to_TpM': dist_to_TpM}
            continue
        
        move_amount = min(required_dist - dist_to_TpM, max_perturb)
        v_pert = v + direction * move_amount
        
        perturbed[idx] = v_pert
        info[idx] = {
            'case': 2, 'action': 'moved',
            'v_orig': v.copy(), 'v_pert': v_pert.copy(),
            'direction': direction.copy(), 'move_amount': move_amount
        }
    
    return perturbed, info


def find_intersection_on_edge(v0, v1, f, grad_f, tol=1e-10):
    """Find intersection point on edge (v0, v1) with manifold M."""
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


def subdivide_triangle_figure2(tri_verts, tri_indices, f, grad_f, new_vertex_start_idx):
    """
    Subdivide triangle according to Figure 2 construction.
    
    For intersected triangles: creates 5 triangles using midpoint of K edge.
    For non-intersected: keeps original triangle.
    
    Returns:
        new_vertices: List of new vertices (intersection points + barycenter)
        new_triangles: List of triangles as (idx0, idx1, idx2)
        K_edge: The K edge (p1_idx, p2_idx) if exists
        barycenter_idx: Index of barycenter if exists
    """
    v0, v1, v2 = tri_verts
    i0, i1, i2 = tri_indices
    
    # Find intersections on each edge
    # Edge 0: v0-v1, Edge 1: v1-v2, Edge 2: v2-v0
    edges_verts = [(v0, v1), (v1, v2), (v2, v0)]
    edges_indices = [(i0, i1), (i1, i2), (i2, i0)]
    
    intersections = {}  # edge_idx -> intersection point
    for edge_idx, (va, vb) in enumerate(edges_verts):
        p = find_intersection_on_edge(va, vb, f, grad_f)
        if p is not None:
            intersections[edge_idx] = p
    
    new_vertices = []
    new_triangles = []
    K_edge = None
    barycenter_idx = None
    
    if len(intersections) != 2:
        # No proper intersection - keep original triangle
        new_triangles.append((i0, i1, i2))
        return new_vertices, new_triangles, K_edge, barycenter_idx
    
    # Get the two intersected edges
    int_edges = sorted(intersections.keys())
    e0, e1 = int_edges
    p1 = intersections[e0]
    p2 = intersections[e1]
    
    # Create new vertices: p1, p2, and barycenter b
    p1_idx = new_vertex_start_idx
    p2_idx = new_vertex_start_idx + 1
    b = (p1 + p2) / 2  # midpoint = barycenter
    b_idx = new_vertex_start_idx + 2
    
    new_vertices = [p1, p2, b]
    K_edge = (p1_idx, p2_idx)
    barycenter_idx = b_idx
    
    # Create 5 triangles by connecting b to all vertices and intersection points
    # The structure depends on which two edges are intersected
    
    # Edge definitions:
    # Edge 0: v0-v1 (vertices at local indices 0,1)
    # Edge 1: v1-v2 (vertices at local indices 1,2)
    # Edge 2: v2-v0 (vertices at local indices 2,0)
    
    # For each edge, identify its endpoints (local vertex indices)
    edge_endpoints = {0: (0, 1), 1: (1, 2), 2: (2, 0)}
    
    # Get local vertex indices for the intersected edges
    e0_v0, e0_v1 = edge_endpoints[e0]  # p1 is on this edge
    e1_v0, e1_v1 = edge_endpoints[e1]  # p2 is on this edge
    
    # Map local indices to actual vertex indices
    local_to_global = {0: i0, 1: i1, 2: i2}
    local_to_pos = {0: v0, 1: v1, 2: v2}
    
    # Find the shared vertex between the two edges (if adjacent)
    e0_set = set([e0_v0, e0_v1])
    e1_set = set([e1_v0, e1_v1])
    shared = e0_set & e1_set
    
    # Create the 5 triangles
    # We traverse around the triangle, creating triangles with b
    
    if len(shared) == 1:
        # Adjacent edges - one shared vertex
        shared_v = shared.pop()
        
        # Find the other vertices
        other_e0 = (e0_set - {shared_v}).pop()  # vertex on e0 not shared
        other_e1 = (e1_set - {shared_v}).pop()  # vertex on e1 not shared
        
        # The 5 triangles going around:
        # Starting from other_e0, going to p1, then shared_v, then p2, then other_e1, back to other_e0
        
        # Triangle 1: (other_e0, p1, b)
        new_triangles.append((local_to_global[other_e0], p1_idx, b_idx))
        
        # Triangle 2: (p1, shared_v, b)
        new_triangles.append((p1_idx, local_to_global[shared_v], b_idx))
        
        # Triangle 3: (shared_v, p2, b)
        new_triangles.append((local_to_global[shared_v], p2_idx, b_idx))
        
        # Triangle 4: (p2, other_e1, b)
        new_triangles.append((p2_idx, local_to_global[other_e1], b_idx))
        
        # Triangle 5: (other_e1, other_e0, b)
        new_triangles.append((local_to_global[other_e1], local_to_global[other_e0], b_idx))
        
    else:
        # Non-adjacent edges (e.g., edges 0 and 2)
        # This means one vertex is "alone" between the two intersection points
        
        # For edges 0 (v0-v1) and 2 (v2-v0): shared vertex is v0
        # Actually for non-adjacent, there's no shared vertex
        # Edges 0 and 2 share v0
        # Let me reconsider - in a triangle, any two edges share exactly one vertex
        
        # Actually edges are always adjacent in a triangle, so shared should always have 1 element
        # Let me handle the general case more carefully
        
        all_verts = {0, 1, 2}
        verts_on_edges = e0_set | e1_set
        
        # If edges share a vertex, we have 3 distinct vertices touched
        # The shared one is in the "middle"
        
        # Fallback: just create triangles around b
        for local_v in [0, 1, 2]:
            next_v = (local_v + 1) % 3
            # Check if this edge has an intersection
            edge_of_this = local_v  # edge local_v connects local_v to next_v
            
        # Actually let's just enumerate based on which edges
        if (e0, e1) == (0, 1):
            # p1 on v0-v1, p2 on v1-v2, shared = v1
            # Going around: v0 -> p1 -> v1 -> p2 -> v2 -> v0
            new_triangles = [
                (i0, p1_idx, b_idx),      # v0, p1, b
                (p1_idx, i1, b_idx),      # p1, v1, b
                (i1, p2_idx, b_idx),      # v1, p2, b
                (p2_idx, i2, b_idx),      # p2, v2, b
                (i2, i0, b_idx),          # v2, v0, b
            ]
        elif (e0, e1) == (0, 2):
            # p1 on v0-v1, p2 on v2-v0, shared = v0
            # Going around: v0 -> p1 -> v1 -> v2 -> p2 -> v0
            # But we need to connect through b
            # Actually: v1 -> v2 is an uninterrupted edge
            new_triangles = [
                (i0, p1_idx, b_idx),      # v0, p1, b
                (p1_idx, i1, b_idx),      # p1, v1, b
                (i1, i2, b_idx),          # v1, v2, b
                (i2, p2_idx, b_idx),      # v2, p2, b
                (p2_idx, i0, b_idx),      # p2, v0, b
            ]
        elif (e0, e1) == (1, 2):
            # p1 on v1-v2, p2 on v2-v0, shared = v2
            # Going around: v0 -> v1 -> p1 -> v2 -> p2 -> v0
            new_triangles = [
                (i0, i1, b_idx),          # v0, v1, b
                (i1, p1_idx, b_idx),      # v1, p1, b
                (p1_idx, i2, b_idx),      # p1, v2, b
                (i2, p2_idx, b_idx),      # v2, p2, b
                (p2_idx, i0, b_idx),      # p2, v0, b
            ]
        else:
            # Shouldn't happen
            new_triangles = [(i0, i1, i2)]
    
    return new_vertices, new_triangles, K_edge, barycenter_idx


def build_subdivided_complex(vertices, triangles, f, grad_f):
    """
    Build complete subdivided complex with Figure 2 construction.
    """
    all_vertices = list(vertices.values())
    vertex_id_to_idx = {vid: idx for idx, vid in enumerate(vertices.keys())}
    
    all_triangles = []
    K_edges = []
    barycenter_indices = []
    
    for tri in triangles:
        tri_verts = [vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]]
        tri_indices = [vertex_id_to_idx[tri[0]], vertex_id_to_idx[tri[1]], vertex_id_to_idx[tri[2]]]
        
        new_vert_start = len(all_vertices)
        
        new_verts, new_tris, K_edge, b_idx = subdivide_triangle_figure2(
            tri_verts, tri_indices, f, grad_f, new_vert_start
        )
        
        all_vertices.extend(new_verts)
        all_triangles.extend(new_tris)
        
        if K_edge is not None:
            K_edges.append(K_edge)
        if b_idx is not None:
            barycenter_indices.append(b_idx)
    
    # Classify triangles as inside/outside based on centroid
    inside_mask = []
    outside_mask = []
    all_verts_arr = np.array(all_vertices)
    
    for tri in all_triangles:
        centroid = (all_verts_arr[tri[0]] + all_verts_arr[tri[1]] + all_verts_arr[tri[2]]) / 3
        if f(centroid) < 0:
            inside_mask.append(True)
            outside_mask.append(False)
        else:
            inside_mask.append(False)
            outside_mask.append(True)
    
    return (all_verts_arr, all_triangles, K_edges, barycenter_indices,
            np.array(inside_mask), np.array(outside_mask))


def plot_implicit_curve(ax, f, box_min, box_max, n_points=300, **kwargs):
    """Plot implicit curve f(x) = 0."""
    x = np.linspace(box_min[0], box_max[0], n_points)
    y = np.linspace(box_min[1], box_max[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([X[i,j], Y[i,j]])) for j in range(len(x))] for i in range(len(y))])
    defaults = {'colors': ['#0066CC'], 'linewidths': [2]}
    defaults.update(kwargs)
    ax.contour(X, Y, Z, levels=[0], **defaults)


def visualize_whitney_figure2(
    f, grad_f, box_min, box_max, L,
    required_dist_factor=1.0,
    output_path=None,
    title="",
    show_colors=True
):
    """
    Visualize Whitney algorithm with Figure 2 subdivision.
    """
    # Generate and perturb
    vertices, triangles = generate_coxeter_A2(box_min, box_max, L)
    perturbed, info = perturb_vertices(vertices, triangles, f, grad_f, L, required_dist_factor)
    
    # Build subdivided complex
    all_verts, all_tris, K_edges, barycenters, inside_mask, outside_mask = build_subdivided_complex(
        perturbed, triangles, f, grad_f
    )
    
    n_moved = sum(1 for i in info.values() if i.get('action') == 'moved')
    n_inside = np.sum(inside_mask)
    n_outside = np.sum(outside_mask)
    n_intersected = len(K_edges)
    
    print(f"\n{title}")
    print(f"  T̃: {len(vertices)} vertices, {len(triangles)} triangles")
    print(f"  Perturbed: {n_moved} vertices")
    print(f"  Intersected triangles: {n_intersected}")
    print(f"  Subdivided: {len(all_verts)} vertices, {len(all_tris)} triangles")
    print(f"  (Original: {len(triangles)}, Added: {n_intersected * 4} from subdivision)")
    print(f"  Inside: {n_inside}, Outside: {n_outside}")
    print(f"  K edges: {len(K_edges)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if show_colors:
        # Draw inside triangles (light blue)
        inside_patches = []
        for i, tri in enumerate(all_tris):
            if inside_mask[i]:
                verts = [all_verts[tri[0]], all_verts[tri[1]], all_verts[tri[2]]]
                inside_patches.append(Polygon(verts, closed=True))
        if inside_patches:
            pc = PatchCollection(inside_patches, facecolor='#ADD8E6', edgecolor='#4682B4', 
                                linewidth=0.5, alpha=0.7, zorder=1)
            ax.add_collection(pc)
        
        # Draw outside triangles (light pink)
        outside_patches = []
        for i, tri in enumerate(all_tris):
            if outside_mask[i]:
                verts = [all_verts[tri[0]], all_verts[tri[1]], all_verts[tri[2]]]
                outside_patches.append(Polygon(verts, closed=True))
        if outside_patches:
            pc = PatchCollection(outside_patches, facecolor='#FFB6C1', edgecolor='#CD5C5C',
                                linewidth=0.5, alpha=0.7, zorder=1)
            ax.add_collection(pc)
    
    # Draw all triangle edges
    drawn_edges = set()
    for tri in all_tris:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            if e not in drawn_edges:
                v0, v1 = all_verts[e[0]], all_verts[e[1]]
                ax.plot([v0[0], v1[0]], [v0[1], v1[1]], '-',
                       color='gray', linewidth=0.5, zorder=2)
                drawn_edges.add(e)
    
    # Draw K edges (green, thick)
    # for e in K_edges:
    #     v0, v1 = all_verts[e[0]], all_verts[e[1]]
    #     ax.plot([v0[0], v1[0]], [v0[1], v1[1]], '-',
    #            color='#228B22', linewidth=3.5, zorder=10)
    
    # Draw K vertices (intersection points)
    K_vert_indices = set()
    for e in K_edges:
        K_vert_indices.add(e[0])
        K_vert_indices.add(e[1])
    K_verts = np.array([all_verts[i] for i in K_vert_indices])
    if len(K_verts) > 0:
        ax.scatter(K_verts[:, 0], K_verts[:, 1], s=1, c='#228B22',
                  edgecolors='darkgreen', linewidths=1, zorder=11)
    
    # Draw barycenters (midpoints of K edges)
    if barycenters:
        b_verts = np.array([all_verts[i] for i in barycenters])
        ax.scatter(b_verts[:, 0], b_verts[:, 1], s=1, c='orange',
                  edgecolors='darkorange', linewidths=1, zorder=12, marker='s')
    
    # Draw original manifold M for reference
    plot_implicit_curve(ax, f, box_min - 2*L, box_max + 2*L, 
                       colors=['#0066CC'], linewidths=[0.25], linestyles=['--'])
    
    # Draw perturbation arrows
    # for idx, v_orig in vertices.items():
    #     v_pert = perturbed[idx]
    #     inf = info[idx]
        
    #     if inf.get('action') == 'moved':
    #         ax.scatter([v_orig[0]], [v_orig[1]], s=10, facecolors='white',
    #                   edgecolors='#888888', linewidths=1.5, zorder=5)
            
    #         dx = v_pert[0] - v_orig[0]
    #         dy = v_pert[1] - v_orig[1]
    #         arrow_len = np.sqrt(dx**2 + dy**2)
    #         min_len = L * 0.15
    #         if arrow_len > 1e-10 and arrow_len < min_len:
    #             scale = min_len / arrow_len
    #             dx *= scale
    #             dy *= scale
            
    #         ax.annotate('',
    #                    xy=(v_orig[0] + dx, v_orig[1] + dy),
    #                    xytext=(v_orig[0], v_orig[1]),
    #                    arrowprops=dict(arrowstyle='-|>', color='red', lw=2,
    #                                   mutation_scale=12),
    #                    zorder=6)
            
    #         ax.scatter([v_pert[0]], [v_pert[1]], s=10, c='red',
    #                   edgecolors='darkred', linewidths=1, zorder=7)
    
    # Styling
    ax.set_xlim(box_min[0] - 0.05, box_max[0] + 0.05)
    ax.set_ylim(box_min[1] - 0.05, box_max[1] + 0.05)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n{len(triangles)} orig triangles → {len(all_tris)} subdivided | K: {len(K_edges)} edges",
                fontsize=11)
    ax.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ADD8E6', edgecolor='#4682B4', alpha=0.7, label='Inside (f < 0)'),
        Patch(facecolor='#FFB6C1', edgecolor='#CD5C5C', alpha=0.7, label='Outside (f > 0)'),
        Line2D([0], [0], color='#228B22', linewidth=3.5, label='K (boundary)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#228B22',
               markeredgecolor='darkgreen', markersize=8, label='K vertices (p₁, p₂)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange',
               markeredgecolor='darkorange', markersize=8, label='Barycenter (midpoint)'),
        Line2D([0], [0], linestyle='--', color='#0066CC', linewidth=2, label='M (manifold)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=8, label='Perturbed vertex'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
    else:
        plt.show()
    
    return {
        'all_vertices': all_verts,
        'all_triangles': all_tris,
        'K_edges': K_edges,
        'barycenters': barycenters
    }


# Example manifolds
def circle(center=(0, 0), radius=1.0):
    cx, cy = center
    f = lambda p: (p[0] - cx)**2 + (p[1] - cy)**2 - radius**2
    grad_f = lambda p: np.array([2*(p[0] - cx), 2*(p[1] - cy)])
    return f, grad_f, radius


def ellipse(a=1.5, b=0.5):
    f = lambda p: (p[0]/a)**2 + (p[1]/b)**2 - 1
    grad_f = lambda p: np.array([2*p[0]/a**2, 2*p[1]/b**2])
    return f, grad_f, b**2/a


if __name__ == "__main__":
    output_dir = "./"
    
    # # Example 1: Circle (zoomed) - shows detail
    # print("=" * 60)
    # print("Example 1: Circle zoomed - Figure 2 subdivision")
    # print("=" * 60)
    # f, grad_f, reach = circle(radius=1.0)
    # L = 0.1
    # visualize_whitney_figure2(
    #     f, grad_f,
    #     box_min=np.array([-0.5, 0.5]),
    #     box_max=np.array([0.5, 1.15]),
    #     L=L,
    #     required_dist_factor=50,
    #     output_path=f"{output_dir}/circle_zoom.png",
    #     title=f"Circle (L={L})"
    # )
    
    # Example 2: Full circle
    print("\n" + "=" * 60)
    print("Example 2: Full circle - Figure 2 subdivision")
    print("=" * 60)
    f, grad_f, reach = circle(radius=1.0)
    L = 0.1
    visualize_whitney_figure2(
        f, grad_f,
        box_min=np.array([-1.3, -1.3]),
        box_max=np.array([1.3, 1.3]),
        L=L,
        required_dist_factor=50,
        output_path=f"{output_dir}/circle_full.pdf",
        title=f"Full Circle (L={L})"
    )
    
    # # Example 3: Ellipse
    # print("\n" + "=" * 60)
    # print("Example 3: Ellipse - Figure 2 subdivision")
    # print("=" * 60)
    # f, grad_f, reach = ellipse(a=1.2, b=0.5)
    # L = 0.15
    # visualize_whitney_figure2(
    #     f, grad_f,
    #     box_min=np.array([-1.5, -0.8]),
    #     box_max=np.array([1.5, 0.8]),
    #     L=L,
    #     required_dist_factor=50,
    #     output_path=f"{output_dir}/whitney_fig2_ellipse.png",
    #     title=f"Whitney Figure 2: Ellipse (L={L})"
    # )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)