"""
Whitney's Triangulation - Paper-Style Visualization
====================================================

Matching the style from Boissonnat, Kachanovich, Wintraecken (2021)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

# =============================================================================
# Coxeter Ã_2 Triangulation
# =============================================================================

def coxeter_A2_direct(box_min, box_max, L):
    """Generate Coxeter Ã_2 triangulation with equilateral triangles."""
    e1 = np.array([1.0, 0.0]) * L
    e2 = np.array([0.5, np.sqrt(3)/2]) * L
    
    B = np.column_stack([e1, e2])
    B_inv = np.linalg.inv(B)
    
    corners = [B_inv @ np.array([box_min[0], box_min[1]]),
               B_inv @ np.array([box_max[0], box_min[1]]),
               B_inv @ np.array([box_min[0], box_max[1]]),
               B_inv @ np.array([box_max[0], box_max[1]])]
    
    i_min = int(np.floor(min(c[0] for c in corners))) - 2
    i_max = int(np.ceil(max(c[0] for c in corners))) + 2
    j_min = int(np.floor(min(c[1] for c in corners))) - 2
    j_max = int(np.ceil(max(c[1] for c in corners))) + 2
    
    vertices = {}
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            p = i * e1 + j * e2
            if box_min[0] - L <= p[0] <= box_max[0] + L and \
               box_min[1] - L <= p[1] <= box_max[1] + L:
                vertices[(i, j)] = p
    
    triangles = []
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            v00, v10, v01, v11 = (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            if v00 in vertices and v10 in vertices and v01 in vertices:
                triangles.append([v00, v10, v01])
            if v10 in vertices and v11 in vertices and v01 in vertices:
                triangles.append([v10, v11, v01])
    
    return vertices, triangles

# =============================================================================
# Algorithm Constants
# =============================================================================

def compute_constants_2d(reach):
    d, n = 2, 1
    t = np.sqrt(2 * (d + 1) / (d * (d + 2)))
    mu_0 = np.sqrt((d + 1) / (2 * d))
    delta = (np.sqrt(d**2 + 2*d + 24) - np.sqrt(d**2 + 2*d)) / np.sqrt(12 * (d + 1))
    N = 3
    k = 1
    rho_1 = (2**(2*k - 2) * math.factorial(k)**2) / (np.pi * math.factorial(2*k) * N)
    L_est = 0.99 * (reach / 54)
    term1 = t * mu_0 * delta / (18 * d * L_est)
    term2 = t**2 / 24
    c_tilde = min(term1, term2, 1/24)
    return {'rho_1': rho_1, 'c_tilde': c_tilde, 'L': L_est}

# =============================================================================
# Implicit Curve
# =============================================================================

def circle_f(p, radius=1.0):
    return p[0]**2 + p[1]**2 - radius**2

def circle_grad(p):
    return np.array([2 * p[0], 2 * p[1]])

# =============================================================================
# Perturbation
# =============================================================================

def perturb_vertices(vertices, f, grad_f, L, c_tilde, rho_1, exaggerate=1.0, force_perturb=False):
    """Perturb vertices. exaggerate factor makes arrows more visible.
       force_perturb=True makes perturbations more likely for visualization."""
    perturbed = {}
    info = {}
    
    for idx, v in vertices.items():
        p = v.copy()
        for _ in range(20):
            val = f(p)
            if abs(val) < 1e-12:
                break
            g = grad_f(p)
            g2 = np.dot(g, g)
            if g2 < 1e-14:
                break
            p = p - (val / g2) * g
        
        dist = np.linalg.norm(v - p)
        
        if dist >= 1.5 * L:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 1, 'perturb': np.zeros(2), 'perturb_vis': np.zeros(2)}
        else:
            g = grad_f(p)
            g_norm = np.linalg.norm(g)
            normal = g / g_norm if g_norm > 1e-10 else np.array([1.0, 0.0])
            
            signed_dist = np.dot(v - p, normal)
            
            # For visualization, use larger required distance to trigger more perturbations
            if force_perturb:
                required_dist = 0.20 * L  # Threshold for perturbation
                max_perturb = 0.25 * L    # Larger visible perturbation amount
            else:
                required_dist = rho_1 * c_tilde * L
                max_perturb = c_tilde * L
            
            if abs(signed_dist) >= required_dist:
                perturbed[idx] = v.copy()
                info[idx] = {'case': 2, 'subcase': 'ok', 'perturb': np.zeros(2), 'perturb_vis': np.zeros(2)}
            else:
                target_dist = required_dist if signed_dist >= 0 else -required_dist
                perturb_amount = target_dist - signed_dist
                if abs(perturb_amount) > max_perturb:
                    perturb_amount = np.sign(perturb_amount) * max_perturb
                
                perturb_vec = perturb_amount * normal
                perturbed[idx] = v + perturb_vec
                # Exaggerated version for visualization
                perturb_vis = perturb_vec * exaggerate
                info[idx] = {'case': 2, 'subcase': 'moved', 'perturb': perturb_vec, 'perturb_vis': perturb_vis}
    
    return perturbed, info

# =============================================================================
# Build K
# =============================================================================

def find_edge_intersections(vertices, triangles, f, grad_f):
    intersections = {}
    
    def project(p):
        for _ in range(20):
            val = f(p)
            if abs(val) < 1e-12:
                break
            g = grad_f(p)
            g2 = np.dot(g, g)
            if g2 < 1e-14:
                break
            p = p - (val / g2) * g
        return p
    
    for tri in triangles:
        vals = [f(vertices[tri[i]]) for i in range(3)]
        for e in [(0, 1), (1, 2), (0, 2)]:
            i0, i1 = e
            if vals[i0] * vals[i1] > 0:
                continue
            edge_key = tuple(sorted([tri[i0], tri[i1]]))
            if edge_key in intersections:
                continue
            p0, p1 = vertices[tri[i0]], vertices[tri[i1]]
            t = vals[i0] / (vals[i0] - vals[i1]) if abs(vals[i0] - vals[i1]) > 1e-14 else 0.5
            p_int = p0 + t * (p1 - p0)
            intersections[edge_key] = project(p_int)
    
    return intersections

def build_K_2d(vertices, triangles, edge_intersections, f):
    K_vertices = []
    K_edges = []
    vertex_map = {}
    
    def get_vertex_idx(p):
        key = (round(p[0], 10), round(p[1], 10))
        if key not in vertex_map:
            vertex_map[key] = len(K_vertices)
            K_vertices.append(p)
        return vertex_map[key]
    
    for tri in triangles:
        vals = [f(vertices[tri[i]]) for i in range(3)]
        if min(vals) > 0 or max(vals) < 0:
            continue
        
        intersecting = []
        for e in [(0, 1), (1, 2), (0, 2)]:
            edge_key = tuple(sorted([tri[e[0]], tri[e[1]]]))
            if edge_key in edge_intersections:
                intersecting.append(edge_intersections[edge_key])
        
        if len(intersecting) < 2:
            continue
        
        v_tri = np.mean(intersecting, axis=0)
        
        for v_edge in intersecting:
            i0 = get_vertex_idx(v_edge)
            i1 = get_vertex_idx(v_tri)
            if i0 != i1:
                K_edges.append((i0, i1))
    
    return K_vertices, K_edges

# =============================================================================
# Paper-Style Visualization
# =============================================================================

def plot_curve(ax, f, box_min, box_max, n_points=500, color='#0066CC', linewidth=2):
    """Plot implicit curve."""
    x = np.linspace(box_min[0], box_max[0], n_points)
    y = np.linspace(box_min[1], box_max[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([X[j,i], Y[j,i]])) for i in range(len(x))] for j in range(len(y))])
    ax.contour(X, Y, Z, levels=[0], colors=[color], linewidths=[linewidth])

def draw_triangulation(ax, vertices, triangles, edge_color='black', vertex_color='black', 
                       edge_width=1.0, vertex_size=60, alpha=1.0):
    """Draw triangulation in paper style."""
    # Collect all edges
    edges = set()
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edges.add(e)
    
    # Draw edges
    for e in edges:
        if e[0] in vertices and e[1] in vertices:
            p0, p1 = vertices[e[0]], vertices[e[1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', 
                   color=edge_color, linewidth=edge_width, alpha=alpha, zorder=1)
    
    # Draw vertices
    verts = np.array([vertices[idx] for idx in vertices])
    ax.scatter(verts[:, 0], verts[:, 1], s=vertex_size, c=vertex_color, 
              edgecolors='black', linewidths=0.5, zorder=10, alpha=alpha)

def visualize_paper_style(f, grad_f, reach, box_min, box_max, output_prefix):
    """Create paper-style visualization."""
    
    const = compute_constants_2d(reach)
    c_tilde, rho_1, L = const['c_tilde'], const['rho_1'], const['L']
    
    # Generate triangulation
    vertices, triangles = coxeter_A2_direct(box_min, box_max, L)
    
    # For visualization, use larger perturbations that are actually visible
    # The paper's figures show visible perturbations for illustration
    # perturbed_vis, perturb_info_vis = perturb_vertices(vertices, f, grad_f, L, c_tilde, rho_1,
    #                                                     exaggerate=1.0, force_perturb=True)
    
    # For K computation, use the real (tiny) perturbations
    perturbed_real, perturb_info_real = perturb_vertices(vertices, f, grad_f, L, c_tilde, rho_1,
                                                          exaggerate=1.0, force_perturb=False)
    
    # Use visualization perturbations for drawing
    perturbed = perturbed_real
    perturb_info = perturb_info_real
    
    # Find intersections and build K
    edge_int = find_edge_intersections(perturbed, triangles, f, grad_f)
    K_verts, K_edges = build_K_2d(perturbed, triangles, edge_int, f)
    
    print(f"Vertices: {len(vertices)}, Triangles: {len(triangles)}")
    print(f"Edge intersections: {len(edge_int)}")
    print(f"K: {len(K_verts)} vertices, {len(K_edges)} edges")
    
    # =========================================================================
    # Figure 1: Perturbation (like top panel of Image 2)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Draw triangulation with gray for edges near M
    edges = set()
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edges.add(e)
    
    for e in edges:
        if e[0] in vertices and e[1] in vertices:
            p0_orig, p1_orig = vertices[e[0]], vertices[e[1]]
            p0_pert, p1_pert = perturbed[e[0]], perturbed[e[1]]
            
            # Check if edge is near M (at least one endpoint perturbed or near)
            info0, info1 = perturb_info[e[0]], perturb_info[e[1]]
            near_M = (info0['case'] == 2 or info1['case'] == 2)
            
            if near_M:
                # Draw original in gray
                ax1.plot([p0_orig[0], p1_orig[0]], [p0_orig[1], p1_orig[1]], '-',
                        color='gray', linewidth=1.0, alpha=0.6, zorder=1)
            # Draw perturbed in black
            ax1.plot([p0_pert[0], p1_pert[0]], [p0_pert[1], p1_pert[1]], '-',
                    color='black', linewidth=1.0, zorder=2)
    
    # Draw vertices
    for idx, v_orig in vertices.items():
        v_pert = perturbed[idx]
        info = perturb_info[idx]
        
        if info['case'] == 2 and info['subcase'] == 'moved':
            # Gray circle for original position
            ax1.scatter([v_orig[0]], [v_orig[1]], s=70, c='gray', 
                       edgecolors='darkgray', linewidths=1, zorder=8)
            # Black circle for perturbed position (actual moved position)
            ax1.scatter([v_pert[0]], [v_pert[1]], s=70, c='black', 
                       edgecolors='black', linewidths=0.5, zorder=10)
            # Red arrow from original to perturbed (no shrink)
            ax1.annotate('', xy=v_pert, xytext=v_orig,
                        arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5,
                                       mutation_scale=12), zorder=9)
        elif info['case'] == 2:
            # Near M but not moved - just black
            ax1.scatter([v_pert[0]], [v_pert[1]], s=80, c='black',
                       edgecolors='black', linewidths=0.5, zorder=10)
        else:
            # Far from M - black
            ax1.scatter([v_pert[0]], [v_pert[1]], s=80, c='black',
                       edgecolors='black', linewidths=0.5, zorder=10)
    
    # Draw curve M in blue
    plot_curve(ax1, f, box_min, box_max, color='#0066CC', linewidth=2.5)
    
    ax1.set_xlim(box_min[0] - 0.1, box_max[0] + 0.1)
    ax1.set_ylim(box_min[1] - 0.1, box_max[1] + 0.1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Perturbation Step\n(gray = original, red arrows = perturbation, blue = M)', fontsize=11)
    
    fig1.tight_layout()
    fig1.savefig(f'{output_prefix}_perturbation.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # =========================================================================
    # Figure 2: Final K (like bottom panel of Image 2 and Image 1)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Draw triangulation in black
    for e in edges:
        if e[0] in perturbed and e[1] in perturbed:
            p0, p1 = perturbed[e[0]], perturbed[e[1]]
            ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], '-',
                    color='black', linewidth=1.0, zorder=1)
    
    # Draw perturbed vertices in black
    verts = np.array(list(perturbed.values()))
    ax2.scatter(verts[:, 0], verts[:, 1], s=80, c='black',
               edgecolors='black', linewidths=0.5, zorder=5)
    
    # Draw curve M in blue
    plot_curve(ax2, f, box_min, box_max, color='#0066CC', linewidth=2.5)
    
    # Draw K in green
    for i0, i1 in K_edges:
        p0, p1 = K_verts[i0], K_verts[i1]
        ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], '-',
                color='#228B22', linewidth=2.5, zorder=15)
    
    if K_verts:
        K_arr = np.array(K_verts)
        ax2.scatter(K_arr[:, 0], K_arr[:, 1], s=50, c='#228B22',
                   edgecolors='#228B22', linewidths=0.5, zorder=20)
    
    ax2.set_xlim(box_min[0] - 0.1, box_max[0] + 0.1)
    ax2.set_ylim(box_min[1] - 0.1, box_max[1] + 0.1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Triangulation K\n(black = ambient T̃, blue = M, green = K)', fontsize=11)
    
    fig2.tight_layout()
    fig2.savefig(f'{output_prefix}_result.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # =========================================================================
    # Figure 3: Combined two-panel (like Image 2)
    # =========================================================================
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Perturbation
    for e in edges:
        if e[0] in vertices and e[1] in vertices:
            p0_orig, p1_orig = vertices[e[0]], vertices[e[1]]
            p0_pert, p1_pert = perturbed[e[0]], perturbed[e[1]]
            info0, info1 = perturb_info[e[0]], perturb_info[e[1]]
            near_M = (info0['case'] == 2 or info1['case'] == 2)
            
            if near_M:
                ax3a.plot([p0_orig[0], p1_orig[0]], [p0_orig[1], p1_orig[1]], '-',
                         color='gray', linewidth=1.0, alpha=0.6, zorder=1)
            ax3a.plot([p0_pert[0], p1_pert[0]], [p0_pert[1], p1_pert[1]], '-',
                     color='black', linewidth=1.0, zorder=2)
    
    for idx, v_orig in vertices.items():
        v_pert = perturbed[idx]
        info = perturb_info[idx]
        
        if info['case'] == 2 and info['subcase'] == 'moved':
            ax3a.scatter([v_orig[0]], [v_orig[1]], s=70, c='gray',
                        edgecolors='darkgray', linewidths=1, zorder=8)
            ax3a.scatter([v_pert[0]], [v_pert[1]], s=70, c='black',
                        edgecolors='black', linewidths=0.5, zorder=10)
            ax3a.annotate('', xy=v_pert, xytext=v_orig,
                         arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5,
                                        mutation_scale=12), zorder=9)
        else:
            ax3a.scatter([v_pert[0]], [v_pert[1]], s=80, c='black',
                        edgecolors='black', linewidths=0.5, zorder=10)
    
    plot_curve(ax3a, f, box_min, box_max, color='#0066CC', linewidth=2.5)
    ax3a.set_xlim(box_min[0] - 0.1, box_max[0] + 0.1)
    ax3a.set_ylim(box_min[1] - 0.1, box_max[1] + 0.1)
    ax3a.set_aspect('equal')
    ax3a.axis('off')
    
    # Bottom: Result
    for e in edges:
        if e[0] in perturbed and e[1] in perturbed:
            p0, p1 = perturbed[e[0]], perturbed[e[1]]
            ax3b.plot([p0[0], p1[0]], [p0[1], p1[1]], '-',
                     color='black', linewidth=1.0, zorder=1)
    
    verts = np.array(list(perturbed.values()))
    ax3b.scatter(verts[:, 0], verts[:, 1], s=80, c='black',
                edgecolors='black', linewidths=0.5, zorder=5)
    
    plot_curve(ax3b, f, box_min, box_max, color='#0066CC', linewidth=2.5)
    
    for i0, i1 in K_edges:
        p0, p1 = K_verts[i0], K_verts[i1]
        ax3b.plot([p0[0], p1[0]], [p0[1], p1[1]], '-',
                 color='#228B22', linewidth=2.5, zorder=15)
    
    if K_verts:
        K_arr = np.array(K_verts)
        ax3b.scatter(K_arr[:, 0], K_arr[:, 1], s=50, c='#228B22',
                    edgecolors='#228B22', linewidths=0.5, zorder=20)
    
    ax3b.set_xlim(box_min[0] - 0.1, box_max[0] + 0.1)
    ax3b.set_ylim(box_min[1] - 0.1, box_max[1] + 0.1)
    ax3b.set_aspect('equal')
    ax3b.axis('off')
    
    fig3.tight_layout()
    fig3.savefig(f'{output_prefix}_combined.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # =========================================================================
    # Figure 4: K only (like Image 1)
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    # Draw triangulation in black
    for e in edges:
        if e[0] in perturbed and e[1] in perturbed:
            p0, p1 = perturbed[e[0]], perturbed[e[1]]
            ax4.plot([p0[0], p1[0]], [p0[1], p1[1]], '-',
                    color='black', linewidth=1.0, zorder=1)
    
    # Draw vertices in black
    verts = np.array(list(perturbed.values()))
    ax4.scatter(verts[:, 0], verts[:, 1], s=80, c='black',
               edgecolors='black', linewidths=0.5, zorder=5)
    
    # Draw K in green (no M curve)
    for i0, i1 in K_edges:
        p0, p1 = K_verts[i0], K_verts[i1]
        ax4.plot([p0[0], p1[0]], [p0[1], p1[1]], '-',
                color='#228B22', linewidth=2.5, zorder=15)
    
    if K_verts:
        K_arr = np.array(K_verts)
        ax4.scatter(K_arr[:, 0], K_arr[:, 1], s=50, c='#228B22',
                   edgecolors='#006400', linewidths=1, zorder=20)
    
    ax4.set_xlim(box_min[0] - 0.1, box_max[0] + 0.1)
    ax4.set_ylim(box_min[1] - 0.1, box_max[1] + 0.1)
    ax4.set_aspect('equal')
    ax4.axis('off')
    
    fig4.tight_layout()
    fig4.savefig(f'{output_prefix}_K_only.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.close('all')
    
    return vertices, perturbed, K_verts, K_edges

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Use a circle with center below the view region so we see an arc
    radius = 2.5
    center_y = -2.0  # Circle center below the viewing region
    
    f = lambda p: (p[0])**2 + (p[1] - center_y)**2 - radius**2
    grad_f = lambda p: np.array([2 * p[0], 2 * (p[1] - center_y)])
    
    # Rectangular region showing part of the circle
    box_min = np.array([-0.5, 0.3])
    box_max = np.array([0.5, 0.7])
    
    # L = 0.55  # Slightly coarser for better visibility
    
    print("Generating paper-style visualizations...")
    visualize_paper_style(f, grad_f, radius, box_min, box_max, 
                         'whitney_paper')
    
    # Coarser example to see individual simplices
    # print("\nCoarser example...")
    # # L2 = 0.75
    # visualize_paper_style(f, grad_f, radius, box_min, box_max,
    #                      'whitney_paper_coarse')
    
    print("\nDone!")