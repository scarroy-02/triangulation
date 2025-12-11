"""
Visualize Whitney perturbation with many moved vertices.

To get more perturbations, we:
1. Use a larger required_dist threshold (for visualization purposes)
2. Use manifolds where vertices naturally land near tangent lines
"""

import numpy as np
import matplotlib.pyplot as plt
from coxeter import generate_coxeter_A2, get_edges
from perturb import find_nearest_point_on_M, distance_to_tangent_space
from utils import rho_1, c_tilde


def perturb_vertices_visual(
    vertices, triangles, f, grad_f, L,
    required_dist_factor=1.0,  # Multiply required_dist by this for more perturbations
    d=2, n=1
):
    """
    Perturb vertices with adjustable threshold for visualization.
    
    required_dist_factor: Multiply the theoretical required_dist by this factor.
                         Use >1 to force more perturbations for visualization.
    """
    rho = rho_1(d, n)
    c = c_tilde(d)
    
    # Use amplified required distance for visualization
    required_dist = rho * c * L * required_dist_factor
    max_perturb = c * L * required_dist_factor  # Also scale max_perturb
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
        
        # Case 1: Far from M
        if dist_to_M >= near_threshold:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 1, 'dist_to_M': dist_to_M}
            continue
        
        # Case 2: Near M
        p = find_nearest_point_on_M(v, f, grad_f)
        dist_to_TpM, direction = distance_to_tangent_space(v, p, grad_f)
        
        if dist_to_TpM >= required_dist:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 2, 'action': 'already_far', 'dist_to_TpM': dist_to_TpM}
            continue
        
        # Need to move
        move_amount = min(required_dist - dist_to_TpM, max_perturb)
        v_pert = v + direction * move_amount
        
        perturbed[idx] = v_pert
        info[idx] = {
            'case': 2,
            'action': 'moved',
            'v_orig': v.copy(),
            'v_pert': v_pert.copy(),
            'direction': direction.copy(),
            'move_amount': move_amount,
            'dist_to_TpM': dist_to_TpM
        }
    
    return perturbed, info


def plot_implicit_curve(ax, f, box_min, box_max, n_points=300, **kwargs):
    """Plot implicit curve f(x) = 0."""
    x = np.linspace(box_min[0], box_max[0], n_points)
    y = np.linspace(box_min[1], box_max[1], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(y)):
        for j in range(len(x)):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    defaults = {'colors': ['#0066CC'], 'linewidths': [2.5]}
    defaults.update(kwargs)
    ax.contour(X, Y, Z, levels=[0], **defaults)


def visualize_perturbation(
    f, grad_f, box_min, box_max, L,
    required_dist_factor=1.0,
    output_path=None,
    title="",
    arrow_scale=1.0
):
    """
    Create visualization showing vertex perturbations with arrows.
    """
    # Generate triangulation
    vertices, triangles = generate_coxeter_A2(box_min, box_max, L)
    
    # Perturb with amplified threshold
    perturbed, info = perturb_vertices_visual(
        vertices, triangles, f, grad_f, L,
        required_dist_factor=required_dist_factor
    )
    
    # Count statistics
    n_case1 = sum(1 for i in info.values() if i.get('case') == 1)
    n_case2_far = sum(1 for i in info.values() if i.get('case') == 2 and i.get('action') == 'already_far')
    n_moved = sum(1 for i in info.values() if i.get('case') == 2 and i.get('action') == 'moved')
    
    print(f"\n{title}")
    print(f"  Vertices: {len(vertices)}, Triangles: {len(triangles)}")
    print(f"  Case 1 (far from M): {n_case1}")
    print(f"  Case 2 (near, already OK): {n_case2_far}")
    print(f"  Case 2 (MOVED): {n_moved}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw edges of perturbed triangulation (light gray)
    edges = get_edges(triangles)
    for e in edges:
        if e[0] in perturbed and e[1] in perturbed:
            p0, p1 = perturbed[e[0]], perturbed[e[1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], '-', 
                   color='#CCCCCC', linewidth=0.5, zorder=1)
    
    # Draw manifold M (blue)
    plot_implicit_curve(ax, f, box_min, box_max)
    
    # Draw all vertices
    for idx, v in vertices.items():
        v_pert = perturbed[idx]
        inf = info[idx]
        
        if inf.get('action') == 'moved':
            # Original position (gray, hollow)
            ax.scatter([v[0]], [v[1]], s=120, facecolors='white', 
                      edgecolors='gray', linewidths=2, zorder=5)
            
            # Arrow from original to perturbed (red) - exaggerated for visibility
            dx = (v_pert[0] - v[0]) * arrow_scale
            dy = (v_pert[1] - v[1]) * arrow_scale
            
            # Make arrow visible even if movement is small
            arrow_len = np.sqrt(dx**2 + dy**2)
            min_arrow_len = L * 0.3  # Minimum visible arrow length
            if arrow_len < min_arrow_len and arrow_len > 1e-10:
                scale_up = min_arrow_len / arrow_len
                dx *= scale_up
                dy *= scale_up
            
            ax.annotate('', 
                       xy=(v[0] + dx, v[1] + dy), 
                       xytext=(v[0], v[1]),
                       arrowprops=dict(arrowstyle='-|>', color='red', lw=2.5,
                                      mutation_scale=15, shrinkA=0, shrinkB=0),
                       zorder=8)
            
            # Perturbed position (red, filled)
            ax.scatter([v_pert[0]], [v_pert[1]], s=100, c='red',
                      edgecolors='darkred', linewidths=1, zorder=10)
        else:
            # Non-moved vertices (black, small)
            ax.scatter([v_pert[0]], [v_pert[1]], s=30, c='black',
                      edgecolors='black', linewidths=0.3, zorder=3)
    
    # Labels
    ax.set_xlim(box_min[0] - 0.1, box_max[0] + 0.1)
    ax.set_ylim(box_min[1] - 0.1, box_max[1] + 0.1)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n{n_moved} vertices perturbed (shown in red)", fontsize=14)
    ax.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=8, label='Unchanged vertices'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='gray', markersize=10, markeredgewidth=1.5,
               label='Original position (moved)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=10, label='Perturbed position'),
        Line2D([0], [0], color='red', linewidth=2, label='Perturbation direction'),
        Line2D([0], [0], color='#0066CC', linewidth=2.5, label='Manifold M'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
    else:
        plt.show()
    
    return n_moved


# =============================================================================
# Example manifolds
# =============================================================================

def circle(center=(0, 0), radius=1.0):
    """Circle centered at (cx, cy) with given radius."""
    cx, cy = center
    f = lambda p: (p[0] - cx)**2 + (p[1] - cy)**2 - radius**2
    grad_f = lambda p: np.array([2*(p[0] - cx), 2*(p[1] - cy)])
    return f, grad_f, radius  # reach = radius for circle


def ellipse(a=1.5, b=0.5):
    """Ellipse with semi-axes a (x) and b (y)."""
    f = lambda p: (p[0]/a)**2 + (p[1]/b)**2 - 1
    grad_f = lambda p: np.array([2*p[0]/a**2, 2*p[1]/b**2])
    reach = b**2 / a  # reach at point of max curvature
    return f, grad_f, reach


def figure_eight(a=1.0):
    """Lemniscate of Bernoulli (figure-8 shape)."""
    # (x^2 + y^2)^2 = a^2 * (x^2 - y^2)
    f = lambda p: (p[0]**2 + p[1]**2)**2 - a**2 * (p[0]**2 - p[1]**2)
    grad_f = lambda p: np.array([
        4*p[0]*(p[0]**2 + p[1]**2) - 2*a**2*p[0],
        4*p[1]*(p[0]**2 + p[1]**2) + 2*a**2*p[1]
    ])
    reach = a / 4  # approximate reach
    return f, grad_f, reach


def sine_wave(amplitude=0.3, frequency=2.0):
    """Sine wave: y = amplitude * sin(frequency * x)."""
    f = lambda p: p[1] - amplitude * np.sin(frequency * p[0])
    grad_f = lambda p: np.array([
        -amplitude * frequency * np.cos(frequency * p[0]),
        1.0
    ])
    # Reach depends on curvature: kappa = A*w^2*sin(wx) / (1 + (Aw*cos(wx))^2)^1.5
    # Max curvature at peaks: kappa_max ≈ A*w^2
    reach = 1.0 / (amplitude * frequency**2)
    return f, grad_f, reach


def cardioid(a=0.5):
    """Cardioid: (x^2 + y^2 - a*x)^2 = a^2*(x^2 + y^2)."""
    f = lambda p: (p[0]**2 + p[1]**2 - a*p[0])**2 - a**2*(p[0]**2 + p[1]**2)
    grad_f = lambda p: np.array([
        2*(p[0]**2 + p[1]**2 - a*p[0])*(2*p[0] - a) - 2*a**2*p[0],
        2*(p[0]**2 + p[1]**2 - a*p[0])*(2*p[1]) - 2*a**2*p[1]
    ])
    reach = a / 4  # approximate
    return f, grad_f, reach


if __name__ == "__main__":
    output_dir = "./"
    
    # Example 1: Circle with high required_dist_factor
    print("=" * 60)
    print("Example 1: Circle with amplified perturbation threshold")
    print("=" * 60)
    f, grad_f, reach = circle(radius=1.0)
    L = 0.2
    visualize_perturbation(
        f, grad_f,
        box_min=np.array([-1.5, -1.5]),
        box_max=np.array([1.5, 1.5]),
        L=L,
        required_dist_factor=100,  # 100x normal threshold
        output_path=f"{output_dir}/perturb_circle.png",
        title=f"Circle (L={L}, threshold=100×)",
        arrow_scale=1.0
    )
    
    # Example 2: Sine wave - lots of tangent crossings
    print("\n" + "=" * 60)
    print("Example 2: Sine wave")
    print("=" * 60)
    f, grad_f, reach = sine_wave(amplitude=0.4, frequency=3.0)
    L = 0.15
    visualize_perturbation(
        f, grad_f,
        box_min=np.array([-2.0, -1.0]),
        box_max=np.array([2.0, 1.0]),
        L=L,
        required_dist_factor=60,
        output_path=f"{output_dir}/perturb_sine.png",
        title=f"Sine wave (L={L}, threshold=60×)",
        arrow_scale=1.0
    )
    
    # Example 3: Ellipse with high eccentricity
    print("\n" + "=" * 60)
    print("Example 3: Ellipse")
    print("=" * 60)
    f, grad_f, reach = ellipse(a=1.5, b=0.4)
    L = 0.12
    visualize_perturbation(
        f, grad_f,
        box_min=np.array([-2.0, -1.0]),
        box_max=np.array([2.0, 1.0]),
        L=L,
        required_dist_factor=80,
        output_path=f"{output_dir}/perturb_ellipse.png",
        title=f"Ellipse (L={L}, threshold=80×)",
        arrow_scale=1.0
    )
    
    # Example 4: Figure-8 (lemniscate)
    print("\n" + "=" * 60)
    print("Example 4: Figure-8 (Lemniscate)")
    print("=" * 60)
    f, grad_f, reach = figure_eight(a=1.0)
    L = 0.12
    visualize_perturbation(
        f, grad_f,
        box_min=np.array([-1.5, -0.8]),
        box_max=np.array([1.5, 0.8]),
        L=L,
        required_dist_factor=60,
        output_path=f"{output_dir}/perturb_figure8.png",
        title=f"Figure-8 (L={L}, threshold=60×)",
        arrow_scale=1.0
    )
    
    # Example 5: Zoomed in region of circle showing detail
    print("\n" + "=" * 60)
    print("Example 5: Circle (zoomed in)")
    print("=" * 60)
    f, grad_f, reach = circle(radius=1.0)
    L = 0.1
    visualize_perturbation(
        f, grad_f,
        box_min=np.array([-0.6, 0.4]),
        box_max=np.array([0.6, 1.2]),
        L=L,
        required_dist_factor=150,
        output_path=f"{output_dir}/perturb_circle_zoom.png",
        title=f"Circle zoomed (L={L}, threshold=150×)",
        arrow_scale=1.0
    )
    
    print("\n" + "=" * 60)
    print("Done! Images saved to /mnt/user-data/outputs/")
    print("=" * 60)