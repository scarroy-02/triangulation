"""
Whitney's Triangulation - Strict Lemma 5.6 Implementation
==========================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# =============================================================================
# Coxeter Ã_2 Triangulation
# =============================================================================

def coxeter_A2_direct(box_min, box_max, L):
    """Generate Coxeter Ã_2 triangulation with equilateral triangles."""
    # Basis vectors for A2 lattice (60 degree separation)
    e1 = np.array([1.0, 0.0]) * L
    e2 = np.array([0.5, np.sqrt(3)/2]) * L
    
    # Create grid covering the box
    B = np.column_stack([e1, e2])
    B_inv = np.linalg.inv(B)
    
    # Find index bounds
    corners = [
        np.array([box_min[0], box_min[1]]),
        np.array([box_max[0], box_min[1]]),
        np.array([box_min[0], box_max[1]]),
        np.array([box_max[0], box_max[1]])
    ]
    
    # Project corners to lattice basis to find min/max indices
    indices = [B_inv @ c for c in corners]
    i_min = int(min(idx[0] for idx in indices)) - 2
    i_max = int(max(idx[0] for idx in indices)) + 2
    j_min = int(min(idx[1] for idx in indices)) - 2
    j_max = int(max(idx[1] for idx in indices)) + 2
    
    vertices = {}
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            p = i * e1 + j * e2
            # Keep vertices slightly outside box to ensure coverage
            if box_min[0] - L <= p[0] <= box_max[0] + L and \
               box_min[1] - L <= p[1] <= box_max[1] + L:
                vertices[(i, j)] = p
    
    triangles = []
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            # A2 tiling consists of two triangles per rhombus
            # Vertices: (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            v00, v10, v01, v11 = (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            
            # Triangle 1: (i,j)-(i+1,j)-(i,j+1)
            if v00 in vertices and v10 in vertices and v01 in vertices:
                triangles.append([v00, v10, v01])
            
            # Triangle 2: (i+1,j)-(i+1,j+1)-(i,j+1)
            if v10 in vertices and v11 in vertices and v01 in vertices:
                triangles.append([v10, v11, v01])
    
    return vertices, triangles

# =============================================================================
# Constants & Projection
# =============================================================================

def compute_constants_2d(reach):
    """Compute theoretical bounds from the paper."""
    # d = 2 
    # # Thickness of A2 simplex
    # t = np.sqrt(2 * (d + 1) / (d * (d + 2))) # approx 0.866
    
    # # Geometry constants (Boissonnat et al 2021)
    # # L should be small relative to reach. The paper requires L ~ reach / 100s
    # # We use a larger L for visibility, but keep proportions.
    # L_est = 0.99 * (reach / 54.0) 
    
    # # c_tilde (perturbation magnitude factor)
    # # Must satisfy Eq (17): c_tilde <= t^2 / 24
    # c_tilde = (t**2) / 24.0 
    
    # # rho_1 (slab width factor) - From Lemma 5.1
    # # For d=2, N_<=0 is small. 
    # # Using the simplified bound rho_1 ~ 1 / (sqrt(d) * N)
    # rho_1 = 0.05 

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
    
    # return {'rho_1': rho_1, 'c_tilde': c_tilde, 'L': L_est}

def project_to_manifold(p, f, grad_f):
    """Newton projection to implicit curve f(x)=0."""
    curr = p.copy()
    for _ in range(15):
        val = f(curr)
        if abs(val) < 1e-12:
            return curr
        g = grad_f(curr)
        g2 = np.dot(g, g)
        if g2 < 1e-14: # Gradient vanish
            return curr
        curr = curr - (val / g2) * g
    return curr

# =============================================================================
# Core Logic: Lemma 5.6 Perturbation
# =============================================================================

def perturb_vertices_strict(vertices, f, grad_f, L, c_tilde, rho_1):
    """
    Implements Lemma 5.6 strictly.
    All vertices within 3L/2 of M MUST be processed to avoid bad spans.
    """
    perturbed = {}
    info = {}
    
    # Max allowed perturbation distance
    max_perturb_dist = c_tilde * L
    # Minimum required distance from Tangent Space (Forbidden Slab half-width)
    min_safe_dist = rho_1 * c_tilde * L
    
    np.random.seed(42) # Deterministic results
    
    for idx, v in vertices.items():
        # 1. Calculate distance to manifold
        p = project_to_manifold(v, f, grad_f)
        dist_v_M = np.linalg.norm(v - p)
        
        # ==========================================
        # CASE 1: Far from Manifold
        # d(v, M) >= 1.5 * L
        # Result: v_tilde = v (No perturbation)
        # ==========================================
        if dist_v_M >= 1.5 * L:
            perturbed[idx] = v.copy()
            info[idx] = {'case': 1}
            continue

        # ==========================================
        # CASE 2: Close to Manifold
        # d(v, M) < 1.5 * L
        # Result: Pick v_tilde in B(v, c*L) avoiding Bad Spans
        # ==========================================
        
        # The "Bad Span" in 2D codimension 1 is the Tangent Line T_p M.
        # We must pick v_tilde such that dist(v_tilde, T_p M) >= rho_1 * c_tilde * L
        
        g = grad_f(p)
        g_norm = np.linalg.norm(g)
        normal = g / g_norm if g_norm > 1e-10 else np.array([1.0, 0.0])
        
        # We attempt to find a valid v_tilde by random sampling in the allowed ball.
        # This mimics the "volume argument" existence proof.
        valid_found = False
        v_tilde = v.copy()
        
        # Attempt 1: Check if original vertex is valid? 
        # The prompt implies "all vertices... should be perturbed". 
        # So we force a random perturbation even if original was safe, 
        # to ensure the "general position" argument holds visually.
        
        for attempt in range(100):
            # Pick random offset in ball B(0, max_perturb_dist)
            angle = np.random.uniform(0, 2*np.pi)
            # Square root for uniform sampling in 2D disk
            r = np.sqrt(np.random.uniform(0, 1)) * max_perturb_dist
            
            offset = np.array([r * np.cos(angle), r * np.sin(angle)])
            candidate = v + offset
            
            # Check safety condition (Lemma 5.6 Eq 1)
            # Distance from candidate to Tangent Plane at p
            # Since p is on M and normal is N, dist = |dot(candidate - p, N)|
            dist_to_span = abs(np.dot(candidate - p, normal))
            
            if dist_to_span >= min_safe_dist:
                v_tilde = candidate
                valid_found = True
                break
        
        # Fallback: If random sampling failed (unlikely), project strictly out of slab
        if not valid_found:
            # Force move along normal to be safe
            sign = np.sign(np.dot(v - p, normal))
            if sign == 0: sign = 1.0
            # Move to boundary of forbidden slab + tiny epsilon
            safe_dist = min_safe_dist * 1.01
            # We construct a vector that satisfies both constraints if possible
            # For visualization, we prioritize the safety constraint
            v_tilde = p + (v - p) - np.dot(v-p, normal)*normal + sign * safe_dist * normal

        perturbed[idx] = v_tilde
        info[idx] = {
            'case': 2,
            'p': p,           # Projection on M
            'normal': normal, # Normal vector
            'v_orig': v
        }

    return perturbed, info

# =============================================================================
# Visualization
# =============================================================================

def plot_paper_figure(f, grad_f, reach, box_min, box_max, output_name):
    
    # 1. Setup
    const = compute_constants_2d(reach)
    L = const['L']
    c_tilde = const['c_tilde']
    rho_1 = const['rho_1']
    
    print(f"L = {L:.4f}")
    print(f"Perturbation Radius (c~L) = {c_tilde*L:.4f}")
    print(f"Forbidden Slab Width (2*rho*c~L) = {2*rho_1*c_tilde*L:.4f}")
    
    # 2. Generate Grid
    vertices, triangles = coxeter_A2_direct(box_min, box_max, L)
    
    # 3. Apply Strict Perturbation
    perturbed, info = perturb_vertices_strict(vertices, f, grad_f, L, c_tilde, rho_1)

    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # --- Draw Forbidden Zones & Allowed Regions (for close vertices) ---
    # We only draw this for a few vertices to avoid clutter, or draw lightly
    for idx, data in info.items():
        if data['case'] == 2:
            v = data['v_orig']
            p = data['p']
            n = data['normal']
            vt = perturbed[idx]
            
            # 1. Allowed Region: Ball B(v, c_tilde * L)
            radius = c_tilde * L
            circle = Circle(v, radius, color='green', alpha=0.05)
            ax.add_patch(circle)
            
            # 2. Forbidden Slab: Region around Tangent Space T_p M
            # Width = 2 * rho_1 * c_tilde * L
            # It's a strip perpendicular to normal
            slab_width = rho_1 * c_tilde * L
            
            # Draw Tangent line segment (approx length for vis)
            tangent = np.array([-n[1], n[0]])
            t_start = p - tangent * L
            t_end = p + tangent * L
            ax.plot([t_start[0], t_end[0]], [t_start[1], t_end[1]], 
                   color='blue', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Draw Slab (light red)
            # Create a polygon for the slab visualization around the tangent
            slab_poly = Polygon([
                t_start + n * slab_width,
                t_end + n * slab_width,
                t_end - n * slab_width,
                t_start - n * slab_width
            ], color='red', alpha=0.1)
            ax.add_patch(slab_poly)

            # Draw Arrow from Original to Perturbed
            ax.annotate('', xy=vt, xytext=v,
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

    # --- Draw Triangulation ---
    for tri in triangles:
        # Check if triangle is active (intersects M approx)
        # We just draw all edges
        pts = [perturbed[v] for v in tri]
        pts.append(pts[0]) # close loop
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color='gray', linewidth=0.5, zorder=1)

    # --- Draw Vertices ---
    for idx, v in perturbed.items():
        color = 'red' if info[idx]['case'] == 2 else 'black'
        size = 20 if info[idx]['case'] == 2 else 10
        ax.scatter(v[0], v[1], c=color, s=size, zorder=2)
        
    # --- Draw Manifold ---
    # Create dense points for contour
    gx = np.linspace(box_min[0], box_max[0], 200)
    gy = np.linspace(box_min[1], box_max[1], 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    for i in range(200):
        for j in range(200):
            GZ[j,i] = f(np.array([GX[j,i], GY[j,i]]))
    ax.contour(GX, GY, GZ, levels=[0], colors='blue', linewidths=2)
    
    # --- Draw "3L/2" Boundary (Approximate) ---
    # We can visualize the region where vertices are red
    # Just purely implicitly visualized by the dot colors

    ax.set_xlim(box_min[0], box_max[0])
    ax.set_ylim(box_min[1], box_max[1])
    ax.set_aspect('equal')
    ax.set_title(f"Whitney Perturbation (Lemma 5.6)\nRed Vertices = Case 2 (d < 1.5L)\nRed Strip = Forbidden Slab | Green Circle = Allowed Region", fontsize=12)
    
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_name}")

# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    # Define implicit circle
    radius = 2.0
    center = np.array([0.0, -1.5]) # Shifted so we see the arc
    
    def f_circ(p): return np.sum((p - center)**2) - radius**2
    def grad_circ(p): return 2 * (p - center)
    
    # Viewing box
    b_min = np.array([-0.6, 0.0])
    b_max = np.array([0.6, 1.0])
    
    # Use a visually reasonable reach (radius)
    # Note: Theoretical L is tiny. For the visual to look like the paper's diagrams,
    # L needs to be large enough to see the triangles. 
    # The code calculates L based on reach, but we might scale reach input to get good L.
    
    plot_paper_figure(f_circ, grad_circ, reach=radius*2.0, 
                     box_min=b_min, box_max=b_max, 
                     output_name="whitney_strict_perturbation.png")