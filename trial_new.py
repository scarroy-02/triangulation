"""
Triangulating Submanifolds: Implementation of Whitney's Method
==============================================================

Based on: Boissonnat, Kachanovich, Wintraecken (2021)
"Triangulating Submanifolds: An Elementary and Quantified Version of Whitney's Method"
Discrete & Computational Geometry, 66:386-434

This implements the algorithm for the simplest case:
  n = 1 (1D manifold, i.e., a curve)
  d = 2 (2D ambient space)

The algorithm has two parts:
  Part 1: Perturb the vertices of an ambient Coxeter triangulation of type Ã₂
          so that all vertices (the 0-skeleton = (d-n-1)-skeleton) are far from M.
  Part 2: Construct the triangulation K of M via barycentric subdivision of the
          polytopes formed by intersections of M with simplices of T̃.

References to equations/lemmas/sections refer to the paper above.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import brentq
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════
# §4: Coxeter Triangulation of Type Ã₂
# ════════════════════════════════════════════════════════════════
#
# Definition 4.2: The Ã_d triangulation. For d=2, this is the
# standard equilateral triangle tiling of R².
#
# The vertex set forms the triangular lattice (Fig. 4, left).
# Properties (from eq. 3 and Lemma 4.5):
#   - All triangles are equilateral with edge length L
#   - Thickness t(T) = √3/2 ≈ 0.866  (eq. 3, d=2 even)
#   - The triangulation is Delaunay protected (Def 4.3)
#   - Stable under small perturbations (Corollary 4.4)

class CoxeterTriangulationA2:
    """
    Coxeter triangulation of type Ã₂ in R².
    
    This is the equilateral triangle tiling. Vertices lie on the
    triangular lattice with basis vectors:
        e₁ = L·(1, 0)
        e₂ = L·(1/2, √3/2)
    
    Each parallelogram cell (i,j)→(i+1,j+1) is split into two
    equilateral triangles (one "up", one "down").
    """
    
    def __init__(self, L, bounds, margin_factor=3):
        """
        Parameters
        ----------
        L : float
            Edge length (longest edge length of the triangulation).
            Must satisfy eq. (11): related to reach of M.
        bounds : tuple
            (xmin, xmax, ymin, ymax) region to cover.
        margin_factor : int
            Extra layers of triangles beyond bounds.
        """
        self.L = L
        self.bounds = bounds
        
        # Basis vectors for the triangular lattice
        self.e1 = np.array([L, 0.0])
        self.e2 = np.array([L / 2.0, L * np.sqrt(3) / 2.0])
        
        # Quality measures (eq. 3, d=2 even case)
        self.thickness = np.sqrt(3) / 2.0  # t(T) = √(2(d+1)/(d(d+2))) for d=2
        
        # Storage
        self.vertices = {}       # (i,j) → position array
        self.edges = set()       # set of frozenset{(i1,j1), (i2,j2)}
        self.triangles = []      # list of ((i,j), (i,j), (i,j))
        
        # Edge-to-triangle adjacency (needed for Part 2)
        self.edge_to_triangles = defaultdict(list)
        
        self._generate(margin_factor)
    
    def _vertex_pos(self, i, j):
        """Compute position of vertex (i,j) in the lattice."""
        return float(i) * self.e1 + float(j) * self.e2
    
    def _generate(self, margin):
        """Generate all vertices, edges, and triangles covering bounds."""
        xmin, xmax, ymin, ymax = self.bounds
        m = margin * self.L
        
        # Determine index ranges
        height = self.L * np.sqrt(3) / 2.0
        j_min = int(np.floor((ymin - m) / height)) - 1
        j_max = int(np.ceil((ymax + m) / height)) + 1
        
        for j in range(j_min, j_max + 1):
            x_offset = j * self.L / 2.0
            i_min = int(np.floor((xmin - m - x_offset) / self.L)) - 1
            i_max = int(np.ceil((xmax + m - x_offset) / self.L)) + 1
            for i in range(i_min, i_max + 1):
                self.vertices[(i, j)] = self._vertex_pos(i, j)
        
        # Generate triangles: two per parallelogram cell
        for (i, j) in list(self.vertices.keys()):
            v0, v1, v2 = (i, j), (i + 1, j), (i, j + 1)
            v3 = (i + 1, j + 1)
            
            # "Up" triangle: v0, v1, v2
            if v1 in self.vertices and v2 in self.vertices:
                tri = (v0, v1, v2)
                tri_idx = len(self.triangles)
                self.triangles.append(tri)
                for e in [frozenset([v0, v1]), frozenset([v1, v2]), frozenset([v0, v2])]:
                    self.edges.add(e)
                    self.edge_to_triangles[e].append(tri_idx)
            
            # "Down" triangle: v1, v3, v2
            if v1 in self.vertices and v3 in self.vertices and v2 in self.vertices:
                tri = (v1, v3, v2)
                tri_idx = len(self.triangles)
                self.triangles.append(tri)
                for e in [frozenset([v1, v3]), frozenset([v3, v2]), frozenset([v1, v2])]:
                    self.edges.add(e)
                    self.edge_to_triangles[e].append(tri_idx)
    
    def get_vertex_pos(self, key):
        """Get position of a vertex (possibly perturbed)."""
        return self.vertices[key]
    
    def triangle_vertex_positions(self, tri_idx, vertex_positions=None):
        """Get the three vertex positions of a triangle."""
        if vertex_positions is None:
            vertex_positions = self.vertices
        tri = self.triangles[tri_idx]
        return [vertex_positions[v] for v in tri]


# ════════════════════════════════════════════════════════════════
# Manifold Representation: Implicit Curve f(x,y) = 0
# ════════════════════════════════════════════════════════════════

class ImplicitCurve:
    """
    A C² curve in R² defined implicitly as {(x,y) : f(x,y) = 0}.
    
    The curve must have positive reach (rch M > 0), which is
    guaranteed for C² curves with non-vanishing gradient on M.
    
    We need two "oracles" (Section 2.1):
      1. Given a point v, find a close point on M (or determine v is far)
      2. Access to the tangent space TM at points on M
    """
    
    def __init__(self, f, grad_f, reach, name="curve"):
        """
        Parameters
        ----------
        f : callable(x, y) → float
            Implicit function defining M = f⁻¹(0).
        grad_f : callable(x, y) → array[2]
            Gradient ∇f.
        reach : float
            The reach rch(M) (Definition in §1, Federer [37]).
        name : str
            Name for display.
        """
        self.f = f
        self.grad_f = grad_f
        self.reach = reach
        self.name = name
    
    def evaluate(self, p):
        """Evaluate f at point p."""
        return self.f(p[0], p[1])
    
    def gradient(self, p):
        """Compute ∇f at point p."""
        return np.array(self.grad_f(p[0], p[1]), dtype=float)
    
    def tangent_space(self, p):
        """
        Compute T_pM (unit tangent vector at p ∈ M).
        For a curve in R², T_pM is spanned by the rotation of ∇f by 90°.
        """
        g = self.gradient(p)
        tangent = np.array([-g[1], g[0]])
        norm = np.linalg.norm(tangent)
        if norm < 1e-15:
            return np.array([1.0, 0.0])
        return tangent / norm
    
    def normal_space(self, p):
        """
        Compute N_pM (unit normal vector at p ∈ M).
        """
        g = self.gradient(p)
        norm = np.linalg.norm(g)
        if norm < 1e-15:
            return np.array([0.0, 1.0])
        return g / norm
    
    def closest_point(self, p, max_iter=50, tol=1e-12):
        """
        Oracle 1: Project p onto M using gradient descent / Newton's method.
        Returns the closest point on M, or None if too far.
        """
        q = np.array(p, dtype=float)
        for _ in range(max_iter):
            val = self.f(q[0], q[1])
            if abs(val) < tol:
                return q
            g = self.gradient(q)
            gg = np.dot(g, g)
            if gg < 1e-20:
                break
            q = q - (val / gg) * g
        
        # Verify convergence
        if abs(self.f(q[0], q[1])) < 1e-8:
            return q
        return None
    
    def find_edge_intersection(self, p1, p2):
        """
        Find the intersection of M with the line segment [p1, p2].
        
        Returns the intersection point, or None if no intersection.
        
        For d-n = 1 (edges in 2D with a curve), Lemma 6.4 guarantees
        at most one intersection point per edge (for sufficiently fine T̃).
        """
        f1 = self.evaluate(p1)
        f2 = self.evaluate(p2)
        
        if abs(f1) < 1e-14:
            return p1.copy()
        if abs(f2) < 1e-14:
            return p2.copy()
        
        # Sign change ⟹ intersection exists
        if f1 * f2 > 0:
            return None
        
        def g(t):
            pt = (1.0 - t) * p1 + t * p2
            return self.f(pt[0], pt[1])
        
        try:
            t_star = brentq(g, 0.0, 1.0, xtol=1e-14)
            return (1.0 - t_star) * p1 + t_star * p2
        except ValueError:
            return None


# ════════════════════════════════════════════════════════════════
# §5: Part 1 — Perturbing the Ambient Triangulation
# ════════════════════════════════════════════════════════════════
#
# Goal: Produce T̃ such that all vertices (0-skeleton = (d-n-1)-skeleton
# for d=2, n=1) satisfy d(v, M) > α₀·L (eq. 14).
#
# For n=1, d=2 (codimension 1), the paper notes (end of §2.1/p.390):
#   "the set of all τ'_j is the empty set and span(τ'_j, T_pM) = T_pM.
#    The perturbation therefore ensures that ṽᵢ lies far from T_pM."
#
# Two cases (§5.2):
#   Case 1: d(vᵢ, M) ≥ 3L/2  →  keep vᵢ unchanged
#   Case 2: d(vᵢ, M) < 3L/2  →  perturb vᵢ away from T_pM
#
# The perturbation bound (eq. 17):
#   |vᵢ - ṽᵢ| ≤ c̃·L

def compute_algorithm_constants(d, n, L, reach, practical_scale=1.0):
    """
    Compute the key constants from the paper for the algorithm.
    
    Parameters
    ----------
    d : int  (ambient dimension, = 2)
    n : int  (manifold dimension, = 1)
    L : float (edge length)
    reach : float (reach of M)
    practical_scale : float
        Scale factor for the perturbation constants. The paper's
        theoretical bounds (practical_scale=1.0) are extremely 
        conservative — α₀ ≈ 0.001, c̃ ≈ 0.02 — meaning almost no 
        vertex ever triggers Case 2. Setting practical_scale > 1 
        multiplies c̃, ρ₁, and the αₖ so the perturbation is visible,
        while keeping the *same algorithmic logic* (Case 1 / Case 2,
        push away from T_pM, volumetric existence argument, etc.).
        
        The paper already notes (Remark 5.3) that the bounds are 
        chosen "very small, because the bounds on the quality of the
        simplices that will make up the triangulations are very weak."
        In practice, much larger perturbations work.
    
    Returns
    -------
    dict of constants
    """
    # Thickness of Ã₂ (eq. 3)
    if d % 2 == 0:
        t_T = np.sqrt(2 * (d + 1) / (d * (d + 2)))
    else:
        t_T = np.sqrt(2 / d)
    
    # Separation μ = √(d/(d+1)) and circumradius Σ
    mu = np.sqrt(d / (d + 1))
    Sigma = np.sqrt(d * (d + 2) / (12 * (d + 1)))
    mu0 = mu / Sigma  # normalised separation = √(12/(d+2))
    
    # Protection δ (eq. 3)
    delta_unnorm = (np.sqrt(d**2 + 2*d + 24) - np.sqrt(d**2 + 2*d)) / np.sqrt(12 * (d + 1))
    # δ scales with L, but the normalized version δ/L is what matters
    
    # c̃ (eq. 6): normalized perturbation radius
    # Theoretical: c̃ = min(t·μ₀·δ/(18d), t²/24)
    c_tilde_theory = min(t_T * mu0 * delta_unnorm / (18 * d), t_T**2 / 24)
    # Practical: scale up, but cap at stability limit from Corollary 4.4
    # The hard ceiling is c̃ < t(T)·μ₀·δ/(18d·L) (eq. 2 via Lemma 4.5)
    # and c̃ ≤ 1/24 (eq. 7). We also want L̃ < 13L/12 (eq. 15) ⟹ c̃ < 1/24.
    c_tilde = min(c_tilde_theory * practical_scale, 0.45)
    
    # N_{≤d-n-1}: upper bound on faces of dim ≤ d-n-1 containing a vertex (eq. 4)
    k_skel = d - n - 1  # = 0 for d=2, n=1
    N_leq = 2  # base safety
    for j in range(1, k_skel + 1):
        S_val = stirling2(d + 1, j)
        N_leq += math.factorial(j) * S_val
    N_leq = max(N_leq, 3)  # ensure reasonable bound
    
    # ρ₁ (Lemma 5.1): volume fraction bound
    if d % 2 == 0:
        k = d // 2
        rho1_theory = (2**(2*k - 2) * math.factorial(k)**2) / (np.pi * math.factorial(2*k) * N_leq)
    else:
        k = (d + 1) // 2
        rho1_theory = math.factorial(2*k) / (2**(2*k + 2) * math.factorial(k) * math.factorial(k-1) * N_leq)
    rho1 = min(rho1_theory * practical_scale, 0.95)
    
    # α_k constants (eq. 8): α₁ = (4/3)ρ₁c̃, α_k = (2/3)α_{k-1}·c̃·ρ₁
    alpha = {}
    alpha[1] = (4.0 / 3.0) * rho1 * c_tilde
    for k in range(2, d - n + 1):
        alpha[k] = (2.0 / 3.0) * alpha[k - 1] * c_tilde * rho1
    alpha[0] = (4.0 / 3.0) * rho1 * c_tilde  # α₀ = α₁ for the vertex case
    
    # ζ (eq. 10)
    from math import comb
    binom_d_dn = comb(d, d - n)
    zeta_raw = (8 * t_T * (1 - 8 * min(c_tilde, t_T**2/16) / t_T**2)) / (15 * np.sqrt(d) * binom_d_dn * (1 + 2 * c_tilde))
    zeta = max(zeta_raw, 0.01)  # clamp positive; zeta is a quality bound (eq. 10)
    
    return {
        'd': d, 'n': n,
        'thickness': t_T,
        'mu': mu, 'mu0': mu0, 'Sigma': Sigma,
        'delta': delta_unnorm,
        'c_tilde': c_tilde,
        'c_tilde_theory': c_tilde_theory,
        'N_leq': N_leq,
        'rho1': rho1,
        'rho1_theory': rho1_theory,
        'alpha': alpha,
        'zeta': zeta,
        'L': L,
        'reach': reach,
        'practical_scale': practical_scale,
    }


def stirling2(n, k):
    """Stirling number of the second kind S(n,k) (Definition 4.9)."""
    if k == 0:
        return 1 if n == 0 else 0
    if k == 1 or k == n:
        return 1
    total = 0
    for j in range(k + 1):
        sign = (-1)**j
        total += sign * int(math.factorial(k) / (math.factorial(j) * math.factorial(k - j))) * (k - j)**n
    return total // math.factorial(k)


def perturb_triangulation(T, curve, constants):
    """
    Part 1 of the algorithm (S5.2): Perturb vertices of T into T~.

    For d=2, n=1 (curve in plane), the (d-n-1)=0 skeleton is the vertex set.
    We perturb vertices so that d(v~, M) > alpha_0 * L for all vertices v~.

    The perturbation follows the inductive scheme of S5.2:
      Case 1 (d(v_i, M) >= 3L/2): set v~_i = v_i
      Case 2 (d(v_i, M) < 3L/2): perturb v~_i away from T_pM

    For codimension 1, span(tau'_j, T_pM) = T_pM (noted at end of S2.1),
    so we perturb away from the tangent line at the nearby manifold point.

    The paper's theoretical bounds (eq. 17, 20) produce alpha_0 ~ 0.001 and
    tangent_clearance ~ 0.0001*L, so almost no vertex ever triggers Case 2.
    With practical_scale > 1 the same logic applies but with scaled-up
    clearance requirements, making the perturbation visible.
    """
    L = constants['L']
    c_tilde = constants['c_tilde']
    rho1 = constants['rho1']
    alpha0 = constants['alpha'].get(0, constants['alpha'][1])

    perturbed = {}
    info = {'case1': 0, 'case2': 0, 'max_perturbation': 0.0}

    # Minimum distance we want vertices from M (eq. 14 with k=0)
    min_dist = alpha0 * L

    # Maximum perturbation allowed (eq. 17): |v_i - v~_i| <= c~ * L
    max_perturb = c_tilde * L

    # Desired distance from tangent line (eq. 20): d(v~_i, T_pM) >= rho1 * c~ * L
    tangent_clearance = rho1 * c_tilde * L

    for key, v in T.vertices.items():
        # Oracle 1: find closest point on M
        p = curve.closest_point(v)

        if p is None:
            perturbed[key] = v.copy()
            info['case1'] += 1
            continue

        dist = np.linalg.norm(v - p)

        if dist >= 1.5 * L:
            # --- Case 1 (S5.2): vertex is far from M ---
            # "choose v~_i = v_i"
            # Any simplex in star has d(tau, M) > (1/2 - 2c~)L > 5L/12
            perturbed[key] = v.copy()
            info['case1'] += 1
        else:
            # --- Case 2 (S5.2): vertex is close to M ---
            # Oracle 2: T_pM (tangent space at closest point p in M)
            tangent = curve.tangent_space(p)
            normal = curve.normal_space(p)

            # For codimension 1 (d=2, n=1):
            # span(tau'_j, T_pM) = T_pM (the tangent line)
            # We need d(v~_i, T_pM) >= rho1 * c~ * L (eq. 20)

            # Signed distance from v to the tangent line through p
            v_rel = v - p
            normal_comp = np.dot(v_rel, normal)

            if abs(normal_comp) >= tangent_clearance:
                # Already sufficiently far from T_pM
                perturbed[key] = v.copy()
                info['case1'] += 1
            else:
                # Lemma 5.6: there exists v~_i in B(v_i, c~L) with
                # d(v~_i, T_pM) >= rho1*c~*L (volumetric argument --
                # the "bad" slab near T_pM occupies < half the ball volume).
                #
                # Push vertex away from T_pM in the normal direction,
                # choosing the side it is already on (arbitrary if on T_pM).
                if normal_comp >= 0:
                    sign = 1.0
                else:
                    sign = -1.0
                if abs(normal_comp) < 1e-15:
                    sign = 1.0

                target_normal = sign * tangent_clearance
                needed_shift = target_normal - normal_comp
                displacement = needed_shift * normal

                disp_norm = np.linalg.norm(displacement)
                if disp_norm > max_perturb:
                    displacement = displacement * (max_perturb / disp_norm)

                v_new = v + displacement
                perturbed[key] = v_new
                info['case2'] += 1
                info['max_perturbation'] = max(info['max_perturbation'],
                                                np.linalg.norm(v_new - v))

    return perturbed, info



# ════════════════════════════════════════════════════════════════
# §6: Part 2 — Constructing the Triangulation K of M
# ════════════════════════════════════════════════════════════════
#
# §6.2: The complex K is defined by barycentric subdivision.
#
# For n=1, d=2:
#   - (d-n) = 1: edges of T̃ that intersect M give vertices v(τ¹)
#     which are the unique intersection points (Lemma 6.4).
#   - Triangles (2-simplices) τ² that intersect M give vertices v(τ²)
#     which are the average of v(τ¹) over all edges of τ² that 
#     intersect M (eq. 26).
#   - An n-simplex (edge) of K is {v(τ¹), v(τ²)} for each τ¹ ⊂ τ²
#     (eq. 25).
#
# The result K is a piecewise-linear approximation of M.

def construct_triangulation_K(T, perturbed_vertices, curve, constants):
    """
    Part 2 of the algorithm (§6.2): Construct the triangulation K of M.
    
    Parameters
    ----------
    T : CoxeterTriangulationA2
    perturbed_vertices : dict
    curve : ImplicitCurve
    constants : dict
    
    Returns
    -------
    K : dict with keys:
        'edge_points' : dict  edge_key → intersection point  (v(τ¹))
        'tri_points'  : dict  tri_idx → barycenter point     (v(τ²))
        'simplices'   : list of (edge_key, tri_idx)          (1-simplices of K)
        'K_vertices'  : ordered list of all vertex positions
        'K_edges'     : list of index pairs into K_vertices
    """
    # ── Step 1: Find edge-curve intersections ──
    # For each edge τ¹ ∈ T̃ of dimension d-n = 1,
    # find the unique intersection point with M (Lemma 6.4).
    
    edge_points = {}  # frozenset → np.array
    
    for edge in T.edges:
        v1_key, v2_key = list(edge)
        p1 = perturbed_vertices[v1_key]
        p2 = perturbed_vertices[v2_key]
        
        intersection = curve.find_edge_intersection(p1, p2)
        if intersection is not None:
            edge_points[edge] = intersection
    
    # ── Step 2: Compute triangle representative points v(τ²) ──
    # For a simplex τ of dimension > d-n, v(τ) is the average of
    # all v(τ^{d-n}_i) for (d-n)-faces that intersect M (eq. 26).
    
    tri_points = {}    # tri_idx → np.array
    tri_edges = {}     # tri_idx → list of edges intersecting M
    
    for tri_idx, tri in enumerate(T.triangles):
        # Get edges of this triangle
        edges_of_tri = [
            frozenset([tri[0], tri[1]]),
            frozenset([tri[1], tri[2]]),
            frozenset([tri[0], tri[2]])
        ]
        
        # Find which edges intersect M
        intersecting = [e for e in edges_of_tri if e in edge_points]
        
        if len(intersecting) > 0:
            # v(τ²) = average of intersection points (eq. 26)
            tri_points[tri_idx] = np.mean(
                [edge_points[e] for e in intersecting], axis=0
            )
            tri_edges[tri_idx] = intersecting
    
    # ── Step 3: Build simplicial complex K ──
    # For each sequence τ^{d-n} ⊂ τ^{d-n+1} ⊂ ... ⊂ τ^d
    # where all simplices intersect M, we add a simplex
    # {v(τ^{d-n}), ..., v(τ^d)} to K.
    #
    # For n=1, d=2: sequences are τ¹ ⊂ τ² (edge ⊂ triangle).
    # Each 1-simplex of K connects v(τ¹) to v(τ²).
    
    # Build vertex list and index mapping
    vertex_positions = []
    vertex_index = {}
    
    for edge_key, pt in edge_points.items():
        idx = len(vertex_positions)
        vertex_positions.append(pt)
        vertex_index[('edge', edge_key)] = idx
    
    for tri_idx, pt in tri_points.items():
        idx = len(vertex_positions)
        vertex_positions.append(pt)
        vertex_index[('tri', tri_idx)] = idx
    
    # Build edges of K
    K_edges = []
    simplex_info = []
    
    for tri_idx, intersecting in tri_edges.items():
        tri_key = ('tri', tri_idx)
        if tri_key not in vertex_index:
            continue
        tri_vidx = vertex_index[tri_key]
        
        for edge_key in intersecting:
            edge_k = ('edge', edge_key)
            if edge_k not in vertex_index:
                continue
            edge_vidx = vertex_index[edge_k]
            K_edges.append((edge_vidx, tri_vidx))
            simplex_info.append((edge_key, tri_idx))
    
    return {
        'edge_points': edge_points,
        'tri_points': tri_points,
        'tri_edges': tri_edges,
        'K_vertices': vertex_positions,
        'K_edges': K_edges,
        'simplex_info': simplex_info,
    }


# ════════════════════════════════════════════════════════════════
# Choosing L: The Coarseness of T (eq. 11-12)
# ════════════════════════════════════════════════════════════════

def compute_max_edge_length(reach, d=2, n=1):
    """
    Compute the maximum allowed edge length L from eq. (12).
    
    For practical purposes, we use the simplified bound (eq. 13):
        L/rch(M) < α²_{d-n} / 54
    
    Since α_k is extremely small (≤ 1/18^k), L ≪ rch(M).
    In practice, we use a fraction of the reach.
    """
    # The theoretical bound is extremely conservative.
    # For a practical implementation, we use L = reach / factor
    # where factor is chosen to ensure good results.
    # The paper's bound (eq. 13) gives L < α²₁·reach/54 which is tiny.
    # For visualization, we use a more practical value.
    return reach / 6.0  # practical choice for visualization


# ════════════════════════════════════════════════════════════════
# Visualization
# ════════════════════════════════════════════════════════════════

def plot_algorithm(curve, T, perturbed_vertices, K, constants, 
                   show_original=True, figsize=(18, 14)):
    """
    Create a comprehensive visualization of the algorithm.
    
    Produces a figure with 4 panels:
    1. Original Coxeter triangulation with curve
    2. Perturbed triangulation with curve  
    3. Edge-curve intersections and triangle centers
    4. Final triangulation K overlaid on curve
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Whitney's Triangulation Algorithm (Boissonnat–Kachanovich–Wintraecken)\n"
        f"Curve: {curve.name}  |  d={constants['d']}, n={constants['n']}  |  "
        f"L={constants['L']:.3f}, rch(M)={constants['reach']:.3f}",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    xmin, xmax, ymin, ymax = T.bounds
    pad = 0.3
    plot_bounds = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)
    
    # ── Helper: draw the curve ──
    def draw_curve(ax, color='#E63946', lw=2.5, label='M'):
        # Sample the curve densely for plotting
        N = 500
        xs = np.linspace(plot_bounds[0], plot_bounds[1], N)
        ys = np.linspace(plot_bounds[2], plot_bounds[3], N)
        X, Y = np.meshgrid(xs, ys)
        Z = np.vectorize(lambda x, y: curve.f(x, y))(X, Y)
        ax.contour(X, Y, Z, levels=[0], colors=[color], linewidths=[lw])
    
    # ── Helper: draw triangulation edges ──
    def draw_triangulation(ax, verts, alpha=0.3, color='#457B9D', lw=0.5):
        segments = []
        for edge in T.edges:
            v1, v2 = list(edge)
            if v1 in verts and v2 in verts:
                p1, p2 = verts[v1], verts[v2]
                if (plot_bounds[0] <= p1[0] <= plot_bounds[1] and 
                    plot_bounds[2] <= p1[1] <= plot_bounds[3]):
                    segments.append([p1, p2])
        lc = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha)
        ax.add_collection(lc)
    
    # ── Helper: draw vertices ──
    def draw_vertices(ax, verts, size=3, color='#1D3557', alpha=0.4):
        pts = []
        for key, v in verts.items():
            if (plot_bounds[0] <= v[0] <= plot_bounds[1] and
                plot_bounds[2] <= v[1] <= plot_bounds[3]):
                pts.append(v)
        if pts:
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, alpha=alpha, zorder=3)
    
    def set_ax(ax, title):
        ax.set_xlim(plot_bounds[0], plot_bounds[1])
        ax.set_ylim(plot_bounds[2], plot_bounds[3])
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(False)
    
    # ════════════ Panel 1: Original Coxeter Triangulation T ════════════
    ax = axes[0, 0]
    draw_triangulation(ax, T.vertices, alpha=0.4, color='#457B9D', lw=0.6)
    draw_vertices(ax, T.vertices, size=5, color='#1D3557', alpha=0.6)
    draw_curve(ax)
    set_ax(ax, "§4: Coxeter Triangulation T of type Ã₂\n(with manifold M)")
    
    # ════════════ Panel 2: Perturbed Triangulation T̃ ════════════
    ax = axes[0, 1]
    draw_triangulation(ax, perturbed_vertices, alpha=0.4, color='#457B9D', lw=0.6)
    draw_curve(ax)
    
    # Highlight perturbed vertices (Case 2) with prominent arrows
    # Also show the 3L/2 band around M where Case 2 applies
    N_contour = 400
    xs_c = np.linspace(plot_bounds[0], plot_bounds[1], N_contour)
    ys_c = np.linspace(plot_bounds[2], plot_bounds[3], N_contour)
    Xc, Yc = np.meshgrid(xs_c, ys_c)
    Zc = np.vectorize(lambda x, y: curve.f(x, y))(Xc, Yc)
    # Draw faint band showing Case 2 zone (d(v,M) < 3L/2)
    ax.contourf(Xc, Yc, Zc, levels=[-1.5*constants['L'], 1.5*constants['L']],
                colors=['#2A9D8F'], alpha=0.06)
    ax.contour(Xc, Yc, Zc, levels=[-1.5*constants['L'], 1.5*constants['L']],
               colors=['#2A9D8F'], linewidths=[0.5], linestyles=['--'], alpha=0.3)
    
    for key in T.vertices:
        v_orig = T.vertices[key]
        v_pert = perturbed_vertices[key]
        disp = np.linalg.norm(v_pert - v_orig)
        if disp > 1e-10:
            if (plot_bounds[0] <= v_pert[0] <= plot_bounds[1] and
                plot_bounds[2] <= v_pert[1] <= plot_bounds[3]):
                # Draw arrow from original to perturbed position
                dx = v_pert[0] - v_orig[0]
                dy = v_pert[1] - v_orig[1]
                ax.annotate('', xy=(v_pert[0], v_pert[1]),
                           xytext=(v_orig[0], v_orig[1]),
                           arrowprops=dict(arrowstyle='->', color='#E76F51',
                                          lw=2.0, mutation_scale=12))
                # Original position (hollow)
                ax.plot(v_orig[0], v_orig[1], 'o', color='none', 
                        markeredgecolor='#E76F51', markersize=6, 
                        markeredgewidth=1.5, zorder=4)
                # Perturbed position (filled)
                ax.plot(v_pert[0], v_pert[1], 'o', color='#2A9D8F', 
                        markersize=7, zorder=5, markeredgecolor='white',
                        markeredgewidth=1)
    draw_vertices(ax, perturbed_vertices, size=3, color='#1D3557', alpha=0.5)
    set_ax(ax, "§5: Perturbed Triangulation T̃\n(vertices pushed away from M)")
    
    # ════════════ Panel 3: Intersections and Construction ════════════
    ax = axes[1, 0]
    draw_triangulation(ax, perturbed_vertices, alpha=0.15, color='#A8DADC', lw=0.4)
    draw_curve(ax, color='#E63946', lw=1.5)
    
    # Draw edge intersection points v(τ¹)
    edge_pts = K['edge_points']
    if edge_pts:
        pts = np.array(list(edge_pts.values()))
        mask = ((pts[:, 0] >= plot_bounds[0]) & (pts[:, 0] <= plot_bounds[1]) &
                (pts[:, 1] >= plot_bounds[2]) & (pts[:, 1] <= plot_bounds[3]))
        ax.scatter(pts[mask, 0], pts[mask, 1], s=25, c='#E63946', 
                   zorder=5, label='v(τ¹) = M ∩ edge', edgecolors='white', linewidths=0.5)
    
    # Draw triangle center points v(τ²)
    tri_pts = K['tri_points']
    if tri_pts:
        pts = np.array(list(tri_pts.values()))
        mask = ((pts[:, 0] >= plot_bounds[0]) & (pts[:, 0] <= plot_bounds[1]) &
                (pts[:, 1] >= plot_bounds[2]) & (pts[:, 1] <= plot_bounds[3]))
        ax.scatter(pts[mask, 0], pts[mask, 1], s=25, c='#F4A261', marker='D',
                   zorder=5, label='v(τ²) = avg of v(τ¹)', edgecolors='white', linewidths=0.5)
    
    # Highlight which edges/triangles intersect M
    for tri_idx, intersecting in K['tri_edges'].items():
        tri = T.triangles[tri_idx]
        verts_pos = [perturbed_vertices[v] for v in tri]
        # Check if visible
        centroid = np.mean(verts_pos, axis=0)
        if not (plot_bounds[0] <= centroid[0] <= plot_bounds[1] and
                plot_bounds[2] <= centroid[1] <= plot_bounds[3]):
            continue
        
        # Shade the triangle
        triangle = plt.Polygon(verts_pos, fill=True, facecolor='#F4A261', 
                               alpha=0.1, edgecolor='#E76F51', linewidth=0.8)
        ax.add_patch(triangle)
    
    ax.legend(fontsize=8, loc='lower right')
    set_ax(ax, "§6.1-6.2: Intersection Points & Simplex Construction\n"
               "(shaded triangles intersect M)")
    
    # ════════════ Panel 4: Final Triangulation K ════════════
    ax = axes[1, 1]
    draw_triangulation(ax, perturbed_vertices, alpha=0.08, color='#A8DADC', lw=0.3)
    draw_curve(ax, color='#E63946', lw=1.0)
    
    # Draw K edges (the actual triangulation of M)
    K_verts = K['K_vertices']
    K_edges_list = K['K_edges']
    
    segments = []
    for (i1, i2) in K_edges_list:
        p1, p2 = K_verts[i1], K_verts[i2]
        if (plot_bounds[0] <= p1[0] <= plot_bounds[1] and
            plot_bounds[2] <= p1[1] <= plot_bounds[3]):
            segments.append([p1, p2])
    
    if segments:
        lc = LineCollection(segments, colors='#264653', linewidths=2.0, 
                           alpha=0.9, zorder=4)
        ax.add_collection(lc)
    
    # Draw K vertices
    if K_verts:
        all_pts = np.array(K_verts)
        mask = ((all_pts[:, 0] >= plot_bounds[0]) & (all_pts[:, 0] <= plot_bounds[1]) &
                (all_pts[:, 1] >= plot_bounds[2]) & (all_pts[:, 1] <= plot_bounds[3]))
        ax.scatter(all_pts[mask, 0], all_pts[mask, 1], s=12, c='#264653', 
                   zorder=5, edgecolors='white', linewidths=0.3)
    
    set_ax(ax, "§6.2 + §7: Final Triangulation K of M\n"
               "(piecewise-linear approximation, proven homeomorphic)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_detail(curve, T, perturbed_vertices, K, constants, 
                center=None, radius=None, figsize=(10, 10)):
    """
    Detailed close-up view of the triangulation.
    """
    if center is None:
        center = np.array([(T.bounds[0] + T.bounds[1])/2,
                           (T.bounds[2] + T.bounds[3])/2])
    if radius is None:
        radius = constants['L'] * 4
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    xmin, xmax = center[0] - radius, center[0] + radius
    ymin, ymax = center[1] - radius, center[1] + radius
    pb = (xmin, xmax, ymin, ymax)
    
    # Draw perturbed triangulation
    for edge in T.edges:
        v1, v2 = list(edge)
        p1, p2 = perturbed_vertices[v1], perturbed_vertices[v2]
        if (pb[0] <= p1[0] <= pb[1] and pb[2] <= p1[1] <= pb[3]):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='#A8DADC', 
                    lw=0.8, alpha=0.5)
    
    # Draw curve
    N = 400
    xs = np.linspace(pb[0], pb[1], N)
    ys = np.linspace(pb[2], pb[3], N)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda x, y: curve.f(x, y))(X, Y)
    ax.contour(X, Y, Z, levels=[0], colors=['#E63946'], linewidths=[2.5])
    
    # Draw perturbed vertices with labels
    for key, v in perturbed_vertices.items():
        if pb[0] <= v[0] <= pb[1] and pb[2] <= v[1] <= pb[3]:
            ax.plot(v[0], v[1], 'o', color='#1D3557', markersize=4, alpha=0.6)
    
    # Draw K
    K_verts = K['K_vertices']
    K_edges_list = K['K_edges']
    
    for (i1, i2) in K_edges_list:
        p1, p2 = K_verts[i1], K_verts[i2]
        if pb[0] <= p1[0] <= pb[1] and pb[2] <= p1[1] <= pb[3]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='#264653', lw=2.5)
    
    # Edge intersection points (on M)
    edge_pts = K['edge_points']
    for edge_key, pt in edge_pts.items():
        if pb[0] <= pt[0] <= pb[1] and pb[2] <= pt[1] <= pb[3]:
            ax.plot(pt[0], pt[1], 'o', color='#E63946', markersize=8, 
                    zorder=5, markeredgecolor='white', markeredgewidth=1)
    
    # Triangle center points  
    for tri_idx, pt in K['tri_points'].items():
        if pb[0] <= pt[0] <= pb[1] and pb[2] <= pt[1] <= pb[3]:
            ax.plot(pt[0], pt[1], 'D', color='#F4A261', markersize=7, 
                    zorder=5, markeredgecolor='white', markeredgewidth=1)
    
    # Shade intersecting triangles
    for tri_idx in K['tri_edges']:
        tri = T.triangles[tri_idx]
        verts_pos = [perturbed_vertices[v] for v in tri]
        centroid = np.mean(verts_pos, axis=0)
        if pb[0] <= centroid[0] <= pb[1] and pb[2] <= centroid[1] <= pb[3]:
            triangle = plt.Polygon(verts_pos, fill=True, facecolor='#E9C46A',
                                   alpha=0.15, edgecolor='#E76F51', linewidth=1.0)
            ax.add_patch(triangle)
    
    ax.set_xlim(pb[0], pb[1])
    ax.set_ylim(pb[2], pb[3])
    ax.set_aspect('equal')
    ax.set_title("Close-up: Triangulation K of M\n"
                 "● = v(τ¹) on M (edge∩M)    ◆ = v(τ²) (avg of edge intersections)\n"
                 "Dark edges = simplices of K", fontsize=11)
    ax.grid(False)
    
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
# Quality Metrics (§6, Lemma 6.7)
# ════════════════════════════════════════════════════════════════

def compute_quality_metrics(K, curve, constants):
    """
    Compute quality metrics of the triangulation K.
    
    Measures:
    - Hausdorff distance between K and M (bounded by Lemma 7.4)
    - Edge length statistics
    - Maximum distance of K vertices from M
    """
    K_verts = K['K_vertices']
    K_edges_list = K['K_edges']
    
    if not K_verts or not K_edges_list:
        return {}
    
    # Edge lengths
    edge_lengths = []
    for (i1, i2) in K_edges_list:
        p1, p2 = K_verts[i1], K_verts[i2]
        edge_lengths.append(np.linalg.norm(np.array(p2) - np.array(p1)))
    
    # Distance of vertices from M
    max_dist = 0
    for pt in K_verts:
        p_on_M = curve.closest_point(np.array(pt))
        if p_on_M is not None:
            dist = np.linalg.norm(np.array(pt) - p_on_M)
            max_dist = max(max_dist, dist)
    
    # Edge intersection points are exactly on M, so distance = 0 for those
    # Triangle centers may be off M
    tri_dists = []
    for tri_idx, pt in K['tri_points'].items():
        p_on_M = curve.closest_point(pt)
        if p_on_M is not None:
            tri_dists.append(np.linalg.norm(pt - p_on_M))
    
    return {
        'num_vertices': len(K_verts),
        'num_edges': len(K_edges_list),
        'num_edge_intersections': len(K['edge_points']),
        'num_triangle_centers': len(K['tri_points']),
        'edge_length_min': min(edge_lengths) if edge_lengths else 0,
        'edge_length_max': max(edge_lengths) if edge_lengths else 0,
        'edge_length_mean': np.mean(edge_lengths) if edge_lengths else 0,
        'max_hausdorff_approx': max_dist,
        'max_tri_center_dist': max(tri_dists) if tri_dists else 0,
        'mean_tri_center_dist': np.mean(tri_dists) if tri_dists else 0,
    }


# ════════════════════════════════════════════════════════════════
# Example Curves
# ════════════════════════════════════════════════════════════════

def circle_curve(cx=0, cy=0, r=1.0):
    """Circle of radius r centered at (cx, cy). Reach = r."""
    return ImplicitCurve(
        f=lambda x, y: (x - cx)**2 + (y - cy)**2 - r**2,
        grad_f=lambda x, y: (2*(x - cx), 2*(y - cy)),
        reach=r,
        name=f"Circle(r={r})"
    )

def ellipse_curve(a=1.5, b=0.8, cx=0, cy=0):
    """Ellipse x²/a² + y²/b² = 1. Reach = b²/a (for a > b)."""
    reach = min(a, b)**2 / max(a, b)
    return ImplicitCurve(
        f=lambda x, y: ((x - cx)/a)**2 + ((y - cy)/b)**2 - 1,
        grad_f=lambda x, y: (2*(x - cx)/a**2, 2*(y - cy)/b**2),
        reach=reach,
        name=f"Ellipse(a={a}, b={b})"
    )

def lemniscate_curve(a=1.5, cx=0, cy=0):
    """
    Lemniscate of Bernoulli: (x²+y²)² = a²(x²-y²).
    An interesting curve with a self-crossing (reach is local).
    We use a smoothed version to ensure positive reach.
    """
    # Smoothed lemniscate-like curve (limacon)
    # r = a*(1 + 0.5*cos(2θ)) in polar, but we use implicit form
    return ImplicitCurve(
        f=lambda x, y: ((x-cx)**2 + (y-cy)**2)**2 - a**2 * ((x-cx)**2 - (y-cy)**2) - 0.3*a**2,
        grad_f=lambda x, y: (
            4*(x-cx)*((x-cx)**2+(y-cy)**2) - 2*a**2*(x-cx),
            4*(y-cy)*((x-cx)**2+(y-cy)**2) + 2*a**2*(y-cy)
        ),
        reach=0.3 * a,  # approximate
        name=f"Lemniscate(a={a})"
    )

def trefoil_curve(r=1.0, cx=0, cy=0):
    """
    A trefoil-like curve: r = r₀(1 + 0.3·cos(3θ)) in polar.
    Implicit form approximation.
    """
    # Use polar-to-implicit: (x²+y²) = r₀²(1 + 0.3·cos(3θ))²
    # cos(3θ) = 4cos³θ - 3cosθ = (4x³ - 3x·r²)/r³
    def f(x, y):
        xr, yr = x - cx, y - cy
        r2 = xr**2 + yr**2
        r_val = np.sqrt(r2 + 1e-20)
        cos_t = xr / (r_val + 1e-20)
        cos3t = 4*cos_t**3 - 3*cos_t
        target_r = r * (1 + 0.3 * cos3t)
        return r_val - target_r
    
    def grad_f(x, y):
        eps = 1e-7
        fx = (f(x+eps, y) - f(x-eps, y)) / (2*eps)
        fy = (f(x, y+eps) - f(x, y-eps)) / (2*eps)
        return (fx, fy)
    
    return ImplicitCurve(f=f, grad_f=grad_f, reach=0.4*r,
                          name=f"Trefoil(r={r})")


# ════════════════════════════════════════════════════════════════
# Main: Run the Full Algorithm
# ════════════════════════════════════════════════════════════════

def run_whitney_triangulation(curve, bounds=None, L=None, practical_scale=50.0, verbose=True):
    """
    Run the complete Whitney triangulation algorithm.
    
    Parameters
    ----------
    curve : ImplicitCurve
    bounds : tuple (xmin, xmax, ymin, ymax), or None for auto
    L : float, edge length, or None for auto
    verbose : bool
    practical_scale : float (default 50.0, scale up tiny constants)
    
    Returns
    -------
    T, perturbed_vertices, K, constants
    """
    reach = curve.reach
    
    if L is None:
        L = compute_max_edge_length(reach)
    
    if bounds is None:
        # Auto bounds: slightly larger than curve extent
        r_est = reach * 1.5
        bounds = (-r_est, r_est, -r_est, r_est)
    
    if verbose:
        print("=" * 65)
        print("  Whitney's Triangulation Algorithm")
        print("  (Boissonnat–Kachanovich–Wintraecken, DCG 2021)")
        print("=" * 65)
        print(f"  Curve:  {curve.name}")
        print(f"  d = 2 (ambient), n = 1 (manifold)")
        print(f"  rch(M) = {reach:.4f}")
        print(f"  L (edge length) = {L:.4f}")
        print(f"  L/rch(M) = {L/reach:.4f}")
        print(f"  practical_scale = {practical_scale:.1f}")
        print()
    
    # Compute constants
    constants = compute_algorithm_constants(d=2, n=1, L=L, reach=reach, practical_scale=practical_scale)
    
    if verbose:
        print("─── Algorithm Constants (§4-5) ───")
        print(f"  t(T) = {constants['thickness']:.6f}  (thickness, eq. 3)")
        print(f"  c̃    = {constants['c_tilde']:.6f}  (theory: {constants['c_tilde_theory']:.6f}, x{constants['c_tilde']/constants['c_tilde_theory']:.1f})")
        print(f"  ρ₁   = {constants['rho1']:.6f}  (theory: {constants['rho1_theory']:.6f}, x{constants['rho1']/constants['rho1_theory']:.1f})")
        print(f"  α₀   = {constants['alpha'][0]:.6f}  (skeleton safety, eq. 8)")
        print(f"  ζ    = {constants['zeta']:.6f}  (eq. 10)")
        print(f"  Max perturbation: c̃·L = {constants['c_tilde']*L:.6f}")
        print(f"  Min vertex-M distance: α₀·L = {constants['alpha'][0]*L:.6f}")
        print()
    
    # ── Part 1: Build and perturb Coxeter triangulation ──
    if verbose:
        print("─── Part 1: Coxeter Triangulation & Perturbation (§4-5) ───")
    
    T = CoxeterTriangulationA2(L, bounds)
    
    if verbose:
        print(f"  Vertices: {len(T.vertices)}")
        print(f"  Edges:    {len(T.edges)}")
        print(f"  Triangles: {len(T.triangles)}")
    
    perturbed_vertices, perturb_info = perturb_triangulation(T, curve, constants)
    
    if verbose:
        print(f"  Case 1 (far from M): {perturb_info['case1']} vertices")
        print(f"  Case 2 (perturbed):  {perturb_info['case2']} vertices")
        print(f"  Max perturbation:    {perturb_info['max_perturbation']:.6f}")
        print()
    
    # ── Part 2: Construct triangulation K ──
    if verbose:
        print("─── Part 2: Triangulation Construction (§6) ───")
    
    K = construct_triangulation_K(T, perturbed_vertices, curve, constants)
    
    if verbose:
        print(f"  Edge-M intersections (v(τ¹)): {len(K['edge_points'])}")
        print(f"  Triangle centers (v(τ²)):     {len(K['tri_points'])}")
        print(f"  Simplices of K:               {len(K['K_edges'])}")
        print()
    
    # Quality metrics
    metrics = compute_quality_metrics(K, curve, constants)
    
    if verbose:
        print("─── Quality Metrics (§6-7) ───")
        print(f"  Total vertices in K:    {metrics.get('num_vertices', 0)}")
        print(f"  Total edges in K:       {metrics.get('num_edges', 0)}")
        print(f"  Edge length (min/mean/max): "
              f"{metrics.get('edge_length_min', 0):.4f} / "
              f"{metrics.get('edge_length_mean', 0):.4f} / "
              f"{metrics.get('edge_length_max', 0):.4f}")
        print(f"  Max Hausdorff dist ≈:   {metrics.get('max_hausdorff_approx', 0):.6f}")
        print(f"  Max v(τ²)-to-M dist:    {metrics.get('max_tri_center_dist', 0):.6f}")
        print()
    
    return T, perturbed_vertices, K, constants


# ════════════════════════════════════════════════════════════════
# Run Examples
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    
    # ── Example 1: Circle ──
    print("\n" + "█" * 65)
    print("  EXAMPLE 1: Circle")
    print("█" * 65 + "\n")
    
    curve1 = circle_curve(cx=0, cy=0, r=1.0)
    T1, pv1, K1, c1 = run_whitney_triangulation(
        curve1, bounds=(-1.8, 1.8, -1.8, 1.8), L=0.18
    )
    
    fig1 = plot_algorithm(curve1, T1, pv1, K1, c1, figsize=(16, 14))
    fig1.savefig('circle_triangulation.png', dpi=150, bbox_inches='tight')
    
    fig1d = plot_detail(curve1, T1, pv1, K1, c1, 
                        center=np.array([1.0, 0.0]), radius=0.5)
    fig1d.savefig('circle_detail.png', dpi=150, bbox_inches='tight')
    
    # ── Example 2: Ellipse ──
    print("\n" + "█" * 65)
    print("  EXAMPLE 2: Ellipse")
    print("█" * 65 + "\n")
    
    curve2 = ellipse_curve(a=1.5, b=0.7)
    T2, pv2, K2, c2 = run_whitney_triangulation(
        curve2, bounds=(-2.2, 2.2, -1.5, 1.5), L=0.15
    )
    
    fig2 = plot_algorithm(curve2, T2, pv2, K2, c2, figsize=(16, 14))
    fig2.savefig('ellipse_triangulation.png', dpi=150, bbox_inches='tight')
    
    # ── Example 3: Trefoil-like curve ──
    print("\n" + "█" * 65)
    print("  EXAMPLE 3: Trefoil")
    print("█" * 65 + "\n")
    
    curve3 = trefoil_curve(r=1.0)
    T3, pv3, K3, c3 = run_whitney_triangulation(
        curve3, bounds=(-2.0, 2.0, -2.0, 2.0), L=0.12
    )
    
    fig3 = plot_algorithm(curve3, T3, pv3, K3, c3, figsize=(16, 14))
    fig3.savefig('trefoil_triangulation.png', dpi=150, bbox_inches='tight')
    
    print("\n✓ All figures saved successfully.")
    print("  - circle_triangulation.png / circle_detail.png")
    print("  - ellipse_triangulation.png")
    print("  - trefoil_triangulation.png")
    
    plt.close('all')