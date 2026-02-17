"""
Whitney's Triangulation Algorithm — 2D Manifold in 3D
=====================================================

Based on: Boissonnat, Kachanovich, Wintraecken (2021)
"Triangulating Submanifolds: An Elementary and Quantified
 Version of Whitney's Method", DCG 66:386–434

Case: n = 2 (surface), d = 3 (ambient R³)

Ambient triangulation: Coxeter triangulation of type Ã₃.

  Constructed as the intersection of the Freudenthal-Kuhn triangulation
  of Z⁴ with the hyperplane H = {x ∈ R⁴ : Σxᵢ = 0} ≅ R³ (Def 1, §2.1).

  Each 4-simplex of the Z⁴ Freudenthal has vertices w₀,...,w₄ where
  w₄ = w₀ + (1,1,1,1).  Since π(w₄) = π(w₀), intersection with H
  gives a tetrahedron with 4 distinct projected vertices.

  For each Z⁴ cube corner n and permutation σ ∈ S₄ (24 tets per cell):
      wᵢ = n + e_{σ(1)} + ... + e_{σ(i)},  projected to H by
      π(z) = z − (Σz/4)(1,1,1,1).

  Properties:
    - All 24 simplices per cell are CONGRUENT (isosceles tetrahedra)
    - Edge lengths: L·√3/2 (4 edges) and L (2 edges)
    - Edge ratio: 2/√3 ≈ 1.155 (vs √3 ≈ 1.732 for Freudenthal)
    - L_max = L   (vs L√3 for Freudenthal)
    - t(T) = √2/2 ≈ 0.707   (vs 1/√6 ≈ 0.408 for Freudenthal)
    - All faces congruent isosceles triangles (area = L²·√2/4 each)
    - Vertex set = A₃* lattice (permutohedral/FCC) in R³

Part 1 (§5): Perturb vertices so that the 0-skeleton is far from M.
Part 2 (§6): Construct K via barycentric subdivision chains.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import brentq
from collections import defaultdict
from itertools import permutations, combinations
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# §2.1/§4: Coxeter Triangulation of type Ã₃
# ═══════════════════════════════════════════════════════════════════
#
# The Ã₃ Coxeter triangulation is the restriction of the 4D
# Freudenthal-Kuhn triangulation to H = {x ∈ R⁴ : Σxᵢ = 0}.
#
# Vertex set: π(Z⁴) where π(z) = z − (Σz/4)(1,...,1).
# This is the A₃* (weight/FCC) lattice in H ≅ R³.
#
# We identify each vertex by a canonical key (a,b,c) ∈ Z³ where
#   z = (a, b, c, 0) + k(1,1,1,1)  for arbitrary k.
# Then π(z) = ((3a−b−c)/4, (−a+3b−c)/4, (−a−b+3c)/4, (−a−b−c)/4).
#
# Alcoves (tetrahedra): for each cell (a,b,c) and σ ∈ S₄,
# the tet has vertices at canonical keys
#   k₀ = (a,b,c), kᵢ = kᵢ₋₁ + δ_{σ(i)}
# where δ₀=(1,0,0), δ₁=(0,1,0), δ₂=(0,0,1), δ₃=(−1,−1,−1).
# (δ₃ reflects adding e₄ in Z⁴ under the canonical projection.)

# ONB for H ⊂ R⁴
_F = np.array([
    [1, -1,  0,  0],
    [1,  1, -2,  0],
    [1,  1,  1, -3],
], dtype=float)
_F[0] /= np.sqrt(2)
_F[1] /= np.sqrt(6)
_F[2] /= np.sqrt(12)

# Key deltas: adding e₀,e₁,e₂,e₃ in Z⁴ shifts the (a,b,c) key by:
_KEY_DELTA = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1)]


def _pi(z):
    """Project Z⁴ vector to H = {Σxᵢ=0}."""
    z = np.asarray(z, dtype=float)
    return z - (z.sum() / 4.0) * np.ones(4)


def _vertex_pos_from_key(key, L):
    """R³ position from canonical key (a,b,c), scaled by L."""
    a, b, c = key
    h = _pi([a, b, c, 0])
    return L * (_F @ h)


class CoxeterA3Triangulation3D:
    """
    Coxeter triangulation of type Ã₃ of R³.

    All simplices are congruent isosceles tetrahedra.
    Edge lengths: L√3/2 (×4) and L (×2).
    Thickness: t(T) = √2/2 ≈ 0.707.
    """

    def __init__(self, L, bounds, margin=2):
        self.L = L
        self.bounds = bounds
        self.Lmax = L                     # longest edge = L
        self.thickness = np.sqrt(2) / 2   # ≈ 0.707

        # All 24 permutations of S₄
        self._perms4 = list(permutations(range(4)))

        # Storage
        self.vertices = {}       # (a,b,c) -> np.array R³ position
        self.tetrahedra = []     # list of 4-tuples of vertex keys
        self.edges = set()       # frozenset of 2 vertex keys
        self.faces = set()       # frozenset of 3 vertex keys

        # Adjacency
        self.edge_to_faces = defaultdict(set)
        self.edge_to_tets = defaultdict(set)
        self.face_to_tets = defaultdict(set)

        self._generate(margin)
        self._build_adjacency()

    def _generate(self, margin):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        L = self.L
        m = margin

        # Lattice vectors in R³ (shifts when incrementing a, b, c by 1):
        da = _vertex_pos_from_key((1, 0, 0), 1.0)
        db = _vertex_pos_from_key((0, 1, 0), 1.0)
        dc = _vertex_pos_from_key((0, 0, 1), 1.0)

        # Inverse map: R³ → (a,b,c) fractional coords
        M = np.column_stack([da, db, dc])
        Minv = np.linalg.inv(M)

        # Map bounds corners to (a,b,c) space
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin],
            [xmin, ymax, zmin], [xmax, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax],
            [xmin, ymax, zmax], [xmax, ymax, zmax],
        ])
        abc = np.array([Minv @ (c / L) for c in corners])

        a0 = int(np.floor(abc[:, 0].min())) - m
        a1 = int(np.ceil(abc[:, 0].max())) + m
        b0 = int(np.floor(abc[:, 1].min())) - m
        b1 = int(np.ceil(abc[:, 1].max())) + m
        c0 = int(np.floor(abc[:, 2].min())) - m
        c1 = int(np.ceil(abc[:, 2].max())) + m

        def ensure_vertex(k):
            if k not in self.vertices:
                self.vertices[k] = _vertex_pos_from_key(k, L)

        for a in range(a0, a1 + 1):
            for b in range(b0, b1 + 1):
                for c in range(c0, c1 + 1):
                    for perm in self._perms4:
                        # Build tet: 4 vertices via cumulative key deltas
                        k0 = (a, b, c)
                        d0 = _KEY_DELTA[perm[0]]
                        k1 = (k0[0]+d0[0], k0[1]+d0[1], k0[2]+d0[2])
                        d1 = _KEY_DELTA[perm[1]]
                        k2 = (k1[0]+d1[0], k1[1]+d1[1], k1[2]+d1[2])
                        d2 = _KEY_DELTA[perm[2]]
                        k3 = (k2[0]+d2[0], k2[1]+d2[1], k2[2]+d2[2])

                        for k in (k0, k1, k2, k3):
                            ensure_vertex(k)

                        tet = (k0, k1, k2, k3)
                        self.tetrahedra.append(tet)

                        verts = [k0, k1, k2, k3]
                        for i in range(4):
                            for j in range(i + 1, 4):
                                self.edges.add(frozenset([verts[i], verts[j]]))
                        for i in range(4):
                            face = frozenset(verts[j] for j in range(4) if j != i)
                            self.faces.add(face)

    def _build_adjacency(self):
        for tet_idx, tet in enumerate(self.tetrahedra):
            verts = list(tet)
            for a in range(4):
                face = frozenset(verts[b] for b in range(4) if b != a)
                self.face_to_tets[face].add(tet_idx)
                fv = list(face)
                for p in range(3):
                    for q in range(p + 1, 3):
                        edge = frozenset([fv[p], fv[q]])
                        self.edge_to_faces[edge].add(face)
            for a in range(4):
                for b in range(a + 1, 4):
                    edge = frozenset([verts[a], verts[b]])
                    self.edge_to_tets[edge].add(tet_idx)


# Backward compat
FreudenthalTriangulation3D = CoxeterA3Triangulation3D


# ═══════════════════════════════════════════════════════════════════
# Implicit Surface  M = { p ∈ R³ : f(p) = 0 }
# ═══════════════════════════════════════════════════════════════════

class ImplicitSurface:
    """
    Smooth implicit surface in R³.
    f(p) = 0 defines M.  grad_f for normals.  reach = reach(M).
    """
    def __init__(self, f, grad_f, reach, name="surface"):
        self.f = f
        self.grad_f = grad_f
        self.reach = reach
        self.name = name

    def evaluate(self, p):
        return self.f(p[0], p[1], p[2])

    def gradient(self, p):
        return np.array(self.grad_f(p[0], p[1], p[2]))

    def normal(self, p):
        g = self.gradient(p)
        n = np.linalg.norm(g)
        return g / n if n > 1e-15 else np.array([0, 0, 1.0])

    def closest_point(self, p, max_iter=60, tol=1e-12):
        q = np.array(p, dtype=float)
        for _ in range(max_iter):
            fv = self.evaluate(q)
            g = self.gradient(q)
            g2 = np.dot(g, g)
            if g2 < 1e-30:
                return None
            q = q - (fv / g2) * g
            if abs(self.evaluate(q)) < tol:
                return q
        return q if abs(self.evaluate(q)) < 1e-6 else None

    def find_edge_intersection(self, p1, p2):
        """Find the point where the edge p1-p2 crosses M (f=0)."""
        f1 = self.evaluate(p1)
        f2 = self.evaluate(p2)
        if f1 * f2 > 0:
            return None
        if abs(f1) < 1e-14:
            return np.array(p1, dtype=float)
        if abs(f2) < 1e-14:
            return np.array(p2, dtype=float)
        def g(t):
            return self.evaluate((1-t)*p1 + t*p2)
        try:
            t_root = brentq(g, 0.0, 1.0, xtol=1e-14)
            return (1 - t_root) * np.array(p1) + t_root * np.array(p2)
        except ValueError:
            return None


# ═══════════════════════════════════════════════════════════════════
# §4-5: Constants
# ═══════════════════════════════════════════════════════════════════

def compute_constants_3d(L, reach, practical_scale=50.0):
    """Compute algorithm constants for d=3, n=2, Coxeter Ã₃."""
    d, n = 3, 2

    # Ã₃ Coxeter thickness (all alcoves congruent):
    #   t(T) = d·V/(diam·max_face_area) = 3·(1/12)/(1·√2/4) = √2/2
    t_T = np.sqrt(2) / 2   # ≈ 0.707
    Lmax = L                # longest edge = L (the 2-step edge)

    # Theoretical c̃ (eq. 6): c̃ < t(T)²/(8(d-n+1))
    # = (1/2)/(8·2) = 1/32 ≈ 0.03125
    c_tilde_theory = t_T ** 2 / (8.0 * (d - n + 1))
    c_tilde = min(c_tilde_theory * practical_scale, 0.42)

    # ρ₁ (Lemma 5.1)
    N_leq = 3
    rho1_theory = math.factorial(4) / (
        2 ** 6 * math.factorial(2) * math.factorial(1) * N_leq)
    rho1 = min(rho1_theory * practical_scale, 0.90)

    alpha0 = (4.0 / 3.0) * rho1 * c_tilde

    binom_d_dn = math.comb(d, d - n)
    zeta_raw = (8 * t_T * (1 - 8 * min(c_tilde, t_T**2 / 16) / t_T**2)) / (
        15 * np.sqrt(d) * binom_d_dn * (1 + 2 * c_tilde))
    zeta = max(zeta_raw, 0.01)

    return {
        'd': d, 'n': n,
        'L': L, 'Lmax': Lmax, 'reach': reach,
        'thickness': t_T,
        'c_tilde': c_tilde, 'c_tilde_theory': c_tilde_theory,
        'rho1': rho1, 'rho1_theory': rho1_theory,
        'alpha0': alpha0,
        'zeta': zeta,
        'practical_scale': practical_scale,
    }


# ═══════════════════════════════════════════════════════════════════
# §5: Part 1 — Perturbation
# ═══════════════════════════════════════════════════════════════════

def perturb_vertices(T, surface, consts):
    """
    Part 1 (§5.2): Perturb vertices of T so 0-skeleton is far from M.

    Case 1: d(v, M) ≥ 3L_max/2  →  keep v
    Case 2: d(v, M) < 3L_max/2  →  push v away from T_pM
    """
    Lmax = consts['Lmax']
    c_tilde = consts['c_tilde']
    rho1 = consts['rho1']

    max_perturb = c_tilde * Lmax
    tangent_clearance = rho1 * c_tilde * Lmax

    perturbed = {}
    info = {'case1': 0, 'case2': 0, 'max_pert': 0.0}

    for key, v in T.vertices.items():
        cp = surface.closest_point(v)
        if cp is None:
            perturbed[key] = v.copy()
            info['case1'] += 1
            continue

        dist = np.linalg.norm(v - cp)
        if dist >= 1.5 * Lmax:
            perturbed[key] = v.copy()
            info['case1'] += 1
            continue

        # Case 2: push away from tangent plane at cp
        info['case2'] += 1
        n = surface.normal(cp)

        # Signed distance to tangent plane
        signed_d = np.dot(v - cp, n)

        if abs(signed_d) >= tangent_clearance:
            perturbed[key] = v.copy()
        else:
            if signed_d >= 0:
                target_d = tangent_clearance
            else:
                target_d = -tangent_clearance
            shift = (target_d - signed_d) * n
            shift_norm = np.linalg.norm(shift)
            if shift_norm > max_perturb:
                shift = shift * (max_perturb / shift_norm)
            new_v = v + shift
            perturbed[key] = new_v
            info['max_pert'] = max(info['max_pert'], np.linalg.norm(shift))

    return perturbed, info


# ═══════════════════════════════════════════════════════════════════
# §6: Part 2 — Construct K
# ═══════════════════════════════════════════════════════════════════

def construct_K(T, pverts, surface, consts):
    """
    Part 2 (§6): Build K from chains τ¹ ⊂ τ² ⊂ τ³.
    """
    Lmax = consts['Lmax']

    # Step 1: edge-surface intersections v(τ¹)
    edge_pts = {}
    for edge in T.edges:
        v1k, v2k = list(edge)
        p1, p2 = pverts[v1k], pverts[v2k]
        pt = surface.find_edge_intersection(p1, p2)
        if pt is not None:
            edge_pts[edge] = pt

    # Step 2: face centres v(τ²)
    face_pts = {}
    for face in T.faces:
        face_edges = []
        fv = list(face)
        for i in range(3):
            for j in range(i + 1, 3):
                e = frozenset([fv[i], fv[j]])
                if e in edge_pts:
                    face_edges.append(e)
        if len(face_edges) >= 2:
            centroid = np.mean([edge_pts[e] for e in face_edges], axis=0)
            cp = surface.closest_point(centroid)
            if cp is not None:
                face_pts[face] = cp

    # Step 3: tet centres v(τ³)
    tet_pts = {}
    for tet_idx, tet in enumerate(T.tetrahedra):
        tet_faces = []
        verts = list(tet)
        for a in range(4):
            face = frozenset(verts[b] for b in range(4) if b != a)
            if face in face_pts:
                tet_faces.append(face)
        if len(tet_faces) >= 1:
            centroid = np.mean([face_pts[f] for f in tet_faces], axis=0)
            cp = surface.closest_point(centroid)
            if cp is not None:
                tet_pts[tet_idx] = cp

    # Step 4: Build triangles from chains (deduplicate)
    K_verts = []
    K_tris = []
    vert_index = {}
    tri_set = set()

    def add_vert(pt):
        key = tuple(np.round(pt, 10))
        if key not in vert_index:
            vert_index[key] = len(K_verts)
            K_verts.append(pt)
        return vert_index[key]

    for tet_idx, tet in enumerate(T.tetrahedra):
        if tet_idx not in tet_pts:
            continue
        vt = tet_pts[tet_idx]
        i_t = add_vert(vt)

        verts = list(tet)
        for a in range(4):
            face = frozenset(verts[b] for b in range(4) if b != a)
            if face not in face_pts:
                continue
            vf = face_pts[face]
            i_f = add_vert(vf)

            fv = list(face)
            for p in range(3):
                for q in range(p + 1, 3):
                    edge = frozenset([fv[p], fv[q]])
                    if edge in edge_pts:
                        ve = edge_pts[edge]
                        i_e = add_vert(ve)
                        tri_key = tuple(sorted([i_e, i_f, i_t]))
                        if tri_key not in tri_set:
                            tri_set.add(tri_key)
                            K_tris.append((i_e, i_f, i_t))

    return {
        'K_verts': K_verts,
        'K_tris': K_tris,
        'edge_pts': edge_pts,
        'face_pts': face_pts,
        'tet_pts': tet_pts,
    }


# ═══════════════════════════════════════════════════════════════════
# Quality metrics
# ═══════════════════════════════════════════════════════════════════

def quality_metrics(K, surface):
    va = [np.array(v) for v in K['K_verts']]
    areas = []
    hausdorff = 0.0
    for (i0, i1, i2) in K['K_tris']:
        a, b, c = va[i0], va[i1], va[i2]
        area = np.linalg.norm(np.cross(b - a, c - a)) / 2
        areas.append(area)
        centroid = (a + b + c) / 3.0
        cp = surface.closest_point(centroid)
        if cp is not None:
            hausdorff = max(hausdorff, np.linalg.norm(centroid - cp))
    return {
        'total_area': sum(areas),
        'min_area': min(areas) if areas else 0,
        'max_area': max(areas) if areas else 0,
        'hausdorff': hausdorff,
    }


# ═══════════════════════════════════════════════════════════════════
# Surface definitions
# ═══════════════════════════════════════════════════════════════════

def sphere_surface(r=1.0):
    return ImplicitSurface(
        f=lambda x, y, z: x**2 + y**2 + z**2 - r**2,
        grad_f=lambda x, y, z: (2*x, 2*y, 2*z),
        reach=r, name=f"Sphere(r={r})")


def torus_surface(R=1.0, r=0.4):
    return ImplicitSurface(
        f=lambda x, y, z: (np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2,
        grad_f=lambda x, y, z: (
            2*(np.sqrt(x**2+y**2)-R)*x/(np.sqrt(x**2+y**2)+1e-30),
            2*(np.sqrt(x**2+y**2)-R)*y/(np.sqrt(x**2+y**2)+1e-30),
            2*z),
        reach=r, name=f"Torus(R={R},r={r})")


def ellipsoid_surface(a=1.2, b=0.8, c=0.6):
    rch = min(a, b, c)**2 / max(a, b, c)
    return ImplicitSurface(
        f=lambda x, y, z: x**2/a**2 + y**2/b**2 + z**2/c**2 - 1,
        grad_f=lambda x, y, z: (2*x/a**2, 2*y/b**2, 2*z/c**2),
        reach=rch, name=f"Ellipsoid({a},{b},{c})")


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def run(surface, L=None, bounds=None, practical_scale=50.0, verbose=True):
    reach = surface.reach
    if L is None:
        L = reach / 5.0
    if bounds is None:
        s = reach * 1.8
        bounds = (-s, s, -s, s, -s, s)

    if verbose:
        print("=" * 65)
        print("  Whitney's Triangulation — 2D manifold in 3D")
        print("  Coxeter triangulation of type Ã₃ (proper alcove decomposition)")
        print("  (Boissonnat–Kachanovich–Wintraecken, DCG 2021)")
        print("=" * 65)
        print(f"  Surface : {surface.name}")
        print(f"  d=3, n=2   rch(M)={reach:.4f}   L={L:.4f}   L/rch={L/reach:.4f}")
        print()

    consts = compute_constants_3d(L, reach, practical_scale)
    if verbose:
        print("─── Constants (§4-5) ───")
        print(f"  t(T)  = {consts['thickness']:.4f}")
        print(f"  c̃     = {consts['c_tilde']:.4f}  "
              f"(theory {consts['c_tilde_theory']:.6f})")
        print(f"  ρ₁    = {consts['rho1']:.4f}")
        print(f"  α₀    = {consts['alpha0']:.4f}")
        print(f"  c̃·L_max = {consts['c_tilde']*consts['Lmax']:.4f}  "
              f"(max perturbation)")
        print(f"  ρ₁c̃L_max = "
              f"{consts['rho1']*consts['c_tilde']*consts['Lmax']:.4f}  "
              f"(tangent clearance)")
        print()

    if verbose:
        print("─── Part 1: Ambient triangulation & perturbation ───")
    T = CoxeterA3Triangulation3D(L, bounds)
    if verbose:
        print(f"  Vertices:    {len(T.vertices)}")
        print(f"  Edges:       {len(T.edges)}")
        print(f"  Faces:       {len(T.faces)}")
        print(f"  Tetrahedra:  {len(T.tetrahedra)}")

    pverts, pinfo = perturb_vertices(T, surface, consts)
    if verbose:
        print(f"  Case 1 (far):      {pinfo['case1']}")
        print(f"  Case 2 (perturbed): {pinfo['case2']}")
        print(f"  Max perturbation:   {pinfo['max_pert']:.6f}")
        print()

    if verbose:
        print("─── Part 2: Triangulation K ───")
    K = construct_K(T, pverts, surface, consts)
    if verbose:
        print(f"  Edge intersections v(τ¹): {len(K['edge_pts'])}")
        print(f"  Face centres v(τ²):       {len(K['face_pts'])}")
        print(f"  Tet centres v(τ³):        {len(K['tet_pts'])}")
        print(f"  Triangles in K:           {len(K['K_tris'])}")
        print()

    qm = quality_metrics(K, surface)
    if verbose:
        print("─── Quality ───")
        print(f"  Total area of K:   {qm['total_area']:.4f}")
        print(f"  Triangle area range: [{qm['min_area']:.6f}, "
              f"{qm['max_area']:.6f}]")
        print(f"  Max Hausdorff ≈:   {qm['hausdorff']:.6f}")
        print()

    return T, pverts, K, consts


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Testing Coxeter Ã₃ triangulation...\n")

    # Verify tet quality
    T = CoxeterA3Triangulation3D(L=1.0, bounds=(-1, 1, -1, 1, -1, 1))
    print(f"Vertices: {len(T.vertices)}")
    print(f"Tets: {len(T.tetrahedra)}")
    print(f"Lmax = {T.Lmax:.6f}")
    print(f"Thickness = {T.thickness:.6f}")

    import random; random.seed(42)
    sample = random.sample(T.tetrahedra, min(200, len(T.tetrahedra)))
    bad = 0
    for tet in sample:
        pts = [T.vertices[v] for v in tet]
        eds = sorted(round(np.linalg.norm(pts[j]-pts[i]), 4)
                     for i, j in combinations(range(4), 2))
        expected = sorted([0.8660, 0.8660, 0.8660, 0.8660, 1.0, 1.0])
        if eds != expected:
            bad += 1
            if bad <= 3:
                print(f"  BAD: {eds}  keys={tet}")
    print(f"Bad tets: {bad}/{len(sample)}")

    # Run on sphere
    print()
    sph = sphere_surface(r=1.0)
    T2, pv2, K2, c2 = run(sph, L=0.35,
                           bounds=(-1.6, 1.6, -1.6, 1.6, -1.6, 1.6))