"""
Whitney's Triangulation Algorithm — 2D Manifold in 3D
=====================================================

Based on: Boissonnat, Kachanovich, Wintraecken (2021)
"Triangulating Submanifolds: An Elementary and Quantified
 Version of Whitney's Method", DCG 66:386–434

Case: n = 2 (surface), d = 3 (ambient R³)

Ambient triangulation: Coxeter triangulation of type Ã₃.

  The Ã₃ Coxeter triangulation is the Freudenthal-Kuhn triangulation
  (§2.1, Def 1).  Each unit cube of the Z³ lattice is subdivided
  into d! = 6 tetrahedra, one per permutation σ ∈ S₃, with vertices:
      v₀ = corner,  vₖ = vₖ₋₁ + e_{σ(k)}   for k = 1,2,3.

  These are NOT regular tetrahedra — no affine map can make all 6
  tets per cube regular simultaneously (the 7 distinct edge vectors
  {eᵢ, eᵢ+eⱼ, (1,1,1)} are overconstrained for a 3×3 Gram matrix).

  Properties:
      Edge lengths:  L, L√2, L√3   (3 per tet of each type)
      L_max = L√3                   (body diagonal)
      t(T) = 1/√6 ≈ 0.408          (thickness, Def 2.2, paper §2)
        = d·vol(σ) / (diam(σ) · max facet vol)
        = 3·(L³/6) / (L√3 · L²√2/2)
        = L³/2 / (L³√6/2)  =  1/√6

Part 1 (§5): Perturb vertices so that the 0-skeleton (= (d-n-1)-skeleton)
  is far from M.  For codimension 1, push vertices away from T_pM.

Part 2 (§6): Construct K via barycentric subdivision.
  For each chain  τ¹ ⊂ τ² ⊂ τ³  (edge ⊂ face ⊂ tet) intersecting M,
  the 2-simplex  {v(τ¹), v(τ²), v(τ³)}  is a triangle of K.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import brentq
from collections import defaultdict
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
# §2.1/§4: Coxeter Triangulation of type Ã₃ (= Freudenthal-Kuhn)
# ═══════════════════════════════════════════════════════════════════
#
# Definition 1 (paper §2.1): The Coxeter triangulation of type Ã_d
# is the Freudenthal-Kuhn triangulation of R^d.  Each unit cube of
# the integer lattice Z^d is subdivided into d! simplices by the
# coordinate permutations.
#
# For d=3: each unit cube → 6 tetrahedra.  The tetrahedron for
# permutation σ ∈ S₃ has vertices
#     v₀ = (i,j,k),  v₁ = v₀+e_{σ(1)},  v₂ = v₁+e_{σ(2)},
#     v₃ = v₂+e_{σ(3)} = (i+1,j+1,k+1).
#
# Edge lengths: L, L√2, L√3 (NOT regular tetrahedra).
# Thickness: t(T) = 1/√6 ≈ 0.408  (Def 2.2)
# L_max = L√3 (body diagonal)
#
# Note: No affine map can make all 6 tets per cube regular
# simultaneously.  The 7 distinct edge directions {eᵢ, eᵢ+eⱼ,
# (1,1,1)} overconstrain the 3×3 Gram matrix A^T A.


class CoxeterA3Triangulation3D:
    """
    Coxeter triangulation of type Ã₃ (= Freudenthal-Kuhn) of R³.

    Each unit cube of the lattice L·Z³ is subdivided into 6 tetrahedra
    by coordinate permutations.  Vertices lie on L·Z³.

    Simplex for permutation σ ∈ S₃:
        v₀ = L·(i,j,k),  vₘ = vₘ₋₁ + L·e_{σ(m)}  for m=1,2,3.

    Edge lengths: L, L√2, L√3.  Thickness t(T) = 1/√6.
    """

    def __init__(self, L, bounds, margin=2):
        """
        Parameters
        ----------
        L : float   – grid spacing.  Vertex positions on L·Z³.
        bounds : (xmin,xmax,ymin,ymax,zmin,zmax) in R³
        margin : int – extra cells beyond bounds
        """
        self.L = L
        self.bounds = bounds
        self.Lmax = L * np.sqrt(3)    # body diagonal = longest edge

        # Thickness (Def 2.2): t = d·V/(diam·max_face_area)
        # = 3·(L³/6) / (L√3 · L²√2/2) = 1/√6
        self.thickness = 1.0 / np.sqrt(6)

        # The 6 permutations of axes for cell subdivision
        self.perms = list(permutations(range(3)))

        # Storage
        self.vertices = {}           # (i,j,k) -> np.array position
        self.tetrahedra = []         # list of 4-tuples of vertex keys
        self.edges = set()           # frozenset of 2 vertex keys
        self.faces = set()           # frozenset of 3 vertex keys

        # Adjacency (built after generation)
        self.edge_to_faces = defaultdict(set)
        self.edge_to_tets = defaultdict(set)
        self.face_to_tets = defaultdict(set)

        self._generate(margin)
        self._build_adjacency()

    def _generate(self, margin):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        m = margin
        L = self.L

        # Integer index range covering the bounds
        i0 = int(np.floor(xmin / L)) - m
        i1 = int(np.ceil(xmax / L)) + m
        j0 = int(np.floor(ymin / L)) - m
        j1 = int(np.ceil(ymax / L)) + m
        k0 = int(np.floor(zmin / L)) - m
        k1 = int(np.ceil(zmax / L)) + m

        # Create vertices: position = L · (i, j, k)
        for i in range(i0, i1 + 2):
            for j in range(j0, j1 + 2):
                for k in range(k0, k1 + 2):
                    self.vertices[(i, j, k)] = L * np.array(
                        [i, j, k], dtype=float)

        # Create tetrahedra: 6 per cell (one per permutation σ ∈ S₃)
        e = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    corner = (i, j, k)
                    for perm in self.perms:
                        v0 = corner
                        v1 = (v0[0] + e[perm[0]][0],
                              v0[1] + e[perm[0]][1],
                              v0[2] + e[perm[0]][2])
                        v2 = (v1[0] + e[perm[1]][0],
                              v1[1] + e[perm[1]][1],
                              v1[2] + e[perm[1]][2])
                        v3 = (v2[0] + e[perm[2]][0],
                              v2[1] + e[perm[2]][1],
                              v2[2] + e[perm[2]][2])

                        tet = (v0, v1, v2, v3)
                        self.tetrahedra.append(tet)

                        # Edges (6 per tet)
                        verts = [v0, v1, v2, v3]
                        for a in range(4):
                            for b in range(a + 1, 4):
                                self.edges.add(frozenset([verts[a], verts[b]]))
                        # Faces (4 per tet)
                        for a in range(4):
                            face = frozenset(
                                [verts[b] for b in range(4) if b != a])
                            self.faces.add(face)

    def _build_adjacency(self):
        """Build edge→face, edge→tet, face→tet maps."""
        for tet_idx, tet in enumerate(self.tetrahedra):
            verts = list(tet)
            # Faces of this tet
            for a in range(4):
                face = frozenset([verts[b] for b in range(4) if b != a])
                self.face_to_tets[face].add(tet_idx)
                # Edges of this face
                fv = list(face)
                for p in range(3):
                    for q in range(p + 1, 3):
                        edge = frozenset([fv[p], fv[q]])
                        self.edge_to_faces[edge].add(face)
            # Edges of this tet
            for a in range(4):
                for b in range(a + 1, 4):
                    edge = frozenset([verts[a], verts[b]])
                    self.edge_to_tets[edge].add(tet_idx)


# Alias for backward compatibility
FreudenthalTriangulation3D = CoxeterA3Triangulation3D


# ═══════════════════════════════════════════════════════════════════
# Implicit Surface  M = { p ∈ R³ : f(p) = 0 }
# ═══════════════════════════════════════════════════════════════════

class ImplicitSurface:
    """
    C² surface in R³ defined implicitly as f⁻¹(0).
    Oracles (§2.1): closest point on M, tangent plane T_pM.
    """

    def __init__(self, f, grad_f, reach, name="surface"):
        self.f = f
        self.grad_f = grad_f
        self.reach = reach
        self.name = name

    def evaluate(self, p):
        return self.f(p[0], p[1], p[2])

    def gradient(self, p):
        return np.array(self.grad_f(p[0], p[1], p[2]), dtype=float)

    def normal(self, p):
        """Unit normal N_pM at p ∈ M."""
        g = self.gradient(p)
        n = np.linalg.norm(g)
        return g / n if n > 1e-15 else np.array([0, 0, 1.0])

    def closest_point(self, p, max_iter=60, tol=1e-12):
        """Oracle 1: project p onto M via Newton iteration."""
        q = np.array(p, dtype=float)
        for _ in range(max_iter):
            val = self.f(q[0], q[1], q[2])
            if abs(val) < tol:
                return q
            g = self.gradient(q)
            gg = np.dot(g, g)
            if gg < 1e-20:
                break
            q = q - (val / gg) * g
        if abs(self.f(q[0], q[1], q[2])) < 1e-8:
            return q
        return None

    def find_edge_intersection(self, p1, p2):
        """
        Find unique intersection of M with segment [p1, p2].
        Lemma 6.4 guarantees at most one point for d-n = 1 edges.
        """
        f1 = self.evaluate(p1)
        f2 = self.evaluate(p2)
        if abs(f1) < 1e-13:
            return p1.copy()
        if abs(f2) < 1e-13:
            return p2.copy()
        if f1 * f2 > 0:
            return None

        def g(t):
            pt = (1.0 - t) * p1 + t * p2
            return self.f(pt[0], pt[1], pt[2])
        try:
            t_star = brentq(g, 0.0, 1.0, xtol=1e-14)
            return (1.0 - t_star) * p1 + t_star * p2
        except ValueError:
            return None


# ═══════════════════════════════════════════════════════════════════
# Algorithm Constants (§4-5)
# ═══════════════════════════════════════════════════════════════════

def compute_constants_3d(L, reach, practical_scale=50.0):
    """Compute algorithm constants for d=3, n=2, Coxeter Ã₃ (= Freudenthal)."""
    d, n = 3, 2

    # Ã₃ = Freudenthal thickness (Def 2.2):
    #   t(T) = d·vol(σ)/(diam(σ)·max_facet_vol)
    #        = 3·(L³/6)/(L√3 · L²√2/2) = 1/√6 ≈ 0.408
    t_T = 1.0 / np.sqrt(6)
    Lmax = L * np.sqrt(3)  # body diagonal = longest edge

    # Theoretical c̃ (eq. 6) — much better than Freudenthal due to t²
    c_tilde_theory = t_T ** 2 / 24.0   # = (4/6)/24 = 1/36 ≈ 0.0278
    c_tilde = min(c_tilde_theory * practical_scale, 0.42)

    # ρ₁ (Lemma 5.1, d=3 odd, k=(d+1)/2=2)
    N_leq = 3  # N_{≤0} for d=3, n=2
    rho1_theory = math.factorial(4) / (
        2 ** 6 * math.factorial(2) * math.factorial(1) * N_leq)
    rho1 = min(rho1_theory * practical_scale, 0.90)

    # α₀ (eq. 8)
    alpha0 = (4.0 / 3.0) * rho1 * c_tilde

    # ζ (eq. 10) — quality bound, clamped positive
    binom_d_dn = math.comb(d, d - n)  # C(3,1)=3
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
#
# For d=3, n=2 (surface in R³), codimension = 1.
# (d-n-1) = 0: the 0-skeleton (vertices) must be pushed from M.
# span(τ'_j, T_pM) = T_pM (tangent plane).
# Push each close vertex away from the tangent plane at a nearby
# manifold point (§5.2, Case 2, same as codimension-1 in 2D).

def perturb_vertices(T, surface, consts):
    """
    Part 1 (§5.2): Perturb vertices of T so 0-skeleton is far from M.

    Case 1: d(v, M) ≥ 3L_max/2  →  keep v
    Case 2: d(v, M) < 3L_max/2  →  push v away from T_pM
    """
    L = consts['Lmax']  # use longest edge for the 3L/2 threshold
    c_tilde = consts['c_tilde']
    rho1 = consts['rho1']

    max_perturb = c_tilde * L          # eq. 17: |v - v~| ≤ c̃L
    tangent_clearance = rho1 * c_tilde * L  # eq. 20

    perturbed = {}
    info = {'case1': 0, 'case2': 0, 'max_pert': 0.0}

    for key, v in T.vertices.items():
        p = surface.closest_point(v)
        if p is None:
            perturbed[key] = v.copy()
            info['case1'] += 1
            continue

        dist = np.linalg.norm(v - p)
        if dist >= 1.5 * L:
            # Case 1
            perturbed[key] = v.copy()
            info['case1'] += 1
        else:
            # Case 2: push away from T_pM in normal direction
            nrm = surface.normal(p)
            normal_comp = np.dot(v - p, nrm)

            if abs(normal_comp) >= tangent_clearance:
                perturbed[key] = v.copy()
                info['case1'] += 1
            else:
                sign = 1.0 if normal_comp >= 0 else -1.0
                if abs(normal_comp) < 1e-15:
                    sign = 1.0
                target = sign * tangent_clearance
                disp = (target - normal_comp) * nrm
                dn = np.linalg.norm(disp)
                if dn > max_perturb:
                    disp *= max_perturb / dn
                perturbed[key] = v + disp
                info['case2'] += 1
                info['max_pert'] = max(info['max_pert'], np.linalg.norm(disp))

    return perturbed, info


# ═══════════════════════════════════════════════════════════════════
# §6: Part 2 — Construct triangulation K of M
# ═══════════════════════════════════════════════════════════════════
#
# For n=2, d=3 the barycentric-subdivision complex K has:
#   - Vertices v(τ¹) on edges of T̃ that intersect M  (on M, unique)
#   - Vertices v(τ²) on faces of T̃ that intersect M  (avg of edge pts)
#   - Vertices v(τ³) on tets of T̃ that intersect M   (avg of edge pts)
#   - 2-simplices: {v(τ¹), v(τ²), v(τ³)} for chains τ¹⊂τ²⊂τ³
#     where all three simplices intersect M  (eq. 25)

def construct_K(T, pverts, surface, consts):
    """
    Part 2 (§6.2): Build the simplicial complex K.

    Returns dict with vertex positions and triangle index lists.
    """
    # ── Step 1: edge–surface intersections → v(τ¹) ──
    edge_pts = {}  # edge_key → np.array on M
    for edge in T.edges:
        v1k, v2k = list(edge)
        p1, p2 = pverts[v1k], pverts[v2k]
        pt = surface.find_edge_intersection(p1, p2)
        if pt is not None:
            edge_pts[edge] = pt

    # ── Step 2: face representative points → v(τ²) ──
    face_pts = {}    # face_key → np.array
    face_edges = {}  # face_key → list of edges intersecting M
    for face in T.faces:
        fv = list(face)
        # 3 edges of this triangle
        f_edges = [frozenset([fv[a], fv[b]])
                   for a in range(3) for b in range(a + 1, 3)]
        hit = [e for e in f_edges if e in edge_pts]
        if hit:
            face_pts[face] = np.mean([edge_pts[e] for e in hit], axis=0)
            face_edges[face] = hit

    # ── Step 3: tet representative points → v(τ³) ──
    tet_pts = {}
    tet_faces = {}   # tet_idx → list of faces intersecting M
    for tet_idx, tet in enumerate(T.tetrahedra):
        verts = list(tet)
        t_edges = [frozenset([verts[a], verts[b]])
                   for a in range(4) for b in range(a + 1, 4)]
        hit = [e for e in t_edges if e in edge_pts]
        if hit:
            tet_pts[tet_idx] = np.mean([edge_pts[e] for e in hit], axis=0)
            # faces of this tet that intersect M
            t_faces = [frozenset([verts[b] for b in range(4) if b != a])
                       for a in range(4)]
            tet_faces[tet_idx] = [f for f in t_faces if f in face_pts]

    # ── Step 4: build K's vertex list & triangles ──
    # Vertices of K
    K_verts = []
    vert_idx = {}

    for ek, pt in edge_pts.items():
        vert_idx[('e', ek)] = len(K_verts)
        K_verts.append(pt)
    for fk, pt in face_pts.items():
        vert_idx[('f', fk)] = len(K_verts)
        K_verts.append(pt)
    for ti, pt in tet_pts.items():
        vert_idx[('t', ti)] = len(K_verts)
        K_verts.append(pt)

    # Triangles of K: one per chain  edge ⊂ face ⊂ tet  (eq. 25)
    K_triangles = []
    for tet_idx, faces_hit in tet_faces.items():
        ti_key = ('t', tet_idx)
        if ti_key not in vert_idx:
            continue
        ti_vi = vert_idx[ti_key]
        for face in faces_hit:
            fi_key = ('f', face)
            if fi_key not in vert_idx:
                continue
            fi_vi = vert_idx[fi_key]
            # edges of this face that intersect M
            for edge in face_edges.get(face, []):
                ei_key = ('e', edge)
                if ei_key not in vert_idx:
                    continue
                ei_vi = vert_idx[ei_key]
                K_triangles.append((ei_vi, fi_vi, ti_vi))

    return {
        'edge_pts': edge_pts,
        'face_pts': face_pts,
        'tet_pts': tet_pts,
        'tet_faces': tet_faces,
        'K_verts': K_verts,
        'K_tris': K_triangles,
    }


# ═══════════════════════════════════════════════════════════════════
# Visualization  (3D)
# ═══════════════════════════════════════════════════════════════════

def plot_result_3d(surface, T, pverts, K, consts, elev=25, azim=135):
    """Four-panel 3D visualization."""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Whitney's Triangulation — Coxeter Ã₃ (Freudenthal) — 2D in 3D\n"
        f"{surface.name}  |  d=3, n=2  |  "
        f"L={consts['L']:.3f}, rch(M)={consts['reach']:.3f}",
        fontsize=14, fontweight='bold', y=0.97)

    xmin, xmax, ymin, ymax, zmin, zmax = T.bounds
    pad = consts['L'] * 2

    # ─── helper: draw the reference surface (wireframe) ───
    def draw_surface(ax, alpha=0.15, color='#E63946'):
        u = np.linspace(0, 2 * np.pi, 80)
        v = np.linspace(0, np.pi, 40) if surface.name.startswith('Sphere') \
            else np.linspace(0, 2 * np.pi, 80)
        U, V = np.meshgrid(u, v)
        X, Y, Z = _parametric_surface(surface.name, U, V)
        ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha,
                          linewidth=0.3, rstride=2, cstride=2)

    def setup_ax(ax, title):
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_zlim(zmin - pad, zmax + pad)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=0)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # ════════ Panel 1: ambient T with surface ════════
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    draw_surface(ax1, alpha=0.25)
    # draw a sample of ambient edges near the surface
    _draw_ambient_edges_near_M(ax1, T, T.vertices, surface,
                                consts, color='#457B9D', alpha=0.08)
    setup_ax(ax1, "§4: Ambient Triangulation T\n(with surface M)")

    # ════════ Panel 2: perturbed T̃ with highlighted verts ════════
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    draw_surface(ax2, alpha=0.20)
    _draw_ambient_edges_near_M(ax2, T, pverts, surface,
                                consts, color='#457B9D', alpha=0.06)

    # Show perturbation arrows
    pert_orig, pert_new = [], []
    for key in T.vertices:
        d = np.linalg.norm(pverts[key] - T.vertices[key])
        if d > 1e-10:
            pert_orig.append(T.vertices[key])
            pert_new.append(pverts[key])
    if pert_new:
        pn = np.array(pert_new)
        ax2.scatter(pn[:, 0], pn[:, 1], pn[:, 2], c='#2A9D8F',
                    s=18, zorder=5, depthshade=False, edgecolors='w',
                    linewidths=0.3)
    setup_ax(ax2,
             f"§5: Perturbed T̃\n({len(pert_new)} vertices pushed from M)")

    # ════════ Panel 3: edge intersections + face/tet centres ════════
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    draw_surface(ax3, alpha=0.12)

    if K['edge_pts']:
        ep = np.array(list(K['edge_pts'].values()))
        ax3.scatter(ep[:, 0], ep[:, 1], ep[:, 2], c='#E63946',
                    s=10, zorder=5, label=f"v(τ¹) ×{len(ep)}",
                    depthshade=False, edgecolors='w', linewidths=0.2)
    if K['face_pts']:
        fp = np.array(list(K['face_pts'].values()))
        ax3.scatter(fp[:, 0], fp[:, 1], fp[:, 2], c='#F4A261',
                    s=8, zorder=5, marker='D',
                    label=f"v(τ²) ×{len(fp)}", depthshade=False)
    if K['tet_pts']:
        tp = np.array(list(K['tet_pts'].values()))
        ax3.scatter(tp[:, 0], tp[:, 1], tp[:, 2], c='#264653',
                    s=6, zorder=5, marker='s',
                    label=f"v(τ³) ×{len(tp)}", depthshade=False)
    ax3.legend(fontsize=8, loc='lower left')
    setup_ax(ax3, "§6: Intersection Points\n"
                   "(edge●, face◆, tet■ representatives)")

    # ════════ Panel 4: final triangulation K ════════
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    draw_surface(ax4, alpha=0.08, color='#E6939A')

    verts_arr = [np.array(v) for v in K['K_verts']]
    if K['K_tris'] and verts_arr:
        polys = []
        for (i0, i1, i2) in K['K_tris']:
            polys.append([verts_arr[i0], verts_arr[i1], verts_arr[i2]])
        pc = Poly3DCollection(polys, alpha=0.45,
                               facecolor='#457B9D', edgecolor='#1D3557',
                               linewidths=0.3)
        ax4.add_collection3d(pc)

    setup_ax(ax4,
             f"§6+7: Triangulation K of M\n"
             f"({len(K['K_tris'])} triangles, "
             f"homeomorphic to M)")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _draw_ambient_edges_near_M(ax, T, verts, surface, consts,
                                color='#457B9D', alpha=0.1):
    """Draw ambient edges within ~2L of the surface."""
    threshold = 2.5 * consts['Lmax']
    segments = []
    for edge in T.edges:
        v1k, v2k = list(edge)
        p1, p2 = verts[v1k], verts[v2k]
        mid = 0.5 * (p1 + p2)
        cp = surface.closest_point(mid)
        if cp is not None and np.linalg.norm(mid - cp) < threshold:
            segments.append([p1, p2])
        if len(segments) > 6000:
            break
    if segments:
        lc = Line3DCollection(segments, colors=color, linewidths=0.25,
                               alpha=alpha)
        ax.add_collection3d(lc)


def _parametric_surface(name, U, V):
    """Parametric mesh for known surfaces (for reference wireframe)."""
    if name.startswith('Sphere'):
        # extract radius from name
        r = float(name.split('r=')[1].rstrip(')'))
        X = r * np.sin(V) * np.cos(U)
        Y = r * np.sin(V) * np.sin(U)
        Z = r * np.cos(V)
    elif name.startswith('Torus'):
        parts = name.split(',')
        R = float(parts[0].split('R=')[1])
        r = float(parts[1].split('r=')[1].rstrip(')'))
        X = (R + r * np.cos(V)) * np.cos(U)
        Y = (R + r * np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)
    elif name.startswith('Ellipsoid'):
        parts = name.replace(')', '').split(',')
        a = float(parts[0].split('a=')[1])
        b = float(parts[1].split('b=')[1])
        c = float(parts[2].split('c=')[1])
        X = a * np.sin(V) * np.cos(U)
        Y = b * np.sin(V) * np.sin(U)
        Z = c * np.cos(V)
    else:
        X = np.cos(U)
        Y = np.sin(U)
        Z = 0 * U
    return X, Y, Z


def plot_K_standalone(surface, K, consts, elev=25, azim=135):
    """Large standalone plot of just the triangulation K."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    xmin, xmax, ymin, ymax, zmin, zmax = \
        [-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]  # default

    # Reference surface
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 40) if surface.name.startswith('Sphere') \
        else np.linspace(0, 2 * np.pi, 80)
    U, V = np.meshgrid(u, v)
    X, Y, Z = _parametric_surface(surface.name, U, V)
    ax.plot_wireframe(X, Y, Z, color='#E63946', alpha=0.06,
                      linewidth=0.2, rstride=3, cstride=3)

    # Draw K triangles
    verts_arr = [np.array(v) for v in K['K_verts']]
    if K['K_tris'] and verts_arr:
        polys = []
        for (i0, i1, i2) in K['K_tris']:
            polys.append([verts_arr[i0], verts_arr[i1], verts_arr[i2]])
        pc = Poly3DCollection(polys, alpha=0.5,
                               facecolor='#A8DADC', edgecolor='#1D3557',
                               linewidths=0.35)
        ax.add_collection3d(pc)

    # Edge intersection points (on M)
    if K['edge_pts']:
        ep = np.array(list(K['edge_pts'].values()))
        ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2], c='#E63946',
                   s=4, depthshade=False, zorder=5)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(
        f"Triangulation K of {surface.name}\n"
        f"{len(K['K_tris'])} triangles  |  "
        f"{len(K['edge_pts'])} edge-surface intersections",
        fontsize=13, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Quality Metrics
# ═══════════════════════════════════════════════════════════════════

def quality_metrics(K, surface):
    va = [np.array(v) for v in K['K_verts']]
    tris = K['K_tris']
    if not tris or not va:
        return {}
    areas, max_dist = [], 0.0
    for (i0, i1, i2) in tris:
        a = np.linalg.norm(np.cross(va[i1] - va[i0], va[i2] - va[i0])) / 2
        areas.append(a)
    for pt in K['K_verts']:
        cp = surface.closest_point(np.array(pt))
        if cp is not None:
            max_dist = max(max_dist, np.linalg.norm(np.array(pt) - cp))
    return {
        'n_verts': len(va), 'n_tris': len(tris),
        'n_edge_pts': len(K['edge_pts']),
        'n_face_pts': len(K['face_pts']),
        'n_tet_pts': len(K['tet_pts']),
        'area_min': min(areas), 'area_max': max(areas),
        'area_mean': np.mean(areas), 'total_area': sum(areas),
        'max_hausdorff': max_dist,
    }


# ═══════════════════════════════════════════════════════════════════
# Example Surfaces
# ═══════════════════════════════════════════════════════════════════

def sphere_surface(r=1.0):
    return ImplicitSurface(
        f=lambda x, y, z: x**2 + y**2 + z**2 - r**2,
        grad_f=lambda x, y, z: (2*x, 2*y, 2*z),
        reach=r,
        name=f"Sphere(r={r})")


def torus_surface(R=1.0, r=0.4):
    """Torus with major radius R, minor radius r.  reach ≈ min(r, R-r)."""
    rch = min(r, R - r) if R > r else r * 0.5
    return ImplicitSurface(
        f=lambda x, y, z: (np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2,
        grad_f=lambda x, y, z: (
            2 * x * (1 - R / (np.sqrt(x**2 + y**2) + 1e-30)),
            2 * y * (1 - R / (np.sqrt(x**2 + y**2) + 1e-30)),
            2 * z),
        reach=rch,
        name=f"Torus(R={R},r={r})")


def ellipsoid_surface(a=1.2, b=0.8, c=0.6):
    rch = min(a, b, c)**2 / max(a, b, c)
    return ImplicitSurface(
        f=lambda x, y, z: (x/a)**2 + (y/b)**2 + (z/c)**2 - 1,
        grad_f=lambda x, y, z: (2*x/a**2, 2*y/b**2, 2*z/c**2),
        reach=rch,
        name=f"Ellipsoid(a={a},b={b},c={c})")


# ═══════════════════════════════════════════════════════════════════
# Main Driver
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
        print("  Coxeter triangulation of type Ã₃ (= Freudenthal-Kuhn)")
        print("  (Boissonnat–Kachanovich–Wintraecken, DCG 2021)")
        print("=" * 65)
        print(f"  Surface : {surface.name}")
        print(f"  d=3, n=2   rch(M)={reach:.4f}   L={L:.4f}   "
              f"L/rch={L/reach:.4f}")
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

    # Part 1
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

    # Part 2
    if verbose:
        print("─── Part 2: Triangulation K ───")
    K = construct_K(T, pverts, surface, consts)
    if verbose:
        print(f"  Edge intersections v(τ¹): {len(K['edge_pts'])}")
        print(f"  Face centres v(τ²):       {len(K['face_pts'])}")
        print(f"  Tet centres v(τ³):        {len(K['tet_pts'])}")
        print(f"  Triangles in K:           {len(K['K_tris'])}")
        print()

    m = quality_metrics(K, surface)
    if verbose and m:
        print("─── Quality ───")
        print(f"  Total area of K:   {m['total_area']:.4f}")
        print(f"  Triangle area range: [{m['area_min']:.6f}, "
              f"{m['area_max']:.6f}]")
        print(f"  Max Hausdorff ≈:   {m['max_hausdorff']:.6f}")
        print()

    return T, pverts, K, consts


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── Sphere ──
    print("\n" + "█" * 65)
    print("  SPHERE")
    print("█" * 65 + "\n")
    sph = sphere_surface(r=1.0)
    T1, pv1, K1, c1 = run(sph, L=0.35,
                           bounds=(-1.6, 1.6, -1.6, 1.6, -1.6, 1.6))

    fig1 = plot_result_3d(sph, T1, pv1, K1, c1)
    fig1.savefig('sphere_3d.png', dpi=150, bbox_inches='tight')

    fig1s = plot_K_standalone(sph, K1, c1)
    fig1s.savefig('sphere_K.png', dpi=150, bbox_inches='tight')

    # ── Torus ──
    print("\n" + "█" * 65)
    print("  TORUS")
    print("█" * 65 + "\n")
    tor = torus_surface(R=1.0, r=0.4)
    T2, pv2, K2, c2 = run(tor, L=0.22,
                           bounds=(-1.8, 1.8, -1.8, 1.8, -0.8, 0.8))

    fig2 = plot_result_3d(tor, T2, pv2, K2, c2, elev=30, azim=120)
    fig2.savefig('torus_3d.png', dpi=150, bbox_inches='tight')

    fig2s = plot_K_standalone(tor, K2, c2, elev=30, azim=120)
    fig2s.savefig('torus_K.png', dpi=150, bbox_inches='tight')

    print("\n✓ All saved.")
    plt.close('all')