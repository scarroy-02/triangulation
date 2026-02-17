# Whitney Triangulation — Coxeter Ã₃ Implementation

## Line-by-Line Code Walkthrough

`whitney3d_coxeter.py` — 626 lines total.

---

## Lines 1–46: Header and Imports

```python
"""
Whitney's Triangulation Algorithm — 2D Manifold in 3D
...
"""
```

The docstring lays out the mathematical setting. We have:
- A smooth surface M (2-manifold, n=2) embedded in R³ (ambient dimension d=3)
- The paper's algorithm produces a triangulation K of M that is provably homeomorphic to M

The Ã₃ Coxeter triangulation is our ambient triangulation T of R³. It's constructed by a specific procedure: take the Freudenthal-Kuhn triangulation of Z⁴ (the standard way to subdivide 4-dimensional cubes into simplices), then intersect the entire thing with the 3-dimensional hyperplane H = {x ∈ R⁴ : x₁ + x₂ + x₃ + x₄ = 0}. The result is a space-filling tetrahedral tiling of H ≅ R³.

Why go through 4D? Because this particular construction produces tetrahedra that are all **congruent** — every single tet in the infinite tiling has exactly the same shape. They're isosceles tetrahedra with 4 short edges of length L√3/2 and 2 long edges of length L. The **thickness** (a measure of simplex quality, Definition 2.2 in the paper) is t(T) = √2/2 ≈ 0.707, which is the best achievable for any Coxeter triangulation of R³.

The imports:
```python
import math                          # factorial, comb
import numpy as np                   # all linear algebra
import matplotlib.pyplot as plt      # (available but not used in this file)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import brentq    # root-finding for edge-surface intersections
from collections import defaultdict  # adjacency maps
from itertools import permutations, combinations
import warnings
warnings.filterwarnings('ignore')    # suppress matplotlib deprecation noise
```

`brentq` is critical — it's Brent's method for finding roots of a scalar function on an interval. We use it to find the exact point where an edge of T crosses the surface M (where f changes sign). It's guaranteed to converge when f(a) and f(b) have opposite signs.

---

## Lines 49–80: The Ã₃ Lattice Machinery

### The hyperplane and ONB (lines 69–77)

```python
_F = np.array([
    [1, -1,  0,  0],
    [1,  1, -2,  0],
    [1,  1,  1, -3],
], dtype=float)
_F[0] /= np.sqrt(2)
_F[1] /= np.sqrt(6)
_F[2] /= np.sqrt(12)
```

This is the orthonormal basis (ONB) for H ⊂ R⁴. The hyperplane H = {x ∈ R⁴ : Σxᵢ = 0} is a 3-dimensional subspace of R⁴. To work in R³, we need to pick three orthonormal vectors that span H.

Before normalization, the rows are:
- f₁ = (1, −1, 0, 0) — already orthogonal to (1,1,1,1)
- f₂ = (1, 1, −2, 0) — orthogonal to f₁ and (1,1,1,1)
- f₃ = (1, 1, 1, −3) — orthogonal to f₁, f₂, and (1,1,1,1)

These are the standard Gram-Schmidt vectors for the A₃ root system. After dividing by their norms (√2, √6, √12), they become orthonormal. The 3×4 matrix `_F` maps a 4-vector in H to its R³ coordinates: if h ∈ H ⊂ R⁴, then its R³ position is `_F @ h`.

### Key deltas (line 80)

```python
_KEY_DELTA = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1)]
```

This is the heart of the Ã₃ construction. In Z⁴, the four unit basis vectors are e₁, e₂, e₃, e₄. A Freudenthal simplex in 4D walks from a corner by adding these one at a time in some permutation order. But we index our vertices by only 3 coordinates (a, b, c), using the convention z = (a, b, c, 0) as the canonical representative of the equivalence class z + k(1,1,1,1).

Under this convention:
- Adding e₁ in Z⁴ shifts the key by **(1, 0, 0)**
- Adding e₂ in Z⁴ shifts the key by **(0, 1, 0)**
- Adding e₃ in Z⁴ shifts the key by **(0, 0, 1)**
- Adding e₄ in Z⁴ shifts the key by... well, e₄ = (0,0,0,1), so the Z⁴ vector becomes (a, b, c, 1). The canonical representative with 4th component 0 is (a−1, b−1, c−1, 0), so the key shifts by **(−1, −1, −1)**.

This is the key insight: the 4th direction "wraps around" in key space. A permutation σ ∈ S₄ of the four directions {0,1,2,3} produces a walk through 4 vertices in key space, giving one tetrahedron.

### Projection functions (lines 83–93)

```python
def _pi(z):
    """Project Z⁴ vector to H = {Σxᵢ=0}."""
    z = np.asarray(z, dtype=float)
    return z - (z.sum() / 4.0) * np.ones(4)
```

This is the orthogonal projection from R⁴ onto H. For any z ∈ R⁴, we subtract the component along (1,1,1,1): the mean of its coordinates times (1,1,1,1). The result has Σxᵢ = 0, so it lies in H.

For example: π((1,0,0,0)) = (3/4, −1/4, −1/4, −1/4).

```python
def _vertex_pos_from_key(key, L):
    """R³ position from canonical key (a,b,c), scaled by L."""
    a, b, c = key
    h = _pi([a, b, c, 0])
    return L * (_F @ h)
```

This converts a vertex key (a,b,c) to its actual R³ position:
1. Form the Z⁴ vector (a, b, c, 0)
2. Project to H: subtract the mean
3. Map from H ⊂ R⁴ to R³ via the ONB matrix F
4. Scale by L

The resulting vertex positions form the **A₃\* lattice** (also called the permutohedral lattice or FCC lattice) in R³. This is the weight lattice of the A₃ root system — a face-centered cubic arrangement.

---

## Lines 96–203: `CoxeterA3Triangulation3D` Class

### `__init__` (lines 105–126)

```python
def __init__(self, L, bounds, margin=2):
    self.L = L
    self.bounds = bounds
    self.Lmax = L                     # longest edge = L
    self.thickness = np.sqrt(2) / 2   # ≈ 0.707
```

`L` is the grid spacing parameter. Unlike the Freudenthal version where the longest edge was L√3 (the cube body diagonal), here the longest edge is just L itself. This is because the 4D→3D projection shrinks distances — the "2-step" edges (connecting vertices that differ by 2 of the 4 basis directions) have length L, while "1-step" edges (differing by 1 basis direction) have length L√3/2.

The thickness √2/2 comes from:
- Volume of one tet = L³/12
- Diameter (longest edge) = L
- Max face area = L²√2/4 (all 4 faces are congruent isosceles triangles)
- t = 3V/(diam · max_face_area) = 3(L³/12)/(L · L²√2/4) = √2/2

```python
    self._perms4 = list(permutations(range(4)))
```

There are 4! = 24 permutations of {0,1,2,3}. Each one defines one tetrahedron per lattice cell. This is the S₄ action that generates the alcoves of the Ã₃ Coxeter group.

```python
    self.vertices = {}       # (a,b,c) -> np.array R³ position
    self.tetrahedra = []     # list of 4-tuples of vertex keys
    self.edges = set()       # frozenset of 2 vertex keys
    self.faces = set()       # frozenset of 3 vertex keys
```

All combinatorial data stored by integer keys. Positions are computed once and cached. Using frozensets for edges and faces means {A,B} = {B,A} automatically — no ordering issues.

```python
    self.edge_to_faces = defaultdict(set)
    self.edge_to_tets = defaultdict(set)
    self.face_to_tets = defaultdict(set)
```

Adjacency maps needed later for the chain construction in §6. `face_to_tets` tells us which tetrahedra share a given face; `edge_to_faces` tells us which faces contain a given edge.

### `_generate` (lines 128–187)

```python
    da = _vertex_pos_from_key((1, 0, 0), 1.0)
    db = _vertex_pos_from_key((0, 1, 0), 1.0)
    dc = _vertex_pos_from_key((0, 0, 1), 1.0)
    M = np.column_stack([da, db, dc])
    Minv = np.linalg.inv(M)
```

The three lattice vectors `da`, `db`, `dc` are the R³ displacements when incrementing a, b, or c by 1. They span R³ (the A₃\* lattice has full rank). The matrix M maps (a,b,c) integer space to R³ space; its inverse Minv maps R³ back to fractional (a,b,c) coordinates.

```python
    corners = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        ...
    ])
    abc = np.array([Minv @ (c / L) for c in corners])
```

To figure out which integer keys (a,b,c) we need, we take the 8 corners of the bounding box, map them back to key space (dividing by L because vertex positions are L times the unit-lattice positions), and find the min/max in each coordinate.

```python
    a0 = int(np.floor(abc[:, 0].min())) - m
    a1 = int(np.ceil(abc[:, 0].max())) + m
    ...
```

The `margin` adds extra cells beyond the strict bounds. This ensures we have complete tetrahedra even at the boundary — important because M might curve near the edge of the bounding box.

```python
    def ensure_vertex(k):
        if k not in self.vertices:
            self.vertices[k] = _vertex_pos_from_key(k, L)
```

Lazy vertex creation — only compute the R³ position when a vertex is first referenced. This avoids creating vertices that no tetrahedron uses.

```python
    for a in range(a0, a1 + 1):
        for b in range(b0, b1 + 1):
            for c in range(c0, c1 + 1):
                for perm in self._perms4:
```

Triple loop over all lattice cells, then 24 permutations per cell = 24 tets per cell.

```python
                    k0 = (a, b, c)
                    d0 = _KEY_DELTA[perm[0]]
                    k1 = (k0[0]+d0[0], k0[1]+d0[1], k0[2]+d0[2])
                    d1 = _KEY_DELTA[perm[1]]
                    k2 = (k1[0]+d1[0], k1[1]+d1[1], k1[2]+d1[2])
                    d2 = _KEY_DELTA[perm[2]]
                    k3 = (k2[0]+d2[0], k2[1]+d2[1], k2[2]+d2[2])
```

This builds one tetrahedron. Starting at corner key k₀ = (a,b,c), we walk 3 steps. At each step, we add one of the 4 key deltas, in the order specified by the permutation. The permutation uses only 3 of its 4 values (indices 0,1,2) because we need 4 vertices from 3 steps; the 4th step would return us to the starting vertex (since the 4 deltas sum to (0,0,0)).

Why do the 4 deltas sum to zero? Because (1,0,0) + (0,1,0) + (0,0,1) + (−1,−1,−1) = (0,0,0). This is the projection of e₁+e₂+e₃+e₄ = (1,1,1,1) down to key space, where it becomes zero — reflecting the fact that the 4D walk returns to the same equivalence class.

The 4th permutation element `perm[3]` is implicitly determined (it's the unused direction). We only need 3 steps to define 4 vertices.

```python
                    for k in (k0, k1, k2, k3):
                        ensure_vertex(k)
                    tet = (k0, k1, k2, k3)
                    self.tetrahedra.append(tet)
```

Register all 4 vertices and store the tet as an ordered tuple. The ordering matters for consistent orientation.

```python
                    verts = [k0, k1, k2, k3]
                    for i in range(4):
                        for j in range(i + 1, 4):
                            self.edges.add(frozenset([verts[i], verts[j]]))
                    for i in range(4):
                        face = frozenset(verts[j] for j in range(4) if j != i)
                        self.faces.add(face)
```

Each tetrahedron has C(4,2) = 6 edges and C(4,3) = 4 triangular faces. Using sets/frozensets means shared edges and faces between adjacent tets are automatically deduplicated.

### `_build_adjacency` (lines 189–203)

```python
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
```

For each tetrahedron (indexed by its position in `self.tetrahedra`):
- Record which tets share each face (`face_to_tets`). In a space-filling tiling, interior faces are shared by exactly 2 tets.
- Record which faces contain each edge (`edge_to_faces`). An interior edge is typically shared by many faces.
- Record which tets contain each edge (`edge_to_tets`).

These maps are needed in §6 to walk chains τ¹ ⊂ τ² ⊂ τ³ (edge ⊂ face ⊂ tet).

---

## Lines 214–265: `ImplicitSurface` Class

The surface M is defined implicitly as the zero set of a smooth function f: R³ → R.

### `__init__` (lines 219–223)

```python
def __init__(self, f, grad_f, reach, name="surface"):
    self.f = f          # f(x,y,z) -> scalar. M = {p : f(p) = 0}
    self.grad_f = grad_f  # ∇f(x,y,z) -> (fx, fy, fz)
    self.reach = reach    # reach(M): distance to medial axis
    self.name = name
```

The **reach** of M is the largest r such that every point within distance r of M has a unique closest point on M. Equivalently, it's the distance from M to its medial axis. For a sphere of radius R, reach = R. For a torus with tube radius r, reach = r.

The reach controls the maximum grid spacing L that the algorithm can tolerate. We need L ≪ reach(M) so that the ambient grid is fine enough relative to the surface's curvature.

### `evaluate`, `gradient`, `normal` (lines 225–234)

```python
def evaluate(self, p):
    return self.f(p[0], p[1], p[2])

def gradient(self, p):
    return np.array(self.grad_f(p[0], p[1], p[2]))

def normal(self, p):
    g = self.gradient(p)
    n = np.linalg.norm(g)
    return g / n if n > 1e-15 else np.array([0, 0, 1.0])
```

The outward unit normal to M at a point p is ∇f/|∇f|. The gradient of the implicit function is always normal to the level set. The fallback to (0,0,1) handles the degenerate case where ∇f = 0 (which shouldn't happen on M if f is a proper implicit function).

### `closest_point` (lines 236–247)

```python
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
```

This finds the closest point on M to a given point p using **Newton projection**. The update step:

$$q \leftarrow q - \frac{f(q)}{|\nabla f(q)|^2} \nabla f(q)$$

This moves q along the gradient direction until f(q) = 0. It's not finding the globally closest point on M — it's projecting along the gradient flow. For points within reach(M) of the surface, this converges to the unique closest point.

Geometrically: f(q)/|∇f|² is (approximately) the signed distance from q to M divided by the gradient magnitude, so multiplying by ∇f gives a step of approximately the right length in the right direction.

The tolerance check `abs(f(q)) < 1e-12` uses the function value rather than position change because for well-conditioned surfaces, f = 0 is the precise condition for being on M.

### `find_edge_intersection` (lines 249–265)

```python
def find_edge_intersection(self, p1, p2):
    f1 = self.evaluate(p1)
    f2 = self.evaluate(p2)
    if f1 * f2 > 0:
        return None
```

If f has the same sign at both endpoints, the edge doesn't cross M (by the intermediate value theorem, assuming f is continuous, a sign change is necessary for a zero). This is the fast rejection test — most edges in T don't cross M.

```python
    if abs(f1) < 1e-14:
        return np.array(p1, dtype=float)
    if abs(f2) < 1e-14:
        return np.array(p2, dtype=float)
```

If an endpoint is already on M (within floating-point tolerance), return it directly. This avoids numerical issues with Brent's method when the root is at an endpoint.

```python
    def g(t):
        return self.evaluate((1-t)*p1 + t*p2)
    try:
        t_root = brentq(g, 0.0, 1.0, xtol=1e-14)
        return (1 - t_root) * np.array(p1) + t_root * np.array(p2)
    except ValueError:
        return None
```

Parameterize the edge as p(t) = (1−t)p₁ + tp₂ for t ∈ [0,1]. Then g(t) = f(p(t)) changes sign from g(0) to g(1). Brent's method finds the root to tolerance 1e-14.

`brentq` combines bisection with inverse quadratic interpolation. It's guaranteed to converge (unlike Newton's method) and typically converges superlinearly. The `ValueError` catch handles edge cases where the sign change disappears due to floating-point issues.

The intersection point is the **v(τ¹)** in the paper's notation — the "vertex of the barycentric subdivision" assigned to the 1-simplex (edge) τ¹.

---

## Lines 268–308: `compute_constants_3d`

```python
def compute_constants_3d(L, reach, practical_scale=50.0):
    d, n = 3, 2
```

d = 3 (ambient R³), n = 2 (surface is 2-dimensional). The codimension is d − n = 1.

### Thickness and Lmax (lines 276–279)

```python
    t_T = np.sqrt(2) / 2   # ≈ 0.707
    Lmax = L                # longest edge = L (the 2-step edge)
```

These are intrinsic properties of the Ã₃ Coxeter triangulation.

### c̃ — perturbation bound (lines 281–284)

```python
    c_tilde_theory = t_T ** 2 / (8.0 * (d - n + 1))
    c_tilde = min(c_tilde_theory * practical_scale, 0.42)
```

Equation (6) in the paper: c̃ < t(T)² / (8(d−n+1)).

For our values: c̃ < (1/2) / (8·2) = 1/32 = 0.03125.

This bounds how far we're allowed to move vertices during perturbation, as a fraction of Lmax. The paper's bound is very conservative — it comes from ensuring that perturbed simplices remain "thick enough" for the transversality argument in §6. In practice, we multiply by `practical_scale` = 50 and cap at 0.42.

### ρ₁ — volume fraction (lines 286–290)

```python
    N_leq = 3
    rho1_theory = math.factorial(4) / (
        2 ** 6 * math.factorial(2) * math.factorial(1) * N_leq)
    rho1 = min(rho1_theory * practical_scale, 0.90)
```

From Lemma 5.1. This controls the tangent-plane clearance: how far from the tangent plane a vertex must be after perturbation. The formula involves:
- 4! = 24 (from the number of simplices in the Coxeter arrangement)
- 2⁶ = 64 (power of 2 scaling)
- 2! · 1! (factorials depending on k = (d+1)/2)
- N_≤ = 3 (number of simplices in the star of a (d−n−1)-face)

The theoretical value is 24/(64·2·1·3) = 24/384 = 0.0625. We scale to 0.90.

### Derived constants (lines 292–297)

```python
    alpha0 = (4.0 / 3.0) * rho1 * c_tilde
```

α₀ is the "skeleton safety margin" — the minimum distance from the (d−n−1)-skeleton of the perturbed T to M. For codimension 1, the (d−n−1)-skeleton is the 0-skeleton (vertices). So α₀ controls how far perturbed vertices stay from M.

```python
    binom_d_dn = math.comb(d, d - n)  # C(3,1) = 3
    zeta_raw = (8 * t_T * (1 - 8 * min(c_tilde, t_T**2 / 16) / t_T**2)) / (
        15 * np.sqrt(d) * binom_d_dn * (1 + 2 * c_tilde))
    zeta = max(zeta_raw, 0.01)
```

ζ is the quality bound on the output triangles (Equation 10). The formula assumes c̃ is small relative to t²; when c̃ is large (as in our practical scaling), the formula can go negative, so we clamp to 0.01. This is fine because ζ is only used in the paper's *proof* that K is homeomorphic to M — the algorithm itself doesn't use ζ.

---

## Lines 311–367: `perturb_vertices` — Part 1 (§5)

This is the first half of the algorithm. The paper calls it "making the 0-skeleton transverse to M."

### Setup (lines 315–330)

```python
def perturb_vertices(T, surface, consts):
    Lmax = consts['Lmax']
    c_tilde = consts['c_tilde']
    rho1 = consts['rho1']

    max_perturb = c_tilde * Lmax
    tangent_clearance = rho1 * c_tilde * Lmax
```

Two key distances:
- `max_perturb` = c̃ · Lmax ≈ 0.252: the maximum allowed displacement of any vertex. Moving a vertex more than this could break the thickness bound on the perturbed simplices.
- `tangent_clearance` = ρ₁ · c̃ · Lmax ≈ 0.227: the minimum signed distance from the tangent plane that a vertex must achieve. If a vertex is closer than this to M's tangent plane, it gets pushed.

### Per-vertex loop (lines 332–367)

```python
    for key, v in T.vertices.items():
        cp = surface.closest_point(v)
        if cp is None:
            perturbed[key] = v.copy()
            info['case1'] += 1
            continue
```

For each vertex, find the closest point on M. If projection fails (vertex is far from M or near a degenerate point), keep it unchanged. This is always Case 1.

```python
        dist = np.linalg.norm(v - cp)
        if dist >= 1.5 * Lmax:
            perturbed[key] = v.copy()
            info['case1'] += 1
            continue
```

**Case 1 (§5.2):** If the vertex is farther than 3Lmax/2 from M, it's already far enough — no perturbation needed. The factor 3/2 comes from the paper's analysis: vertices this far from M cannot be part of any simplex that intersects M, so their position doesn't matter.

```python
        # Case 2: push away from tangent plane at cp
        info['case2'] += 1
        n = surface.normal(cp)
        signed_d = np.dot(v - cp, n)
```

**Case 2 (§5.2):** The vertex is within 3Lmax/2 of M and needs checking. We compute the signed distance from v to the tangent plane T_{cp}M at the closest point. The tangent plane has normal `n` and passes through `cp`, so the signed distance is (v − cp) · n.

A positive signed distance means v is on the "outside" of M (same side as the outward normal); negative means "inside."

```python
        if abs(signed_d) >= tangent_clearance:
            perturbed[key] = v.copy()
```

If the vertex is already far enough from the tangent plane (in either direction), no perturbation needed. Even though it's close to M in 3D distance, it has enough clearance in the normal direction. This matters because the transversality argument in §6 only needs vertices to be off the tangent plane — their distance to M in other directions is irrelevant.

```python
        else:
            if signed_d >= 0:
                target_d = tangent_clearance
            else:
                target_d = -tangent_clearance
            shift = (target_d - signed_d) * n
```

If the vertex is too close to the tangent plane, push it away. The direction of push preserves the sign: if v was slightly above the tangent plane (signed_d ≥ 0), push it further above to reach `tangent_clearance`; if below, push further below to reach `−tangent_clearance`.

The shift vector is purely in the normal direction: `(target − current) × n̂`.

```python
            shift_norm = np.linalg.norm(shift)
            if shift_norm > max_perturb:
                shift = shift * (max_perturb / shift_norm)
            new_v = v + shift
            perturbed[key] = new_v
            info['max_pert'] = max(info['max_pert'], np.linalg.norm(shift))
```

Clamp the shift to `max_perturb` — we can't move any vertex more than c̃ · Lmax without violating the simplex quality bounds. In practice, this clamp rarely activates because `tangent_clearance < max_perturb` (since ρ₁ < 1).

The perturbed vertex positions are stored in a new dictionary `perturbed` (same keys, possibly different positions). The original positions in `T.vertices` are unchanged.

---

## Lines 370–465: `construct_K` — Part 2 (§6)

This is the core of the algorithm: constructing the output triangulation K of M from the perturbed ambient triangulation.

### Theoretical framework

The key idea (§6 of the paper): after perturbation, the (d−n−1)-skeleton of T̃ (the perturbed triangulation) is transverse to M. This means:
- No vertex of T̃ lies on M (guaranteed by Part 1)
- Each edge of T̃ crosses M at most once (guaranteed by L ≪ reach(M))
- The intersections have a clean combinatorial structure

We build K using the **barycentric subdivision** of T̃. For each chain of simplices τ¹ ⊂ τ² ⊂ τ³ (edge ⊂ face ⊂ tet) where each simplex "intersects" M, we create a triangle in K with one vertex from each simplex.

For our case (n=2, d=3, codimension 1):
- v(τ¹) = point where edge τ¹ crosses M (found by root-finding)
- v(τ²) = projection of the centroid of τ²'s intersection points onto M
- v(τ³) = projection of the centroid of τ³'s intersection points onto M
- Each valid chain τ¹ ⊂ τ² ⊂ τ³ gives one triangle {v(τ¹), v(τ²), v(τ³)} in K

### Step 1: Edge intersections v(τ¹) (lines 380–387)

```python
    edge_pts = {}
    for edge in T.edges:
        v1k, v2k = list(edge)
        p1, p2 = pverts[v1k], pverts[v2k]
        pt = surface.find_edge_intersection(p1, p2)
        if pt is not None:
            edge_pts[edge] = pt
```

For every edge of T̃ (using perturbed vertex positions `pverts`), check if it crosses M. If f(p1) and f(p2) have opposite signs, the edge crosses M; find the exact crossing point via Brent's method.

These are the **vertices of type v(τ¹)** — they lie exactly on M. Each such point is assigned to the edge that created it.

For a sphere with L=0.6, about 278 out of ~53,000 edges cross M.

### Step 2: Face centres v(τ²) (lines 389–403)

```python
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
```

For each triangular face of T̃, find how many of its 3 edges cross M. A face "intersects M" (in the paper's sense) if ≥ 2 of its edges cross M. In codimension 1, when M is a surface and the face is a triangle, M typically enters through one edge and exits through another, so exactly 2 edges cross.

The **v(τ²) point** is computed as:
1. Average the edge crossing points (centroid of the 2 or 3 intersection points)
2. Project this centroid onto M via `closest_point`

This gives a point on M that represents the "center" of how M passes through this face. In the paper, this is the barycentre of M ∩ τ², projected to M.

### Step 3: Tet centres v(τ³) (lines 405–418)

```python
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
```

Same logic, one dimension up. For each tetrahedron, check how many of its 4 faces intersect M. Average the face points and project to M.

A tet "intersects M" when M passes through it. Typically, M enters through one face and exits through another, slicing the tet. The v(τ³) point represents the center of this slice.

### Step 4: Build triangles from chains (lines 420–457)

```python
    K_verts = []
    K_tris = []
    vert_index = {}
    tri_set = set()
```

`K_verts` is the flat list of vertex positions; `K_tris` is the list of index triples. `vert_index` deduplicates vertices by their (rounded) position. `tri_set` deduplicates triangles — critical for the Coxeter triangulation because the denser adjacency structure means the same chain can be discovered from multiple tets.

```python
    def add_vert(pt):
        key = tuple(np.round(pt, 10))
        if key not in vert_index:
            vert_index[key] = len(K_verts)
            K_verts.append(pt)
        return vert_index[key]
```

Vertex deduplication: round to 10 decimal places (enough to distinguish genuinely different points while merging numerically identical ones) and use as a dictionary key.

```python
    for tet_idx, tet in enumerate(T.tetrahedra):
        if tet_idx not in tet_pts:
            continue
        vt = tet_pts[tet_idx]
        i_t = add_vert(vt)
```

For each tet that intersects M, get its v(τ³) point and register it as a K vertex.

```python
        verts = list(tet)
        for a in range(4):
            face = frozenset(verts[b] for b in range(4) if b != a)
            if face not in face_pts:
                continue
            vf = face_pts[face]
            i_f = add_vert(vf)
```

For each face of this tet that also intersects M, get its v(τ²) point. Now we have a pair (face, tet) with τ² ⊂ τ³.

```python
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
```

For each edge of this face that crosses M, we now have a complete chain:

**τ¹ ⊂ τ² ⊂ τ³ (edge ⊂ face ⊂ tet)**

This chain produces one triangle in K with vertices:
- **i_e** = v(τ¹): the edge-M intersection point (lies exactly on M)
- **i_f** = v(τ²): the face centre (lies on M)
- **i_t** = v(τ³): the tet centre (lies on M)

All three vertices lie on M, so the triangle is an approximation of a small patch of M.

The deduplication via `tri_set` is essential: in the Coxeter Ã₃ triangulation, faces are shared by multiple tets, so the same (edge, face) pair can be encountered from different tets, producing the same triangle. Without deduplication, we'd get ~4× too many triangles.

---

## Lines 468–489: `quality_metrics`

```python
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
```

For each triangle in K:
- **Area** = ||(b−a) × (c−a)|| / 2 (standard cross-product formula)
- **Hausdorff estimate**: the distance from the triangle's centroid to M. Since all three vertices lie on M, the centroid is at most ~(Lmax)² away from M (second-order error from curvature). Taking the max over all triangles gives an approximate one-sided Hausdorff distance.

The total area should approximately equal the surface area of M (12.566 for a unit sphere, 15.791 for the torus). Getting within 0.3% validates that K is a good approximation.

---

## Lines 492–518: Surface Definitions

### Sphere (lines 496–500)

```python
def sphere_surface(r=1.0):
    return ImplicitSurface(
        f=lambda x, y, z: x**2 + y**2 + z**2 - r**2,
        grad_f=lambda x, y, z: (2*x, 2*y, 2*z),
        reach=r, name=f"Sphere(r={r})")
```

f(x,y,z) = x² + y² + z² − r². Zero set is the sphere of radius r. Gradient is (2x, 2y, 2z), pointing radially outward. Reach = r (the center is the medial axis, at distance r from M).

### Torus (lines 503–510)

```python
def torus_surface(R=1.0, r=0.4):
    return ImplicitSurface(
        f=lambda x, y, z: (np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2,
        ...
        reach=r, name=f"Torus(R={R},r={r})")
```

f = (√(x²+y²) − R)² + z² − r². This is the standard torus: tube radius r, center-to-tube-center distance R. The zero set is the set of points at distance r from the circle of radius R in the xy-plane.

The gradient has the `+1e-30` in the denominator to avoid division by zero on the z-axis (where √(x²+y²) = 0). Reach = r because the inner tube's curvature is 1/r.

### Ellipsoid (lines 513–518)

```python
def ellipsoid_surface(a=1.2, b=0.8, c=0.6):
    rch = min(a, b, c)**2 / max(a, b, c)
```

f = x²/a² + y²/b² + z²/c² − 1. The reach of an ellipsoid is the minimum radius of curvature, which is min(a,b,c)²/max(a,b,c). This occurs at the endpoints of the longest axis, where the surface curves most sharply.

---

## Lines 521–593: `run` — Main Pipeline

```python
def run(surface, L=None, bounds=None, practical_scale=50.0, verbose=True):
    reach = surface.reach
    if L is None:
        L = reach / 5.0
```

Default L = reach/5. This gives L/reach = 0.2, which is comfortably within the regime where the paper's algorithm works. Smaller L gives more triangles but better approximation.

```python
    consts = compute_constants_3d(L, reach, practical_scale)
```

Compute all the §4–5 constants based on L, reach, and the practical scaling factor.

```python
    T = CoxeterA3Triangulation3D(L, bounds)
```

Build the ambient Ã₃ triangulation covering the bounding box. This is the most memory-intensive step — for L=0.6 and bounds ±1.6, this creates ~8,000 vertices and ~150,000 tets.

```python
    pverts, pinfo = perturb_vertices(T, surface, consts)
```

Part 1: perturb vertices near M so no vertex lies on M's tangent plane.

```python
    K = construct_K(T, pverts, surface, consts)
```

Part 2: build the output triangulation K.

```python
    qm = quality_metrics(K, surface)
```

Measure how good K is: total area, triangle size range, Hausdorff distance.

```python
    return T, pverts, K, consts
```

Return everything so the caller can do further analysis, export OBJs, visualize, etc.

---

## Lines 597–625: `__main__` Test

```python
if __name__ == '__main__':
    T = CoxeterA3Triangulation3D(L=1.0, bounds=(-1, 1, -1, 1, -1, 1))
    ...
    sample = random.sample(T.tetrahedra, min(200, len(T.tetrahedra)))
    bad = 0
    for tet in sample:
        pts = [T.vertices[v] for v in tet]
        eds = sorted(round(np.linalg.norm(pts[j]-pts[i]), 4)
                     for i, j in combinations(range(4), 2))
        expected = sorted([0.8660, 0.8660, 0.8660, 0.8660, 1.0, 1.0])
```

Verification: sample 200 random tets and check that every one has exactly the expected edge pattern — 4 edges of √3/2 ≈ 0.8660 and 2 edges of 1.0. This confirms all alcoves are congruent.

```python
    sph = sphere_surface(r=1.0)
    T2, pv2, K2, c2 = run(sph, L=0.35,
                           bounds=(-1.6, 1.6, -1.6, 1.6, -1.6, 1.6))
```

End-to-end test on a unit sphere.

---

## Summary: Data Flow

```
                    ┌─────────────────┐
                    │ ImplicitSurface │
                    │  f, ∇f, reach   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            v                v                v
    ┌───────────────┐  ┌──────────┐  ┌─────────────┐
    │ Coxeter Ã₃ T  │  │ Constants│  │  Bounds, L  │
    │ vertices,     │  │ c̃, ρ₁,   │  │             │
    │ edges, faces, │  │ α₀, etc  │  │             │
    │ tets, adj.    │  │          │  │             │
    └───────┬───────┘  └────┬─────┘  └─────────────┘
            │               │
            v               v
    ┌─────────────────────────────┐
    │  perturb_vertices (§5)      │
    │  For each vertex near M:    │
    │  push away from tangent     │
    │  plane at closest point     │
    │  → pverts dict              │
    └──────────────┬──────────────┘
                   │
                   v
    ┌──────────────────────────────────────┐
    │  construct_K (§6)                    │
    │                                      │
    │  Step 1: edges crossing M → v(τ¹)   │
    │  Step 2: face centroids   → v(τ²)   │
    │  Step 3: tet centroids    → v(τ³)   │
    │  Step 4: chains τ¹⊂τ²⊂τ³ → K_tris  │
    │          (deduplicated)              │
    └──────────────┬───────────────────────┘
                   │
                   v
    ┌──────────────────────────────┐
    │  K = { K_verts, K_tris }    │
    │  Triangulation of M         │
    │  homeomorphic to M          │
    │  all vertices on M          │
    └──────────────────────────────┘
```
