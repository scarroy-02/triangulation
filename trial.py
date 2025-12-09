"""
Whitney's Triangulation Algorithm - Full Implementation
========================================================

Based on: "Triangulating submanifolds: An elementary and quantified version 
of Whitney's method" by Boissonnat, Kachanovich, Wintraecken (2021)

This implementation follows the paper's equations exactly:
- Section 4: Coxeter triangulation Ã_d
- Section 5.1: Algorithm constants (equations 4-14)
- Section 5.2: Perturbation algorithm
- Section 6.2: Triangulation construction K
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from math import factorial, comb
import struct

# Import reach computation
from reach import ImplicitSurface, compute_reach, ManifoldPoint


# =============================================================================
# Section 4: Coxeter Triangulation Ã_d
# =============================================================================

class CoxeterTriangulation:
    """
    Coxeter triangulation of type Ã_d (Freudenthal/Kuhn triangulation).
    
    This is the standard triangulation of R^d into d-simplices obtained by:
    1. Divide R^d into unit hypercubes
    2. Subdivide each hypercube into d! simplices via Freudenthal subdivision
    
    Properties (from Section 4):
    - Vertex set: Z^d (integer lattice, scaled by L)
    - Edge length: L
    - Thickness t(T): measures "fatness" of simplices
        For d odd:  t = sqrt(2/d)
        For d even: t = sqrt(2(d+1) / (d(d+2)))
    - μ₀: minimum distance from vertex to opposite facet / L
    - δ: Delaunay protection constant
    """
    
    def __init__(self, d: int, L: float):
        """
        Args:
            d: Ambient dimension
            L: Edge length
        """
        self.d = d
        self.L = L
        
        # Thickness t(T) from Section 4
        # For Ã_d (Freudenthal), the thickness depends on parity of d
        if d % 2 == 1:  # odd
            self.thickness = np.sqrt(2.0 / d)
        else:  # even
            self.thickness = np.sqrt(2.0 * (d + 1) / (d * (d + 2)))
        
        # μ₀: ratio of altitude to edge length for regular simplex
        # For Freudenthal triangulation in R^d:
        # The minimum altitude is L * sqrt((d+1)/(2d))
        # So μ₀ = sqrt((d+1)/(2d)) ≈ sqrt(1/2) for large d
        self.mu_0 = np.sqrt((d + 1) / (2.0 * d))
        
        # δ: Delaunay protection - minimum gap in Delaunay condition
        # For Ã_d, from the paper's analysis
        # δ = (sqrt(d² + 2d + 24) - sqrt(d² + 2d)) / sqrt(12(d+1))
        self.delta = (np.sqrt(d**2 + 2*d + 24) - np.sqrt(d**2 + 2*d)) / np.sqrt(12 * (d + 1))
    
    def get_simplex_containing(self, p: np.ndarray) -> np.ndarray:
        """
        Get the d-simplex of T containing point p.
        
        For Freudenthal triangulation:
        1. Find the unit cube containing p
        2. Determine which of the d! simplices contains p
        
        Returns:
            (d+1) x d array of vertex coordinates
        """
        # Scale to unit grid
        q = p / self.L
        
        # Find base cube vertex (floor of coordinates)
        base = np.floor(q).astype(int)
        
        # Fractional part determines which simplex
        frac = q - base
        
        # Sort coordinates to determine permutation
        # The simplex is determined by the ordering of fractional parts
        perm = np.argsort(-frac)  # Descending order
        
        # Build simplex vertices
        # v_0 = base
        # v_i = v_{i-1} + e_{perm[i-1]} for i = 1, ..., d
        vertices = np.zeros((self.d + 1, self.d))
        vertices[0] = base
        for i in range(1, self.d + 1):
            vertices[i] = vertices[i-1].copy()
            vertices[i, perm[i-1]] += 1
        
        return vertices * self.L
    
    def get_simplices_in_box(self, min_corner: np.ndarray, max_corner: np.ndarray) -> List[np.ndarray]:
        """
        Get all d-simplices intersecting a bounding box.
        
        Returns:
            List of simplices, each as (d+1) x d array
        """
        # Determine range of cubes to consider
        min_idx = np.floor(min_corner / self.L).astype(int) - 1
        max_idx = np.ceil(max_corner / self.L).astype(int) + 1
        
        simplices = []
        
        # Iterate over all cubes in range
        ranges = [range(min_idx[i], max_idx[i]) for i in range(self.d)]
        
        from itertools import product, permutations
        
        for base in product(*ranges):
            base = np.array(base)
            
            # Each cube has d! simplices, one for each permutation
            for perm in permutations(range(self.d)):
                vertices = np.zeros((self.d + 1, self.d))
                vertices[0] = base
                for i in range(1, self.d + 1):
                    vertices[i] = vertices[i-1].copy()
                    vertices[i, perm[i-1]] += 1
                
                simplices.append(vertices * self.L)
        
        return simplices


# =============================================================================
# Section 5.1: Algorithm Constants
# =============================================================================

def stirling2(n: int, k: int) -> int:
    """
    Stirling number of the second kind S(n, k).
    
    Counts the number of ways to partition n elements into k non-empty subsets.
    """
    if k == 0:
        return 1 if n == 0 else 0
    if k == n or k == 1:
        return 1
    if k > n:
        return 0
    
    # Use recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    # Build table to avoid recursion
    S = [[0] * (k + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            S[i][j] = j * S[i-1][j] + S[i-1][j-1]
    
    return S[n][k]


def compute_N_leq_k(d: int, k: int) -> int:
    """
    Compute N_{≤k} from equation (4).
    
    N_{≤k} bounds the number of faces of dimension ≤ k that contain a given vertex.
    
    N_{≤k} = 2 + Σ_{j=1}^{k+1} j! · S(d+1, j)
    
    where S(n, k) is Stirling number of second kind.
    """
    if k < 0:
        return 2
    
    result = 2
    for j in range(1, k + 2):
        result += factorial(j) * stirling2(d + 1, j)
    
    return result


def compute_rho_1(d: int, N: int) -> float:
    """
    Compute ρ₁ from equation (5).
    
    ρ₁ is a lower bound on the volume fraction of a ball that lies 
    outside the convex hull of points on its boundary.
    
    For d even (d = 2k):
        ρ₁ = (2^{2k-2} · (k!)²) / (π · (2k)! · N)
    
    For d odd (d = 2k-1):
        ρ₁ = (2k-2)! / (2^{2k} · k! · (k-1)! · N)
    """
    if d % 2 == 0:  # even
        k = d // 2
        numerator = (2 ** (2*k - 2)) * (factorial(k) ** 2)
        denominator = np.pi * factorial(2*k) * N
    else:  # odd
        k = (d + 1) // 2
        numerator = factorial(2*k - 2)
        denominator = (2 ** (2*k)) * factorial(k) * factorial(k - 1) * N
    
    return numerator / denominator


def compute_algorithm_constants(d: int, n: int, reach: float, coxeter: CoxeterTriangulation) -> dict:
    """
    Compute all algorithm constants from Section 5.1.
    
    The key challenge is that c̃ depends on L (equation 6), and the bound 
    on L depends on α_k which depends on c̃ (equations 8, 13).
    
    We solve this by finding L that satisfies all constraints.
    
    Args:
        d: Ambient dimension
        n: Manifold dimension
        reach: Reach of the manifold
        coxeter: Coxeter triangulation (for thickness, μ₀, δ)
    
    Returns:
        Dictionary with all constants including L and c̃
    """
    t = coxeter.thickness
    mu_0 = coxeter.mu_0
    delta = coxeter.delta
    
    print("\n" + "=" * 60)
    print("COMPUTING ALGORITHM CONSTANTS (Section 5.1)")
    print("=" * 60)
    
    print(f"\nInput parameters:")
    print(f"  d (ambient dimension) = {d}")
    print(f"  n (manifold dimension) = {n}")
    print(f"  reach(M) = {reach}")
    print(f"  t(T) (thickness) = {t:.6f}")
    print(f"  μ₀ = {mu_0:.6f}")
    print(f"  δ = {delta:.6f}")
    
    # Equation (4): N_{≤k}
    k = d - n - 1  # We need N_{≤(d-n-1)}
    N = compute_N_leq_k(d, k)
    print(f"\nEquation (4): N_{{≤{k}}} = {N}")
    
    # Equation (5): ρ₁
    rho_1 = compute_rho_1(d, N)
    print(f"Equation (5): ρ₁ = {rho_1:.10f}")
    
    # Now we need to solve for L and c̃ together
    # 
    # From equation (6): c̃ = min(t·μ₀·δ / (18·d·L), t²/24)
    # From equation (7): c̃ ≤ 1/24
    # From equation (8): α₁ = (4/3)·ρ₁·c̃, α_k = (2/3)·α_{k-1}·c̃·ρ₁
    # From equation (13): L/reach ≤ α_{d-n}^{d-n} / 54
    #
    # Approach: Iterate to find consistent L
    
    codim = d - n  # codimension
    
    # Start with a reasonable guess for L
    L = reach / 10
    
    print(f"\nSolving coupled system (equations 6, 8, 13)...")
    
    for iteration in range(100):
        # Equation (6): c̃ = min(t·μ₀·δ / (18·d·L), t²/24)
        term1 = t * mu_0 * delta / (18 * d * L)
        term2 = t**2 / 24
        c_tilde = min(term1, term2, 1.0/24)  # Also enforce equation (7)
        
        # Equation (8): α_k
        # α₁ = (4/3)·ρ₁·c̃
        # α_k = α₁ · ((2/3)·c̃·ρ₁)^{k-1}
        alpha_1 = (4.0/3) * rho_1 * c_tilde
        
        def alpha(kk):
            if kk <= 0:
                return 1.0
            if kk == 1:
                return alpha_1
            return alpha_1 * ((2.0/3) * c_tilde * rho_1) ** (kk - 1)
        
        # Equation (13): L/reach ≤ α_{d-n}^{d-n} / 54
        alpha_codim = alpha(codim)
        L_bound_ratio = (alpha_codim ** codim) / 54
        L_bound = L_bound_ratio * reach
        
        # Check if current L satisfies the bound
        if L <= L_bound:
            break
        
        # Update L to satisfy bound
        L_new = L_bound * 0.99  # Slightly inside bound
        
        if abs(L_new - L) / L < 1e-10:
            break
        
        L = L_new
        
        if L < reach * 1e-15:
            print("  Warning: L becoming extremely small, using minimum")
            L = reach * 1e-15
            break
    
    # Final computation of c̃ with the determined L
    term1 = t * mu_0 * delta / (18 * d * L)
    term2 = t**2 / 24
    c_tilde = min(term1, term2, 1.0/24)
    
    # Recompute α_k with final c̃
    alpha_1 = (4.0/3) * rho_1 * c_tilde
    
    def alpha(kk):
        if kk <= 0:
            return 1.0
        if kk == 1:
            return alpha_1
        return alpha_1 * ((2.0/3) * c_tilde * rho_1) ** (kk - 1)
    
    alpha_codim = alpha(codim)
    L_bound_ratio = (alpha_codim ** codim) / 54
    
    # Equation (10): ζ (quality bound)
    binom_d_codim = comb(d, codim)
    zeta_num = 8 * t * (1 - 8 * c_tilde / t**2)
    zeta_denom = 15 * np.sqrt(d) * binom_d_codim * (1 + 2 * c_tilde)
    zeta = zeta_num / zeta_denom if zeta_denom > 0 else 0
    
    print(f"\nFinal constants:")
    print(f"  Equation (6): c̃ = min({term1:.6e}, {term2:.6e}) = {c_tilde:.10f}")
    print(f"  Equation (7): c̃ ≤ 1/24 = {1/24:.10f} {'✓' if c_tilde <= 1/24 else '✗'}")
    print(f"  Equation (8): α₁ = {alpha_1:.10e}")
    print(f"               α_{codim} = {alpha_codim:.10e}")
    print(f"  Equation (10): ζ = {zeta:.10f}")
    print(f"  Equation (13): L/reach ≤ {L_bound_ratio:.10e}")
    print(f"                 L ≤ {L_bound_ratio * reach:.10e}")
    print(f"                 Using L = {L:.10e}")
    
    return {
        'd': d,
        'n': n,
        'reach': reach,
        'N': N,
        'rho_1': rho_1,
        'c_tilde': c_tilde,
        'alpha': alpha,
        'zeta': zeta,
        'L': L,
        'L_bound_ratio': L_bound_ratio,
        't': t,
        'mu_0': mu_0,
        'delta': delta,
    }


# =============================================================================
# Section 5.2: Perturbation Algorithm
# =============================================================================

class PerturbationAlgorithm:
    """
    Perturbation algorithm from Section 5.2.
    
    Perturbs vertices of T to get T̃ such that simplices of dimension ≤ d-n-1
    are "protected" (far from M).
    
    Two cases for each vertex v_i:
    
    Case 1: d(v_i, M) ≥ 3L/2
        Keep v_i unchanged: ṽ_i = v_i
    
    Case 2: d(v_i, M) < 3L/2
        Let p be closest point on M to v_i
        Perturb v_i away from tangent plane T_p M
        Requirements:
        - |v_i - ṽ_i| ≤ c̃·L  (equation 17)
        - d(ṽ_i, T_p M) ≥ ρ₁·c̃·L  (equation 19)
    """
    
    def __init__(self, constants: dict, surface: ImplicitSurface):
        self.constants = constants
        self.surface = surface
        
        self.d = constants['d']
        self.n = constants['n']
        self.L = constants['L']
        self.c_tilde = constants['c_tilde']
        self.rho_1 = constants['rho_1']
        
        # Maximum perturbation: c̃·L (equation 17)
        self.max_perturbation = self.c_tilde * self.L
        
        # Required distance from tangent plane: ρ₁·c̃·L (equation 19)
        self.required_distance = self.rho_1 * self.c_tilde * self.L
        
        # Case threshold: 3L/2
        self.case_threshold = 1.5 * self.L
    
    def perturb_vertex(self, v: np.ndarray) -> np.ndarray:
        """
        Perturb a single vertex according to Section 5.2.
        
        Args:
            v: Original vertex position
            
        Returns:
            Perturbed vertex position
        """
        # Find closest point on M
        p = self.surface.project_to_surface(v)
        
        if p is None:
            return v  # Can't find projection, keep original
        
        dist_to_M = np.linalg.norm(v - p)
        
        # Case 1: Far from M, no perturbation needed
        if dist_to_M >= self.case_threshold:
            return v
        
        # Case 2: Close to M, need to perturb
        # Get normal at closest point (defines tangent plane)
        normal = self.surface.normal(p)
        
        # Current signed distance from tangent plane T_p M
        # T_p M passes through p with normal = normal
        # Distance of v from T_p M = (v - p) · normal
        current_dist = np.dot(v - p, normal)
        
        # We need |distance from T_p M| ≥ ρ₁·c̃·L
        if abs(current_dist) >= self.required_distance:
            return v  # Already far enough from tangent plane
        
        # Perturb in normal direction to achieve required distance
        # Choose direction that moves away from M
        if current_dist >= 0:
            target_dist = self.required_distance
        else:
            target_dist = -self.required_distance
        
        # Perturbation vector
        perturbation = (target_dist - current_dist) * normal
        
        # Clamp perturbation magnitude (equation 17)
        pert_magnitude = np.linalg.norm(perturbation)
        if pert_magnitude > self.max_perturbation:
            perturbation = perturbation * (self.max_perturbation / pert_magnitude)
        
        return v + perturbation
    
    def perturb_all_vertices(self, vertices: Dict[tuple, np.ndarray]) -> Dict[tuple, np.ndarray]:
        """
        Perturb all vertices.
        
        Args:
            vertices: Dictionary mapping vertex key to position
            
        Returns:
            Dictionary mapping vertex key to perturbed position
        """
        perturbed = {}
        
        case1_count = 0
        case2_count = 0
        
        for key, v in vertices.items():
            v_new = self.perturb_vertex(v)
            perturbed[key] = v_new
            
            # Track statistics
            p = self.surface.project_to_surface(v)
            if p is not None:
                if np.linalg.norm(v - p) >= self.case_threshold:
                    case1_count += 1
                else:
                    case2_count += 1
        
        print(f"  Case 1 (far from M): {case1_count} vertices")
        print(f"  Case 2 (perturbed): {case2_count} vertices")
        
        return perturbed


# =============================================================================
# Section 6.2: Building the Triangulation K
# =============================================================================

class WhitneyTriangulationK:
    """
    Build triangulation K from Section 6.2.
    
    For each simplex τ of T̃ that intersects M, we define a vertex v(τ) ∈ M.
    
    From equation (26):
    - For dim(τ) = d-n: v(τ) is the unique intersection of τ with M
    - For dim(τ) > d-n: v(τ) = average of v(τ_i) over (d-n)-faces τ_i
    
    The simplices of K are defined by equation (25):
    For each chain τ_{d-n} ⊂ τ_{d-n+1} ⊂ ... ⊂ τ_d where τ_d ∩ M ≠ ∅,
    the simplex [v(τ_{d-n}), v(τ_{d-n+1}), ..., v(τ_d)] is in K.
    """
    
    def __init__(self, surface: ImplicitSurface, constants: dict, 
                 perturbed_simplices: List[np.ndarray]):
        """
        Args:
            surface: The implicit surface M
            constants: Algorithm constants
            perturbed_simplices: List of perturbed d-simplices
        """
        self.surface = surface
        self.constants = constants
        self.d = constants['d']
        self.n = constants['n']
        
        # Build the triangulation
        self.vertices = []  # List of vertices in K
        self.simplices = []  # List of n-simplices as tuples of vertex indices
        
        self._build(perturbed_simplices)
    
    def _simplex_intersects_M(self, simplex: np.ndarray) -> bool:
        """Check if a simplex intersects M by checking sign changes of f."""
        values = [self.surface.f(v) for v in simplex]
        return min(values) <= 0 <= max(values)
    
    def _find_intersection(self, simplex: np.ndarray) -> Optional[np.ndarray]:
        """
        Find intersection of simplex with M.
        
        For codimension 1 (surface in R³), a (d-n)-simplex is an edge.
        We find where f changes sign along the edge.
        """
        dim = len(simplex) - 1
        
        if dim == 1:  # Edge (1-simplex)
            v0, v1 = simplex[0], simplex[1]
            f0, f1 = self.surface.f(v0), self.surface.f(v1)
            
            if f0 * f1 > 0:
                return None  # No sign change
            
            if abs(f0 - f1) < 1e-14:
                return (v0 + v1) / 2
            
            # Linear interpolation to find zero crossing
            t = f0 / (f0 - f1)
            p = v0 + t * (v1 - v0)
            
            # Refine with projection
            p_refined = self.surface.project_to_surface(p)
            return p_refined if p_refined is not None else p
        
        else:
            # For higher dimensional simplices, use centroid and project
            centroid = np.mean(simplex, axis=0)
            return self.surface.project_to_surface(centroid)
    
    def _get_faces(self, simplex: np.ndarray, dim: int) -> List[np.ndarray]:
        """Get all faces of given dimension from a simplex."""
        from itertools import combinations
        
        n_vertices = len(simplex)
        faces = []
        
        for indices in combinations(range(n_vertices), dim + 1):
            face = simplex[list(indices)]
            faces.append(face)
        
        return faces
    
    def _simplex_key(self, simplex: np.ndarray) -> tuple:
        """Create a hashable key for a simplex based on sorted vertex coordinates."""
        # Round and sort vertices for consistent key
        rounded = [tuple(np.round(v, 10)) for v in simplex]
        return tuple(sorted(rounded))
    
    def _build(self, ambient_simplices: List[np.ndarray]):
        """Build the triangulation K."""
        print(f"Building K from {len(ambient_simplices)} ambient {self.d}-simplices...")
        
        codim = self.d - self.n  # d - n = 1 for surfaces in R³
        
        # Cache for v(τ) values
        v_cache = {}  # simplex_key -> vertex on M
        
        # Find all simplices that intersect M
        intersecting = []
        for sigma in ambient_simplices:
            if self._simplex_intersects_M(sigma):
                intersecting.append(sigma)
        
        print(f"  {len(intersecting)} simplices intersect M")
        
        if len(intersecting) == 0:
            print("  Warning: No simplices intersect M!")
            return
        
        # Compute v(τ) for all faces of intersecting simplices
        # Start from (d-n)-faces and work up
        
        # First, compute v(τ) for (d-n)-simplices (base case - edges for surfaces)
        # These are the only faces that directly intersect M
        for sigma in intersecting:
            base_faces = self._get_faces(sigma, codim)
            for face in base_faces:
                key = self._simplex_key(face)
                if key not in v_cache:
                    intersection = self._find_intersection(face)
                    if intersection is not None:
                        v_cache[key] = intersection
        
        print(f"  Found {len(v_cache)} edge intersections with M")
        
        # Now compute v(τ) for higher dimensional faces (recursive averaging)
        # For each face τ of dim > d-n, v(τ) = average of v(e) over all (d-n)-faces e ⊂ τ
        for dim in range(codim + 1, self.d + 1):
            count = 0
            for sigma in intersecting:
                faces = self._get_faces(sigma, dim)
                for face in faces:
                    key = self._simplex_key(face)
                    if key not in v_cache:
                        # Average over all (d-n)-subfaces that have v(τ) defined
                        subfaces = self._get_faces(face, codim)
                        v_values = []
                        for sf in subfaces:
                            sf_key = self._simplex_key(sf)
                            if sf_key in v_cache:
                                v_values.append(v_cache[sf_key])
                        
                        if len(v_values) > 0:
                            v_cache[key] = np.mean(v_values, axis=0)
                            count += 1
            
            print(f"  Computed v(τ) for {count} {dim}-faces")
        
        # Build K by enumerating chains
        # A chain is τ_{d-n} ⊂ τ_{d-n+1} ⊂ ... ⊂ τ_d
        # Each chain gives an n-simplex in K: [v(τ_{d-n}), v(τ_{d-n+1}), ..., v(τ_d)]
        
        vertex_map = {}  # vertex position tuple -> index
        
        for sigma in intersecting:
            # Get all (d-n)-faces that intersect M
            base_faces = self._get_faces(sigma, codim)
            
            for base_face in base_faces:
                base_key = self._simplex_key(base_face)
                if base_key not in v_cache:
                    continue  # This edge doesn't intersect M
                
                # Build all chains starting from this base_face up to sigma
                chains = self._build_chains_from_face(base_face, sigma, codim)
                
                for chain in chains:
                    # chain = [base_face, face_{codim+1}, ..., sigma]
                    simplex_vertices = []
                    valid_chain = True
                    
                    for face in chain:
                        key = self._simplex_key(face)
                        if key not in v_cache:
                            valid_chain = False
                            break
                        simplex_vertices.append(v_cache[key])
                    
                    if not valid_chain:
                        continue
                    
                    if len(simplex_vertices) != self.n + 1:
                        continue
                    
                    # Add vertices and simplex to K
                    vertex_indices = []
                    for v in simplex_vertices:
                        v_key = tuple(np.round(v, 10))
                        if v_key not in vertex_map:
                            vertex_map[v_key] = len(self.vertices)
                            self.vertices.append(v)
                        vertex_indices.append(vertex_map[v_key])
                    
                    self.simplices.append(tuple(vertex_indices))
        
        # Remove duplicate simplices
        self.simplices = list(set(self.simplices))
        
        print(f"  K has {len(self.vertices)} vertices, {len(self.simplices)} {self.n}-simplices")
    
    def _build_chains_from_face(self, start_face: np.ndarray, sigma: np.ndarray, 
                                 start_dim: int) -> List[List[np.ndarray]]:
        """
        Build all chains from start_face up to sigma.
        
        A chain is a sequence of faces: start_face ⊂ face_{dim+1} ⊂ ... ⊂ sigma
        where each face has dimension one more than the previous.
        """
        chains = []
        
        def extend_chain(current_face: np.ndarray, chain_so_far: List[np.ndarray], current_dim: int):
            if current_dim == self.d:
                # Reached sigma
                chains.append(chain_so_far)
                return
            
            # Find all faces of sigma of dimension current_dim + 1 that contain current_face
            next_dim = current_dim + 1
            next_faces = self._get_faces(sigma, next_dim)
            
            for next_face in next_faces:
                if self._is_subface(current_face, next_face):
                    extend_chain(next_face, chain_so_far + [next_face], next_dim)
        
        extend_chain(start_face, [start_face], start_dim)
        
        return chains
    
    def _is_subface(self, small: np.ndarray, large: np.ndarray) -> bool:
        """Check if small is a face of large."""
        small_set = set(tuple(np.round(v, 10)) for v in small)
        large_set = set(tuple(np.round(v, 10)) for v in large)
        return small_set.issubset(large_set)
    
    def euler_characteristic(self) -> int:
        """Compute Euler characteristic of K."""
        V = len(self.vertices)
        F = len(self.simplices)
        
        # Count edges
        edges = set()
        for simplex in self.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        E = len(edges)
        
        chi = V - E + F
        print(f"\nEuler characteristic: V - E + F = {V} - {E} + {F} = {chi}")
        return chi


# =============================================================================
# Mesh Export
# =============================================================================

def export_stl_binary(K: WhitneyTriangulationK, filename: str):
    """Export triangulation to binary STL file."""
    vertices = [np.array(v) for v in K.vertices]
    
    with open(filename, 'wb') as f:
        f.write(b'\x00' * 80)  # Header
        f.write(struct.pack('<I', len(K.simplices)))  # Triangle count
        
        for simplex in K.simplices:
            v0 = vertices[simplex[0]]
            v1 = vertices[simplex[1]]
            v2 = vertices[simplex[2]]
            
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-14:
                normal = normal / norm
            
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))  # Attribute
    
    print(f"Exported {len(K.simplices)} triangles to {filename}")


def export_mesh_medit(K: WhitneyTriangulationK, filename: str):
    """Export triangulation to Medit .mesh format."""
    with open(filename, 'w') as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n\n")
        
        f.write("Vertices\n")
        f.write(f"{len(K.vertices)}\n")
        for v in K.vertices:
            f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f} 0\n")
        f.write("\n")
        
        f.write("Triangles\n")
        f.write(f"{len(K.simplices)}\n")
        for tri in K.simplices:
            f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1} 0\n")
        f.write("\n")
        
        f.write("End\n")
    
    print(f"Exported to {filename}")


# =============================================================================
# Main Algorithm
# =============================================================================

def whitney_triangulate(surface: ImplicitSurface,
                        bounds: Tuple[np.ndarray, np.ndarray],
                        reach: float = None,
                        L_override: float = None) -> WhitneyTriangulationK:
    """
    Whitney's triangulation algorithm.
    
    Args:
        surface: Implicit surface to triangulate
        bounds: Bounding box (min_corner, max_corner)
        reach: Reach of M (computed if not provided)
        L_override: Override computed L (for testing)
    
    Returns:
        WhitneyTriangulationK object
    """
    d = 3  # Ambient dimension
    n = 2  # Surface dimension
    
    print("\n" + "=" * 60)
    print("WHITNEY'S TRIANGULATION ALGORITHM")
    print("=" * 60)
    
    # Step 1: Compute reach if not provided
    if reach is None:
        print("\nStep 1: Computing reach...")
        reach, _ = compute_reach(surface, bounds, n_samples=500, refine=False)
    else:
        print(f"\nStep 1: Using provided reach = {reach}")
    
    # Step 2: Create Coxeter triangulation (to get thickness, etc.)
    print("\nStep 2: Setting up Coxeter triangulation...")
    coxeter = CoxeterTriangulation(d, L=1.0)  # L will be determined
    
    # Step 3: Compute algorithm constants
    print("\nStep 3: Computing algorithm constants...")
    constants = compute_algorithm_constants(d, n, reach, coxeter)
    
    # Use override L if provided
    if L_override is not None:
        print(f"\n*** Using override L = {L_override} ***")
        constants['L'] = L_override
        # Recompute c_tilde for the new L
        t = constants['t']
        mu_0 = constants['mu_0']
        delta = constants['delta']
        term1 = t * mu_0 * delta / (18 * d * L_override)
        term2 = t**2 / 24
        constants['c_tilde'] = min(term1, term2, 1.0/24)
    
    L = constants['L']
    
    # Step 4: Generate Coxeter triangulation T with the computed L
    print(f"\nStep 4: Generating Coxeter triangulation with L = {L:.6e}...")
    coxeter = CoxeterTriangulation(d, L)
    ambient_simplices = coxeter.get_simplices_in_box(bounds[0], bounds[1])
    print(f"  Generated {len(ambient_simplices)} ambient 3-simplices")
    
    # Extract unique vertices
    vertex_dict = {}
    for simplex in ambient_simplices:
        for v in simplex:
            key = tuple(np.round(v, 10))
            if key not in vertex_dict:
                vertex_dict[key] = v.copy()
    print(f"  {len(vertex_dict)} unique vertices")
    
    # Step 5: Perturb vertices
    print(f"\nStep 5: Perturbation algorithm (Section 5.2)...")
    print(f"  c̃ = {constants['c_tilde']:.10f}")
    print(f"  Max perturbation c̃·L = {constants['c_tilde'] * L:.10e}")
    print(f"  Required distance ρ₁·c̃·L = {constants['rho_1'] * constants['c_tilde'] * L:.10e}")
    
    perturber = PerturbationAlgorithm(constants, surface)
    perturbed_vertices = perturber.perturb_all_vertices(vertex_dict)
    
    # Apply perturbation to simplices
    perturbed_simplices = []
    for simplex in ambient_simplices:
        new_simplex = np.zeros_like(simplex)
        for i, v in enumerate(simplex):
            key = tuple(np.round(v, 10))
            new_simplex[i] = perturbed_vertices[key]
        perturbed_simplices.append(new_simplex)
    
    # Step 6: Build triangulation K
    print(f"\nStep 6: Building triangulation K (Section 6.2)...")
    K = WhitneyTriangulationK(surface, constants, perturbed_simplices)
    
    # Step 7: Verify
    chi = K.euler_characteristic()
    
    return K


# =============================================================================
# Examples
# =============================================================================

def triangulate_sphere(radius: float = 1.0, L_override: float = None):
    """Triangulate a sphere."""
    print("\n" + "#" * 60)
    print(f"# SPHERE (radius = {radius})")
    print("#" * 60)
    
    def f(p):
        return p[0]**2 + p[1]**2 + p[2]**2 - radius**2
    
    def grad_f(p):
        return 2 * p
    
    def hess_f(p):
        return 2 * np.eye(3)
    
    surface = ImplicitSurface(f, grad_f, hess_f)
    bounds = (np.array([-radius-0.1]*3), np.array([radius+0.1]*3))
    
    K = whitney_triangulate(surface, bounds, reach=radius, L_override=L_override)
    
    return K


def triangulate_torus(R: float = 1.0, r: float = 0.4, L_override: float = None):
    """Triangulate a torus."""
    print("\n" + "#" * 60)
    print(f"# TORUS (R = {R}, r = {r})")
    print("#" * 60)
    
    def f(p):
        rho = np.sqrt(p[0]**2 + p[1]**2)
        return (rho - R)**2 + p[2]**2 - r**2
    
    def grad_f(p):
        rho = np.sqrt(p[0]**2 + p[1]**2)
        if rho < 1e-10:
            return np.array([0., 0., 2*p[2]])
        return np.array([
            2 * (rho - R) * p[0] / rho,
            2 * (rho - R) * p[1] / rho,
            2 * p[2]
        ])
    
    surface = ImplicitSurface(f, grad_f)
    bounds = (np.array([-(R+r+0.1), -(R+r+0.1), -(r+0.1)]),
              np.array([R+r+0.1, R+r+0.1, r+0.1]))
    
    K = whitney_triangulate(surface, bounds, reach=r, L_override=L_override)
    
    return K


if __name__ == "__main__":
    # Test with sphere
    # The theoretical L is extremely small, so we also test with practical L
    # print("=" * 70)
    # print("Testing with theoretical L from paper bounds")
    # print("=" * 70)
    
    # K_sphere = triangulate_sphere(radius=1.0)
    
    print("\n" + "=" * 70)
    print("Now testing with practical L values for visualization")
    print("=" * 70)
    
    # With practical L for visualization
    K_sphere_practical = triangulate_sphere(radius=1.0, L_override=0.1)
    export_stl_binary(K_sphere_practical, "sphere_whitney.stl")
    export_mesh_medit(K_sphere_practical, "sphere_whitney.mesh")
    
    K_torus_practical = triangulate_torus(R=1.0, r=0.4, L_override=0.05)
    export_stl_binary(K_torus_practical, "torus_whitney.stl")
    export_mesh_medit(K_torus_practical, "torus_whitney.mesh")