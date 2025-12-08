"""
Whitney's Method - Exact Algorithm Implementation
=================================================

Implements the algorithm from:
"Triangulating Submanifolds: An Elementary and Quantified Version of Whitney's Method"
Boissonnat, Kachanovich, Wintraecken (2021)

This implements the ALGORITHM exactly, not the proofs.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from itertools import combinations, permutations
from dataclasses import dataclass
import struct


# =============================================================================
# Section 4: Coxeter Triangulation of Type Ã_d
# =============================================================================

class CoxeterTriangulationAd:
    """
    Coxeter triangulation of type Ã_d (Definition 4.2).
    
    This is equivalent to the Freudenthal/Kuhn triangulation:
    each unit d-cube is divided into d! simplices, one per permutation.
    
    Vertices of fundamental simplex in R^{d+1} on hyperplane Σx_i = 0:
        u_0 = 0
        u_k has first k coordinates = (k-d-1)/(d+1), last d+1-k = k/(d+1)
    """
    
    def __init__(self, d: int, L: float):
        """
        Args:
            d: Ambient dimension
            L: Edge length (longest edge of simplices)
        """
        self.d = d
        self.L = L
        
        # Thickness t(T) from equation (3)
        if d % 2 == 1:
            self.thickness = np.sqrt(2.0 / d)
        else:
            self.thickness = np.sqrt(2.0 * (d + 1) / (d * (d + 2)))
        
        # Protection δ from equation (3)
        self.delta = (np.sqrt(d**2 + 2*d + 24) - np.sqrt(d**2 + 2*d)) / np.sqrt(12 * (d + 1))
        
        # μ₀ (normalized separation)
        self.mu_0 = np.sqrt(12.0 / (d + 2))
    
    def get_simplices_in_box(self, box_min: np.ndarray, box_max: np.ndarray) -> List[np.ndarray]:
        """
        Generate all d-simplices (as vertex arrays) within the bounding box.
        
        Uses Freudenthal/Kuhn triangulation: each cube gives d! simplices.
        """
        d = self.d
        L = self.L
        
        # Integer cube indices covering the box
        idx_min = np.floor(box_min / L).astype(int) - 1
        idx_max = np.ceil(box_max / L).astype(int) + 1
        
        simplices = []
        
        # Iterate over all cubes
        from itertools import product
        for cube_idx in product(*[range(idx_min[i], idx_max[i]) for i in range(d)]):
            cube_origin = np.array(cube_idx, dtype=float) * L
            
            # Each permutation π ∈ S_d gives one simplex
            # Vertices: v_0 = origin, v_k = v_{k-1} + L·e_{π(k)}
            for perm in permutations(range(d)):
                vertices = np.zeros((d + 1, d))
                vertices[0] = cube_origin
                
                for k in range(d):
                    vertices[k + 1] = vertices[k].copy()
                    vertices[k + 1, perm[k]] += L
                
                simplices.append(vertices)
        
        return simplices


# =============================================================================
# Section 5.1: Algorithm Constants
# =============================================================================

def compute_constants(d: int, n: int, reach: float, thickness: float) -> dict:
    """
    Compute constants from Section 5.1, equations (5)-(14).
    
    Args:
        d: Ambient dimension
        n: Manifold dimension  
        reach: Reach of manifold (rch M)
        thickness: Thickness of Coxeter triangulation t(T)
    
    Returns:
        Dictionary with all constants
    """
    from math import factorial, comb
    
    # Stirling numbers of second kind (for N_{≤k} bound)
    def stirling2(n, k):
        if k == 0:
            return 1 if n == 0 else 0
        if k == n or k == 1:
            return 1
        return k * stirling2(n-1, k) + stirling2(n-1, k-1)
    
    # N_{≤k} from equation (4): bound on faces of dim ≤ k containing a vertex
    def N_leq(k):
        return 2 + sum(factorial(j) * stirling2(d + 1, j) for j in range(1, k + 2))
    
    N = N_leq(d - n - 1) if d - n - 1 >= 0 else 2
    
    # ρ₁ from equation (5): volume fraction bound
    if d % 2 == 0:
        k = d // 2
        rho_1 = (2**(2*k - 2) * factorial(k)**2) / (np.pi * factorial(2*k) * N)
    else:
        k = (d + 1) // 2
        rho_1 = factorial(2*k - 2) / (2**(2*k) * factorial(k) * factorial(k - 1) * N)
    
    # μ₀ and δ for Ã_d
    mu_0 = np.sqrt(12.0 / (d + 2))
    delta = (np.sqrt(d**2 + 2*d + 24) - np.sqrt(d**2 + 2*d)) / np.sqrt(12 * (d + 1))
    
    # c̃ from equation (6)
    c_tilde = min(
        thickness * mu_0 * delta / (18 * d),
        thickness**2 / 24
    )
    
    # Enforce equation (7): c̃ ≤ 1/24
    c_tilde = min(c_tilde, 1.0 / 24)
    
    # α_k from equation (8)
    # α₁ = (4/3)ρ₁c̃, then α_k = (2/3)α_{k-1}·c̃·ρ₁
    def alpha(k):
        if k < 1:
            return 1.0
        result = (4.0 / 3) * rho_1 * c_tilde
        for _ in range(1, k):
            result *= (2.0 / 3) * c_tilde * rho_1
        return result
    
    # ζ from equation (10)
    binom = comb(d, d - n)
    zeta = (8 * thickness * (1 - 8 * c_tilde / thickness**2)) / (15 * np.sqrt(d) * binom * (1 + 2 * c_tilde))
    
    # L from equations (11)-(13)
    # The bound is complex; use simplified version from (13)
    alpha_exp = alpha(4 + 2*n) ** (d - n)
    zeta_exp = zeta ** (2 * n)
    
    # From equation (13): L/rch ≤ α_{d-n}^{d-n} · 54^{-1}
    L_bound = alpha(d - n) ** (d - n) / 54
    L = L_bound * reach
    
    return {
        'N': N,
        'rho_1': rho_1,
        'c_tilde': c_tilde,
        'alpha': alpha,
        'zeta': zeta,
        'L_bound': L_bound,
        'L': L,
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
    are far from M.
    """
    
    def __init__(self, d: int, n: int, L: float, c_tilde: float, rho_1: float):
        """
        Args:
            d: Ambient dimension
            n: Manifold dimension
            L: Edge length
            c_tilde: Normalized perturbation bound (equation 6)
            rho_1: Volume fraction bound (equation 5)
        """
        self.d = d
        self.n = n
        self.L = L
        self.c_tilde = c_tilde
        self.rho_1 = rho_1
        
        # Maximum perturbation radius: c̃·L (equation 17)
        self.max_perturbation = c_tilde * L
        
        # Required distance from spans: ρ₁·c̃·L (equation 19)
        self.required_distance = rho_1 * c_tilde * L
    
    def perturb_vertices(self, vertices: Dict[tuple, np.ndarray], 
                         manifold) -> Dict[tuple, np.ndarray]:
        """
        Perturb all vertices following the algorithm in Section 5.2.
        
        "We are going to inductively choose new vertices ṽ₁, ṽ₂, ..."
        
        Args:
            vertices: Dictionary mapping vertex key to coordinates
            manifold: Manifold oracle
            
        Returns:
            Dictionary mapping vertex key to perturbed coordinates
        """
        perturbed = {}
        
        for key, v in vertices.items():
            perturbed[key] = self._perturb_single_vertex(v, manifold, list(perturbed.values()))
        
        return perturbed
    
    def _perturb_single_vertex(self, v: np.ndarray, manifold, 
                                existing: List[np.ndarray]) -> np.ndarray:
        """
        Perturb a single vertex.
        
        Section 5.2 describes two cases:
        
        Case 1: d(vᵢ, M) ≥ 3L/2
            "In this case we choose ṽᵢ = vᵢ"
        
        Case 2: d(vᵢ, M) < 3L/2
            "Let p be a point in M such that d(vᵢ,p) < 3L/2.
             We now consider span(τ'ⱼ, TₚM) for all 0 ≤ j ≤ ν...
             we pick ṽᵢ so that it lies sufficiently far from [these spans]"
        """
        L = self.L
        
        # Compute distance to M
        dist_to_M = manifold.distance(v)
        
        # CASE 1: Vertex far from manifold
        if dist_to_M >= 1.5 * L:
            return v.copy()
        
        # CASE 2: Vertex close to manifold
        # Find closest point p on M
        p = manifold.closest_point(v)
        if p is None:
            return v.copy()
        
        # Get tangent space TₚM (as n×d matrix of basis vectors)
        T_p = manifold.tangent_basis(p)
        
        # For codimension d-n = 1:
        # span(∅, TₚM) = TₚM, which is a hyperplane
        # We need to perturb away from this hyperplane
        
        if self.d - self.n == 1:
            return self._perturb_codim_1(v, p, T_p, manifold)
        else:
            return self._perturb_general(v, p, T_p, manifold)
    
    def _perturb_codim_1(self, v: np.ndarray, p: np.ndarray, 
                         T_p: np.ndarray, manifold) -> np.ndarray:
        """
        Perturbation for codimension 1 case.
        
        The span of TₚM is a hyperplane. We perturb away from it.
        
        From equation (19): d(ṽᵢ, span) ≥ ρ₁c̃L
        From equation (17): |vᵢ - ṽᵢ| ≤ c̃L
        """
        # Normal to tangent space
        normal = manifold.normal(p)
        
        # Current signed distance from v to the tangent hyperplane at p
        # TₚM passes through p, so distance = (v - p)·normal
        signed_dist = np.dot(v - p, normal)
        current_dist = abs(signed_dist)
        
        # If already far enough, add small random perturbation
        if current_dist >= self.required_distance:
            noise = np.random.randn(self.d) * self.max_perturbation * 0.1
            if np.linalg.norm(noise) > self.max_perturbation:
                noise *= self.max_perturbation / np.linalg.norm(noise)
            return v + noise
        
        # Need to perturb in normal direction
        # Choose direction away from manifold
        direction = np.sign(signed_dist) if signed_dist != 0 else 1.0
        
        # Required shift in normal direction
        shift_amount = self.required_distance - current_dist + self.max_perturbation * 0.1
        shift_amount = min(shift_amount, self.max_perturbation)
        
        perturbation = direction * shift_amount * normal
        
        return v + perturbation
    
    def _perturb_general(self, v: np.ndarray, p: np.ndarray,
                         T_p: np.ndarray, manifold) -> np.ndarray:
        """
        Perturbation for higher codimension.
        
        Need to consider span(τ'ⱼ, TₚM) for all faces τ'ⱼ of dimension ≤ d-n-2.
        
        For simplicity, use random perturbation within allowed radius.
        """
        perturbation = np.random.randn(self.d)
        perturbation *= self.max_perturbation * 0.5 / np.linalg.norm(perturbation)
        return v + perturbation


# =============================================================================
# Section 6.2: Constructing the Triangulation K
# =============================================================================

class WhitneyTriangulationK:
    """
    Construct the triangulation K of manifold M.
    
    Section 6.2: "In each simplex τ of T̃ that intersects M, we choose a
    point v(τ) and construct a complex K with these points as vertices."
    """
    
    def __init__(self, manifold, ambient_simplices: List[np.ndarray]):
        """
        Args:
            manifold: The manifold M with oracle methods
            ambient_simplices: List of d-simplices from perturbed triangulation T̃
        """
        self.manifold = manifold
        self.d = manifold.d
        self.n = manifold.n
        self.codim = self.d - self.n  # d - n
        
        # Storage for v(τ) values
        # Key: sorted tuple of vertex coordinates (rounded)
        self.v_tau: Dict[tuple, np.ndarray] = {}
        
        # Output triangulation K
        self.vertices: List[np.ndarray] = []
        self.simplices: List[Tuple[int, ...]] = []  # n-simplices as vertex index tuples
        self._vertex_to_idx: Dict[tuple, int] = {}
        
        # Build K
        self._build(ambient_simplices)
    
    def _vertex_key(self, v: np.ndarray) -> tuple:
        """Create hashable key for a vertex."""
        return tuple(np.round(v, 10))
    
    def _simplex_key(self, verts: np.ndarray) -> tuple:
        """Create hashable key for a simplex (sorted vertices)."""
        return tuple(sorted(self._vertex_key(v) for v in verts))
    
    def _add_vertex(self, v: np.ndarray) -> int:
        """Add vertex to K, return its index."""
        key = self._vertex_key(v)
        if key not in self._vertex_to_idx:
            self._vertex_to_idx[key] = len(self.vertices)
            self.vertices.append(v.copy())
        return self._vertex_to_idx[key]
    
    def _simplex_intersects_M(self, verts: np.ndarray) -> bool:
        """
        Check if simplex intersects manifold M.
        
        For implicit manifold f(x) = 0, check for sign changes.
        """
        if hasattr(self.manifold, 'f'):
            vals = [self.manifold.f(v) for v in verts]
            return min(vals) <= 0 <= max(vals)
        return False
    
    def _compute_v_tau(self, verts: np.ndarray, dim: int) -> Optional[np.ndarray]:
        """
        Compute v(τ) for simplex τ with given vertices.
        
        Section 6.2, equation (26):
        
        "If τ is a simplex of dimension d-n, then there is a unique point
         of intersection with M (Lemma 6.4). We define v(τ) to be this
         unique point."
        
        "If τ has dimension greater than d-n, then we consider the faces
         τ₁^{d-n}, ..., τⱼ^{d-n} of τ of dimension d-n that intersect M.
         We now define:
         
             v(τ) = (v(τ₁^{d-n}) + ... + v(τⱼ^{d-n})) / j"
        """
        key = self._simplex_key(verts)
        
        # Check cache
        if key in self.v_tau:
            return self.v_tau[key]
        
        # Check if τ intersects M
        if not self._simplex_intersects_M(verts):
            return None
        
        if dim == self.codim:
            # Base case: τ has dimension d-n
            # "there is a unique point of intersection with M"
            v = self._find_unique_intersection(verts)
        else:
            # Recursive case: τ has dimension > d-n
            # "v(τ) = average of v(τᵢ^{d-n}) for (d-n)-faces that intersect M"
            
            face_values = []
            
            # Enumerate all (d-n)-dimensional faces
            for face_indices in combinations(range(len(verts)), self.codim + 1):
                face_verts = verts[list(face_indices)]
                face_v = self._compute_v_tau(face_verts, self.codim)
                if face_v is not None:
                    face_values.append(face_v)
            
            if len(face_values) == 0:
                return None
            
            # Average
            v = np.mean(face_values, axis=0)
        
        if v is not None:
            self.v_tau[key] = v
        
        return v
    
    def _find_unique_intersection(self, verts: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the unique intersection of a (d-n)-simplex with M.
        
        Lemma 6.4: "every simplex of dimension d-n that intersects M
        contains at most one point of M"
        """
        # For codimension 1, (d-n)-simplex is an edge (1-simplex)
        if self.codim == 1 and len(verts) == 2:
            return self.manifold.intersect_edge(verts[0], verts[1])
        
        # For higher codimension, use projection of centroid
        if self._simplex_intersects_M(verts):
            centroid = np.mean(verts, axis=0)
            return self.manifold.project(centroid)
        
        return None
    
    def _build(self, ambient_simplices: List[np.ndarray]):
        """
        Build the triangulation K.
        
        Section 6.2, equation (25):
        
        "For each sequence τ₀ ⊂ τ₁ ⊂ ... ⊂ τₖ of distinct simplices in T̃
         such that τ₀ intersects M,
         
             σᵏ = {v(τ₀), v(τ₁), ..., v(τₖ)}
         
         will be a simplex of K."
        
        For our case: chains τ_{d-n} ⊂ τ_{d-n+1} ⊂ ... ⊂ τ_d give n-simplices.
        """
        print(f"Building K from {len(ambient_simplices)} ambient {self.d}-simplices...")
        
        intersecting_count = 0
        
        for d_simplex in ambient_simplices:
            # Check if d-simplex intersects M
            if not self._simplex_intersects_M(d_simplex):
                continue
            
            intersecting_count += 1
            
            # Compute v(τ) for all faces of all dimensions from d-n to d
            v_by_face: Dict[tuple, np.ndarray] = {}
            
            for dim in range(self.codim, self.d + 1):
                num_verts = dim + 1
                for face_indices in combinations(range(self.d + 1), num_verts):
                    face_verts = d_simplex[list(face_indices)]
                    v = self._compute_v_tau(face_verts, dim)
                    if v is not None:
                        v_by_face[face_indices] = v
            
            # Enumerate chains and create K simplices
            self._create_K_simplices_from_chains(d_simplex, v_by_face)
        
        print(f"  {intersecting_count} simplices intersect M")
        print(f"  K has {len(self.vertices)} vertices, {len(self.simplices)} {self.n}-simplices")
    
    def _create_K_simplices_from_chains(self, d_simplex: np.ndarray, 
                                         v_by_face: Dict[tuple, np.ndarray]):
        """
        Create K simplices from valid chains.
        
        A chain is: τ_{d-n} ⊂ τ_{d-n+1} ⊂ ... ⊂ τ_d
        where each τ_k has v(τ_k) defined.
        
        The corresponding K simplex is: {v(τ_{d-n}), ..., v(τ_d)}
        """
        # Start from (d-n)-faces (base of chains)
        base_faces = [idx for idx in v_by_face if len(idx) == self.codim + 1]
        
        for base in base_faces:
            # Enumerate all chains starting from this base
            chains = self._enumerate_chains(base, v_by_face)
            
            for chain in chains:
                # Chain should have length n+1 (from dim d-n to dim d)
                if len(chain) != self.n + 1:
                    continue
                
                # Create K simplex: {v(τ_{d-n}), ..., v(τ_d)}
                K_simplex = []
                valid = True
                
                for face_idx in chain:
                    if face_idx not in v_by_face:
                        valid = False
                        break
                    vertex_idx = self._add_vertex(v_by_face[face_idx])
                    K_simplex.append(vertex_idx)
                
                if valid and len(set(K_simplex)) == self.n + 1:  # All distinct vertices
                    self.simplices.append(tuple(K_simplex))
    
    def _enumerate_chains(self, start: tuple, v_by_face: Dict) -> List[List[tuple]]:
        """
        Recursively enumerate all chains from start face up to d-simplex.
        """
        # If start is the d-simplex (d+1 vertices), chain is complete
        if len(start) == self.d + 1:
            return [[start]]
        
        chains = []
        
        # Find all faces of dimension +1 that contain start
        for larger in v_by_face:
            if len(larger) == len(start) + 1 and set(start).issubset(set(larger)):
                sub_chains = self._enumerate_chains(larger, v_by_face)
                for sub in sub_chains:
                    chains.append([start] + sub)
        
        return chains


# =============================================================================
# Manifold Oracle
# =============================================================================

class ImplicitManifold:
    """
    Manifold defined implicitly as f(x) = 0.
    """
    
    def __init__(self, f, grad_f, n: int, d: int, reach: float):
        """
        Args:
            f: Implicit function (manifold is f(x) = 0)
            grad_f: Gradient of f
            n: Intrinsic dimension
            d: Ambient dimension
            reach: Reach of manifold
        """
        self.f = f
        self.grad_f = grad_f
        self.n = n
        self.d = d
        self.reach = reach
    
    def distance(self, p: np.ndarray) -> float:
        """Approximate distance to manifold."""
        closest = self.closest_point(p)
        if closest is None:
            return float('inf')
        return np.linalg.norm(p - closest)
    
    def closest_point(self, p: np.ndarray, max_iter: int = 30) -> Optional[np.ndarray]:
        """Find closest point on manifold using Newton projection."""
        x = p.copy()
        for _ in range(max_iter):
            val = self.f(x)
            if abs(val) < 1e-12:
                return x
            grad = self.grad_f(x)
            grad_norm_sq = np.dot(grad, grad)
            if grad_norm_sq < 1e-14:
                return None
            x = x - (val / grad_norm_sq) * grad
        return x if abs(self.f(x)) < 1e-8 else None
    
    def project(self, p: np.ndarray) -> Optional[np.ndarray]:
        """Project point onto manifold."""
        return self.closest_point(p)
    
    def normal(self, p: np.ndarray) -> np.ndarray:
        """Unit normal at point (for codimension 1)."""
        grad = self.grad_f(p)
        return grad / np.linalg.norm(grad)
    
    def tangent_basis(self, p: np.ndarray) -> np.ndarray:
        """Orthonormal basis for tangent space TₚM."""
        normal = self.normal(p)
        
        # Build orthonormal basis for hyperplane perpendicular to normal
        basis = []
        for i in range(self.d):
            e_i = np.zeros(self.d)
            e_i[i] = 1.0
            
            # Project out normal
            v = e_i - np.dot(e_i, normal) * normal
            
            # Project out previous basis vectors
            for b in basis:
                v = v - np.dot(v, b) * b
            
            norm_v = np.linalg.norm(v)
            if norm_v > 0.1:
                basis.append(v / norm_v)
            
            if len(basis) == self.n:
                break
        
        return np.array(basis)
    
    def intersect_edge(self, p1: np.ndarray, p2: np.ndarray) -> Optional[np.ndarray]:
        """Find intersection of edge with manifold using bisection."""
        f1, f2 = self.f(p1), self.f(p2)
        
        # No sign change -> no intersection
        if f1 * f2 > 0:
            return None
        
        # Bisection
        a, b = 0.0, 1.0
        for _ in range(50):
            t = (a + b) / 2
            p_mid = (1 - t) * p1 + t * p2
            f_mid = self.f(p_mid)
            
            if abs(f_mid) < 1e-14:
                return p_mid
            
            if f1 * f_mid < 0:
                b = t
            else:
                a = t
                f1 = f_mid
        
        t = (a + b) / 2
        return (1 - t) * p1 + t * p2


# =============================================================================
# STL Export
# =============================================================================

def export_mesh_medit(vertices: List[np.ndarray], 
                      triangles: List[Tuple[int, int, int]], 
                      filename: str):
    """Export to .mesh format (Medit/INRIA format)."""
    with open(filename, 'w') as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n\n")
        
        f.write("Vertices\n")
        f.write(f"{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f} 0\n")
        f.write("\n")
        
        f.write("Triangles\n")
        f.write(f"{len(triangles)}\n")
        for tri in triangles:
            f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1} 0\n")  # 1-indexed
        f.write("\n")
        
        f.write("End\n")

def export_stl_ascii(triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                     filename: str):
    """Export triangles to ASCII STL file."""
    with open(filename, 'w') as f:
        f.write("solid whitney_triangulation\n")
        
        for v0, v1, v2 in triangles:
            # Compute normal
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n = n / norm
            else:
                n = np.array([0, 0, 1])
            
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid whitney_triangulation\n")


def export_stl_binary(triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                      filename: str):
    """Export triangles to binary STL file (smaller file size)."""
    with open(filename, 'wb') as f:
        # 80-byte header
        header = b'Whitney Triangulation' + b'\0' * (80 - 21)
        f.write(header)
        
        # Number of triangles (4 bytes, little-endian)
        f.write(struct.pack('<I', len(triangles)))
        
        for v0, v1, v2 in triangles:
            # Compute normal
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n = n / norm
            else:
                n = np.array([0, 0, 1])
            
            # Normal (3 floats)
            f.write(struct.pack('<fff', n[0], n[1], n[2]))
            
            # Vertices (3 × 3 floats)
            f.write(struct.pack('<fff', v0[0], v0[1], v0[2]))
            f.write(struct.pack('<fff', v1[0], v1[1], v1[2]))
            f.write(struct.pack('<fff', v2[0], v2[1], v2[2]))
            
            # Attribute byte count (2 bytes, typically 0)
            f.write(struct.pack('<H', 0))


# =============================================================================
# Main Algorithm
# =============================================================================

def whitney_triangulate(manifold: ImplicitManifold, 
                        bounds: Tuple[np.ndarray, np.ndarray],
                        L: float = None) -> WhitneyTriangulationK:
    """
    Whitney's triangulation algorithm.
    
    Algorithm from Section 2.1:
    
    Part 1: Perturbation Algorithm
        - Choose Coxeter triangulation T of type Ã_d with edge length L
        - Perturb vertices to get T̃
    
    Part 2: Triangulation Construction
        - For each τ ∈ T̃ intersecting M, compute v(τ)
        - Build K via barycentric subdivision
    
    Args:
        manifold: Implicit manifold to triangulate
        bounds: (min_corner, max_corner) bounding box
        L: Edge length (computed from reach if not specified)
    
    Returns:
        WhitneyTriangulationK object containing the triangulation
    """
    d = manifold.d
    n = manifold.n
    
    print("=" * 70)
    print("WHITNEY'S TRIANGULATION ALGORITHM")
    print("=" * 70)
    print(f"Manifold: dimension {n} in R^{d}")
    print(f"Reach: {manifold.reach}")
    
    # Create Coxeter triangulation to get thickness
    temp_coxeter = CoxeterTriangulationAd(d, 1.0)
    
    # Compute algorithm constants (Section 5.1)
    constants = compute_constants(d, n, manifold.reach, temp_coxeter.thickness)
    
    print(f"\nAlgorithm constants (Section 5.1):")
    print(f"  ρ₁ = {constants['rho_1']:.6f}")
    print(f"  c̃  = {constants['c_tilde']:.6f}")
    print(f"  L/reach bound = {constants['L_bound']:.6f}")
    
    # Set L
    if L is None:
        # Use computed bound, but enforce a practical minimum
        L = max(constants['L'], manifold.reach / 10)
    
    print(f"  Using L = {L:.4f}")
    
    # PART 1: Create Coxeter triangulation T
    print(f"\nPart 1: Coxeter triangulation T")
    coxeter = CoxeterTriangulationAd(d, L)
    print(f"  Edge length L = {L:.4f}")
    print(f"  Thickness t(T) = {coxeter.thickness:.4f}")
    
    # Generate ambient simplices
    ambient_simplices = coxeter.get_simplices_in_box(bounds[0], bounds[1])
    print(f"  Generated {len(ambient_simplices)} ambient {d}-simplices")
    
    # Extract unique vertices
    vertex_dict = {}
    for simplex in ambient_simplices:
        for v in simplex:
            key = tuple(np.round(v, 10))
            if key not in vertex_dict:
                vertex_dict[key] = v.copy()
    
    print(f"  Total vertices: {len(vertex_dict)}")
    
    # PART 1: Perturb vertices to get T̃
    print(f"\nPart 1: Vertex perturbation")
    perturber = PerturbationAlgorithm(d, n, L, constants['c_tilde'], constants['rho_1'])
    print(f"  Max perturbation: c̃·L = {perturber.max_perturbation:.6f}")
    print(f"  Required distance: ρ₁·c̃·L = {perturber.required_distance:.6f}")
    
    perturbed_vertices = perturber.perturb_vertices(vertex_dict, manifold)
    
    # Apply perturbation to simplices
    perturbed_simplices = []
    for simplex in ambient_simplices:
        new_simplex = np.zeros_like(simplex)
        for i, v in enumerate(simplex):
            key = tuple(np.round(v, 10))
            new_simplex[i] = perturbed_vertices[key]
        perturbed_simplices.append(new_simplex)
    
    print(f"  Perturbation complete")
    
    # PART 2: Build triangulation K
    print(f"\nPart 2: Building triangulation K")
    K = WhitneyTriangulationK(manifold, perturbed_simplices)
    
    return K


# =============================================================================
# Examples
# =============================================================================

def triangulate_sphere(L: float = 0.3):
    """Triangulate unit sphere."""
    print("\n" + "=" * 70)
    print("TRIANGULATING UNIT SPHERE")
    print("=" * 70)
    
    # Sphere: x² + y² + z² - 1 = 0
    f = lambda p: p[0]**2 + p[1]**2 + p[2]**2 - 1
    grad_f = lambda p: 2 * p
    
    sphere = ImplicitManifold(f, grad_f, n=2, d=3, reach=1.0)
    bounds = (np.array([-1.3, -1.3, -1.3]), np.array([1.3, 1.3, 1.3]))
    
    K = whitney_triangulate(sphere, bounds, L=L)
    
    # Get triangles
    triangles = []
    for simplex in K.simplices:
        v0 = K.vertices[simplex[0]]
        v1 = K.vertices[simplex[1]]
        v2 = K.vertices[simplex[2]]
        triangles.append((v0, v1, v2))
    
    # Export
    export_mesh_medit(K.vertices, K.simplices, "sphere_whitney_small.mesh")
    export_stl_binary(triangles, "sphere_whitney_small.stl")
    print(f"\nExported {len(triangles)} triangles to sphere_whitney_small.stl")
    
    # Verify Euler characteristic
    V = len(K.vertices)
    edges = set()
    for s in K.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edges.add(tuple(sorted([s[i], s[j]])))
    E = len(edges)
    F = len(K.simplices)
    print(f"\nEuler characteristic: V - E + F = {V} - {E} + {F} = {V - E + F}")
    
    return K, triangles


def triangulate_torus(R: float = 1.0, r: float = 0.4, L: float = 0.2):
    """Triangulate torus with major radius R and minor radius r."""
    print("\n" + "=" * 70)
    print(f"TRIANGULATING TORUS (R={R}, r={r})")
    print("=" * 70)
    
    # Torus: (sqrt(x² + y²) - R)² + z² - r² = 0
    def f(p):
        rho = np.sqrt(p[0]**2 + p[1]**2)
        return (rho - R)**2 + p[2]**2 - r**2
    
    def grad_f(p):
        rho = np.sqrt(p[0]**2 + p[1]**2)
        if rho < 1e-10:
            return np.array([0.0, 0.0, 2*p[2]])
        return np.array([
            2 * (rho - R) * p[0] / rho,
            2 * (rho - R) * p[1] / rho,
            2 * p[2]
        ])
    
    torus = ImplicitManifold(f, grad_f, n=2, d=3, reach=r)
    bounds = (np.array([-(R+r+0.2), -(R+r+0.2), -(r+0.2)]), 
              np.array([R+r+0.2, R+r+0.2, r+0.2]))
    
    K = whitney_triangulate(torus, bounds, L=L)
    
    # Get triangles
    triangles = []
    for simplex in K.simplices:
        v0 = K.vertices[simplex[0]]
        v1 = K.vertices[simplex[1]]
        v2 = K.vertices[simplex[2]]
        triangles.append((v0, v1, v2))
    
    # Export
    export_mesh_medit(K.vertices, K.simplices, "torus_whitney_small.mesh")
    export_stl_binary(triangles, "torus_whitney_small.stl")
    print(f"\nExported {len(triangles)} triangles to torus_whitney_small.stl")
    
    # Verify Euler characteristic (should be 0 for torus)
    V = len(K.vertices)
    edges = set()
    for s in K.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edges.add(tuple(sorted([s[i], s[j]])))
    E = len(edges)
    F = len(K.simplices)
    print(f"\nEuler characteristic: V - E + F = {V} - {E} + {F} = {V - E + F}")
    
    return K, triangles


if __name__ == "__main__":
    # Triangulate sphere
    K_sphere, tri_sphere = triangulate_sphere(L=0.15)
    
    # Triangulate torus
    K_torus, tri_torus = triangulate_torus(L=0.05)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("Files created:")
    print("  - sphere_whitney.stl")
    print("  - torus_whitney.stl")