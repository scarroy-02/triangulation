"""
Proper Coxeter Ã₃ triangulation for 3D.

The Ã₃ (A₃-tilde) Coxeter triangulation is the Delaunay triangulation of the 
A₃* lattice, which in 3D corresponds to the BCC (body-centered cubic) lattice.

BCC lattice vertices:
- Cubic vertices: (i, j, k) * L for integers i, j, k
- Body-center vertices: (i+0.5, j+0.5, k+0.5) * L for integers i, j, k

The Delaunay triangulation of BCC creates 24 congruent tetrahedra per cubic cell.

Reference: Boissonnat et al., "Triangulating smooth submanifolds with light scaffolding"
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from scipy.spatial import Delaunay
import gudhi


def generate_bcc_lattice(box_min: np.ndarray, box_max: np.ndarray, L: float):
    """
    Generate BCC lattice points within a bounding box.
    
    The BCC lattice has two types of vertices:
    1. Cubic vertices at integer multiples of L
    2. Body-centered vertices at (i+0.5, j+0.5, k+0.5) * L
    
    Returns:
        vertices: Dict mapping index to position
        vertex_type: Dict mapping index to type ('cubic' or 'body')
    """
    # Extend box slightly to ensure coverage
    margin = L
    
    # Compute grid ranges
    n_min = np.floor((box_min - margin) / L).astype(int)
    n_max = np.ceil((box_max + margin) / L).astype(int)
    
    vertices = {}
    vertex_type = {}
    idx = 0
    
    # Cubic vertices
    for i in range(n_min[0], n_max[0] + 1):
        for j in range(n_min[1], n_max[1] + 1):
            for k in range(n_min[2], n_max[2] + 1):
                pos = np.array([i, j, k], dtype=float) * L
                vertices[idx] = pos
                vertex_type[idx] = 'cubic'
                idx += 1
    
    # Body-centered vertices
    for i in range(n_min[0], n_max[0] + 1):
        for j in range(n_min[1], n_max[1] + 1):
            for k in range(n_min[2], n_max[2] + 1):
                pos = np.array([i + 0.5, j + 0.5, k + 0.5], dtype=float) * L
                vertices[idx] = pos
                vertex_type[idx] = 'body'
                idx += 1
    
    return vertices, vertex_type


def generate_coxeter_A3(box_min: np.ndarray, box_max: np.ndarray, L: float):
    """
    Generate Coxeter Ã₃ triangulation using BCC lattice and Delaunay.
    
    Returns:
        vertices: Dict mapping index to position
        tetrahedra: List of 4-tuples of vertex indices
    """
    # Generate BCC lattice
    vertices, vertex_type = generate_bcc_lattice(box_min, box_max, L)
    
    # Convert to array for Delaunay
    n_verts = len(vertices)
    points = np.array([vertices[i] for i in range(n_verts)])
    
    # Compute Delaunay triangulation
    delaunay = Delaunay(points)
    
    # Filter tetrahedra to those intersecting the bounding box
    tetrahedra = []
    for simplex in delaunay.simplices:
        # Get tetrahedron vertices
        tet_verts = points[simplex]
        centroid = np.mean(tet_verts, axis=0)
        
        # Keep if centroid is within extended box
        margin = L
        if (np.all(centroid >= box_min - margin) and 
            np.all(centroid <= box_max + margin)):
            tetrahedra.append(tuple(simplex))
    
    return vertices, tetrahedra


def generate_coxeter_A3_gudhi(box_min: np.ndarray, box_max: np.ndarray, L: float):
    """
    Generate Coxeter Ã₃ triangulation using GUDHI's Delaunay complex.
    
    Returns:
        vertices: Dict mapping index to position
        tetrahedra: List of 4-tuples of vertex indices
    """
    # Generate BCC lattice
    vertices, vertex_type = generate_bcc_lattice(box_min, box_max, L)
    
    # Convert to list for GUDHI
    n_verts = len(vertices)
    points = [list(vertices[i]) for i in range(n_verts)]
    
    # Use GUDHI DelaunayComplex
    delaunay = gudhi.DelaunayComplex(points=points)
    st = delaunay.create_simplex_tree()
    
    # Extract tetrahedra (3-simplices)
    tetrahedra = []
    for simplex, _ in st.get_simplices():
        if len(simplex) == 4:  # Tetrahedron
            # Get centroid
            tet_verts = np.array([vertices[i] for i in simplex])
            centroid = np.mean(tet_verts, axis=0)
            
            # Keep if within box
            margin = L
            if (np.all(centroid >= box_min - margin) and 
                np.all(centroid <= box_max + margin)):
                tetrahedra.append(tuple(simplex))
    
    return vertices, tetrahedra


def get_tetrahedron_edges(tet: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """Get all 6 edges of a tetrahedron."""
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edges.append(tuple(sorted([tet[i], tet[j]])))
    return edges


def get_all_edges(tetrahedra: List[Tuple[int, int, int, int]]) -> Set[Tuple[int, int]]:
    """Get all unique edges from tetrahedra."""
    edges = set()
    for tet in tetrahedra:
        for e in get_tetrahedron_edges(tet):
            edges.add(e)
    return edges


def analyze_triangulation(vertices, tetrahedra, L):
    """Analyze properties of the triangulation."""
    edges = get_all_edges(tetrahedra)
    
    # Edge lengths
    edge_lengths = []
    for e in edges:
        v0, v1 = vertices[e[0]], vertices[e[1]]
        edge_lengths.append(np.linalg.norm(v1 - v0))
    
    edge_lengths = np.array(edge_lengths)
    
    print(f"Triangulation statistics:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Tetrahedra: {len(tetrahedra)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Edge lengths:")
    print(f"    min: {edge_lengths.min():.6f}")
    print(f"    max: {edge_lengths.max():.6f}")
    print(f"    mean: {edge_lengths.mean():.6f}")
    print(f"  Expected edge length L: {L}")
    print(f"  Edge length / L ratio: {edge_lengths.min()/L:.4f} to {edge_lengths.max()/L:.4f}")
    
    # For BCC Delaunay, we expect edges of length:
    # - L (between adjacent cubic vertices)
    # - L * sqrt(3)/2 ≈ 0.866 L (between cubic and body-center)
    print(f"  Expected ratios: 1.0 (cubic-cubic), {np.sqrt(3)/2:.4f} (cubic-body)")
    
    return edge_lengths


if __name__ == "__main__":
    # Test
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([2.0, 2.0, 2.0])
    L = 0.5
    
    print("Testing Coxeter Ã₃ triangulation (BCC Delaunay)")
    print("=" * 50)
    
    vertices, tetrahedra = generate_coxeter_A3(box_min, box_max, L)
    analyze_triangulation(vertices, tetrahedra, L)