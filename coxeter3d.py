"""
Coxeter Ã_3 triangulation for 3D.

The Ã_3 lattice uses 4 basis vectors in 3D that form regular tetrahedra.
Each cube of the BCC lattice is divided into 24 tetrahedra.

For simplicity, we use the Freudenthal/Kuhn triangulation which divides
each cube into 6 tetrahedra. This is simpler and still works for the algorithm.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from itertools import combinations


def generate_kuhn_triangulation(box_min: np.ndarray, box_max: np.ndarray, L: float):
    """
    Generate Kuhn/Freudenthal triangulation of 3D box.
    
    Each unit cube is divided into 6 tetrahedra.
    
    Args:
        box_min: Minimum corner of box
        box_max: Maximum corner of box
        L: Edge length (approximately)
    
    Returns:
        vertices: Dict mapping index to position
        tetrahedra: List of 4-tuples of vertex indices
    """
    # Compute grid size
    box_size = box_max - box_min
    n = np.ceil(box_size / L).astype(int) + 1
    
    # Generate vertices on regular grid
    vertices = {}
    idx = 0
    vertex_grid = {}  # (i,j,k) -> vertex index
    
    for i in range(n[0] + 1):
        for j in range(n[1] + 1):
            for k in range(n[2] + 1):
                pos = box_min + np.array([i, j, k]) * L
                vertices[idx] = pos
                vertex_grid[(i, j, k)] = idx
                idx += 1
    
    # Generate tetrahedra using Kuhn triangulation
    # Each cube [i,i+1] x [j,j+1] x [k,k+1] is divided into 6 tetrahedra
    tetrahedra = []
    
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                # Get the 8 vertices of the cube
                v000 = vertex_grid[(i, j, k)]
                v001 = vertex_grid[(i, j, k+1)]
                v010 = vertex_grid[(i, j+1, k)]
                v011 = vertex_grid[(i, j+1, k+1)]
                v100 = vertex_grid[(i+1, j, k)]
                v101 = vertex_grid[(i+1, j, k+1)]
                v110 = vertex_grid[(i+1, j+1, k)]
                v111 = vertex_grid[(i+1, j+1, k+1)]
                
                # Kuhn triangulation: 6 tetrahedra per cube
                # Based on the main diagonal from v000 to v111
                tetrahedra.append((v000, v100, v110, v111))
                tetrahedra.append((v000, v110, v010, v111))
                tetrahedra.append((v000, v010, v011, v111))
                tetrahedra.append((v000, v011, v001, v111))
                tetrahedra.append((v000, v001, v101, v111))
                tetrahedra.append((v000, v101, v100, v111))
    
    return vertices, tetrahedra


def get_tetrahedron_edges(tet: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """Get all 6 edges of a tetrahedron."""
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edges.append(tuple(sorted([tet[i], tet[j]])))
    return edges


def get_tetrahedron_faces(tet: Tuple[int, int, int, int]) -> List[Tuple[int, int, int]]:
    """Get all 4 faces of a tetrahedron."""
    faces = []
    for i in range(4):
        face = tuple(sorted([tet[j] for j in range(4) if j != i]))
        faces.append(face)
    return faces


def get_all_edges(tetrahedra: List[Tuple[int, int, int, int]]) -> Set[Tuple[int, int]]:
    """Get all unique edges from tetrahedra."""
    edges = set()
    for tet in tetrahedra:
        for e in get_tetrahedron_edges(tet):
            edges.add(e)
    return edges


def get_all_faces(tetrahedra: List[Tuple[int, int, int, int]]) -> Set[Tuple[int, int, int]]:
    """Get all unique faces from tetrahedra."""
    faces = set()
    for tet in tetrahedra:
        for f in get_tetrahedron_faces(tet):
            faces.add(f)
    return faces


if __name__ == "__main__":
    # Test
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([1.0, 1.0, 1.0])
    L = 0.5
    
    vertices, tetrahedra = generate_kuhn_triangulation(box_min, box_max, L)
    edges = get_all_edges(tetrahedra)
    faces = get_all_faces(tetrahedra)
    
    print(f"Box: {box_min} to {box_max}, L={L}")
    print(f"Vertices: {len(vertices)}")
    print(f"Tetrahedra: {len(tetrahedra)}")
    print(f"Edges: {len(edges)}")
    print(f"Faces: {len(faces)}")
    
    # Check edge lengths
    edge_lengths = []
    for e in edges:
        v0, v1 = vertices[e[0]], vertices[e[1]]
        edge_lengths.append(np.linalg.norm(v1 - v0))
    
    print(f"Edge lengths: min={min(edge_lengths):.4f}, max={max(edge_lengths):.4f}")