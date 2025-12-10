"""
Coxeter Ã_2 triangulation for 2D ambient space.

The Coxeter Ã_2 triangulation tiles R^2 with equilateral triangles.
Basis vectors:
    e_1 = (1, 0) * L
    e_2 = (1/2, sqrt(3)/2) * L

All edges have length L (the longest edge length).
"""

import numpy as np
from typing import Dict, List, Tuple, Set

def generate_coxeter_A2(box_min: np.ndarray, box_max: np.ndarray, L: float) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, int, int]]]:
    """
    Generate Coxeter Ã_2 triangulation covering a bounding box.
    
    Args:
        box_min: Lower-left corner of bounding box
        box_max: Upper-right corner of bounding box
        L: Edge length (all edges are equal in Coxeter Ã_2)
    
    Returns:
        vertices: Dict mapping vertex index to position
        triangles: List of triangles as (i, j, k) vertex indices
    """
    # Basis vectors for Coxeter Ã_2
    e1 = np.array([1.0, 0.0]) * L
    e2 = np.array([0.5, np.sqrt(3)/2]) * L
    
    # Extend box slightly to ensure coverage
    margin = 2 * L
    x_min, y_min = box_min[0] - margin, box_min[1] - margin
    x_max, y_max = box_max[0] + margin, box_max[1] + margin
    
    # Compute grid range
    # A point at grid (i, j) has position i*e1 + j*e2
    # We need to find range of i, j that covers the box
    
    # Height of one row
    row_height = e2[1]
    j_min = int(np.floor(y_min / row_height)) - 1
    j_max = int(np.ceil(y_max / row_height)) + 1
    
    # Width depends on j due to the offset
    i_min = int(np.floor((x_min - j_max * e2[0]) / e1[0])) - 1
    i_max = int(np.ceil((x_max - j_min * e2[0]) / e1[0])) + 1
    
    # Generate vertices
    vertices = {}
    vertex_index = {}  # (i, j) -> index
    idx = 0
    
    for j in range(j_min, j_max + 1):
        for i in range(i_min, i_max + 1):
            pos = i * e1 + j * e2
            vertices[idx] = pos
            vertex_index[(i, j)] = idx
            idx += 1
    
    # Generate triangles
    # Each parallelogram (i,j), (i+1,j), (i,j+1), (i+1,j+1) is split into 2 triangles
    # Split along SHORT diagonal for equilateral triangles:
    #   Triangle 1: (i,j), (i+1,j), (i,j+1)
    #   Triangle 2: (i+1,j), (i+1,j+1), (i,j+1)
    triangles = []
    
    for j in range(j_min, j_max):
        for i in range(i_min, i_max):
            if (i, j) in vertex_index and (i+1, j) in vertex_index and \
               (i, j+1) in vertex_index and (i+1, j+1) in vertex_index:
                v00 = vertex_index[(i, j)]
                v10 = vertex_index[(i+1, j)]
                v01 = vertex_index[(i, j+1)]
                v11 = vertex_index[(i+1, j+1)]
                
                # Two triangles per parallelogram
                triangles.append((v00, v10, v01))
                triangles.append((v10, v11, v01))
    
    return vertices, triangles


def get_edges(triangles: List[Tuple[int, int, int]]) -> Set[Tuple[int, int]]:
    """Extract all edges from triangles."""
    edges = set()
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1) % 3]]))
            edges.add(e)
    return edges


def verify_edge_lengths(vertices: Dict[int, np.ndarray], triangles: List[Tuple[int, int, int]], L: float, tol: float = 1e-10) -> bool:
    """Verify all edges have length L."""
    edges = get_edges(triangles)
    for e in edges:
        v0, v1 = vertices[e[0]], vertices[e[1]]
        length = np.linalg.norm(v1 - v0)
        if abs(length - L) > tol:
            print(f"Edge {e}: length = {length}, expected = {L}")
            return False
    return True


if __name__ == "__main__":
    # Test
    box_min = np.array([-1.0, -1.0])
    box_max = np.array([1.0, 1.0])
    L = 0.5
    
    vertices, triangles = generate_coxeter_A2(box_min, box_max, L)
    print(f"Generated {len(vertices)} vertices, {len(triangles)} triangles")
    
    if verify_edge_lengths(vertices, triangles, L):
        print("All edges have correct length L")
    else:
        print("ERROR: Edge length mismatch")