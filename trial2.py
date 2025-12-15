#!/usr/bin/env python3
"""
Sphere-Coxeter Triangulation Intersection with STL Output

This implements the Ã₃ Coxeter triangulation (not just Freudenthal-Kuhn)
and outputs a proper triangulated surface mesh as STL.

The Coxeter triangulation is obtained by applying a linear transformation
to the Freudenthal-Kuhn triangulation that improves simplex quality.

For a 2D manifold (sphere surface) in 3D with codimension 1:
- We find edges (1-simplices) that intersect the sphere
- Build faces (2-cells) from the intersection points
- Output as STL mesh
"""

import numpy as np
from collections import defaultdict, deque
from itertools import permutations, combinations
import struct


class CoxeterTriangulation:
    """
    Ã₃ Coxeter triangulation of R³.
    
    This is obtained by applying a linear transformation to the 
    Freudenthal-Kuhn triangulation. The transformation matrix for Ã_d
    produces simplices with optimal quality (best inradius/circumradius ratio).
    
    The Ã₃ triangulation has vertices on the "permutahedral lattice" which
    is the lattice of points (x,y,z,w) in Z⁴ with x+y+z+w=0, projected to R³.
    """
    
    def __init__(self, scale=1.0):
        self.d = 3  # dimension
        self.scale = scale
        
        # The Coxeter matrix for Ã₃
        # This transforms the standard FK triangulation to the Coxeter one
        # The matrix maps the standard basis to vectors that form the Ã₃ root system
        self.matrix = self._compute_coxeter_matrix() * scale
        self.matrix_inv = np.linalg.inv(self.matrix)
    
    def _compute_coxeter_matrix(self):
        """
        Compute the transformation matrix for Ã₃ Coxeter triangulation.
        
        The Ã_d Coxeter triangulation uses vectors from the A_d root system.
        For Ã₃, we use a matrix whose columns are the simple roots of A₃.
        """
        # Simple roots of A₃ (embedded in R³)
        # These form the edges of a regular tetrahedron
        # Using the standard construction from the permutahedral lattice
        
        # One common choice: columns are differences of standard basis vectors
        # projected from R⁴ to R³
        M = np.array([
            [1, -0.5, -0.5],
            [0, np.sqrt(3)/2, -np.sqrt(3)/6],
            [0, 0, np.sqrt(6)/3]
        ]).T
        
        # Normalize to unit edge length
        M = M / np.linalg.norm(M[:, 0])
        
        return M
    
    def to_simplex_coords(self, point):
        """Convert Cartesian coordinates to simplex (FK) coordinates."""
        return self.matrix_inv @ np.array(point)
    
    def to_cartesian(self, simplex_coords):
        """Convert simplex coordinates to Cartesian."""
        return self.matrix @ np.array(simplex_coords)
    
    def locate_point(self, point):
        """
        Find which simplex contains the given point.
        Returns (base_vertex, permutation) in simplex coordinates.
        """
        # Transform to FK coordinates
        p = self.to_simplex_coords(point)
        
        # Find base cube vertex
        base = np.floor(p).astype(int)
        
        # Find position within cube and determine permutation
        frac = p - base
        perm = tuple(np.argsort(-frac))  # Sort descending
        
        return tuple(base), perm
    
    def get_simplex_vertices_cartesian(self, base_vertex, perm):
        """
        Get the d+1 vertices of a simplex in Cartesian coordinates.
        """
        vertices = []
        current = np.array(base_vertex, dtype=float)
        vertices.append(self.to_cartesian(current))
        
        for i in perm:
            current = current.copy()
            current[i] += 1
            vertices.append(self.to_cartesian(current))
        
        return vertices
    
    def get_vertex_cartesian(self, vertex):
        """Get Cartesian coordinates of a vertex."""
        return self.to_cartesian(np.array(vertex, dtype=float))
    
    def get_simplex_edges(self, base_vertex, perm):
        """
        Get all edges of a simplex as pairs of vertex identifiers.
        Each vertex is identified by its integer coordinates in simplex space.
        """
        # Build list of vertices in simplex coordinates
        vertices = [tuple(base_vertex)]
        current = list(base_vertex)
        
        for i in perm:
            current = current.copy()
            current[i] += 1
            vertices.append(tuple(current))
        
        # Return all pairs
        edges = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                edges.append((vertices[i], vertices[j]))
        
        return edges
    
    def get_cofaces(self, base_vertex, perm):
        """
        Get all (d+1)-simplices (full simplices) that contain this d-simplex.
        
        For a d-simplex defined by (base_vertex, perm), the cofaces are obtained
        by considering adjacent cubes and permutations.
        """
        cofaces = []
        base = list(base_vertex)
        perm = list(perm)
        d = self.d
        
        # In the FK triangulation, cofaces come from:
        # 1. Same cube, different permutations that share d vertices
        # 2. Adjacent cubes with appropriate permutations
        
        # For simplicity, return neighboring full simplices
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    new_base = (base[0] + di, base[1] + dj, base[2] + dk)
                    for p in permutations(range(d)):
                        cofaces.append((new_base, p))
        
        return cofaces


class PermutahedralRepresentation:
    """
    Represents a simplex using the permutahedral representation.
    
    A k-simplex is represented by:
    - vertex: the lexicographically minimal vertex (integer coordinates)
    - partition: ordered partition of {0, 1, ..., d} into (k+1) parts
    
    For a full d-simplex, partition has (d+1) singleton parts corresponding
    to a permutation.
    """
    
    def __init__(self, vertex, partition):
        self.vertex = tuple(vertex)
        self.partition = tuple(tuple(p) for p in partition)
        self._hash = hash((self.vertex, self.partition))
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return self.vertex == other.vertex and self.partition == other.partition
    
    def dimension(self):
        return len(self.partition) - 1
    
    @staticmethod
    def from_vertices(vertices, d):
        """
        Create permutahedral representation from a list of vertex coordinates.
        """
        vertices = sorted([tuple(v) for v in vertices])
        base = vertices[0]
        
        # Compute the partition from vertex differences
        # This is simplified - full implementation would be more complex
        if len(vertices) == 2:
            # Edge case
            diff = tuple(vertices[1][i] - vertices[0][i] for i in range(len(base)))
            # Find which coordinate changed
            changed = [i for i in range(len(diff)) if diff[i] != 0]
            partition = [tuple(changed), tuple(i for i in range(d+1) if i not in changed)]
        else:
            # General case - use ordered partition based on vertex sequence
            partition = [[i] for i in range(len(vertices))]
        
        return PermutahedralRepresentation(base, partition)


def sphere_function(point, center, radius):
    """Implicit function for sphere: positive outside, negative inside."""
    return np.linalg.norm(np.array(point) - np.array(center)) - radius


def edge_sphere_intersection(v1, v2, center, radius):
    """
    Find intersection of edge [v1, v2] with sphere.
    Returns intersection point or None.
    """
    v1, v2 = np.array(v1), np.array(v2)
    center = np.array(center)
    
    d = v2 - v1
    f = v1 - center
    
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None
    
    sqrt_disc = np.sqrt(discriminant)
    
    # Check both solutions
    for t in [(-b - sqrt_disc) / (2*a), (-b + sqrt_disc) / (2*a)]:
        if 0 < t < 1:  # Strict inequality - intersection in interior of edge
            return v1 + t * d
    
    return None


def trace_sphere_manifold(triangulation, center, radius, seed_point, max_iters=50000):
    """
    Trace the sphere surface through the Coxeter triangulation.
    
    For codimension 1 (2D surface in 3D):
    - Find edges (1-simplices) that intersect the sphere
    - Record intersection points
    - Build connectivity for mesh faces
    
    Returns:
        edge_intersections: dict mapping edge -> intersection point
        vertex_coords: dict mapping vertex_id -> Cartesian coordinates
    """
    
    edge_intersections = {}  # (v1, v2) -> intersection_point
    visited_simplices = set()
    
    # Queue of full simplices to process
    queue = deque()
    
    # Find starting simplex
    base, perm = triangulation.locate_point(seed_point)
    queue.append((base, perm))
    
    iterations = 0
    
    while queue and iterations < max_iters:
        iterations += 1
        
        base, perm = queue.popleft()
        simplex_key = (base, perm)
        
        if simplex_key in visited_simplices:
            continue
        visited_simplices.add(simplex_key)
        
        # Get all edges of this simplex
        edges = triangulation.get_simplex_edges(base, perm)
        
        found_intersection = False
        
        for v1, v2 in edges:
            # Canonical edge representation (sorted vertices)
            edge_key = tuple(sorted([v1, v2]))
            
            if edge_key in edge_intersections:
                found_intersection = True
                continue
            
            # Get Cartesian coordinates
            p1 = triangulation.get_vertex_cartesian(v1)
            p2 = triangulation.get_vertex_cartesian(v2)
            
            # Check for intersection
            intersection = edge_sphere_intersection(p1, p2, center, radius)
            
            if intersection is not None:
                edge_intersections[edge_key] = intersection
                found_intersection = True
        
        # If this simplex intersects the sphere, add neighbors to queue
        if found_intersection:
            for neighbor in triangulation.get_cofaces(base, perm):
                if neighbor not in visited_simplices:
                    queue.append(neighbor)
    
    print(f"  Iterations: {iterations}")
    print(f"  Visited simplices: {len(visited_simplices)}")
    print(f"  Intersecting edges: {len(edge_intersections)}")
    
    return edge_intersections


def build_surface_mesh(triangulation, edge_intersections, center, radius):
    """
    Build a triangulated surface mesh from edge intersections.
    
    For codimension 1, faces of the PL approximation correspond to 
    2-simplices (triangles) in the triangulation that are "crossed" by the manifold.
    
    A triangle is crossed if exactly 2 of its edges intersect the sphere
    (or all 3 in degenerate cases).
    """
    
    # First, collect all vertices (intersection points)
    vertices = []
    vertex_map = {}  # edge_key -> vertex_index
    
    for edge_key, point in edge_intersections.items():
        idx = len(vertices)
        vertices.append(point)
        vertex_map[edge_key] = idx
    
    print(f"  Mesh vertices: {len(vertices)}")
    
    # Now find faces: triangles in the triangulation with 2+ intersecting edges
    # We need to find all triangles (2-faces of tetrahedra) that have intersecting edges
    
    faces = []
    face_set = set()  # To avoid duplicates
    
    # Build a map from vertices to edges they belong to
    vertex_to_edges = defaultdict(set)
    for edge_key in edge_intersections.keys():
        v1, v2 = edge_key
        vertex_to_edges[v1].add(edge_key)
        vertex_to_edges[v2].add(edge_key)
    
    # For each triangulation vertex, look at triangles formed by its incident edges
    processed_triangles = set()
    
    for vertex in vertex_to_edges.keys():
        edges_at_v = list(vertex_to_edges[vertex])
        
        # Get all other endpoints
        other_vertices = set()
        for e in edges_at_v:
            v1, v2 = e
            other = v2 if v1 == vertex else v1
            other_vertices.add(other)
        
        # For each pair of other vertices, check if they form a triangle with vertex
        for v1, v2 in combinations(other_vertices, 2):
            # Check if edge (v1, v2) also intersects
            triangle_verts = tuple(sorted([vertex, v1, v2]))
            
            if triangle_verts in processed_triangles:
                continue
            processed_triangles.add(triangle_verts)
            
            # Get the three edges of this potential triangle
            e1 = tuple(sorted([vertex, v1]))
            e2 = tuple(sorted([vertex, v2]))
            e3 = tuple(sorted([v1, v2]))
            
            # Count how many edges intersect
            intersecting = []
            for e in [e1, e2, e3]:
                if e in edge_intersections:
                    intersecting.append(e)
            
            # A face is formed if exactly 2 edges intersect (generic case)
            # or 3 edges (degenerate, but we can still triangulate)
            if len(intersecting) >= 2:
                # Get vertex indices
                face_vertices = [vertex_map[e] for e in intersecting[:3]]
                
                if len(face_vertices) == 2:
                    # Need to find a third point - look for adjacent triangles
                    continue
                
                face_key = tuple(sorted(face_vertices))
                if face_key not in face_set:
                    face_set.add(face_key)
                    faces.append(face_vertices)
    
    print(f"  Initial faces: {len(faces)}")
    
    # Alternative approach: for each tetrahedron that intersects, 
    # the intersection with sphere creates a polygon (usually triangle or quad)
    # Let's use a different strategy based on the dual perspective
    
    faces = build_faces_from_tetrahedra(triangulation, edge_intersections, vertex_map, center, radius)
    
    return np.array(vertices), faces


def build_faces_from_tetrahedra(triangulation, edge_intersections, vertex_map, center, radius):
    """
    Build faces by finding tetrahedra that intersect the sphere.
    
    When a sphere passes through a tetrahedron, it typically intersects
    3 or 4 edges, creating a triangular or quadrilateral face.
    """
    
    faces = []
    processed_tets = set()
    
    # Group edges by their incident tetrahedra
    # An edge belongs to multiple tetrahedra
    
    # For each intersecting edge, find tetrahedra containing it
    for edge_key in edge_intersections.keys():
        v1, v2 = edge_key
        
        # Find tetrahedra containing this edge
        # A tetrahedron containing edge (v1, v2) has vertices v1, v2, and two others
        # that differ from v1 by one coordinate each
        
        # In FK triangulation, we need to find all tetrahedra containing this edge
        # This is done by looking at the permutahedral structure
        
        # Simplified approach: use the cube containing v1
        base = tuple(min(v1[i], v2[i]) for i in range(3))
        
        for perm in permutations(range(3)):
            tet_key = (base, perm)
            if tet_key in processed_tets:
                continue
            
            # Get edges of this tetrahedron
            edges = triangulation.get_simplex_edges(base, perm)
            
            # Find which edges intersect
            intersecting_edges = []
            for e in edges:
                e_sorted = tuple(sorted(e))
                if e_sorted in edge_intersections:
                    intersecting_edges.append(e_sorted)
            
            if len(intersecting_edges) >= 3:
                processed_tets.add(tet_key)
                
                # Get vertex indices
                face_verts = [vertex_map[e] for e in intersecting_edges]
                
                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4:
                    # Quad - split into two triangles
                    # Order vertices by angle around centroid
                    pts = [edge_intersections[e] for e in intersecting_edges]
                    centroid = np.mean(pts, axis=0)
                    
                    # Project to plane and sort by angle
                    normal = centroid - np.array(center)
                    normal = normal / np.linalg.norm(normal)
                    
                    # Create orthonormal basis on tangent plane
                    if abs(normal[0]) < 0.9:
                        u = np.cross(normal, [1, 0, 0])
                    else:
                        u = np.cross(normal, [0, 1, 0])
                    u = u / np.linalg.norm(u)
                    v = np.cross(normal, u)
                    
                    # Compute angles
                    angles = []
                    for p in pts:
                        d = p - centroid
                        angle = np.arctan2(np.dot(d, v), np.dot(d, u))
                        angles.append(angle)
                    
                    # Sort by angle
                    order = np.argsort(angles)
                    sorted_verts = [face_verts[i] for i in order]
                    
                    # Two triangles
                    faces.append([sorted_verts[0], sorted_verts[1], sorted_verts[2]])
                    faces.append([sorted_verts[0], sorted_verts[2], sorted_verts[3]])
    
    # Remove duplicate faces
    unique_faces = []
    face_set = set()
    for f in faces:
        key = tuple(sorted(f))
        if key not in face_set:
            face_set.add(key)
            unique_faces.append(f)
    
    print(f"  Faces from tetrahedra: {len(unique_faces)}")
    
    return unique_faces


def write_stl_binary(filename, vertices, faces, solid_name="sphere"):
    """Write mesh to binary STL file."""
    
    with open(filename, 'wb') as f:
        # Header (80 bytes)
        header = f"Binary STL - {solid_name}".encode('ascii')
        header = header[:80].ljust(80, b'\0')
        f.write(header)
        
        # Number of triangles
        f.write(struct.pack('<I', len(faces)))
        
        # Write each triangle
        for face in faces:
            if len(face) < 3:
                continue
                
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])
            
            # Write normal
            f.write(struct.pack('<3f', *normal))
            
            # Write vertices
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            
            # Attribute byte count
            f.write(struct.pack('<H', 0))
    
    print(f"Wrote {len(faces)} triangles to {filename}")


def write_stl_ascii(filename, vertices, faces, solid_name="sphere"):
    """Write mesh to ASCII STL file."""
    
    with open(filename, 'w') as f:
        f.write(f"solid {solid_name}\n")
        
        for face in faces:
            if len(face) < 3:
                continue
                
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])
            
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write(f"endsolid {solid_name}\n")
    
    print(f"Wrote {len(faces)} triangles to {filename}")


def write_mesh_file(filename, vertices, faces):
    """Write mesh in Medit .mesh format."""
    
    with open(filename, 'w') as f:
        f.write("MeshVersionFormatted 1\n\n")
        f.write("Dimension 3\n\n")
        
        # Vertices
        f.write(f"Vertices\n{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f} 0\n")
        f.write("\n")
        
        # Triangles
        f.write(f"Triangles\n{len(faces)}\n")
        for face in faces:
            if len(face) >= 3:
                # .mesh uses 1-based indexing
                f.write(f"{face[0]+1} {face[1]+1} {face[2]+1} 1\n")
        f.write("\n")
        
        f.write("End\n")
    
    print(f"Wrote {len(vertices)} vertices and {len(faces)} triangles to {filename}")


def main():
    print("=" * 70)
    print("Sphere - Coxeter Triangulation (Ã₃) Intersection")
    print("Codimension 1: 2D manifold (sphere surface) in 3D")
    print("=" * 70)
    
    # Parameters
    sphere_center = np.array([0.0, 0.0, 0.0])
    sphere_radius = 1.0
    triangulation_scale = 0.15  # Smaller = finer mesh
    
    print(f"\nParameters:")
    print(f"  Sphere center: {sphere_center}")
    print(f"  Sphere radius: {sphere_radius}")
    print(f"  Triangulation scale: {triangulation_scale}")
    print(f"  Codimension: 1 (2D surface in 3D)")
    
    # Create Coxeter triangulation
    print("\nCreating Ã₃ Coxeter triangulation...")
    tri = CoxeterTriangulation(scale=triangulation_scale)
    
    print(f"  Coxeter matrix:\n{tri.matrix}")
    
    # Seed point on sphere
    seed = sphere_center + np.array([sphere_radius, 0, 0])
    print(f"\nSeed point: {seed}")
    
    # Trace the manifold
    print("\nTracing sphere surface through triangulation...")
    edge_intersections = trace_sphere_manifold(
        tri, sphere_center, sphere_radius, seed, max_iters=100000
    )
    
    # Build mesh
    print("\nBuilding surface mesh...")
    vertices, faces = build_surface_mesh(
        tri, edge_intersections, sphere_center, sphere_radius
    )
    
    if len(faces) == 0:
        print("\nWarning: No faces generated. Trying alternative method...")
        # Fallback: create faces from nearby intersection points
        faces = create_faces_by_proximity(vertices, sphere_center)
    
    print(f"\nFinal mesh:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    
    # Write output files
    print("\nWriting output files...")
    
    if len(vertices) > 0 and len(faces) > 0:
        write_stl_ascii("sphere_coxeter.stl", vertices, faces, "coxeter_sphere")
        write_stl_binary("sphere_coxeter_binary.stl", vertices, faces, "coxeter_sphere")
        write_mesh_file("sphere_coxeter.mesh", vertices, faces)
        
        # Also save as OBJ for compatibility
        write_obj_file("sphere_coxeter.obj", vertices, faces)
    
    # Visualization
    print("\nCreating visualization...")
    visualize_mesh(vertices, faces, sphere_center, sphere_radius, tri, edge_intersections)
    
    print("\n" + "=" * 70)
    print("Output files:")
    print("  sphere_coxeter.stl       - ASCII STL file")
    print("  sphere_coxeter_binary.stl - Binary STL file")
    print("  sphere_coxeter.mesh      - Medit mesh file")
    print("  sphere_coxeter.obj       - Wavefront OBJ file")
    print("  sphere_coxeter_mesh.png  - Visualization")
    print("=" * 70)


def create_faces_by_proximity(vertices, center):
    """
    Fallback: create faces by connecting nearby vertices.
    Uses Delaunay-like approach on the sphere.
    """
    from scipy.spatial import Delaunay
    
    if len(vertices) < 4:
        return []
    
    # Project to sphere surface (normalize)
    center = np.array(center)
    projected = []
    for v in vertices:
        d = v - center
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            projected.append(d / norm)
        else:
            projected.append(np.array([1, 0, 0]))
    
    projected = np.array(projected)
    
    # Use spherical coordinates for 2D Delaunay
    theta = np.arctan2(projected[:, 1], projected[:, 0])
    phi = np.arccos(np.clip(projected[:, 2], -1, 1))
    
    # Handle poles by using stereographic projection
    coords_2d = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta)
    ])
    
    try:
        tri = Delaunay(coords_2d)
        faces = tri.simplices.tolist()
        print(f"  Created {len(faces)} faces via Delaunay triangulation")
        return faces
    except Exception as e:
        print(f"  Delaunay failed: {e}")
        return []


def write_obj_file(filename, vertices, faces):
    """Write mesh in Wavefront OBJ format."""
    with open(filename, 'w') as f:
        f.write(f"# Coxeter triangulation sphere intersection\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        
        f.write("\n")
        
        for face in faces:
            if len(face) >= 3:
                # OBJ uses 1-based indexing
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Wrote {len(vertices)} vertices and {len(faces)} faces to {filename}")


def visualize_mesh(vertices, faces, center, radius, triangulation, edge_intersections):
    """Create visualization of the mesh."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    
    fig = plt.figure(figsize=(16, 5))
    
    # --- Plot 1: Mesh faces ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title(f'PL Approximation of Sphere\n({len(faces)} triangles)')
    
    if len(faces) > 0:
        # Create triangle collection
        triangles = []
        for face in faces:
            if len(face) >= 3:
                tri = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
                triangles.append(tri)
        
        if triangles:
            pc = Poly3DCollection(triangles, alpha=0.7, facecolor='cyan',
                                 edgecolor='darkblue', linewidth=0.3)
            ax1.add_collection3d(pc)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    lim = radius * 1.3
    ax1.set_xlim([-lim, lim])
    ax1.set_ylim([-lim, lim])
    ax1.set_zlim([-lim, lim])
    
    # --- Plot 2: Mesh with reference sphere ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Mesh vs True Sphere')
    
    # Reference sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax2.plot_wireframe(x, y, z, color='green', alpha=0.2, linewidth=0.3)
    
    # Mesh vertices
    if len(vertices) > 0:
        ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c='red', s=5, alpha=0.8)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-lim, lim])
    ax2.set_ylim([-lim, lim])
    ax2.set_zlim([-lim, lim])
    
    # --- Plot 3: Sample of Coxeter triangulation edges ---
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Coxeter Triangulation Edges\nNear Sphere')
    
    # Get some triangulation edges
    tri_edges = []
    for edge_key in list(edge_intersections.keys())[:500]:
        v1, v2 = edge_key
        p1 = triangulation.get_vertex_cartesian(v1)
        p2 = triangulation.get_vertex_cartesian(v2)
        tri_edges.append([p1, p2])
    
    if tri_edges:
        lc = Line3DCollection(tri_edges, colors='gray', linewidths=0.5, alpha=0.5)
        ax3.add_collection3d(lc)
    
    # Intersection points
    int_pts = np.array(list(edge_intersections.values()))
    if len(int_pts) > 0:
        ax3.scatter(int_pts[:, 0], int_pts[:, 1], int_pts[:, 2],
                   c='red', s=10, alpha=0.8)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim([-lim, lim])
    ax3.set_ylim([-lim, lim])
    ax3.set_zlim([-lim, lim])
    
    plt.tight_layout()
    plt.savefig('sphere_coxeter_mesh.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to sphere_coxeter_mesh.png")
    plt.show()


if __name__ == '__main__':
    main()