"""
Export Whitney 3D meshes using proper Coxeter Ã₃ triangulation.
"""

import numpy as np
from stl import mesh as stl_mesh
import meshio
from coxeter_A3 import generate_coxeter_A3, get_all_edges, get_tetrahedron_edges
from visualize_detailed import (
    perturb_vertices_3d, find_edge_intersection, build_K_surface,
    sphere
)


def export_stl(vertices, faces, filename):
    """Export triangular mesh to STL."""
    n_faces = len(faces)
    stl = stl_mesh.Mesh(np.zeros(n_faces, dtype=stl_mesh.Mesh.dtype))
    for i, (i0, i1, i2) in enumerate(faces):
        stl.vectors[i][0] = vertices[i0]
        stl.vectors[i][1] = vertices[i1]
        stl.vectors[i][2] = vertices[i2]
    stl.save(filename)
    print(f"Saved STL: {filename} ({n_faces} triangles)")


def export_medit(vertices, tetrahedra, filename, triangles=None, tet_labels=None):
    """Export to MEDIT .mesh format."""
    with open(filename, 'w') as f:
        f.write("MeshVersionFormatted 1\n\nDimension 3\n\n")
        
        f.write(f"Vertices\n{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f} 0\n")
        f.write("\n")
        
        if triangles:
            f.write(f"Triangles\n{len(triangles)}\n")
            for i, (i0, i1, i2) in enumerate(triangles):
                f.write(f"{i0+1} {i1+1} {i2+1} 1\n")
            f.write("\n")
        
        if tetrahedra:
            f.write(f"Tetrahedra\n{len(tetrahedra)}\n")
            for i, tet in enumerate(tetrahedra):
                label = tet_labels[i] if tet_labels else 1
                f.write(f"{tet[0]+1} {tet[1]+1} {tet[2]+1} {tet[3]+1} {label}\n")
            f.write("\n")
        
        f.write("End\n")
    print(f"Saved MEDIT: {filename}")


def export_vtk(vertices, tetrahedra, filename, cell_data=None):
    """Export to VTK format."""
    cells = [("tetra", np.array(tetrahedra))]
    mesh = meshio.Mesh(points=vertices, cells=cells, cell_data=cell_data or {})
    mesh.write(filename)
    print(f"Saved VTK: {filename}")


def generate_and_export_coxeter(name, f, grad_f, box_min, box_max, L, output_dir,
                                required_dist_factor=50.0):
    """Generate Whitney mesh using Coxeter Ã₃ and export."""
    
    print(f"\n{'='*60}")
    print(f"Generating: {name} (Coxeter Ã₃)")
    print(f"{'='*60}")
    
    # Generate Coxeter triangulation
    vertices, tetrahedra = generate_coxeter_A3(box_min, box_max, L)
    print(f"Coxeter Ã₃: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra")
    
    # Perturb vertices
    perturbed, info = perturb_vertices_3d(vertices, tetrahedra, f, grad_f, L, required_dist_factor)
    n_moved = sum(1 for i in info.values() if i.get('action') == 'moved')
    print(f"Perturbed: {n_moved} vertices")
    
    # Build K surface
    K_vertices, K_faces, _ = build_K_surface(perturbed, tetrahedra, f, grad_f)
    print(f"K surface: {len(K_vertices)} vertices, {len(K_faces)} faces")
    
    # Convert to arrays
    n_verts = len(perturbed)
    vert_array = np.zeros((n_verts, 3))
    for idx, pos in perturbed.items():
        vert_array[idx] = pos
    
    # Classify tetrahedra
    tet_labels = []
    for tet in tetrahedra:
        signs = [np.sign(f(vert_array[tet[i]])) for i in range(4)]
        if all(s <= 0 for s in signs):
            tet_labels.append(1)  # Inside
        elif all(s >= 0 for s in signs):
            tet_labels.append(2)  # Outside
        else:
            tet_labels.append(3)  # Boundary
    
    n_inside = sum(1 for l in tet_labels if l == 1)
    n_outside = sum(1 for l in tet_labels if l == 2)
    n_boundary = sum(1 for l in tet_labels if l == 3)
    print(f"Inside: {n_inside}, Outside: {n_outside}, Boundary: {n_boundary}")
    
    prefix = f"{output_dir}/{name}_coxeter"
    
    # Export K surface
    if K_faces:
        K_arr = np.array(K_vertices)
        export_stl(K_arr, K_faces, f"{prefix}_K.stl")
    
    # Export full mesh
    export_medit(vert_array, tetrahedra, f"{prefix}_full.mesh", tet_labels=tet_labels)
    export_vtk(vert_array, tetrahedra, f"{prefix}_full.vtk", 
               cell_data={"region": [np.array(tet_labels)]})
    
    # Export boundary tetrahedra only
    boundary_tets = [tet for tet, label in zip(tetrahedra, tet_labels) if label == 3]
    if boundary_tets:
        export_medit(vert_array, boundary_tets, f"{prefix}_boundary.mesh")
    
    # Export inside tetrahedra only
    inside_tets = [tet for tet, label in zip(tetrahedra, tet_labels) if label == 1]
    if inside_tets:
        export_medit(vert_array, inside_tets, f"{prefix}_inside.mesh")
    
    return {
        'vertices': vert_array,
        'tetrahedra': tetrahedra,
        'K_vertices': K_vertices,
        'K_faces': K_faces,
        'tet_labels': tet_labels
    }


def torus(R=1.0, r=0.4):
    """Torus with major radius R and minor radius r."""
    def f(p):
        x, y, z = p
        return (np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2
    def grad_f(p):
        x, y, z = p
        rho = np.sqrt(x**2 + y**2)
        if rho < 1e-12:
            return np.array([0.0, 0.0, 2*z])
        factor = 2 * (rho - R) / rho
        return np.array([factor * x, factor * y, 2*z])
    return f, grad_f, r


if __name__ == "__main__":
    output_dir = "./"
    
    # Sphere with Coxeter triangulation
    # f, grad_f, _ = sphere(radius=1.0)
    # generate_and_export_coxeter(
    #     "sphere",
    #     f, grad_f,
    #     box_min=np.array([-1.3, -1.3, -1.3]),
    #     box_max=np.array([1.3, 1.3, 1.3]),
    #     L=0.25,
    #     output_dir=output_dir,
    #     required_dist_factor=50
    # )
    
    # # Torus with Coxeter triangulation
    # f, grad_f, _ = torus(R=1.0, r=0.4)
    # generate_and_export_coxeter(
    #     "torus",
    #     f, grad_f,
    #     box_min=np.array([-1.6, -1.6, -0.6]),
    #     box_max=np.array([1.6, 1.6, 0.6]),
    #     L=0.2,
    #     output_dir=output_dir,
    #     required_dist_factor=50
    # )
    
    # Small section of sphere for detailed viewing
    f, grad_f, _ = sphere(radius=1.0)
    generate_and_export_coxeter(
        "sphere_section",
        f, grad_f,
        box_min=np.array([-0.3, -0.3, 0.8]),
        box_max=np.array([0.3, 0.3, 1.1]),
        L=0.05,
        output_dir=output_dir,
        required_dist_factor=80
    )
    
    print("\n" + "="*60)
    print("All Coxeter meshes exported!")
    print("="*60)
    print("\nFiles:")
    print("  *_coxeter_K.stl      - K surface (Whitney approximation)")
    print("  *_coxeter_full.mesh  - Full tetrahedral mesh")
    print("  *_coxeter_full.vtk   - Full mesh (ParaView)")
    print("  *_coxeter_boundary.mesh - Boundary tetrahedra only")
    print("  *_coxeter_inside.mesh   - Inside tetrahedra only")