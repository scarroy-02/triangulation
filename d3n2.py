"""
Whitney 3D — Export + Visualization
====================================

Companion to whitney3d.py.  Produces:
  • OBJ / PLY exports for Blender
  • 3D close-ups  (mpl 3D axes)
  • 2D tangent-plane projections  (the clearest view)

Usage:
    python whitney3d_visualize.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.collections import LineCollection, PolyCollection
from whitney3d import (sphere_surface, torus_surface, run,
                        _parametric_surface)


# ═══════════════════════════════════════════════════════════════
# OBJ + PLY Export
# ═══════════════════════════════════════════════════════════════

def export_obj(K, filename, surface_name="WhitneyK"):
    """Export K as Wavefront OBJ (triangles only)."""
    verts = K['K_verts']
    tris = K['K_tris']
    with open(filename, 'w') as f:
        f.write(f"# Whitney triangulation of {surface_name}\n")
        f.write(f"# {len(verts)} vertices, {len(tris)} faces\n")
        f.write(f"o {surface_name}\n")
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for (i0, i1, i2) in tris:
            f.write(f"f {i0+1} {i1+1} {i2+1}\n")
    print(f"  OBJ: {filename}  ({len(verts)} verts, {len(tris)} tris)")


def export_ply(K, filename, surface_name="WhitneyK"):
    """Export K as PLY (Blender, MeshLab, etc.)."""
    verts = K['K_verts']
    tris = K['K_tris']
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"comment Whitney triangulation of {surface_name}\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(tris)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for (i0, i1, i2) in tris:
            f.write(f"3 {i0} {i1} {i2}\n")
    print(f"  PLY: {filename}  ({len(verts)} verts, {len(tris)} tris)")


def export_ambient_edges_obj(T, pverts, surface, consts, filename, radius=None):
    """Export ambient edges near M as OBJ line elements."""
    if radius is None:
        radius = 2.0 * consts['Lmax']
    vert_map, vert_list, lines = {}, [], []
    idx = 0
    for edge in T.edges:
        v1k, v2k = list(edge)
        p1, p2 = pverts[v1k], pverts[v2k]
        mid = 0.5 * (p1 + p2)
        cp = surface.closest_point(mid)
        if cp is not None and np.linalg.norm(mid - cp) < radius:
            for vk in [v1k, v2k]:
                if vk not in vert_map:
                    vert_map[vk] = idx
                    vert_list.append(pverts[vk])
                    idx += 1
            lines.append((vert_map[v1k], vert_map[v2k]))
    with open(filename, 'w') as f:
        f.write(f"# Ambient edges near surface ({len(lines)} edges)\n")
        for v in vert_list:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for (a, b) in lines:
            f.write(f"l {a+1} {b+1}\n")
    print(f"  Ambient OBJ: {filename}  ({len(vert_list)} verts, {len(lines)} edges)")


# ═══════════════════════════════════════════════════════════════
# 3D Close-up (matplotlib)
# ═══════════════════════════════════════════════════════════════

def closeup_3d(surface, T, pverts, K, consts, center, radius,
               elev=20, azim=60, title_extra=""):
    """
    Two-panel 3D close-up:
      Left:  ambient T̃ edges + v(τ¹), v(τ²), v(τ³) + perturbation arrows
      Right: K triangles
    """
    fig = plt.figure(figsize=(20, 10))
    cx, cy, cz = center
    r = radius
    va = [np.array(v) for v in K['K_verts']]

    # Surface patch
    u = np.linspace(0, 2*np.pi, 120)
    v_p = (np.linspace(0, np.pi, 60) if surface.name.startswith('Sphere')
           else np.linspace(0, 2*np.pi, 120))
    U, V = np.meshgrid(u, v_p)
    X, Y, Z = _parametric_surface(surface.name, U, V)
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)
    X[dist > r*1.2] = np.nan

    def setup(ax, title):
        ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=2)
        ax.set_xlabel('x', fontsize=7); ax.set_ylabel('y', fontsize=7)
        ax.set_zlabel('z', fontsize=7); ax.tick_params(labelsize=6)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor('lightgray')

    def nearby(pt):
        return np.linalg.norm(np.array(pt) - center) < r * 1.3

    # ── Left: ambient + intersection points ──
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_wireframe(X, Y, Z, color='#E63946', alpha=0.30, linewidth=0.5,
                       rstride=1, cstride=1)
    segs = []
    for edge in T.edges:
        v1k, v2k = list(edge)
        p1, p2 = pverts[v1k], pverts[v2k]
        if nearby(0.5*(p1+p2)):
            segs.append([p1, p2])
    if segs:
        ax1.add_collection3d(Line3DCollection(
            segs, colors='#457B9D', linewidths=0.6, alpha=0.25))
    for key in T.vertices:
        vo, vn = T.vertices[key], pverts[key]
        if np.linalg.norm(vn-vo) > 1e-10 and (nearby(vo) or nearby(vn)):
            ax1.plot([vo[0],vn[0]], [vo[1],vn[1]], [vo[2],vn[2]],
                     color='#E76F51', linewidth=2.5, alpha=0.8, zorder=6)
            ax1.scatter(*vn, c='#2A9D8F', s=40, zorder=7, depthshade=False,
                        edgecolors='white', linewidths=0.8)
    for pt in K['edge_pts'].values():
        if nearby(pt):
            ax1.scatter(*pt, c='#E63946', s=50, zorder=7, depthshade=False,
                        edgecolors='white', linewidths=0.8)
    for pt in K['face_pts'].values():
        if nearby(pt):
            ax1.scatter(*pt, c='#F4A261', s=35, zorder=7, marker='D',
                        depthshade=False, edgecolors='white', linewidths=0.5)
    for pt in K['tet_pts'].values():
        if nearby(pt):
            ax1.scatter(*pt, c='#264653', s=25, zorder=7, marker='s',
                        depthshade=False, edgecolors='white', linewidths=0.5)
    setup(ax1, "Ambient T̃ + Intersection Points\n"
               "● v(τ¹) on M  ◆ v(τ²) face avg  ■ v(τ³) tet avg\n"
               "Orange arrows = perturbation")

    # ── Right: K triangles ──
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_wireframe(X, Y, Z, color='#E63946', alpha=0.10, linewidth=0.3,
                       rstride=1, cstride=1)
    local_tris = []
    for (i0, i1, i2) in K['K_tris']:
        tc = (va[i0]+va[i1]+va[i2])/3.0
        if np.linalg.norm(tc - center) < r*1.1:
            local_tris.append([va[i0], va[i1], va[i2]])
    if local_tris:
        ax2.add_collection3d(Poly3DCollection(
            local_tris, alpha=0.55, facecolor='#A8DADC',
            edgecolor='#1D3557', linewidths=0.7))
    for pt in K['edge_pts'].values():
        if nearby(pt):
            ax2.scatter(*pt, c='#E63946', s=30, zorder=7, depthshade=False,
                        edgecolors='white', linewidths=0.5)
    setup(ax2, f"Triangulation K (close-up)\n"
               f"{len(local_tris)} triangles in view")

    fig.suptitle(f"Close-up: {surface.name}{title_extra}",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ═══════════════════════════════════════════════════════════════
# 2D Tangent-Plane Projection (clearest view)
# ═══════════════════════════════════════════════════════════════

def tangent_frame(surface, p):
    """Orthonormal frame (e1, e2, n) at p on surface."""
    n = surface.normal(p)
    up = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1, 0, 0.0])
    e1 = np.cross(n, up); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1); e2 /= np.linalg.norm(e2)
    return e1, e2, n


def project_2d(pts, origin, e1, e2):
    """Project 3D points to 2D tangent coordinates."""
    rel = np.atleast_2d(pts) - origin
    return np.column_stack([rel @ e1, rel @ e2])


def tangent_plane_6panel(surface, T, pv, K, consts, center_3d, patch_r,
                          title, filename):
    """
    6-panel 2D tangent-plane projection showing the full algorithm:
      ① Before perturbation    ② After perturbation     ③ Edge-M intersections
      ④ Barycentric subdivision ⑤ K + vertex types       ⑥ Clean K
    """
    uv = T.vertices
    va = [np.array(v) for v in K['K_verts']]
    R = patch_r

    # Tangent frame
    cp = surface.closest_point(center_3d)
    if cp is None: cp = center_3d
    e1, e2, nrm = tangent_frame(surface, cp)
    origin = cp

    def nearby(pt, scale=1.0):
        return np.linalg.norm(np.array(pt) - center_3d) < R * scale

    def proj(pts_3d):
        if len(pts_3d) == 0: return np.empty((0, 2))
        return project_2d(np.array(pts_3d), origin, e1, e2)

    def proj_segs(segs_3d):
        return [proj(np.array(s)) for s in segs_3d]

    # ── Collect local geometry ──
    Lmax = consts['Lmax']
    edges_b, edges_a = [], []
    for edge in T.edges:
        v1k, v2k = list(edge)
        mid = 0.5*(pv[v1k] + pv[v2k])
        if nearby(mid, 1.5):
            edges_b.append([uv[v1k], uv[v2k]])
            edges_a.append([pv[v1k], pv[v2k]])

    verts_near, verts_far = [], []
    for key, v in uv.items():
        if nearby(v, 1.4):
            c = surface.closest_point(v)
            if c is not None and np.linalg.norm(v-c) < 1.5*Lmax:
                verts_near.append(v)
            else:
                verts_far.append(v)

    pert_pairs = [(uv[k], pv[k]) for k in T.vertices
                  if (nearby(uv[k], 1.4) or nearby(pv[k], 1.4))
                  and np.linalg.norm(pv[k]-uv[k]) > 1e-10]

    verts_still = [pv[k] for k in pv
                   if nearby(pv[k], 1.3) and np.linalg.norm(pv[k]-uv[k]) < 1e-10]

    local_ep = {ek: pt for ek, pt in K['edge_pts'].items() if nearby(pt, 1.2)}
    crossing_edges = [[pv[list(ek)[0]], pv[list(ek)[1]]] for ek in local_ep]
    local_fp = {fk: pt for fk, pt in K['face_pts'].items() if nearby(pt, 1.2)}
    local_tp = {tk: pt for tk, pt in K['tet_pts'].items() if nearby(pt, 1.2)}

    local_tris = []
    for (i0, i1, i2) in K['K_tris']:
        tc = (va[i0]+va[i1]+va[i2])/3.0
        if nearby(tc, 1.1):
            local_tris.append((va[i0], va[i1], va[i2]))

    # Surface sample points
    surf_pts = []
    if surface.name.startswith('Sphere'):
        theta = np.linspace(0, 2*np.pi, 800)
        for t in theta:
            p = cp + R*1.3*(np.cos(t)*e1 + np.sin(t)*e2)
            pp = surface.closest_point(p)
            if pp is not None: surf_pts.append(pp)
    else:
        u = np.linspace(0, 2*np.pi, 250)
        v = np.linspace(0, 2*np.pi, 250)
        Rr, rr = 1.0, 0.4
        for ui in u:
            for vi in v:
                p = np.array([(Rr+rr*np.cos(vi))*np.cos(ui),
                               (Rr+rr*np.cos(vi))*np.sin(ui),
                               rr*np.sin(vi)])
                if nearby(p, 1.3): surf_pts.append(p)

    # ── Project ──
    surf_2d = proj(surf_pts)
    eb_2d = proj_segs(edges_b)
    ea_2d = proj_segs(edges_a)
    vnear_2d = proj(verts_near)
    vfar_2d = proj(verts_far) if verts_far else np.empty((0, 2))
    vstill_2d = proj(verts_still) if verts_still else np.empty((0, 2))
    pp_orig_2d = proj([p[0] for p in pert_pairs]) if pert_pairs else np.empty((0,2))
    pp_new_2d = proj([p[1] for p in pert_pairs]) if pert_pairs else np.empty((0,2))
    ep_2d = proj(list(local_ep.values())) if local_ep else np.empty((0,2))
    cross_2d = proj_segs(crossing_edges)
    fp_2d = proj(list(local_fp.values())) if local_fp else np.empty((0,2))
    tp_2d = proj(list(local_tp.values())) if local_tp else np.empty((0,2))
    tris_2d = [proj(np.array([a,b,c])) for (a,b,c) in local_tris]

    # Example chain triangles
    ex_tris = []
    used = []
    for t2 in tris_2d:
        tc = t2.mean(axis=0)
        if np.linalg.norm(tc) < R*0.5:
            if not any(np.linalg.norm(tc-u) < 0.04 for u in used):
                ex_tris.append(t2)
                used.append(tc)
                if len(ex_tris) >= 15: break

    # ── Plot ──
    lim = R * 1.1
    fig, axes = plt.subplots(2, 3, figsize=(27, 18))
    fig.patch.set_facecolor('white')

    def setup(ax, ttl):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(ttl, fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.1)

    def draw_surface(ax, alpha=0.4):
        if len(surf_2d):
            ax.scatter(surf_2d[:,0], surf_2d[:,1], c='#E63946',
                       s=0.3, alpha=alpha, zorder=1)

    # ① Before perturbation
    ax = axes[0, 0]
    draw_surface(ax)
    if eb_2d:
        ax.add_collection(LineCollection(eb_2d, colors='#457B9D',
                                          linewidths=0.7, alpha=0.35))
    if len(vnear_2d):
        ax.scatter(vnear_2d[:,0], vnear_2d[:,1], c='#E76F51', s=50,
                   zorder=5, edgecolors='white', linewidths=0.8,
                   label=f'Near M ({len(vnear_2d)})')
    if len(vfar_2d):
        ax.scatter(vfar_2d[:,0], vfar_2d[:,1], c='#1D3557', s=15,
                   alpha=0.5, zorder=3, label=f'Far ({len(vfar_2d)})')
    ax.legend(fontsize=8, loc='upper right')
    setup(ax, "① BEFORE Perturbation\n"
              "Orange = vertices within 3L_max/2 of M (Case 2)\n"
              "Red cloud = surface M")

    # ② After perturbation
    ax = axes[0, 1]
    draw_surface(ax)
    if ea_2d:
        ax.add_collection(LineCollection(ea_2d, colors='#457B9D',
                                          linewidths=0.7, alpha=0.35))
    for i in range(len(pp_orig_2d)):
        o, n = pp_orig_2d[i], pp_new_2d[i]
        ax.annotate('', xy=n, xytext=o,
                     arrowprops=dict(arrowstyle='->', color='#E76F51',
                                    lw=2.5, mutation_scale=14))
        ax.scatter(*o, c='none', s=50, edgecolors='#E76F51',
                   linewidths=1.5, zorder=4)
        ax.scatter(*n, c='#2A9D8F', s=60, zorder=6,
                   edgecolors='white', linewidths=1.0)
    if len(vstill_2d):
        ax.scatter(vstill_2d[:,0], vstill_2d[:,1], c='#1D3557',
                   s=12, alpha=0.4, zorder=3)
    setup(ax, "② AFTER Perturbation (§5.2)\n"
              "○ → ● pushed away from tangent plane T_pM\n"
              f"{len(pp_orig_2d)} vertices perturbed in view")

    # ③ Edge-M intersections
    ax = axes[0, 2]
    draw_surface(ax, 0.3)
    if ea_2d:
        ax.add_collection(LineCollection(ea_2d, colors='#A8DADC',
                                          linewidths=0.4, alpha=0.25))
    if cross_2d:
        ax.add_collection(LineCollection(cross_2d, colors='#457B9D',
                                          linewidths=1.8, alpha=0.6))
    if len(ep_2d):
        ax.scatter(ep_2d[:,0], ep_2d[:,1], c='#E63946', s=70, zorder=8,
                   edgecolors='white', linewidths=1.2,
                   label=f'v(τ¹) ×{len(ep_2d)}')
    ax.legend(fontsize=9, loc='upper right')
    setup(ax, "③ Edge–Surface Intersections v(τ¹)\n"
              "Each edge of T̃ crossing M has one unique\n"
              "intersection point (Lemma 6.4) — red ●")

    # ④ Barycentric subdivision
    ax = axes[1, 0]
    draw_surface(ax, 0.2)
    if ea_2d:
        ax.add_collection(LineCollection(ea_2d, colors='#A8DADC',
                                          linewidths=0.3, alpha=0.12))
    if len(ep_2d):
        ax.scatter(ep_2d[:,0], ep_2d[:,1], c='#E63946', s=50, zorder=7,
                   edgecolors='white', linewidths=0.8, label='v(τ¹) edge')
    if len(fp_2d):
        ax.scatter(fp_2d[:,0], fp_2d[:,1], c='#F4A261', s=40, zorder=7,
                   marker='D', edgecolors='white', linewidths=0.5,
                   label='v(τ²) face')
    if len(tp_2d):
        ax.scatter(tp_2d[:,0], tp_2d[:,1], c='#264653', s=28, zorder=7,
                   marker='s', edgecolors='white', linewidths=0.5,
                   label='v(τ³) tet')
    for t2 in ex_tris:
        closed = np.vstack([t2, t2[0:1]])
        ax.plot(closed[:,0], closed[:,1], color='#2A9D8F',
                linewidth=2.2, alpha=0.7, zorder=6)
    ax.legend(fontsize=8, loc='upper right')
    setup(ax, "④ Barycentric Subdivision (§6.2)\n"
              "v(τ²) = avg of edge pts on face\n"
              "v(τ³) = avg of edge pts in tet\n"
              "Green = example triangles {●,◆,■}")

    # ⑤ K with vertex types
    ax = axes[1, 1]
    if tris_2d:
        ax.add_collection(PolyCollection(tris_2d, facecolors='#A8DADC',
                                          edgecolors='#1D3557',
                                          linewidths=0.5, alpha=0.5))
    if len(ep_2d):
        ax.scatter(ep_2d[:,0], ep_2d[:,1], c='#E63946', s=30, zorder=8,
                   edgecolors='white', linewidths=0.5)
    if len(fp_2d):
        ax.scatter(fp_2d[:,0], fp_2d[:,1], c='#F4A261', s=22, zorder=8,
                   marker='D', edgecolors='white', linewidths=0.3)
    if len(tp_2d):
        ax.scatter(tp_2d[:,0], tp_2d[:,1], c='#264653', s=16, zorder=8,
                   marker='s', edgecolors='white', linewidths=0.3)
    setup(ax, f"⑤ Triangulation K + vertex types\n"
              f"{len(tris_2d)} triangles in view\n"
              "Each has exactly one ●, one ◆, one ■")

    # ⑥ Clean K
    ax = axes[1, 2]
    if tris_2d:
        ax.add_collection(PolyCollection(tris_2d, facecolors='#A8DADC',
                                          edgecolors='#1D3557',
                                          linewidths=0.6, alpha=0.6))
    setup(ax, f"⑥ Final Triangulation K\n"
              f"{len(tris_2d)} triangles — PL surface\n"
              "Homeomorphic to M (§7)")

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':

    outdir = './'

    # ── Sphere ──
    print("═══ Sphere ═══")
    sph = sphere_surface(r=1.0)
    T1, pv1, K1, c1 = run(sph, L=0.35,
                           bounds=(-1.6,1.6,-1.6,1.6,-1.6,1.6))

    export_obj(K1, f'{outdir}/sphere_K.obj', 'Sphere')
    export_ply(K1, f'{outdir}/sphere_K.ply', 'Sphere')
    export_ambient_edges_obj(T1, pv1, sph, c1,
                              f'{outdir}/sphere_ambient.obj')

    fig = closeup_3d(sph, T1, pv1, K1, c1,
                     center=np.array([1.0, 0.0, 0.0]), radius=0.6,
                     elev=15, azim=10, title_extra=" — +x side")
    fig.savefig(f'{outdir}/sphere_closeup.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    tangent_plane_6panel(
        sph, T1, pv1, K1, c1,
        center_3d=np.array([1.0, 0.0, 0.0]), patch_r=0.55,
        title="Whitney's Algorithm — Sphere(r=1.0)\n"
              "Tangent-plane projection at (+1,0,0)  |  d=3, n=2, L=0.350",
        filename=f'{outdir}/sphere_2d_steps.png')

    tangent_plane_6panel(
        sph, T1, pv1, K1, c1,
        center_3d=np.array([0.0, 0.0, 1.0]), patch_r=0.55,
        title="Whitney's Algorithm — Sphere(r=1.0)\n"
              "Tangent-plane projection at north pole  |  d=3, n=2, L=0.350",
        filename=f'{outdir}/sphere_2d_pole.png')

    # ── Torus ──
    print("\n═══ Torus ═══")
    tor = torus_surface(R=1.0, r=0.4)
    T2, pv2, K2, c2 = run(tor, L=0.22,
                           bounds=(-1.8,1.8,-1.8,1.8,-0.8,0.8))

    export_obj(K2, f'{outdir}/torus_K.obj', 'Torus')
    export_ply(K2, f'{outdir}/torus_K.ply', 'Torus')
    export_ambient_edges_obj(T2, pv2, tor, c2,
                              f'{outdir}/torus_ambient.obj')

    fig = closeup_3d(tor, T2, pv2, K2, c2,
                     center=np.array([1.4, 0.0, 0.0]), radius=0.5,
                     elev=10, azim=0, title_extra=" — outer equator")
    fig.savefig(f'{outdir}/torus_closeup.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    tangent_plane_6panel(
        tor, T2, pv2, K2, c2,
        center_3d=np.array([1.4, 0.0, 0.0]), patch_r=0.40,
        title="Whitney's Algorithm — Torus(R=1.0, r=0.4)\n"
              "Tangent-plane projection at outer equator  |  d=3, n=2, L=0.220",
        filename=f'{outdir}/torus_2d_outer.png')

    tangent_plane_6panel(
        tor, T2, pv2, K2, c2,
        center_3d=np.array([0.6, 0.0, 0.0]), patch_r=0.35,
        title="Whitney's Algorithm — Torus(R=1.0, r=0.4)\n"
              "Tangent-plane projection at inner hole  |  d=3, n=2, L=0.220",
        filename=f'{outdir}/torus_2d_inner.png')

    print("\n✓ All outputs generated.")