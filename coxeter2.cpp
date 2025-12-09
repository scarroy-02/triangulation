/**
 * Whitney's Triangulation Algorithm - C++ Implementation
 * =======================================================
 * 
 * Based on: "Triangulating submanifolds: An elementary and quantified version 
 * of Whitney's method" by Boissonnat, Kachanovich, Wintraecken (2021)
 * 
 * Compile:
 *   g++ -O3 -std=c++17 -I/usr/include/eigen3 whitney_cpp.cpp -o whitney_cpp
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <array>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <functional>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// =============================================================================
// Algorithm Constants (Section 5.1)
// =============================================================================

long long stirling2(int n, int k) {
    if (k == 0) return (n == 0) ? 1 : 0;
    if (k == n || k == 1) return 1;
    if (k > n) return 0;
    vector<vector<long long>> S(n + 1, vector<long long>(k + 1, 0));
    S[0][0] = 1;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= min(i, k); j++)
            S[i][j] = j * S[i-1][j] + S[i-1][j-1];
    return S[n][k];
}

long long factorial(int n) {
    long long r = 1;
    for (int i = 2; i <= n; i++) r *= i;
    return r;
}

struct Constants {
    int d, n;
    double reach, thickness, mu_0, delta, rho_1, c_tilde, L, L_theoretical;
    long long N;
    
    void compute(int d_, int n_, double reach_) {
        d = d_; n = n_; reach = reach_;
        
        // Thickness for Ã_d
        thickness = (d % 2 == 1) ? sqrt(2.0 / d) : sqrt(2.0 * (d + 1) / (d * (d + 2)));
        mu_0 = sqrt((d + 1.0) / (2.0 * d));
        delta = (sqrt(d*d + 2*d + 24.0) - sqrt(d*d + 2.0*d)) / sqrt(12.0 * (d + 1));
        
        // Equation (4): N_{≤(d-n-1)}
        int k = d - n - 1;
        N = 2;
        for (int j = 1; j <= k + 1; j++)
            N += factorial(j) * stirling2(d + 1, j);
        
        // Equation (5): ρ₁
        if (d % 2 == 0) {
            int kk = d / 2;
            rho_1 = pow(2.0, 2*kk - 2) * pow(factorial(kk), 2) / (M_PI * factorial(2*kk) * N);
        } else {
            int kk = (d + 1) / 2;
            rho_1 = (double)factorial(2*kk - 2) / (pow(2.0, 2*kk) * factorial(kk) * factorial(kk - 1) * N);
        }
        
        // Solve coupled system for L and c̃
        L = reach / 10.0;
        for (int iter = 0; iter < 100; iter++) {
            // Equation (6)
            double term1 = thickness * mu_0 * delta / (18.0 * d * L);
            double term2 = thickness * thickness / 24.0;
            c_tilde = min({term1, term2, 1.0/24.0});
            
            // Equation (8): α_k
            double alpha_1 = (4.0/3.0) * rho_1 * c_tilde;
            int codim = d - n;
            double alpha_k = alpha_1;
            for (int i = 1; i < codim; i++)
                alpha_k *= (2.0/3.0) * c_tilde * rho_1;
            
            // Equation (13)
            double L_bound = pow(alpha_k, codim) / 54.0 * reach;
            L_theoretical = L_bound;
            
            if (L <= L_bound) break;
            L = L_bound * 0.99;
            if (L < reach * 1e-15) { L = reach * 1e-15; break; }
        }
        
        // Recompute c̃
        double term1 = thickness * mu_0 * delta / (18.0 * d * L);
        double term2 = thickness * thickness / 24.0;
        c_tilde = min({term1, term2, 1.0/24.0});
    }
    
    void print() const {
        cout << "\n============================================================\n";
        cout << "ALGORITHM CONSTANTS (Section 5.1)\n";
        cout << "============================================================\n";
        cout << "d = " << d << ", n = " << n << ", reach = " << reach << "\n";
        cout << "t(T) = " << thickness << ", μ₀ = " << mu_0 << ", δ = " << delta << "\n";
        cout << "N_{≤" << (d-n-1) << "} = " << N << " (eq 4)\n";
        cout << "ρ₁ = " << rho_1 << " (eq 5)\n";
        cout << "c̃ = " << c_tilde << " (eq 6)\n";
        cout << "L_theoretical = " << L_theoretical << " (eq 13)\n";
        cout << "L/reach = " << L_theoretical / reach << "\n";
    }
};

// =============================================================================
// Mesh Output
// =============================================================================

void write_mesh(const string& filename, const vector<Vector3d>& V, 
                const vector<array<int,3>>& T) {
    ofstream ofs(filename);
    ofs << "MeshVersionFormatted 1\nDimension 3\n\n";
    ofs << "Vertices\n" << V.size() << "\n";
    for (const auto& v : V)
        ofs << v(0) << " " << v(1) << " " << v(2) << " 0\n";
    ofs << "\nTriangles\n" << T.size() << "\n";
    for (const auto& t : T)
        ofs << (t[0]+1) << " " << (t[1]+1) << " " << (t[2]+1) << " 0\n";
    ofs << "\nEnd\n";
}

void write_stl(const string& filename, const vector<Vector3d>& V,
               const vector<array<int,3>>& T) {
    ofstream ofs(filename, ios::binary);
    char header[80] = {0};
    ofs.write(header, 80);
    uint32_t num = T.size();
    ofs.write((char*)&num, 4);
    
    for (const auto& tri : T) {
        Vector3d v0 = V[tri[0]], v1 = V[tri[1]], v2 = V[tri[2]];
        Vector3d n = (v1 - v0).cross(v2 - v0);
        double norm = n.norm();
        if (norm > 1e-14) n /= norm;
        
        float data[12] = {(float)n(0), (float)n(1), (float)n(2),
                          (float)v0(0), (float)v0(1), (float)v0(2),
                          (float)v1(0), (float)v1(1), (float)v1(2),
                          (float)v2(0), (float)v2(1), (float)v2(2)};
        ofs.write((char*)data, 48);
        uint16_t attr = 0;
        ofs.write((char*)&attr, 2);
    }
}

// =============================================================================
// Whitney Triangulation
// =============================================================================

struct EdgeKey {
    array<int,3> a, b;
    EdgeKey(array<int,3> x, array<int,3> y) {
        if (x > y) swap(x, y);
        a = x; b = y;
    }
    bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};

struct EdgeHash {
    size_t operator()(const EdgeKey& e) const {
        size_t h = 0;
        for (int i = 0; i < 3; i++) {
            h ^= hash<int>()(e.a[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= hash<int>()(e.b[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

void triangulate(const string& name,
                 function<double(const Vector3d&)> f,
                 function<Vector3d(const Vector3d&)> grad_f,
                 double reach,
                 const Vector3d& box_min, const Vector3d& box_max,
                 double L_override = -1) {
    
    cout << "\n############################################################\n";
    cout << "# " << name << "\n";
    cout << "############################################################\n";
    
    Constants C;
    C.compute(3, 2, reach);
    C.print();
    
    double L = (L_override > 0) ? L_override : C.L;
    if (L_override > 0)
        cout << "\n*** Using L = " << L << " ***\n";
    
    auto t0 = chrono::high_resolution_clock::now();
    
    int nx = (int)ceil((box_max(0) - box_min(0)) / L) + 2;
    int ny = (int)ceil((box_max(1) - box_min(1)) / L) + 2;
    int nz = (int)ceil((box_max(2) - box_min(2)) / L) + 2;
    
    cout << "\nGrid: " << nx << " x " << ny << " x " << nz << "\n";
    cout << "Simplices: " << (long long)nx * ny * nz * 6 << "\n";
    
    if ((long long)nx * ny * nz > 50000000) {
        cout << "Too many simplices! Aborting.\n";
        return;
    }
    
    // Freudenthal permutations
    int perms[6][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
    
    auto get_pt = [&](int i, int j, int k) -> Vector3d {
        return Vector3d(box_min(0) + i*L, box_min(1) + j*L, box_min(2) + k*L);
    };
    
    auto project = [&](Vector3d p) -> Vector3d {
        for (int iter = 0; iter < 20; iter++) {
            double val = f(p);
            if (abs(val) < 1e-12) break;
            Vector3d g = grad_f(p);
            double g2 = g.squaredNorm();
            if (g2 < 1e-14) break;
            p -= (val / g2) * g;
        }
        return p;
    };
    
    // Find edge intersections
    cout << "Finding edge intersections...\n";
    unordered_map<EdgeKey, Vector3d, EdgeHash> edge_int;
    long long simplex_count = 0, intersecting = 0;
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                for (int p = 0; p < 6; p++) {
                    simplex_count++;
                    
                    array<array<int,3>, 4> idx;
                    Vector3d verts[4];
                    double vals[4];
                    
                    idx[0] = {i, j, k};
                    verts[0] = get_pt(i, j, k);
                    vals[0] = f(verts[0]);
                    
                    array<int,3> cur = {i, j, k};
                    for (int v = 1; v <= 3; v++) {
                        cur[perms[p][v-1]]++;
                        idx[v] = cur;
                        verts[v] = get_pt(cur[0], cur[1], cur[2]);
                        vals[v] = f(verts[v]);
                    }
                    
                    double mn = *min_element(vals, vals+4);
                    double mx = *max_element(vals, vals+4);
                    if (mn > 0 || mx < 0) continue;
                    intersecting++;
                    
                    for (int e1 = 0; e1 < 4; e1++) {
                        for (int e2 = e1+1; e2 < 4; e2++) {
                            if (vals[e1] * vals[e2] > 0) continue;
                            EdgeKey key(idx[e1], idx[e2]);
                            if (edge_int.count(key)) continue;
                            
                            double t = vals[e1] / (vals[e1] - vals[e2]);
                            Vector3d pt = verts[e1] + t * (verts[e2] - verts[e1]);
                            edge_int[key] = project(pt);
                        }
                    }
                }
            }
        }
    }
    
    auto t1 = chrono::high_resolution_clock::now();
    cout << "  " << intersecting << " simplices intersect M\n";
    cout << "  " << edge_int.size() << " edge intersections\n";
    cout << "  Time: " << chrono::duration_cast<chrono::milliseconds>(t1-t0).count() << " ms\n";
    
    // Build triangulation K
    cout << "Building triangulation K...\n";
    
    auto pt_hash = [](const Vector3d& p) -> size_t {
        size_t h = 0;
        for (int i = 0; i < 3; i++) {
            long long x = (long long)round(p(i) * 1e7);  // Less precision for better dedup
            h ^= hash<long long>()(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    };
    auto pt_eq = [](const Vector3d& a, const Vector3d& b) {
        return (a - b).squaredNorm() < 1e-14;  // Adjusted tolerance
    };
    
    unordered_map<Vector3d, int, decltype(pt_hash), decltype(pt_eq)> 
        vert_map(0, pt_hash, pt_eq);
    vector<Vector3d> vertices;
    set<array<int,3>> tri_set;
    
    auto get_idx = [&](const Vector3d& p) -> int {
        auto it = vert_map.find(p);
        if (it != vert_map.end()) return it->second;
        int idx = vertices.size();
        vert_map[p] = idx;
        vertices.push_back(p);
        return idx;
    };
    
    int faces[4][3] = {{0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}};
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                for (int p = 0; p < 6; p++) {
                    array<array<int,3>, 4> idx;
                    double vals[4];
                    
                    idx[0] = {i, j, k};
                    vals[0] = f(get_pt(i, j, k));
                    
                    array<int,3> cur = {i, j, k};
                    for (int v = 1; v <= 3; v++) {
                        cur[perms[p][v-1]]++;
                        idx[v] = cur;
                        vals[v] = f(get_pt(cur[0], cur[1], cur[2]));
                    }
                    
                    double mn = *min_element(vals, vals+4);
                    double mx = *max_element(vals, vals+4);
                    if (mn > 0 || mx < 0) continue;
                    
                    // v(tetrahedron) = average of edge intersections
                    Vector3d v_tetra = Vector3d::Zero();
                    int edge_count = 0;
                    for (int e1 = 0; e1 < 4; e1++) {
                        for (int e2 = e1+1; e2 < 4; e2++) {
                            EdgeKey key(idx[e1], idx[e2]);
                            auto it = edge_int.find(key);
                            if (it != edge_int.end()) {
                                v_tetra += it->second;
                                edge_count++;
                            }
                        }
                    }
                    if (edge_count == 0) continue;
                    v_tetra /= edge_count;
                    
                    // For each edge, for each face containing it
                    for (int e1 = 0; e1 < 4; e1++) {
                        for (int e2 = e1+1; e2 < 4; e2++) {
                            EdgeKey ekey(idx[e1], idx[e2]);
                            auto eit = edge_int.find(ekey);
                            if (eit == edge_int.end()) continue;
                            Vector3d v_edge = eit->second;
                            
                            for (int fi = 0; fi < 4; fi++) {
                                bool has1 = false, has2 = false;
                                for (int fv = 0; fv < 3; fv++) {
                                    if (faces[fi][fv] == e1) has1 = true;
                                    if (faces[fi][fv] == e2) has2 = true;
                                }
                                if (!has1 || !has2) continue;
                                
                                // v(face) = average of edge intersections in face
                                Vector3d v_face = Vector3d::Zero();
                                int fc = 0;
                                for (int fe1 = 0; fe1 < 3; fe1++) {
                                    for (int fe2 = fe1+1; fe2 < 3; fe2++) {
                                        EdgeKey fkey(idx[faces[fi][fe1]], idx[faces[fi][fe2]]);
                                        auto fit = edge_int.find(fkey);
                                        if (fit != edge_int.end()) {
                                            v_face += fit->second;
                                            fc++;
                                        }
                                    }
                                }
                                if (fc == 0) continue;
                                v_face /= fc;
                                
                                int i0 = get_idx(v_edge);
                                int i1 = get_idx(v_face);
                                int i2 = get_idx(v_tetra);
                                
                                if (i0 != i1 && i1 != i2 && i0 != i2) {
                                    array<int,3> tri = {i0, i1, i2};
                                    sort(tri.begin(), tri.end());
                                    tri_set.insert(tri);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    vector<array<int,3>> triangles(tri_set.begin(), tri_set.end());
    
    // Euler characteristic
    set<pair<int,int>> edges;
    for (const auto& t : triangles) {
        edges.insert({min(t[0],t[1]), max(t[0],t[1])});
        edges.insert({min(t[1],t[2]), max(t[1],t[2])});
        edges.insert({min(t[0],t[2]), max(t[0],t[2])});
    }
    
    int V = vertices.size(), E = edges.size(), F = triangles.size();
    int chi = V - E + F;
    
    auto t2 = chrono::high_resolution_clock::now();
    cout << "  K: " << V << " vertices, " << F << " triangles\n";
    cout << "  Euler χ = " << V << " - " << E << " + " << F << " = " << chi << "\n";
    cout << "  Time: " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " ms\n";
    
    write_mesh(name + ".mesh", vertices, triangles);
    write_stl(name + ".stl", vertices, triangles);
    cout << "  Wrote " << name << ".mesh and " << name << ".stl\n";
    
    cout << "\nTotal: " << chrono::duration_cast<chrono::milliseconds>(t2-t0).count() << " ms\n";
}

int main() {
    cout << "Whitney's Triangulation Algorithm (C++)\n";
    cout << "========================================\n";
    
    // Sphere
    double sphere_r = 1.0;
    auto sphere_f = [=](const Vector3d& p) { return p.squaredNorm() - sphere_r*sphere_r; };
    auto sphere_g = [](const Vector3d& p) -> Vector3d { return 2.0 * p; };
    
    triangulate("sphere_cpp", sphere_f, sphere_g, sphere_r,
                Vector3d(-1.3, -1.3, -1.3), Vector3d(1.3, 1.3, 1.3), 0.12);
    
    // Torus
    double R = 1.0, r = 0.4;
    auto torus_f = [=](const Vector3d& p) {
        double rho = sqrt(p(0)*p(0) + p(1)*p(1));
        return (rho - R)*(rho - R) + p(2)*p(2) - r*r;
    };
    auto torus_g = [=](const Vector3d& p) -> Vector3d {
        double rho = sqrt(p(0)*p(0) + p(1)*p(1));
        if (rho < 1e-10) return Vector3d(0, 0, 2*p(2));
        return Vector3d(2*(rho-R)*p(0)/rho, 2*(rho-R)*p(1)/rho, 2*p(2));
    };
    
    triangulate("torus_cpp", torus_f, torus_g, r,
                Vector3d(-(R+r+0.2), -(R+r+0.2), -(r+0.2)),
                Vector3d(R+r+0.2, R+r+0.2, r+0.2), 0.05);
    
    return 0;
}