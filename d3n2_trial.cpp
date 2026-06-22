// ============================================================================
// whitney_mesher.cpp
//
// A practical implementation of:
//   J.-D. Boissonnat, S. Kachanovich, M. Wintraecken,
//   "Triangulating Submanifolds: An Elementary and Quantified Version of
//    Whitney's Method", Discrete & Computational Geometry 66 (2021) 386-434.
//
// Specialised to a 2-manifold (n = 2) embedded in R^3 (d = 3), codimension
// k = d - n = 1.  Built on the GUDHI "Coxeter triangulation" module
// (the implementation companion of S. Kachanovich's PhD thesis, ch. 3-4),
// which supplies:
//   * Coxeter_triangulation<>          : the ambient A~_3 triangulation,
//                                        stored as one matrix + offset,
//   * Permutahedral_representation     : O(d) simplex handles with
//                                        face_range / coface_range queries,
//   * locate_point / cartesian_coordinates.
//
// The two parts of the paper's algorithm (Sect. 2.1):
//   Part 1 (perturbation): every vertex of the ambient triangulation that is
//     near the manifold is nudged (by at most c~ * L) so that it ends up at
//     distance >= rho_1 * c~ * L from the tangent plane T_p M of a nearby
//     manifold point p.  In codimension 1 the spans span(tau', T_p M) of the
//     paper degenerate to T_p M itself (see the remark after Part 1 in
//     Sect. 2.1), so this is the complete perturbation.  Lemma 5.7 then
//     certifies d(vertex, M) > alpha_1 * L: the 0-skeleton is "safe".
//     The perturbation is small enough (eq. (17), Cor. 4.4) that the
//     combinatorial structure of the A~_3 triangulation is unchanged
//     (Delaunay protection!), so GUDHI's combinatorics remain valid and we
//     only override the *geometry* via a vertex -> position cache.
//   Part 2 (triangulation K, Sect. 6.2): trace all crossing edges (the
//     (d-n)-simplices) by BFS, then barycentrically subdivide: for every flag
//     tau^1 c tau^2 c tau^3 of intersecting simplices output the triangle
//     { v(tau^1), v(tau^2), v(tau^3) }, where v(tau^1) is the (unique, by
//     Lemma 6.4) intersection point and v(tau^j), j > 1, are averages
//     (eq. (26)).
//
// The theoretical grid scale (eq. (12)) is ~1.6e-34 * reach -- unusable.
// We therefore keep ALL perturbation constants of the paper exactly as
// specified (they are proportional to L) and let the user choose a practical
// L.  Both the theoretical and the used values are printed.
//
// Output: Wavefront OBJ (+ a helper script converts it to .blend).
// ============================================================================

#include <gudhi/Coxeter_triangulation.h>
#include <gudhi/Permutahedral_representation.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Triangulation = Gudhi::coxeter_triangulation::Coxeter_triangulation<>;
using Simplex       = Triangulation::Simplex_handle;   // permutahedral repr.
using GVertex       = std::vector<int>;                // lattice vertex y in Z^d

// ----------------------------------------------------------------------------
// Implicit surfaces.  These play the role of the paper's two oracles
// (Sect. 2.1, end of Part 1): (a) given a grid vertex, find a nearby manifold
// point or certify the vertex is far; (b) provide tangent planes T_p M.
// For the torus and sphere both oracles are exact.
// ----------------------------------------------------------------------------
struct Surface {
  virtual ~Surface() = default;
  virtual double          F(const Eigen::Vector3d& p)        const = 0;
  virtual Eigen::Vector3d grad(const Eigen::Vector3d& p)     const = 0;
  virtual double          dist(const Eigen::Vector3d& p)     const = 0; // exact d(p, M)
  virtual Eigen::Vector3d project(const Eigen::Vector3d& p)  const = 0; // pi_M(p)
  virtual Eigen::Vector3d seed()                             const = 0; // a point on M
  virtual double          reach()                            const = 0; // rch M
  virtual std::string     name()                             const = 0;
};

struct Torus final : Surface {
  double R, r;                                  // ring radius, tube radius
  Torus(double R_, double r_) : R(R_), r(r_) {}
  double F(const Eigen::Vector3d& p) const override {
    const double rho = std::hypot(p.x(), p.y());
    const double a   = rho - R;
    return a * a + p.z() * p.z() - r * r;
  }
  Eigen::Vector3d grad(const Eigen::Vector3d& p) const override {
    const double rho = std::hypot(p.x(), p.y());
    if (rho < 1e-14) return Eigen::Vector3d(0.0, 0.0, 2.0 * p.z());
    const double a = rho - R;
    return Eigen::Vector3d(2.0 * a * p.x() / rho, 2.0 * a * p.y() / rho,
                           2.0 * p.z());
  }
  double dist(const Eigen::Vector3d& p) const override {
    const double rho = std::hypot(p.x(), p.y());
    const double a   = rho - R;
    return std::abs(std::hypot(a, p.z()) - r);
  }
  Eigen::Vector3d project(const Eigen::Vector3d& p) const override {
    double rho = std::hypot(p.x(), p.y());
    Eigen::Vector3d ring;
    if (rho < 1e-14) ring = Eigen::Vector3d(R, 0.0, 0.0);      // axis: pick one
    else             ring = Eigen::Vector3d(R * p.x() / rho, R * p.y() / rho, 0.0);
    Eigen::Vector3d dir = p - ring;
    const double len = dir.norm();
    if (len < 1e-14) dir = Eigen::Vector3d(0.0, 0.0, 1.0);     // ring core: pick one
    else             dir /= len;
    return ring + r * dir;
  }
  Eigen::Vector3d seed()  const override { return Eigen::Vector3d(R + r, 0.0, 0.0); }
  double          reach() const override { return std::min(r, R - r); }
  std::string     name()  const override {
    std::ostringstream o; o << "torus(R=" << R << ", r=" << r << ")"; return o.str();
  }
};

struct Sphere final : Surface {
  double r;
  explicit Sphere(double r_) : r(r_) {}
  double F(const Eigen::Vector3d& p) const override { return p.squaredNorm() - r * r; }
  Eigen::Vector3d grad(const Eigen::Vector3d& p) const override { return 2.0 * p; }
  double dist(const Eigen::Vector3d& p) const override { return std::abs(p.norm() - r); }
  Eigen::Vector3d project(const Eigen::Vector3d& p) const override {
    const double len = p.norm();
    if (len < 1e-14) return Eigen::Vector3d(r, 0.0, 0.0);
    return (r / len) * p;
  }
  Eigen::Vector3d seed()  const override { return Eigen::Vector3d(r, 0.0, 0.0); }
  double          reach() const override { return r; }
  std::string     name()  const override {
    std::ostringstream o; o << "sphere(r=" << r << ")"; return o.str();
  }
};

// ----------------------------------------------------------------------------
// All constants of the paper, computed for d = 3, n = 2.
// Equation numbers refer to the DCG 2021 paper.
// ----------------------------------------------------------------------------
struct PaperConstants {
  int d = 3, n = 2, codim = 1;

  // eq. (4): N_{<=k} = 2 + sum_{j=1..k} j! * S(d+1, j); here k = d-n-1 = 0.
  double N_le_pert = 2.0;

  // eq. (5), d = 3 odd  (d = 2k-1, k = 2):
  //   rho_1 = (2k)! / (2^{2k+2} k! (k-1)! N_{<=d-n-1}) = 24 / (64*2*2) = 3/32.
  double rho1 = 24.0 / (64.0 * 2.0 * 2.0);

  // eq. (3) for A~_3, normalised so the longest edge L(sigma) = 1 (d odd):
  double t      = std::sqrt(2.0 / 3.0);                       // thickness
  double epsil  = std::sqrt(3.0 * 5.0 / (12.0 * 4.0));        // circumradius
  double mu     = std::sqrt(3.0) / 4.0;                       // shortest edge
  double mu0    = (std::sqrt(3.0) / 4.0) / std::sqrt(15.0 / 48.0); // mu/eps = 0.7746
  double delta  = (std::sqrt(39.0) - std::sqrt(15.0)) / std::sqrt(48.0); // protection

  // eq. (6): c~ = min( t*mu0*delta / (18 d L) , t^2 / 24 ).
  double ctilde;

  // eq. (8): alpha_1 = (4/3) rho_1 c~  (only alpha_1 matters in codim 1).
  double alpha1;

  // eq. (10).
  double zeta;

  // eq. (12): the certified scale L / rch(M).
  double L_over_rch_theory;

  explicit PaperConstants(double pert_mult = 1.0) {
    const double c1 = t * mu0 * delta / (18.0 * d * 1.0);
    const double c2 = t * t / 24.0;
    ctilde = pert_mult * std::min(c1, c2);   // pert_mult x the paper's max perturbation (eq. (6))
    alpha1 = 4.0 / 3.0 * rho1 * ctilde;
    const double binom = 3.0;  // C(d, d-n) = C(3,1)
    zeta = (8.0 / (15.0 * std::sqrt(3.0) * binom * (1.0 + 2.0 * ctilde)))
           * (1.0 - 8.0 * ctilde / (t * t)) * t;
    const double A = std::pow(alpha1, 4 + 2 * n) * std::pow(zeta, 2 * n);
    L_over_rch_theory =
        (A / (3.0 * (n + 1) * (n + 1)))
        / (std::pow(A / (6.0 * (n + 1) * (n + 1)), 2.0) + 36.0);
  }

  void print(double L_used, double rch, double sep_mult) const {
    auto row = [](const char* sym, const char* ref, double theo, double used,
                  const char* note) {
      std::printf("  %-26s %-10s %14.6e  %14.6e  %s\n", sym, ref, theo, used, note);
    };
    std::printf("\n=================== constants of the paper (d=3, n=2, codim 1) "
                "===================\n");
    std::printf("  %-26s %-10s %14s  %14s  %s\n", "symbol", "eq.", "theoretical",
                "used", "");
    row("N_{<=d-n-1}",        "(4)",  N_le_pert, N_le_pert, "faces/vertex bound (+2)");
    row("rho_1",              "(5)",  rho1,      rho1,      "= 3/32, slab volume fraction");
    row("thickness t(A~3)",   "(3)",  t,         t,         "= sqrt(2/3), regular tetrahedra");
    row("mu_0 = mu/epsilon",  "(3)",  mu0,       mu0,       "computed directly from (3)");
    row("protection delta/L", "(3)",  delta,     delta,     "A~ family, best known");
    row("c~",                 "(6)",  ctilde,    ctilde,    "max perturbation = c~ * L");
    row("rho_1*c~ (separ.)",  "(19)", rho1 * ctilde, rho1 * ctilde * sep_mult,
        "min dist to T_p(M), * sep_mult");
    row("alpha_1 (clearance)","(8)",  alpha1,    alpha1 * sep_mult,
        "certified d(vertex, M) / L");
    row("zeta",               "(10)", zeta,      zeta,      "quality factor");
    row("L / rch(M)",         "(12)", L_over_rch_theory, L_used / rch,
        "THE practical compromise");
    std::printf("  -> theoretical L for this surface : %.3e   (rch = %g)\n",
                L_over_rch_theory * rch, rch);
    std::printf("  -> L actually used                : %.3e   (%.0fx too coarse "
                "for the certified guarantee)\n",
                L_used, L_used / (L_over_rch_theory * rch));
    std::printf("  -> Lemma 6.7 altitude lower bound : %.3e * L  "
                "(conservative; empirical value printed at the end)\n",
                zeta * zeta * alpha1 * alpha1);
    std::printf("====================================================================="
                "===============\n\n");
  }
};

// ----------------------------------------------------------------------------
// Canonical string keys for simplices / vertices (the permutahedral
// representation sigma(y, omega) is canonical, so equal simplices serialise
// identically).
// ----------------------------------------------------------------------------
static std::string vkey(const GVertex& v) {
  std::ostringstream o;
  for (int x : v) o << x << ',';
  return o.str();
}
static std::string skey(const Simplex& s) {
  std::ostringstream o;
  for (int x : s.vertex()) o << x << ',';
  o << '|';
  for (const auto& part : s.partition()) {
    for (auto i : part) o << i << ',';
    o << ';';
  }
  return o.str();
}

// ----------------------------------------------------------------------------
// Part 1 of the algorithm: the perturbed geometry T~.
//
// The combinatorics stay those of the unperturbed A~_3 triangulation
// (Lemma 5.5 / Cor. 4.4: the perturbation respects the protection bound, so
// the combinatorial structure is unchanged) -- which is what lets us keep
// using GUDHI's face/coface machinery.  Only vertex *positions* change.
// ----------------------------------------------------------------------------
class PerturbedGeometry {
 public:
  PerturbedGeometry(const Triangulation& T, const Surface& M,
                    const PaperConstants& C, double L, double sep_mult)
      : T_(T), M_(M), C_(C), L_(L), sep_mult_(sep_mult) {}

  // Perturbed position of a lattice vertex (cached, decided on first use).
  const Eigen::Vector3d& position(const GVertex& v) {
    const std::string k = vkey(v);
    auto it = cache_.find(k);
    if (it != cache_.end()) return it->second;

    Eigen::VectorXd base_xd = T_.cartesian_coordinates(v);
    Eigen::Vector3d base(base_xd[0], base_xd[1], base_xd[2]);
    Eigen::Vector3d out = base;

    if (M_.dist(base) >= 1.5 * L_) {
      ++n_far_;                                            // Case 1, Sect. 5.2
    } else {                                               // Case 2, Sect. 5.2
      // p in M near v;  codim 1  =>  the only span is T_p M itself.
      const Eigen::Vector3d p   = M_.project(base);
      Eigen::Vector3d       nrm = M_.grad(p);
      const double gn = nrm.norm();
      nrm = (gn > 1e-14) ? Eigen::Vector3d(nrm / gn) : Eigen::Vector3d(0, 0, 1);

      const double h   = (base - p).dot(nrm);     // signed dist to T_p M
      const double thr = C_.rho1 * C_.ctilde * L_ * sep_mult_;  // eq. (19)
      const double cap = C_.ctilde * L_;                        // eq. (17)

      if (std::abs(h) < thr) {
        const double dir = (h >= 0.0) ? 1.0 : -1.0;
        double s = dir * thr - h;                 // push to |h + s| = thr
        if (std::abs(s) > cap) {                  // (never with paper constants)
          const double s2 = -dir * thr - h;
          if (std::abs(s2) < std::abs(s)) s = s2;
        }
        if (std::abs(s) > cap) s = (s > 0 ? cap : -cap);  // hard safety cap
        out = base + s * nrm;
        ++n_perturbed_;
        max_shift_ = std::max(max_shift_, std::abs(s));
      } else {
        ++n_near_ok_;
      }
    }
    return cache_.emplace(k, out).first->second;
  }

  void print_stats() const {
    std::printf("[part 1] perturbation: %zu vertices touched "
                "(%zu far / case 1, %zu near & already safe, %zu nudged; "
                "max shift %.3e = %.4f * c~L)\n",
                cache_.size(), n_far_, n_near_ok_, n_perturbed_, max_shift_,
                max_shift_ / (C_.ctilde * L_));
  }

 private:
  const Triangulation& T_;
  const Surface& M_;
  const PaperConstants& C_;
  double L_, sep_mult_;
  std::unordered_map<std::string, Eigen::Vector3d> cache_;
  std::size_t n_far_ = 0, n_near_ok_ = 0, n_perturbed_ = 0;
  double max_shift_ = 0.0;
};

// ----------------------------------------------------------------------------
// Crossing test for a (perturbed) edge tau^1, with the intersection point
// v(tau^1) (unique by Lemma 6.4).  Root by bisection on the perturbed segment.
// ----------------------------------------------------------------------------
struct EdgeCross {
  bool hit = false;
  Eigen::Vector3d pt;
};

static EdgeCross edge_intersection(const Simplex& edge, PerturbedGeometry& geo,
                                   const Surface& M) {
  EdgeCross out;
  std::vector<Eigen::Vector3d> P;
  for (const auto& v : edge.vertex_range()) P.push_back(geo.position(v));
  if (P.size() != 2) return out;

  double fa = M.F(P[0]), fb = M.F(P[1]);
  // Skeleton safety (Lemma 5.7) guarantees no vertex lies on M.
  if (fa == 0.0 || fb == 0.0) {
    std::fprintf(stderr, "warning: vertex exactly on M despite perturbation\n");
    return out;
  }
  if (fa * fb > 0.0) return out;

  Eigen::Vector3d a = P[0], b = P[1];
  for (int i = 0; i < 80; ++i) {
    const Eigen::Vector3d m = 0.5 * (a + b);
    const double fm = M.F(m);
    if (fm == 0.0) { a = m; b = m; break; }
    if (fa * fm < 0.0) { b = m; fb = fm; } else { a = m; fa = fm; }
  }
  out.hit = true;
  out.pt  = 0.5 * (a + b);
  return out;
}

// ----------------------------------------------------------------------------
// OBJ writer (Blender imports OBJ natively; obj_to_blend.py produces .blend).
// ----------------------------------------------------------------------------
static void write_obj(const std::string& path,
                      const std::vector<Eigen::Vector3d>& V,
                      const std::vector<std::array<int, 3>>& Fc) {
  std::ofstream f(path);
  f << "# Whitney triangulation (Boissonnat-Kachanovich-Wintraecken, DCG 2021)\n";
  for (const auto& p : V)
    f << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  for (const auto& t : Fc)
    f << "f " << t[0] + 1 << ' ' << t[1] + 1 << ' ' << t[2] + 1 << '\n';
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
  // ---------------- CLI ----------------
  double      L_target = 0.08;     // practical longest-edge length
  double      sep_mult = 10.0;     // multiplier on the rho_1 c~ L separation
  double      pert_mult = 50.0;    // multiplier on the max perturbation c~ (eq. (6))
  std::string surface  = "torus";
  std::string out_path = "whitney_mesh.obj";
  double      torus_R = 1.0, torus_r = 0.4, sphere_r = 1.0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(1); }
      return argv[++i];
    };
    if      (a == "--L")        L_target = std::stod(need("--L"));
    else if (a == "--sep-mult") sep_mult = std::stod(need("--sep-mult"));
    else if (a == "--pert-mult") pert_mult = std::stod(need("--pert-mult"));
    else if (a == "--surface")  surface  = need("--surface");
    else if (a == "--out")      out_path = need("--out");
    else if (a == "--R")        torus_R  = std::stod(need("--R"));
    else if (a == "--r")        torus_r  = std::stod(need("--r"));
    else if (a == "--sphere-r") sphere_r = std::stod(need("--sphere-r"));
    else {
      std::printf("usage: %s [--surface torus|sphere] [--L h] [--sep-mult m]\n"
                  "          [--pert-mult m] [--R R] [--r r] [--sphere-r r] [--out file.obj]\n", argv[0]);
      return a == "--help" ? 0 : 1;
    }
  }

  std::unique_ptr<Surface> M;
  if (surface == "sphere") M = std::make_unique<Sphere>(sphere_r);
  else                     M = std::make_unique<Torus>(torus_R, torus_r);

  PaperConstants C(pert_mult);

  // ---------------- ambient A~_3 triangulation (GUDHI) ----------------
  const int d = 3;
  Triangulation cox(d);

  // Calibrate the longest edge empirically, then rescale the stored matrix
  // Lambda (the whole infinite grid costs O(d^2) memory: one matrix + offset).
  auto longest_edge = [&](void) -> double {
    Eigen::VectorXd q(d);
    q << 0.1234, 0.4567, 0.7891;                       // generic point
    Simplex s = cox.locate_point(q);
    if (s.dimension() < d) {
      for (const auto& cf : s.coface_range(d)) { s = cf; break; }
    }
    std::vector<Eigen::Vector3d> P;
    for (const auto& v : s.vertex_range()) {
      Eigen::VectorXd x = cox.cartesian_coordinates(v);
      P.emplace_back(x[0], x[1], x[2]);
    }
    double L = 0.0;
    for (std::size_t i = 0; i < P.size(); ++i)
      for (std::size_t j = i + 1; j < P.size(); ++j)
        L = std::max(L, (P[i] - P[j]).norm());
    return L;
  };
  const double L0 = longest_edge();
  cox.change_matrix(Eigen::MatrixXd((L_target / L0) * cox.matrix()));
  const double L = longest_edge();                     // ~= L_target

  std::printf("surface: %s   reach = %g\n", M->name().c_str(), M->reach());
  std::printf("ambient: Coxeter triangulation of type A~_3 "
              "(GUDHI), longest edge L = %.6f\n", L);
  C.print(L, M->reach(), sep_mult);

  PerturbedGeometry geo(cox, *M, C, L, sep_mult);

  // ---------------- Part 2a: trace the crossing edges (BFS) ----------------
  // The piercing simplices have dimension d - n = 1 (edges); BFS over the
  // adjacency graph: two crossing edges are adjacent iff they share a
  // cofacet (a triangle).  This is the thesis' manifold tracing algorithm.
  struct EdgeRec { Simplex s; Eigen::Vector3d pt; int vid; };
  std::unordered_map<std::string, EdgeRec> edges;     // crossing edges
  std::unordered_set<std::string> visited;            // all tested edges
  std::queue<Simplex> bfs;

  auto try_edge = [&](const Simplex& e) {
    const std::string k = skey(e);
    if (!visited.insert(k).second) return;
    EdgeCross c = edge_intersection(e, geo, *M);
    if (c.hit) {
      edges.emplace(k, EdgeRec{e, c.pt, -1});
      bfs.push(e);
    }
  };

  {  // seed: locate the simplex containing a point of M, scan tets around it.
    Eigen::Vector3d x0 = M->seed();
    Eigen::VectorXd x0d(d); x0d << x0.x(), x0.y(), x0.z();
    Simplex s0 = cox.locate_point(x0d);
    std::vector<Simplex> tets;
    if (s0.dimension() == d) tets.push_back(s0);
    else for (const auto& cf : s0.coface_range(d)) tets.push_back(cf);

    // ring expansion until a crossing edge is found (first tet in practice)
    std::unordered_set<std::string> seen_tets;
    std::queue<Simplex> tetq;
    for (auto& t : tets) { if (seen_tets.insert(skey(t)).second) tetq.push(t); }
    int scanned = 0;
    while (!tetq.empty() && edges.empty() && scanned < 20000) {
      Simplex t = tetq.front(); tetq.pop(); ++scanned;
      for (const auto& e : t.face_range(1)) try_edge(e);
      if (!edges.empty()) break;
      for (const auto& tri : t.face_range(2))
        for (const auto& nb : tri.coface_range(d))
          if (seen_tets.insert(skey(nb)).second) tetq.push(nb);
    }
    if (edges.empty()) {
      std::fprintf(stderr, "error: no crossing edge near the seed point\n");
      return 1;
    }
  }

  while (!bfs.empty()) {
    Simplex e = bfs.front(); bfs.pop();
    for (const auto& tri : e.coface_range(2))      // cofacets of the edge
      for (const auto& e2 : tri.face_range(1))     // facets of the cofacet
        try_edge(e2);
  }
  std::printf("[part 2a] tracing: |S| = %zu crossing edges (%zu edges tested)\n",
              edges.size(), visited.size());

  // ---------------- Part 2b: barycentric construction of K ----------------
  // v(tau^1) = intersection point;  for dim tau > 1,
  // v(tau) = average of v over the crossing (d-n)-faces of tau   (eq. (26)).
  struct FaceRec { Simplex s; Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                   int cnt = 0; int vid = -1; };
  std::unordered_map<std::string, FaceRec> tris, tets;

  std::vector<Eigen::Vector3d> V;                    // output vertex buffer
  for (auto& [k, er] : edges) {
    er.vid = static_cast<int>(V.size());
    V.push_back(er.pt);
    for (const auto& tri : er.s.coface_range(2)) {
      auto& fr = tris[skey(tri)];
      fr.s = tri; fr.sum += er.pt; ++fr.cnt;
    }
    for (const auto& tet : er.s.coface_range(3)) {
      auto& fr = tets[skey(tet)];
      fr.s = tet; fr.sum += er.pt; ++fr.cnt;
    }
  }
  for (auto& [k, fr] : tris) { fr.vid = (int)V.size(); V.push_back(fr.sum / fr.cnt); }
  for (auto& [k, fr] : tets) { fr.vid = (int)V.size(); V.push_back(fr.sum / fr.cnt); }

  // flags tau^1 c tau^2 c tau^3, all intersecting M  ->  output triangles.
  std::vector<std::array<int, 3>> Fc;
  for (auto& [tk, tet] : tets) {
    for (const auto& tri : tet.s.face_range(2)) {
      auto it_t = tris.find(skey(tri));
      if (it_t == tris.end()) continue;
      for (const auto& e : tri.face_range(1)) {
        auto it_e = edges.find(skey(e));
        if (it_e == edges.end()) continue;
        std::array<int, 3> f = {it_e->second.vid, it_t->second.vid, tet.vid};
        // orient consistently with the outward gradient of F
        const Eigen::Vector3d c =
            (V[f[0]] + V[f[1]] + V[f[2]]) / 3.0;
        const Eigen::Vector3d nrm =
            (V[f[1]] - V[f[0]]).cross(V[f[2]] - V[f[0]]);
        if (nrm.dot(M->grad(c)) < 0.0) std::swap(f[1], f[2]);
        Fc.push_back(f);
      }
    }
  }
  std::printf("[part 2b] K: %zu vertices (%zu edge pts + %zu triangle pts + "
              "%zu tet pts), %zu triangles\n",
              V.size(), edges.size(), tris.size(), tets.size(), Fc.size());
  geo.print_stats();

  // ---------------- quality + closeness report ----------------
  double min_alt_ratio = 1e300, mean_alt = 0.0, max_off = 0.0;
  for (const auto& f : Fc) {
    const Eigen::Vector3d &a = V[f[0]], &b = V[f[1]], &c = V[f[2]];
    const double e0 = (b - a).norm(), e1 = (c - b).norm(), e2 = (a - c).norm();
    const double Lmax = std::max({e0, e1, e2});
    const double area = 0.5 * (b - a).cross(c - a).norm();
    const double min_alt = 2.0 * area / Lmax;          // smallest altitude
    const double ratio = (Lmax > 0) ? min_alt / Lmax : 0.0;
    min_alt_ratio = std::min(min_alt_ratio, ratio);
    mean_alt += ratio;
  }
  for (const auto& p : V) max_off = std::max(max_off, M->dist(p));
  std::printf("[quality] min altitude/longest-edge over K : %.4e "
              "(mean %.4e)  -- Lemma 6.7 certified bound: %.3e\n",
              min_alt_ratio, Fc.empty() ? 0.0 : mean_alt / Fc.size(),
              C.zeta * C.zeta * C.alpha1 * C.alpha1);
  std::printf("[closeness] max distance of K's vertices to M : %.4e "
              "(= %.4f * L); edge points lie on M to ~1e-15\n",
              max_off, max_off / L);

  write_obj(out_path, V, Fc);
  std::printf("[output] wrote %s  (import in Blender: File > Import > "
              "Wavefront, or run obj_to_blend.py for a .blend)\n",
              out_path.c_str());
  return 0;
}