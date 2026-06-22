// ============================================================================
// whitney_codim1.cpp
//
// Generalisation of d3n2_trial.cpp to ANY codimension-1 manifold:
//   n = d - 1,  codim k = d - n = 1,  for arbitrary ambient dimension d >= 2.
//
// Same paper: Boissonnat-Kachanovich-Wintraecken, "Triangulating Submanifolds",
// DCG 66 (2021) 386-434.  In codimension 1 the algorithm is dimension-uniform:
//   * the piercing simplices are always the (d-n) = 1-simplices (EDGES);
//   * the spans span(tau', T_p M) all degenerate to the tangent hyperplane
//     T_p M (the remark after Part 1, Sect. 2.1), so Part 1 just pushes every
//     near-M vertex off T_p M along the surface normal;
//   * Part 2 outputs, for every flag tau^1 c tau^2 c ... c tau^d of crossing
//     simplices, the (d-1)-simplex { v(tau^1), ..., v(tau^d) } (eq. (25)/(26)),
//     where v(tau^1) is the edge-M intersection and v(tau^j), j>1, is the
//     average of v over the crossing edges of tau^j.
//
// So the output K triangulates the (d-1)-manifold with (d-1)-simplices:
//   d=2 -> polygonal curve, d=3 -> triangles, d=4 -> tetrahedra in R^4, ...
//
// Built on GUDHI's Coxeter_triangulation<> (ambient A~_d), which is dimension
// generic.  Only vertex *positions* are overridden (perturbation cache); the
// combinatorics stay GUDHI's unperturbed A~_d (Cor. 4.4 / Delaunay protection).
// ============================================================================

#include <gudhi/Coxeter_triangulation.h>
#include <gudhi/Permutahedral_representation.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Triangulation = Gudhi::coxeter_triangulation::Coxeter_triangulation<>;
using Simplex       = Triangulation::Simplex_handle;
using GVertex       = std::vector<int>;
using Vec           = Eigen::VectorXd;            // ambient point in R^d

// ----------------------------------------------------------------------------
// Implicit codim-1 surfaces (the paper's two oracles: closest point + tangent).
// ----------------------------------------------------------------------------
struct Surface {
  virtual ~Surface() = default;
  virtual double F(const Vec& p)       const = 0;
  virtual Vec    grad(const Vec& p)    const = 0;   // normal direction (un-normalised)
  virtual double dist(const Vec& p)    const = 0;   // exact d(p, M)
  virtual Vec    project(const Vec& p) const = 0;   // pi_M(p)
  virtual Vec    seed()                const = 0;   // a point on M
  virtual double reach()               const = 0;   // rch M
  virtual std::string name()           const = 0;
};

// Hypersphere S^{d-1} of radius r, centred at the origin -- exact in any d.
struct Sphere final : Surface {
  int d; double r;
  Sphere(int d_, double r_) : d(d_), r(r_) {}
  double F(const Vec& p) const override { return p.squaredNorm() - r * r; }
  Vec    grad(const Vec& p) const override { return 2.0 * p; }
  double dist(const Vec& p) const override { return std::abs(p.norm() - r); }
  Vec    project(const Vec& p) const override {
    const double len = p.norm();
    if (len < 1e-14) { Vec e = Vec::Zero(d); e[0] = r; return e; }
    return (r / len) * p;
  }
  Vec    seed()  const override { Vec e = Vec::Zero(d); e[0] = r; return e; }
  double reach() const override { return r; }
  std::string name() const override {
    std::ostringstream o; o << "sphere(d=" << d << ", r=" << r << ")"; return o.str();
  }
};

// Torus of revolution in R^3 (d == 3 only) -- kept for parity with d3n2_trial.
struct Torus final : Surface {
  double R, r;
  Torus(double R_, double r_) : R(R_), r(r_) {}
  double F(const Vec& p) const override {
    const double rho = std::hypot(p.x(), p.y());
    const double a = rho - R;
    return a * a + p.z() * p.z() - r * r;
  }
  Vec grad(const Vec& p) const override {
    const double rho = std::hypot(p.x(), p.y());
    Vec g(3);
    if (rho < 1e-14) { g << 0.0, 0.0, 2.0 * p.z(); return g; }
    const double a = rho - R;
    g << 2.0 * a * p.x() / rho, 2.0 * a * p.y() / rho, 2.0 * p.z();
    return g;
  }
  double dist(const Vec& p) const override {
    const double rho = std::hypot(p.x(), p.y());
    return std::abs(std::hypot(rho - R, p.z()) - r);
  }
  Vec project(const Vec& p) const override {
    const double rho = std::hypot(p.x(), p.y());
    Vec ring(3);
    if (rho < 1e-14) ring << R, 0.0, 0.0;
    else             ring << R * p.x() / rho, R * p.y() / rho, 0.0;
    Vec dir = p - ring;
    const double len = dir.norm();
    if (len < 1e-14) dir << 0.0, 0.0, 1.0; else dir /= len;
    return ring + r * dir;
  }
  Vec    seed()  const override { Vec e(3); e << R + r, 0.0, 0.0; return e; }
  double reach() const override { return std::min(r, R - r); }
  std::string name() const override {
    std::ostringstream o; o << "torus(R=" << R << ", r=" << r << ")"; return o.str();
  }
};

// ----------------------------------------------------------------------------
// Paper constants for general d, codim 1 (n = d-1).  Eq. numbers from DCG 2021.
// All quantities are dimensionless (computed in the normalised scale L(sigma)=1).
// ----------------------------------------------------------------------------
struct PaperConstants {
  int d, n = 0, codim = 1;
  double N_le_pert = 2.0;        // N_{<=d-n-1} = N_{<=0} = 2 in codim 1  (eq. 4)
  double rho1 = 0, t = 0, epsil = 0, mu = 0, mu0 = 0, delta = 0;
  double ctilde = 0, alpha1 = 0, zeta = 0, L_over_rch_theory = 0;

  PaperConstants(int d_, double pert_mult) : d(d_), n(d_ - 1) {
    auto fact = [](int m) { double f = 1.0; for (int i = 2; i <= m; ++i) f *= i; return f; };
    const double PI = std::acos(-1.0);

    // eq. (5): rho_1, the slab volume fraction (N_{<=0} = 2).
    if (d % 2 == 0) { int k = d / 2;
      rho1 = std::pow(2.0, 2 * k - 2) * fact(k) * fact(k) / (PI * fact(2 * k) * N_le_pert);
    } else { int k = (d + 1) / 2;
      rho1 = fact(2 * k) / (std::pow(2.0, 2 * k + 2) * fact(k) * fact(k - 1) * N_le_pert);
    }

    // eq. (3): geometry of A~_d (normalised so the longest edge = 1).
    t     = (d % 2 == 1) ? std::sqrt(2.0 / d)
                         : std::sqrt(2.0 * (d + 1) / ((double)d * (d + 2)));
    epsil = std::sqrt((double)d * (d + 2) / (12.0 * (d + 1)));
    mu    = std::sqrt((double)d / (d + 1));               // shortest edge
    mu0   = mu / epsil;                                   // = sqrt(12/(d+2))
    delta = (std::sqrt((double)d * d + 2.0 * d + 24.0) - std::sqrt((double)d * d + 2.0 * d))
            / std::sqrt(12.0 * (d + 1));

    // eq. (6): c~ = min( t mu0 delta / (18 d), t^2/24 ), times the user multiplier.
    const double c1 = t * mu0 * delta / (18.0 * d);
    const double c2 = t * t / 24.0;
    ctilde = pert_mult * std::min(c1, c2);

    // eq. (8): alpha_1 = (4/3) rho_1 c~   (codim 1 -> only alpha_1 matters).
    alpha1 = 4.0 / 3.0 * rho1 * ctilde;

    // eq. (10): zeta, with C(d, d-n) = C(d, 1) = d.
    zeta = (8.0 / (15.0 * std::sqrt((double)d) * (double)d * (1.0 + 2.0 * ctilde)))
           * (1.0 - 8.0 * ctilde / (t * t)) * t;

    // eq. (12): the certified scale L / rch(M).
    const double A = std::pow(alpha1, 4 + 2 * n) * std::pow(zeta, 2 * n);
    L_over_rch_theory = (A / (3.0 * (n + 1) * (n + 1)))
                        / (std::pow(A / (6.0 * (n + 1) * (n + 1)), 2.0) + 36.0);
  }

  void print(double L_used, double rch, double sep_mult) const {
    std::printf("\n========== paper constants (d=%d, n=%d, codim 1) ==========\n", d, n);
    std::printf("  rho_1              (5)  = %.6e\n", rho1);
    std::printf("  thickness t        (3)  = %.6e\n", t);
    std::printf("  mu_0 = mu/eps      (3)  = %.6e\n", mu0);
    std::printf("  protection delta/L (3)  = %.6e\n", delta);
    std::printf("  c~ (max pert)      (6)  = %.6e   (max shift = c~ * L = %.3e)\n",
                ctilde, ctilde * L_used);
    std::printf("  rho_1*c~ (separ.) (19)  = %.6e   (* sep_mult = %.3e)\n",
                rho1 * ctilde, rho1 * ctilde * sep_mult);
    std::printf("  alpha_1            (8)  = %.6e\n", alpha1);
    std::printf("  zeta              (10)  = %.6e%s\n", zeta,
                zeta < 0 ? "   (<0: pert beyond t^2/8; quality diagnostics void)" : "");
    std::printf("  L/rch theoretical (12)  = %.6e   -> certified L = %.3e (rch=%g)\n",
                L_over_rch_theory, L_over_rch_theory * rch, rch);
    std::printf("  L actually used         = %.3e   (%.1ex the certified scale)\n",
                L_used, L_used / (L_over_rch_theory * rch));
    std::printf("===========================================================\n\n");
  }
};

// ----------------------------------------------------------------------------
// Canonical string keys (the permutahedral representation is canonical).
// ----------------------------------------------------------------------------
static std::string vkey(const GVertex& v) {
  std::ostringstream o; for (int x : v) o << x << ','; return o.str();
}
static std::string skey(const Simplex& s) {
  std::ostringstream o;
  for (int x : s.vertex()) o << x << ',';
  o << '|';
  for (const auto& part : s.partition()) { for (auto i : part) o << i << ','; o << ';'; }
  return o.str();
}

// ----------------------------------------------------------------------------
// Part 1: perturbed geometry T~.  Push each near-M vertex off the tangent
// hyperplane T_p M, by at most c~ * L (eq. 17), to distance rho_1 c~ L sep_mult
// (eq. 19).  Combinatorics stay GUDHI's; only positions change.
// ----------------------------------------------------------------------------
class PerturbedGeometry {
 public:
  PerturbedGeometry(const Triangulation& T, const Surface& M, const PaperConstants& C,
                    int d, double L, double sep_mult)
      : T_(T), M_(M), C_(C), d_(d), L_(L), sep_mult_(sep_mult) {}

  const Vec& position(const GVertex& v) {
    const std::string k = vkey(v);
    auto it = cache_.find(k);
    if (it != cache_.end()) return it->second;

    Vec base = T_.cartesian_coordinates(v);             // size d
    Vec out  = base;

    if (M_.dist(base) >= 1.5 * L_) {
      ++n_far_;                                          // Case 1
    } else {                                             // Case 2: codim 1 -> only T_p M
      const Vec p = M_.project(base);
      Vec nrm = M_.grad(p);
      const double gn = nrm.norm();
      if (gn > 1e-14) nrm /= gn; else { nrm = Vec::Zero(d_); nrm[d_ - 1] = 1.0; }

      const double h   = (base - p).dot(nrm);            // signed dist to T_p M
      const double thr = C_.rho1 * C_.ctilde * L_ * sep_mult_;
      const double cap = C_.ctilde * L_;

      if (std::abs(h) < thr) {
        const double dir = (h >= 0.0) ? 1.0 : -1.0;
        double s = dir * thr - h;
        if (std::abs(s) > cap) {
          const double s2 = -dir * thr - h;
          if (std::abs(s2) < std::abs(s)) s = s2;
        }
        if (std::abs(s) > cap) s = (s > 0 ? cap : -cap);
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
                "(%zu far, %zu near&safe, %zu nudged; max shift %.3e = %.4f * c~L)\n",
                cache_.size(), n_far_, n_near_ok_, n_perturbed_, max_shift_,
                C_.ctilde * L_ > 0 ? max_shift_ / (C_.ctilde * L_) : 0.0);
  }

 private:
  const Triangulation& T_;
  const Surface& M_;
  const PaperConstants& C_;
  int d_;
  double L_, sep_mult_;
  std::unordered_map<std::string, Vec> cache_;
  std::size_t n_far_ = 0, n_near_ok_ = 0, n_perturbed_ = 0;
  double max_shift_ = 0.0;
};

// ----------------------------------------------------------------------------
// Edge crossing point (unique by Lemma 6.4), found by bisection.
// ----------------------------------------------------------------------------
struct EdgeCross { bool hit = false; Vec pt; };

static EdgeCross edge_intersection(const Simplex& edge, PerturbedGeometry& geo,
                                   const Surface& M) {
  EdgeCross out;
  std::vector<Vec> P;
  for (const auto& v : edge.vertex_range()) P.push_back(geo.position(v));
  if (P.size() != 2) return out;

  double fa = M.F(P[0]), fb = M.F(P[1]);
  if (fa == 0.0 || fb == 0.0) {
    std::fprintf(stderr, "warning: vertex exactly on M despite perturbation\n");
    return out;
  }
  if (fa * fb > 0.0) return out;

  Vec a = P[0], b = P[1];
  for (int i = 0; i < 80; ++i) {
    Vec m = 0.5 * (a + b);
    const double fm = M.F(m);
    if (fm == 0.0) { a = m; b = m; break; }
    if (fa * fm < 0.0) { b = m; fb = fm; } else { a = m; fa = fm; }
  }
  out.hit = true;
  out.pt  = 0.5 * (a + b);
  return out;
}

// ----------------------------------------------------------------------------
// Volume of an m-simplex (Cayley/Gram) and its minimum altitude, in any R^d.
// ----------------------------------------------------------------------------
static double simplex_volume(const std::vector<Vec>& pts) {
  const int m = (int)pts.size() - 1;                 // m-simplex
  if (m <= 0) return 0.0;
  Eigen::MatrixXd E(m, pts[0].size());
  for (int i = 0; i < m; ++i) E.row(i) = (pts[i + 1] - pts[0]).transpose();
  double det = (E * E.transpose()).determinant();
  if (det < 0) det = 0;
  double f = 1.0; for (int i = 2; i <= m; ++i) f *= i;
  return std::sqrt(det) / f;
}
static double min_altitude(const std::vector<Vec>& pts) {
  const int k = (int)pts.size() - 1;                 // k-simplex
  const double vk = simplex_volume(pts);
  double maxf = 0.0;
  for (int i = 0; i <= k; ++i) {
    std::vector<Vec> facet;
    for (int j = 0; j <= k; ++j) if (j != i) facet.push_back(pts[j]);
    maxf = std::max(maxf, simplex_volume(facet));
  }
  return (maxf < 1e-300) ? 0.0 : k * vk / maxf;
}

// ----------------------------------------------------------------------------
// Output writers.
// ----------------------------------------------------------------------------
static void write_obj(const std::string& path, const std::vector<Vec>& V,
                      const std::vector<std::vector<int>>& Fc) {            // d == 3
  std::ofstream f(path);
  f << "# Whitney triangulation (codim 1, d=3)\n";
  for (const auto& p : V) f << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
  for (const auto& s : Fc) f << "f " << s[0] + 1 << ' ' << s[1] + 1 << ' ' << s[2] + 1 << '\n';
}
static void write_generic(const std::string& path, int d, const std::string& surf,
                          const std::vector<Vec>& V, const std::vector<std::vector<int>>& Fc) {
  std::ofstream f(path);
  f << "# Whitney codim-1 mesh: d=" << d << ", surface=" << surf << "\n";
  f << "# " << V.size() << " vertices, " << Fc.size()
    << " simplices (each: " << d << " vertex indices, 1-based)\n";
  f << "d " << d << '\n';
  for (const auto& p : V) { f << "v"; for (int i = 0; i < d; ++i) f << ' ' << p[i]; f << '\n'; }
  for (const auto& s : Fc) { f << "s"; for (int idx : s) f << ' ' << idx + 1; f << '\n'; }
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
  int         d         = 3;
  double      L_target  = 0.08;
  double      sep_mult  = 10.0;
  double      pert_mult = 50.0;
  std::string surface   = "sphere";
  std::string out_path  = "whitney_codim1.obj";
  double      torus_R = 1.0, torus_r = 0.4, sphere_r = 1.0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(1); }
      return argv[++i];
    };
    if      (a == "--d")         d         = std::stoi(need("--d"));
    else if (a == "--L")         L_target  = std::stod(need("--L"));
    else if (a == "--sep-mult")  sep_mult  = std::stod(need("--sep-mult"));
    else if (a == "--pert-mult") pert_mult = std::stod(need("--pert-mult"));
    else if (a == "--surface")   surface   = need("--surface");
    else if (a == "--out")       out_path  = need("--out");
    else if (a == "--R")         torus_R   = std::stod(need("--R"));
    else if (a == "--r")         torus_r   = std::stod(need("--r"));
    else if (a == "--sphere-r")  sphere_r  = std::stod(need("--sphere-r"));
    else {
      std::printf("usage: %s [--d D] [--surface sphere|torus] [--L h] [--sep-mult m]\n"
                  "          [--pert-mult m] [--R R] [--r r] [--sphere-r r] [--out file]\n", argv[0]);
      return a == "--help" ? 0 : 1;
    }
  }
  if (d < 2) { std::fprintf(stderr, "need d >= 2\n"); return 1; }

  std::unique_ptr<Surface> M;
  if (surface == "torus") {
    if (d != 3) { std::fprintf(stderr, "torus is only defined for d=3\n"); return 1; }
    M = std::make_unique<Torus>(torus_R, torus_r);
  } else {
    M = std::make_unique<Sphere>(d, sphere_r);
  }

  PaperConstants C(d, pert_mult);

  // ---------------- ambient A~_d triangulation ----------------
  Triangulation cox(d);
  auto longest_edge = [&]() -> double {
    Vec q(d);
    for (int i = 0; i < d; ++i) q[i] = 0.1234 + 0.3171 * i;     // generic point
    Simplex s = cox.locate_point(q);
    if (s.dimension() < d) for (const auto& cf : s.coface_range(d)) { s = cf; break; }
    std::vector<Vec> P;
    for (const auto& v : s.vertex_range()) P.push_back(cox.cartesian_coordinates(v));
    double L = 0.0;
    for (std::size_t i = 0; i < P.size(); ++i)
      for (std::size_t j = i + 1; j < P.size(); ++j)
        L = std::max(L, (P[i] - P[j]).norm());
    return L;
  };
  const double L0 = longest_edge();
  cox.change_matrix(Eigen::MatrixXd((L_target / L0) * cox.matrix()));
  const double L = longest_edge();

  std::printf("surface: %s   reach = %g\n", M->name().c_str(), M->reach());
  std::printf("ambient: Coxeter triangulation A~_%d, longest edge L = %.6f\n", d, L);
  C.print(L, M->reach(), sep_mult);

  PerturbedGeometry geo(cox, *M, C, d, L, sep_mult);

  // ---------------- Part 2a: trace crossing edges by BFS ----------------
  struct EdgeRec { Simplex s; Vec pt; };
  std::unordered_map<std::string, EdgeRec> edges;
  std::unordered_set<std::string> visited;
  std::queue<Simplex> bfs;

  auto try_edge = [&](const Simplex& e) {
    const std::string k = skey(e);
    if (!visited.insert(k).second) return;
    EdgeCross c = edge_intersection(e, geo, *M);
    if (c.hit) { edges.emplace(k, EdgeRec{e, c.pt}); bfs.push(e); }
  };

  {  // seed: ring-expand top simplices around M.seed() until a crossing edge appears
    Vec x0 = M->seed();
    Simplex s0 = cox.locate_point(x0);
    std::vector<Simplex> tets;
    if (s0.dimension() == d) tets.push_back(s0);
    else for (const auto& cf : s0.coface_range(d)) tets.push_back(cf);

    std::unordered_set<std::string> seen;
    std::queue<Simplex> tetq;
    for (auto& t : tets) if (seen.insert(skey(t)).second) tetq.push(t);
    int scanned = 0;
    while (!tetq.empty() && edges.empty() && scanned < 50000) {
      Simplex t = tetq.front(); tetq.pop(); ++scanned;
      for (const auto& e : t.face_range(1)) try_edge(e);
      if (!edges.empty()) break;
      for (const auto& f : t.face_range(2))
        for (const auto& nb : f.coface_range(d))
          if (seen.insert(skey(nb)).second) tetq.push(nb);
    }
    if (edges.empty()) { std::fprintf(stderr, "error: no crossing edge near seed\n"); return 1; }
  }

  while (!bfs.empty()) {
    Simplex e = bfs.front(); bfs.pop();
    for (const auto& f : e.coface_range(2))      // 2-simplices sharing the edge
      for (const auto& e2 : f.face_range(1))     // their edges are the neighbours
        try_edge(e2);
  }
  std::printf("[part 2a] |S| = %zu crossing edges (%zu edges tested)\n",
              edges.size(), visited.size());

  // ---------------- Part 2b: barycentric construction of K ----------------
  // v(edge) = crossing point;  v(tau), dim tau > 1, = average of v over the
  // crossing edges of tau (eq. 26).  Build by accumulating each edge point into
  // all its cofaces of every dimension 2..d.
  std::vector<Vec> V;
  std::vector<std::unordered_map<std::string, int>> cross_vid(d + 1);   // dim -> skey -> vid
  std::vector<std::unordered_map<std::string, Vec>> accum(d + 1);
  std::vector<std::unordered_map<std::string, int>> accum_cnt(d + 1);
  std::unordered_map<std::string, Simplex> top_simplices;

  for (auto& [k, er] : edges) {
    cross_vid[1][k] = (int)V.size();
    V.push_back(er.pt);
    for (int dim = 2; dim <= d; ++dim) {
      for (const auto& cf : er.s.coface_range(dim)) {
        const std::string ck = skey(cf);
        auto it = accum[dim].find(ck);
        if (it == accum[dim].end()) { accum[dim][ck] = er.pt; accum_cnt[dim][ck] = 1; }
        else { it->second += er.pt; accum_cnt[dim][ck] += 1; }
        if (dim == d) top_simplices.emplace(ck, cf);
      }
    }
  }
  for (int dim = 2; dim <= d; ++dim)
    for (auto& [ck, sum] : accum[dim]) {
      cross_vid[dim][ck] = (int)V.size();
      V.push_back(sum / accum_cnt[dim][ck]);
    }

  // Enumerate flags tau^d > tau^{d-1} > ... > tau^1 of crossing simplices.
  std::vector<std::vector<int>> Fc;
  std::vector<int> chain;
  std::function<void(const Simplex&, int)> descend = [&](const Simplex& tau, int dim) {
    chain.push_back(cross_vid[dim].at(skey(tau)));
    if (dim == 1) {
      Fc.push_back(chain);                          // d vertices, one per dimension
    } else {
      for (const auto& f : tau.face_range(dim - 1)) {
        auto jt = cross_vid[dim - 1].find(skey(f));
        if (jt != cross_vid[dim - 1].end()) descend(f, dim - 1);
      }
    }
    chain.pop_back();
  };
  for (auto& [ck, s] : top_simplices) descend(s, d);

  // Orient each (d-1)-simplex so its normal agrees with grad F (outward).
  for (auto& f : Fc) {
    Vec c = Vec::Zero(d);
    for (int idx : f) c += V[idx];
    c /= (double)d;
    Eigen::MatrixXd A(d, d);
    for (int i = 1; i < d; ++i) A.row(i - 1) = (V[f[i]] - V[f[0]]).transpose();
    A.row(d - 1) = M->grad(c).transpose();
    if (A.determinant() < 0) std::swap(f[d - 2], f[d - 1]);
  }
  std::printf("[part 2b] K: %zu vertices, %zu (%d-1)-simplices\n", V.size(), Fc.size(), d);
  geo.print_stats();

  // ---------------- quality + closeness ----------------
  double min_ratio = 1e300, mean_ratio = 0.0, max_off = 0.0;
  for (const auto& f : Fc) {
    std::vector<Vec> pts; for (int idx : f) pts.push_back(V[idx]);
    double Lmax = 0.0;
    for (std::size_t i = 0; i < pts.size(); ++i)
      for (std::size_t j = i + 1; j < pts.size(); ++j)
        Lmax = std::max(Lmax, (pts[i] - pts[j]).norm());
    const double ratio = (Lmax > 0) ? min_altitude(pts) / Lmax : 0.0;
    min_ratio = std::min(min_ratio, ratio);
    mean_ratio += ratio;
  }
  for (const auto& p : V) max_off = std::max(max_off, M->dist(p));
  std::printf("[quality] min altitude/longest-edge over K : %.4e (mean %.4e)\n",
              Fc.empty() ? 0.0 : min_ratio, Fc.empty() ? 0.0 : mean_ratio / Fc.size());
  std::printf("[closeness] max dist of K's vertices to M : %.4e (= %.4f * L)\n",
              max_off, max_off / L);

  if (d == 3) write_obj(out_path, V, Fc);
  else        write_generic(out_path, d, M->name(), V, Fc);
  std::printf("[output] wrote %s\n", out_path.c_str());
  return 0;
}
