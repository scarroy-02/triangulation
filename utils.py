import math
from scipy.special import stirling2, binom

# d = ambient dimension
# n = manifold dimension
# reach = reach of the manifold

# Defining the constants
def t(d):
    if d%2 == 1:
        return math.sqrt(2.0 / d)
    else:
        return math.sqrt((2 * (d + 1)) / (d * (d + 2)))

def delta(d):
    return (math.sqrt(d**2 + 2.0 * d + 24.0) - math.sqrt(d**2 + 2.0 * d)) / (math.sqrt(12.0 * (d + 1)))

def c_tilde(d):
    num = (math.sqrt(d**2 + 2.0 * d + 24.0) - math.sqrt(d**2 + 2.0 * d))
    
    if d&2 == 1:
        term1 = math.sqrt(2) * num / (9 * math.pow(d, 1.5) * (d + 1) * (math.sqrt(d + 2)))
    else:
        term1 = math.sqrt(2 * (d + 1)) * num / (9 * math.pow(d, 2) * (math.pow((d + 2), 1.5)))
    
    term2 = math.pow(t(d), 2) / 24.0
    return min(term1, term2, 1.0 / 24)

def Nleqk(d,k):
    tot = 0
    for i in range(1, k+1):
        tot = tot + math.factorial(i) * stirling2(d + 1, i)
    return 2 + tot

def rho_1(d,n):
    return 1.0 / (math.sqrt(d) * Nleqk(d, d - n - 1))

def alpha(d,k,n):
    return (math.pow(2.0, k + 1) * math.pow(rho_1(d,n), k) * math.pow(c_tilde(d), k)) / math.pow(3.0, k)

def zeta(d,n):
    num = 8 * (1 - (8.0 * c_tilde(d) / t(d)**2)) * t(d)
    den = 15 * math.sqrt(d) * binom(d, d - n) * (1 + 2 * c_tilde(d))
    return num / den

def L(d,n,reach):
    num = math.pow(alpha(d, d - n, n), 4 + 2 * n) * math.pow(zeta(d,n), 2 * n) / (3 * (n + 1)**2)
    den = (math.pow(alpha(d, d - n, n), 4 + 2 * n) * math.pow(zeta(d,n), 2 * n) / (6 * (n + 1)**2))**2 + 36.0
    return reach * num / den


def print_constants(d, n, reach):
    """Print all constants for debugging."""
    print(f"d = {d}, n = {n}, reach = {reach}")
    print(f"t(d) = {t(d):.6f}")
    print(f"delta(d) = {delta(d):.6f}")
    print(f"c_tilde(d) = {c_tilde(d):.6f}")
    print(f"Nleqk(d, d-n-1) = Nleqk({d}, {d-n-1}) = {Nleqk(d, d-n-1)}")
    print(f"rho_1(d, n) = {rho_1(d, n):.6f}")
    print(f"zeta(d, n) = {zeta(d, n):.6f}")
    print(f"L_bound = {L(d, n, reach):.30f}")
    print(f"L/reach = {L(d, n, reach)/reach:.30f}")


if __name__ == "__main__":
    # Test for d=2, n=1
    print_constants(2, 1, 1.0)