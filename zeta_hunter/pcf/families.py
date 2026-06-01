from dataclasses import dataclass


@dataclass(frozen=True)
class PCFFamily:
    """
    Describes a family of Polynomial Continued Fractions to sweep.

    CF form:  a(1) / (b(1) + a(2) / (b(2) + a(3) / (...)))

    The sweeper iterates all integer coefficient tuples for a(n) and b(n)
    within coeff_range, fixing the polynomial degrees.
    """
    name: str
    a_degree: int           # degree of numerator polynomial a(n)
    b_degree: int           # degree of denominator polynomial b(n)
    coeff_range: range      # integer range swept for each coefficient
    depth: int              # CF evaluation depth
    targets: tuple          # constant names from constants.CONSTANTS to compare

    @property
    def a_n_coeffs(self) -> int:
        return self.a_degree + 1

    @property
    def b_n_coeffs(self) -> int:
        return self.b_degree + 1

    @property
    def total_combinations(self) -> int:
        r = len(self.coeff_range)
        return r ** (self.a_n_coeffs + self.b_n_coeffs)


# ---------------------------------------------------------------------------
# Named families — ordered by theoretical priority
# ---------------------------------------------------------------------------

APERY = PCFFamily(
    name="Apery",
    a_degree=6,
    b_degree=3,
    coeff_range=range(-60, 61),
    depth=500,
    targets=("zeta3", "zeta5", "zeta7"),
    # Apery's proof uses a(n) = -n^6, b(n) = 34n^3 - 51n^2 + 27n - 5.
    # Searching the neighbourhood of this shape for relatives hitting zeta(5), zeta(7).
)

ZAGIER = PCFFamily(
    name="Zagier",
    a_degree=2,
    b_degree=2,
    coeff_range=range(-30, 31),
    depth=300,
    targets=("zeta3", "zeta5", "catalan", "beta3"),
    # Zagier's integer sequences with a closed orbit live here.
    # Known to produce zeta(2), beta(3), Catalan at small coefficients.
)

RAMANUJAN = PCFFamily(
    name="Ramanujan",
    a_degree=1,
    b_degree=2,
    coeff_range=range(-20, 21),
    depth=300,
    targets=("zeta3", "4_over_pi", "catalan", "zeta5"),
    # Ramanujan's 4/pi formula is this shape (degree 1/2).
    # Existing notebook found a 6-digit hit for 4/pi here.
)

ALL_FAMILIES = (APERY, ZAGIER, RAMANUJAN)
