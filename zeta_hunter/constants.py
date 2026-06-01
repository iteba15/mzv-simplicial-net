import mpmath as mp
from typing import Dict

mp.mp.dps = 500

CONSTANTS: Dict[str, mp.mpf] = {
    # Fundamental
    "pi":         mp.pi,
    "e":          mp.e,
    "phi":        mp.phi,
    "gamma":      mp.euler,
    # Logarithms
    "ln2":        mp.log(2),
    "ln3":        mp.log(3),
    "ln5":        mp.log(5),
    "ln7":        mp.log(7),
    "ln10":       mp.log(10),
    # Square roots
    "sqrt2":      mp.sqrt(2),
    "sqrt3":      mp.sqrt(3),
    "sqrt5":      mp.sqrt(5),
    "sqrt6":      mp.sqrt(6),
    "sqrt7":      mp.sqrt(7),
    # Riemann zeta (even — known closed forms, used for validation)
    "zeta2":      mp.zeta(2),
    "zeta4":      mp.zeta(4),
    "zeta6":      mp.zeta(6),
    "zeta8":      mp.zeta(8),
    # Riemann zeta (odd — primary targets)
    "zeta3":      mp.zeta(3),
    "zeta5":      mp.zeta(5),
    "zeta7":      mp.zeta(7),
    "zeta9":      mp.zeta(9),
    # Catalan's constant and 4/pi
    "catalan":    mp.catalan,
    "4_over_pi":  mp.mpf(4) / mp.pi,
    # Polylogarithms
    "Li2_half":   mp.polylog(2, mp.mpf("0.5")),
    "Li2_neg1":   mp.polylog(2, -1),
    "Li3_half":   mp.polylog(3, mp.mpf("0.5")),
    "Li3_neg1":   mp.polylog(3, -1),
    "Li3_phi_inv":mp.polylog(3, 1 / mp.phi),
    "Li4_half":   mp.polylog(4, mp.mpf("0.5")),
    # Dirichlet beta values
    "beta2":      mp.catalan,            # beta(2) = Catalan
    "beta3":      mp.pi ** 3 / 32,       # beta(3) = pi^3/32
    # Clausen functions (imaginary part of polylogarithm)
    "Cl2_pi2":    mp.catalan,            # Cl_2(pi/2) = Catalan
    "Cl2_pi3":    mp.im(mp.polylog(2, mp.exp(1j * mp.pi / 3))),
    "Cl2_pi4":    mp.im(mp.polylog(2, mp.exp(1j * mp.pi / 4))),
    # Special combinations used by known identities
    "pi2_ln2":    mp.pi ** 2 * mp.log(2),
    "pi_ln2_sq":  mp.pi * mp.log(2) ** 2,
    "ln2_cubed":  mp.log(2) ** 3,
    # Misc
    "glaisher":   mp.glaisher,
    "khinchin":   mp.khinchin,
}

# (constant_name, human_formula, expected_value)
KNOWN_IDENTITIES = [
    ("zeta2",    "pi^2/6",         mp.pi ** 2 / 6),
    ("zeta4",    "pi^4/90",        mp.pi ** 4 / 90),
    ("zeta6",    "pi^6/945",       mp.pi ** 6 / 945),
    ("Li2_neg1", "-pi^2/12",       -mp.pi ** 2 / 12),
    ("Li3_neg1", "-3/4*zeta(3)",   -mp.mpf(3) / 4 * mp.zeta(3)),
    ("beta2",    "catalan",         mp.catalan),
    ("beta3",    "pi^3/32",         mp.pi ** 3 / 32),
    ("Cl2_pi2",  "catalan",         mp.catalan),
]


def verify_identities(min_digits: int = 100) -> bool:
    """Raise AssertionError if any known identity fails at min_digits precision."""
    for name, formula, expected in KNOWN_IDENTITIES:
        diff = abs(CONSTANTS[name] - expected)
        if diff > mp.mpf(10) ** (-min_digits):
            raise AssertionError(
                f"Identity failed: {name} = {formula}, diff = {float(diff):.2e}"
            )
    return True


def get_targets() -> Dict[str, mp.mpf]:
    """Return the primary search targets."""
    keys = ["zeta3", "zeta5", "zeta7", "zeta9", "catalan", "4_over_pi"]
    return {k: CONSTANTS[k] for k in keys}
