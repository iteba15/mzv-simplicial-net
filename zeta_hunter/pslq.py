from dataclasses import dataclass
from typing import Optional

import mpmath as mp

from zeta_hunter.constants import CONSTANTS


@dataclass
class PSLQResult:
    basis_name: str
    coefficients: list
    precision_digits: float
    verdict: str           # TRIVIAL | NOISE | CANDIDATE | IDENTITY
    formula_str: str       # human-readable reconstruction


class PSLQSearcher:
    """
    Searches for integer linear relations: c0*target + c1*b1 + ... + cn*bn = 0
    using mpmath's PSLQ implementation.

    Verdict thresholds:
        IDENTITY  — precision > 200 digits, max|coeff| <= 1000
        CANDIDATE — precision > 50  digits, max|coeff| <= 1_000_000
        NOISE     — large coefficients or low precision
        TRIVIAL   — relation involves only the target itself
    """

    IDENTITY_DIGITS = 200
    CANDIDATE_DIGITS = 50

    def __init__(self, precision: int = 500):
        self.precision = precision
        mp.mp.dps = max(mp.mp.dps, precision)

    def _bases(self) -> dict:
        """Return all seven named bases as {name: (values, labels)}."""
        pi = CONSTANTS["pi"]
        ln2 = CONSTANTS["ln2"]
        gamma = CONSTANTS["gamma"]

        return {
            "pi_powers": (
                [pi, pi**2, pi**3, pi**4, pi**5, pi**6],
                ["pi", "pi^2", "pi^3", "pi^4", "pi^5", "pi^6"],
            ),
            "pi_ln2": (
                [
                    pi**i * ln2**j
                    for i in range(4) for j in range(4)
                    if 0 < i + j <= 4
                ],
                [
                    f"pi^{i}*ln2^{j}"
                    for i in range(4) for j in range(4)
                    if 0 < i + j <= 4
                ],
            ),
            "apery_polylog": (
                [CONSTANTS["Li3_half"], CONSTANTS["Li3_neg1"],
                 CONSTANTS["Li3_phi_inv"], pi**3],
                ["Li3(1/2)", "Li3(-1)", "Li3(1/phi)", "pi^3"],
            ),
            "clausen_beta": (
                [CONSTANTS["Cl2_pi3"], CONSTANTS["beta3"],
                 CONSTANTS["Cl2_pi4"], pi**3],
                ["Cl2(pi/3)", "beta(3)", "Cl2(pi/4)", "pi^3"],
            ),
            "mzv_weight3": (
                [CONSTANTS["zeta3"], CONSTANTS["zeta2"] * CONSTANTS["gamma"]],
                ["zeta(3)", "zeta(2)*gamma"],
            ),
            "gamma_mix": (
                [gamma, gamma * pi**2, gamma**2, CONSTANTS["zeta2"]],
                ["gamma", "gamma*pi^2", "gamma^2", "zeta(2)"],
            ),
            "broadhurst": (
                [CONSTANTS["zeta3"] * CONSTANTS["zeta5"],
                 CONSTANTS["zeta7"], pi**8],
                ["zeta(3)*zeta(5)", "zeta(7)", "pi^8"],
            ),
        }

    def run_all_bases(self, target: mp.mpf) -> list:
        """Run PSLQ against all seven bases. Returns list of PSLQResult."""
        results = []
        for name, (values, labels) in self._bases().items():
            result = self._run_single(target, name, values, labels)
            if result is not None:
                results.append(result)
        return results

    def _run_single(
        self,
        target: mp.mpf,
        basis_name: str,
        basis: list,
        labels: list,
    ) -> Optional[PSLQResult]:
        try:
            vector = [target] + basis
            relation = mp.pslq(
                vector,
                tol=mp.mpf(10) ** (-self.precision // 2),
                maxcoeff=10**8,
                maxsteps=5000,
            )
            if not relation or relation[0] == 0:
                return None

            residual = sum(mp.mpf(r) * v for r, v in zip(relation, vector))
            if residual == 0:
                precision = float(self.precision)
            else:
                precision = -float(
                    mp.log10(abs(residual) + mp.mpf(10) ** (-self.precision))
                )

            coeffs = [int(c) for c in relation]
            verdict = self._classify(coeffs, precision)
            formula = self._reconstruct(coeffs[1:], labels)

            return PSLQResult(
                basis_name=basis_name,
                coefficients=coeffs,
                precision_digits=precision,
                verdict=verdict,
                formula_str=f"target = {formula}",
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"PSLQ failed on basis '{basis_name}': {exc}", stacklevel=2)
            return None

    def _classify(self, coefficients: list, precision: float) -> str:
        nonzero = [c for c in coefficients if c != 0]
        if len(nonzero) <= 1:
            return "TRIVIAL"
        max_c = max(abs(c) for c in coefficients)
        if precision >= self.IDENTITY_DIGITS and max_c <= 1000:
            return "IDENTITY"
        if precision >= self.CANDIDATE_DIGITS and max_c <= 1_000_000:
            return "CANDIDATE"
        return "NOISE"

    def _reconstruct(self, basis_coeffs: list, labels: list) -> str:
        """Build a readable formula string from PSLQ output coefficients."""
        terms = []
        for coeff, label in zip(basis_coeffs, labels):
            if coeff == 0:
                continue
            sign = "+" if coeff > 0 else "-"
            val = abs(coeff)
            terms.append(f"{sign} {val}*{label}" if val != 1 else f"{sign} {label}")
        return " ".join(terms).lstrip("+ ") if terms else "0"
