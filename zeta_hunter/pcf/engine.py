import torch
import mpmath as mp
from typing import Optional


class PCFEngine:
    """
    Evaluates generalised Polynomial Continued Fractions (PCFs).

    CF form:  CF = a(1) / (b(1) + a(2) / (b(2) + a(3) / (...)))

    Two evaluation paths:
      batch_pcf()   — GPU float64, fast, ~1M PCFs/batch, 8-digit precision.
      precise_pcf() — CPU mpmath, arbitrary precision, single formula.
    """

    MAX_DEGREE = 7  # supports polynomials up to degree 7 (Apery uses degree 6)

    def __init__(self, depth: int = 500):
        self.depth = depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Precompute n^k for n=1..depth — shape [depth, MAX_DEGREE+1]
        n_vals = torch.arange(1, depth + 1, dtype=torch.float64, device=self.device)
        self.n_powers = torch.stack(
            [n_vals ** k for k in range(self.MAX_DEGREE + 1)], dim=1
        )

    def _poly_vals(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate polynomials at n=1..depth for a batch of coefficient vectors.

        Args:
            coeffs: [B, D+1] — coefficients of n^0, n^1, ..., n^D
        Returns:
            [B, depth]
        """
        d = coeffs.shape[1]
        coeffs_dev = coeffs.to(self.device, dtype=torch.float64)
        return (self.n_powers[:, :d] @ coeffs_dev.T).T

    def batch_pcf(
        self,
        a_coeffs: torch.Tensor,   # [B, a_degree+1]
        b_coeffs: torch.Tensor,   # [B, b_degree+1]
    ) -> torch.Tensor:
        """
        Evaluate a batch of PCFs via backward recurrence on the GPU.

        Returns: [B] float64 tensor. Entries are NaN if recurrence degenerates.
        """
        a_vals = self._poly_vals(a_coeffs)   # [B, depth]
        b_vals = self._poly_vals(b_coeffs)   # [B, depth]

        # Backward recurrence from the deepest term upward
        result = b_vals[:, -1].clone()       # start: b(depth)

        for i in range(self.depth - 2, -1, -1):
            safe = torch.where(
                result.abs() < 1e-300,
                torch.full_like(result, 1e-300),
                result,
            )
            result = b_vals[:, i] + a_vals[:, i + 1] / safe

        # Final step: CF = a(1) / result
        safe = torch.where(
            result.abs() < 1e-300,
            torch.full_like(result, 1e-300),
            result,
        )
        return a_vals[:, 0] / safe

    def precise_pcf(
        self,
        a_coeffs: list,
        b_coeffs: list,
        depth: Optional[int] = None,
        dps: int = 500,
    ) -> Optional[mp.mpf]:
        """
        High-precision single PCF using mpmath backward recurrence.
        Returns None if the recurrence degenerates (zero denominator).
        """
        mp.mp.dps = dps
        d = depth or self.depth * 2

        def poly(coeffs: list, n: mp.mpf) -> mp.mpf:
            return sum(mp.mpf(c) * mp.power(n, k) for k, c in enumerate(coeffs))

        result = poly(b_coeffs, d)
        for i in range(d - 1, 0, -1):
            an = poly(a_coeffs, i + 1)
            bn = poly(b_coeffs, i)
            if result == 0:
                return None
            result = bn + an / result

        a1 = poly(a_coeffs, 1)
        if result == 0:
            return None
        return a1 / result
