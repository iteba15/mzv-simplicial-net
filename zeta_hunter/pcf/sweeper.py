import itertools
from datetime import datetime, timezone
from typing import Iterator, Optional

import mpmath as mp
import torch
from tqdm import tqdm

from zeta_hunter.constants import CONSTANTS
from zeta_hunter.logger import Hit, RunLogger
from zeta_hunter.pcf.engine import PCFEngine
from zeta_hunter.pcf.families import PCFFamily


class PCFSweeper:
    """
    Orchestrates a theory-guided PCF sweep across three stages.

    Stage 1 — GPU float64:   batch_pcf(), threshold 1e-8
    Stage 2 — mpmath 500d:   precise_pcf(), classify verdict
    Stage 3 — mpmath 2000d:  manual trigger for CANDIDATE hits
    """

    BATCH_SIZE = 500_000
    STAGE1_THRESHOLD = 1e-8
    STAGE2_DPS = 500
    STAGE2_MIN_DIGITS = 50
    IDENTITY_MIN_DIGITS = 200

    def __init__(self, family: PCFFamily, logger: RunLogger):
        self.family = family
        self.logger = logger
        self.engine = PCFEngine(depth=family.depth)
        self.targets_float = {
            name: float(CONSTANTS[name]) for name in family.targets
        }
        self.targets_mp = {name: CONSTANTS[name] for name in family.targets}

    def sweep(self) -> None:
        """Run the full sweep. Logs all Stage 2 hits."""
        self.logger.set_config({
            "family": self.family.name,
            "a_degree": self.family.a_degree,
            "b_degree": self.family.b_degree,
            "coeff_range": [
                self.family.coeff_range.start,
                self.family.coeff_range.stop,
            ],
            "depth": self.family.depth,
            "targets": list(self.family.targets),
            "batch_size": self.BATCH_SIZE,
        })

        total_scanned = 0
        stage1_hits = 0
        stage2_hits = 0
        start = datetime.now(timezone.utc)

        for batch_a, batch_b in tqdm(
            self._generate_batches(),
            desc=f"Sweeping {self.family.name}",
            unit="batch",
        ):
            a_tensor = torch.tensor(batch_a, dtype=torch.float64)
            b_tensor = torch.tensor(batch_b, dtype=torch.float64)
            results = self.engine.batch_pcf(a_tensor, b_tensor)
            total_scanned += len(batch_a)

            for target_name, target_val in self.targets_float.items():
                diff = (results - target_val).abs()
                hit_mask = (diff < self.STAGE1_THRESHOLD) & torch.isfinite(results)
                if not hit_mask.any():
                    continue
                for idx in hit_mask.nonzero(as_tuple=True)[0]:
                    stage1_hits += 1
                    a_hit = batch_a[idx.item()]
                    b_hit = batch_b[idx.item()]
                    err = diff[idx].item()
                    self.logger.log_stage1_hit(a_hit, b_hit, target_name, err)
                    hit = self._verify_stage2(a_hit, b_hit, target_name, err)
                    if hit:
                        stage2_hits += 1
                        self.logger.log_hit(hit)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        self.logger.log_stats({
            "total_scanned": total_scanned,
            "stage1_hits": stage1_hits,
            "stage2_hits": stage2_hits,
            "elapsed_seconds": elapsed,
            "throughput_per_hour": (
                int(total_scanned / elapsed * 3600) if elapsed > 0 else 0
            ),
        })

    def _generate_batches(self) -> Iterator[tuple]:
        """Yield (a_batch, b_batch) pairs, each of length BATCH_SIZE."""
        a_nc = self.family.a_n_coeffs
        b_nc = self.family.b_n_coeffs
        cr = list(self.family.coeff_range)
        batch_a: list = []
        batch_b: list = []
        for combo in itertools.product(cr, repeat=a_nc + b_nc):
            batch_a.append(list(combo[:a_nc]))
            batch_b.append(list(combo[a_nc:]))
            if len(batch_a) == self.BATCH_SIZE:
                yield batch_a, batch_b
                batch_a, batch_b = [], []
        if batch_a:
            yield batch_a, batch_b

    def _verify_stage2(
        self,
        a_coeffs: list,
        b_coeffs: list,
        target_name: str,
        stage1_error: float,
    ) -> Optional[Hit]:
        """Re-compute at 500-digit precision. Return a Hit or None."""
        mp_val = self.engine.precise_pcf(
            a_coeffs, b_coeffs,
            depth=self.family.depth * 2,
            dps=self.STAGE2_DPS,
        )
        if mp_val is None:
            return None

        target_mp = self.targets_mp[target_name]
        diff = abs(mp_val - target_mp)
        precision = (
            float(self.STAGE2_DPS) if diff == 0
            else -float(mp.log10(diff))
        )
        verdict = self._classify(a_coeffs, b_coeffs, precision)

        if precision < self.STAGE2_MIN_DIGITS and verdict not in ("CANDIDATE", "IDENTITY"):
            return None

        return Hit(
            run_id=self.logger.run_id,
            family=self.family.name,
            a_coeffs=a_coeffs,
            b_coeffs=b_coeffs,
            depth=self.family.depth,
            target=target_name,
            stage1_error=stage1_error,
            stage2_precision_digits=precision,
            stage2_value=mp.nstr(mp_val, 40),
            verdict=verdict,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _classify(
        self, a_coeffs: list, b_coeffs: list, precision: float
    ) -> str:
        all_c = a_coeffs + b_coeffs
        max_c = max(abs(c) for c in all_c) if all_c else 0
        if precision >= self.IDENTITY_MIN_DIGITS:
            return "IDENTITY"
        if precision >= self.STAGE2_MIN_DIGITS:
            return "CANDIDATE" if max_c <= 10_000 else "NOISE"
        return "NOISE" if max_c > 1_000_000 else "TRIVIAL"
