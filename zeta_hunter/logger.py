import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass
class Hit:
    run_id: str
    family: str
    a_coeffs: list
    b_coeffs: list
    depth: int
    target: str
    stage1_error: float
    stage2_precision_digits: float
    stage2_value: str
    verdict: str      # TRIVIAL | NOISE | CANDIDATE | IDENTITY
    timestamp: str


class RunLogger:
    """
    Writes per-run JSON logs and maintains a global append-only hit_registry.json.

    Per-run file:  runs/<run_id>.json
    Global file:   hit_registry.json  (CANDIDATE and IDENTITY hits only)
    """

    REGISTRY_PATH = Path("hit_registry.json")

    def __init__(self, run_id: str, runs_dir: str = "runs"):
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)
        self.run_path = self.runs_dir / f"{run_id}.json"
        self._data: dict = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": {},
            "stage1_hits": [],
            "stage2_hits": [],
            "stats": {},
        }

    def set_config(self, config: dict) -> None:
        self._data["config"] = config

    def log_stage1_hit(
        self, a_coeffs: list, b_coeffs: list, target: str, error: float
    ) -> None:
        self._data["stage1_hits"].append(
            {"a_coeffs": a_coeffs, "b_coeffs": b_coeffs,
             "target": target, "error": error}
        )

    def log_hit(self, hit: Hit) -> None:
        """Record a Stage 2 verified hit. Promote CANDIDATE/IDENTITY to registry."""
        self._data["stage2_hits"].append(asdict(hit))
        self._flush()
        if hit.verdict in ("CANDIDATE", "IDENTITY"):
            self._append_registry(hit)
        if hit.verdict == "IDENTITY":
            self._alert(hit)

    def log_stats(self, stats: dict) -> None:
        self._data["stats"] = stats
        self._data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def _flush(self) -> None:
        tmp = self.run_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        os.replace(tmp, self.run_path)

    def _append_registry(self, hit: Hit) -> None:
        lock_path = self.REGISTRY_PATH.with_suffix(".lock")
        # Acquire advisory lock using O_CREAT | O_EXCL (atomic on same filesystem)
        for _ in range(50):
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.1)
        else:
            # Could not acquire lock after 5s — proceed anyway to avoid deadlock
            pass
        try:
            registry: list = []
            if self.REGISTRY_PATH.exists():
                with open(self.REGISTRY_PATH) as f:
                    registry = json.load(f)
            registry.append(asdict(hit))
            tmp = self.REGISTRY_PATH.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(registry, f, indent=2, default=str)
            os.replace(tmp, self.REGISTRY_PATH)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _alert(self, hit: Hit) -> None:
        _log.critical(
            "IDENTITY FOUND: %s | a(n)=%s | b(n)=%s | precision=%.1f digits | value=%s...",
            hit.target, hit.a_coeffs, hit.b_coeffs,
            hit.stage2_precision_digits, hit.stage2_value[:40],
        )
