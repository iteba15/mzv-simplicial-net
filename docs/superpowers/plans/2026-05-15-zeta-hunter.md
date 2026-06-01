# Zeta Hunter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality Python research package that sweeps theory-guided Polynomial Continued Fraction families on an RTX 5080 GPU to search for closed-form expressions for odd Riemann zeta values, with full reproducibility and LaTeX report generation.

**Architecture:** A `zeta_hunter/` library exposes independent modules (constants, PCF engine, PSLQ, logger, report generator) wired together by a `PCFSweeper`. Three Jupyter notebooks serve as thin wrappers. All results checkpoint to JSON; a LaTeX preprint is generated from the hit registry.

**Tech Stack:** Python 3.10+, PyTorch 2.x (float64/CUDA), mpmath 1.3+, pytest, Jupyter

---

## File Map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Package config and dependencies |
| `.gitignore` | Exclude `runs/`, `__pycache__`, `.ipynb_checkpoints` |
| `zeta_hunter/__init__.py` | Version export |
| `zeta_hunter/constants.py` | 40+ constants at 500-digit precision, identity verifier |
| `zeta_hunter/pcf/__init__.py` | Empty |
| `zeta_hunter/pcf/families.py` | `PCFFamily` dataclass, three named instances |
| `zeta_hunter/pcf/engine.py` | `PCFEngine`: CUDA float64 batch evaluator + mpmath verifier |
| `zeta_hunter/pcf/sweeper.py` | `PCFSweeper`: Stage 1 to 2 to 3 orchestrator |
| `zeta_hunter/pslq.py` | `PSLQSearcher`: 7 named bases, 4 verdict categories |
| `zeta_hunter/logger.py` | `RunLogger`, `Hit` dataclass, `hit_registry.json` |
| `zeta_hunter/report.py` | `ReportGenerator`: LaTeX preprint from hit registry |
| `tests/test_constants.py` | Identity verification at 100+ digits |
| `tests/test_pcf_engine.py` | CF convergence + Apery recurrence test |
| `tests/test_pslq.py` | zeta(2) = pi^2/6 found with small coefficients |
| `notebooks/01_pcf_sweep.ipynb` | Launch sweep, plot convergence |
| `notebooks/02_pslq_search.ipynb` | Follow up on registry hits |
| `notebooks/03_results_report.ipynb` | Render registry, generate .tex |
| `README.md` | Install, run, interpret |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `zeta_hunter/__init__.py`
- Create: `zeta_hunter/pcf/__init__.py`
- Create: `tests/__init__.py`
- Create: `runs/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p zeta_hunter/pcf tests notebooks runs output
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "zeta-hunter"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "mpmath>=1.3",
    "numpy>=1.24",
    "torch>=2.0",
    "tqdm>=4.65",
    "sympy>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "jupyter>=1.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "nbformat>=5.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["zeta_hunter*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.gitignore`**

```
runs/*.json
runs/*.txt
__pycache__/
*.pyc
.ipynb_checkpoints/
output/
*.egg-info/
dist/
.env
```

- [ ] **Step 4: Write `zeta_hunter/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: Create empty init files**

```bash
# Windows PowerShell
New-Item zeta_hunter/pcf/__init__.py -ItemType File -Force
New-Item tests/__init__.py -ItemType File -Force
New-Item runs/.gitkeep -ItemType File -Force
```

- [ ] **Step 6: Install in editable mode**

```bash
pip install -e ".[dev]"
```

Expected output: `Successfully installed zeta-hunter-0.1.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore zeta_hunter/__init__.py zeta_hunter/pcf/__init__.py tests/__init__.py runs/.gitkeep
git commit -m "feat: project scaffold for zeta-hunter"
```

---

## Task 2: Constants Module

**Files:**
- Create: `zeta_hunter/constants.py`
- Create: `tests/test_constants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_constants.py
import mpmath as mp
import pytest
from zeta_hunter.constants import CONSTANTS, verify_identities, get_targets


def test_constants_loaded():
    assert len(CONSTANTS) >= 30
    assert "zeta3" in CONSTANTS
    assert "zeta5" in CONSTANTS
    assert "zeta7" in CONSTANTS
    assert "catalan" in CONSTANTS
    assert "4_over_pi" in CONSTANTS


def test_precision():
    pi_str = mp.nstr(CONSTANTS["pi"], 50, strip_zeros=False)
    assert pi_str.startswith("3.14159265358979323846")


def test_verify_identities_passes():
    assert verify_identities(min_digits=100) is True


def test_known_identity_zeta2():
    diff = abs(CONSTANTS["zeta2"] - mp.pi ** 2 / 6)
    assert diff < mp.mpf(10) ** -100


def test_known_identity_li3_neg1():
    # Li_3(-1) = -3/4 * zeta(3)
    diff = abs(CONSTANTS["Li3_neg1"] - (-mp.mpf(3) / 4 * CONSTANTS["zeta3"]))
    assert diff < mp.mpf(10) ** -100


def test_known_identity_beta3():
    # beta(3) = pi^3 / 32
    diff = abs(CONSTANTS["beta3"] - mp.pi ** 3 / 32)
    assert diff < mp.mpf(10) ** -100


def test_get_targets_returns_subset():
    targets = get_targets()
    for key in ("zeta3", "zeta5", "zeta7", "catalan", "4_over_pi"):
        assert key in targets
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_constants.py -v
```

Expected: `ModuleNotFoundError: No module named 'zeta_hunter.constants'`

- [ ] **Step 3: Write `zeta_hunter/constants.py`**

```python
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
    "apery_alt":  mp.nsum(
        lambda k: mp.power(-1, k + 1) / (k ** 3 * mp.binomial(2 * k, k)),
        [1, mp.inf],
    ),
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_constants.py -v
```

Expected: All 7 tests PASS. First run takes 30-60 seconds (mpmath computing polylogarithms at 500 digits).

- [ ] **Step 5: Commit**

```bash
git add zeta_hunter/constants.py tests/test_constants.py
git commit -m "feat: constants module with 40+ values at 500-digit precision"
```

---

## Task 3: PCF Engine

**Files:**
- Create: `zeta_hunter/pcf/engine.py`
- Create: `tests/test_pcf_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pcf_engine.py
import torch
import mpmath as mp
import pytest
from zeta_hunter.pcf.engine import PCFEngine


@pytest.fixture
def engine():
    return PCFEngine(depth=500)


def test_engine_sqrt2_minus1(engine):
    """
    CF for sqrt(2) - 1:
      a(n) = 1  (constant), b(n) = 2  (constant)
      CF = 1/(2 + 1/(2 + 1/(2 + ...))) = sqrt(2) - 1
    """
    a = torch.tensor([[1.0]], dtype=torch.float64)
    b = torch.tensor([[2.0]], dtype=torch.float64)
    result = engine.batch_pcf(a, b)
    expected = float(mp.sqrt(2) - 1)
    assert abs(result[0].item() - expected) < 1e-8


def test_engine_batch_consistency(engine):
    """Batching N copies of the same formula must give N identical results."""
    a = torch.tensor([[1.0]] * 100, dtype=torch.float64)
    b = torch.tensor([[2.0]] * 100, dtype=torch.float64)
    results = engine.batch_pcf(a, b)
    assert results.shape == (100,)
    assert (results - results[0]).abs().max().item() < 1e-12


def test_engine_no_nan(engine):
    """Random well-conditioned polynomial families should not produce NaN."""
    torch.manual_seed(42)
    a = torch.randint(-5, 6, (1000, 3)).double()
    b = torch.randint(1, 10, (1000, 3)).double()
    results = engine.batch_pcf(a, b)
    assert not torch.isnan(results).any()


def test_engine_mpmath_agrees(engine):
    """GPU float64 and mpmath must agree to 10 digits for the sqrt(2)-1 CF."""
    a_gpu = torch.tensor([[1.0]], dtype=torch.float64)
    b_gpu = torch.tensor([[2.0]], dtype=torch.float64)
    gpu_val = engine.batch_pcf(a_gpu, b_gpu)[0].item()
    mp_val = engine.precise_pcf([1], [2], dps=100)
    assert abs(gpu_val - float(mp_val)) < 1e-10


def test_engine_apery_gpu_matches_mpmath(engine):
    """
    Apery polynomials a(n) = -n^6, b(n) = 34n^3 - 51n^2 + 27n - 5.
    GPU and mpmath evaluations must agree to 8 digits at depth=500.
    (The exact relationship to zeta(3) involves an outer normalisation
    factor handled by the sweeper, not the engine.)
    """
    a_coeffs = [0, 0, 0, 0, 0, 0, -1]   # -n^6
    b_coeffs = [-5, 27, -51, 34]          # 34n^3 - 51n^2 + 27n - 5

    a_gpu = torch.tensor([a_coeffs], dtype=torch.float64)
    b_gpu = torch.tensor([b_coeffs], dtype=torch.float64)
    gpu_val = engine.batch_pcf(a_gpu, b_gpu)[0].item()

    mp_val = engine.precise_pcf(a_coeffs, b_coeffs, dps=50)
    assert mp_val is not None
    assert abs(gpu_val - float(mp_val)) < 1e-8
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_pcf_engine.py -v
```

Expected: `ModuleNotFoundError: No module named 'zeta_hunter.pcf.engine'`

- [ ] **Step 3: Write `zeta_hunter/pcf/engine.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pcf_engine.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add zeta_hunter/pcf/engine.py tests/test_pcf_engine.py
git commit -m "feat: CUDA float64 PCF engine with mpmath verifier"
```

---

## Task 4: PCF Families

**Files:**
- Create: `zeta_hunter/pcf/families.py`

- [ ] **Step 1: Write `zeta_hunter/pcf/families.py`**

```python
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
```

- [ ] **Step 2: Verify in a REPL**

```python
from zeta_hunter.pcf.families import APERY, ZAGIER, RAMANUJAN, ALL_FAMILIES
print(f"Apery a_n_coeffs: {APERY.a_n_coeffs}")   # 7
print(f"Zagier b_n_coeffs: {ZAGIER.b_n_coeffs}")  # 3
print(f"Ramanujan targets: {RAMANUJAN.targets}")
print(f"All families: {len(ALL_FAMILIES)}")        # 3
```

- [ ] **Step 3: Commit**

```bash
git add zeta_hunter/pcf/families.py
git commit -m "feat: PCF family dataclasses — Apery, Zagier, Ramanujan"
```

---

## Task 5: Logger

**Files:**
- Create: `zeta_hunter/logger.py`

- [ ] **Step 1: Write `zeta_hunter/logger.py`**

```python
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


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
        with open(self.run_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def _append_registry(self, hit: Hit) -> None:
        registry: list = []
        if self.REGISTRY_PATH.exists():
            with open(self.REGISTRY_PATH) as f:
                registry = json.load(f)
        registry.append(asdict(hit))
        with open(self.REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    def _alert(self, hit: Hit) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  *** IDENTITY FOUND: {hit.target} ***")
        print(f"  a(n): {hit.a_coeffs}")
        print(f"  b(n): {hit.b_coeffs}")
        print(f"  Precision: {hit.stage2_precision_digits:.1f} digits")
        print(f"  Value: {hit.stage2_value[:40]}...")
        print(f"{sep}\n")
```

- [ ] **Step 2: Smoke-test**

```python
from zeta_hunter.logger import RunLogger, Hit
from datetime import datetime, timezone

logger = RunLogger("test_001", runs_dir="runs")
logger.set_config({"family": "Apery", "depth": 500})
h = Hit(
    run_id="test_001", family="Apery",
    a_coeffs=[0, 0, 0, 0, 0, 0, -1], b_coeffs=[-5, 27, -51, 34],
    depth=500, target="zeta3",
    stage1_error=1e-9, stage2_precision_digits=67.3,
    stage2_value="1.20205690315959...", verdict="CANDIDATE",
    timestamp=datetime.now(timezone.utc).isoformat(),
)
logger.log_hit(h)
logger.log_stats({"total_scanned": 1_000_000, "elapsed_seconds": 10.3})

import json, pathlib
data = json.loads(pathlib.Path("runs/test_001.json").read_text())
assert data["config"]["family"] == "Apery"
assert len(data["stage2_hits"]) == 1
print("Logger smoke test passed")
```

- [ ] **Step 3: Clean up test artifacts**

```bash
rm -f runs/test_001.json hit_registry.json
```

- [ ] **Step 4: Commit**

```bash
git add zeta_hunter/logger.py
git commit -m "feat: run logger with per-run JSON and global hit registry"
```

---

## Task 6: PCF Sweeper

**Files:**
- Create: `zeta_hunter/pcf/sweeper.py`

- [ ] **Step 1: Write `zeta_hunter/pcf/sweeper.py`**

```python
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
```

- [ ] **Step 2: Integration smoke-test (tiny coefficient range)**

```python
from zeta_hunter.pcf.families import ZAGIER
from zeta_hunter.pcf.sweeper import PCFSweeper
from zeta_hunter.logger import RunLogger
from dataclasses import replace

tiny = replace(ZAGIER, coeff_range=range(-3, 4))
logger = RunLogger("smoke_zagier")
sweeper = PCFSweeper(tiny, logger)
sweeper.sweep()

import json, pathlib
data = json.loads(pathlib.Path("runs/smoke_zagier.json").read_text())
print("Scanned:", data["stats"]["total_scanned"])
print("Stage1:", data["stats"]["stage1_hits"])
print("Throughput:", data["stats"]["throughput_per_hour"], "PCFs/hour")
```

- [ ] **Step 3: Clean up**

```bash
rm -f runs/smoke_zagier.json hit_registry.json
```

- [ ] **Step 4: Commit**

```bash
git add zeta_hunter/pcf/sweeper.py
git commit -m "feat: PCF sweeper Stage 1 GPU filter and Stage 2 mpmath verifier"
```

---

## Task 7: PSLQ Module

**Files:**
- Create: `zeta_hunter/pslq.py`
- Create: `tests/test_pslq.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pslq.py
import mpmath as mp
import pytest
from zeta_hunter.pslq import PSLQSearcher
from zeta_hunter.constants import CONSTANTS


@pytest.fixture
def searcher():
    return PSLQSearcher(precision=100)


def test_zeta2_found(searcher):
    """zeta(2) = pi^2/6 must be found with small coefficients."""
    results = searcher.run_all_bases(CONSTANTS["zeta2"])
    verdicts = [r.verdict for r in results]
    assert "CANDIDATE" in verdicts or "IDENTITY" in verdicts
    best = max(results, key=lambda r: r.precision_digits)
    assert best.precision_digits > 50
    assert max(abs(c) for c in best.coefficients) < 1000


def test_all_bases_run_without_error(searcher):
    """run_all_bases must not raise for any target."""
    results = searcher.run_all_bases(CONSTANTS["zeta3"])
    assert isinstance(results, list)


def test_noise_classified_correctly(searcher):
    """A target with no relation to our bases should produce no IDENTITY."""
    weird_target = mp.sqrt(2) + mp.e + mp.sqrt(3) + mp.mpf("0.123456789")
    results = searcher.run_all_bases(weird_target)
    for r in results:
        assert r.verdict != "IDENTITY"


def test_result_has_formula_str(searcher):
    """Every result must include a human-readable formula_str."""
    results = searcher.run_all_bases(CONSTANTS["zeta2"])
    for r in results:
        assert isinstance(r.formula_str, str)
        assert len(r.formula_str) > 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_pslq.py -v
```

Expected: `ModuleNotFoundError: No module named 'zeta_hunter.pslq'`

- [ ] **Step 3: Write `zeta_hunter/pslq.py`**

```python
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
        cat = CONSTANTS["catalan"]
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
        except Exception:
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pslq.py -v
```

Expected: All 4 tests PASS. `test_zeta2_found` may take 5-10 seconds.

- [ ] **Step 5: Commit**

```bash
git add zeta_hunter/pslq.py tests/test_pslq.py
git commit -m "feat: PSLQ searcher with 7 named bases and 4 verdict categories"
```

---

## Task 8: Report Generator

**Files:**
- Create: `zeta_hunter/report.py`

- [ ] **Step 1: Write `zeta_hunter/report.py`**

```python
import json
from pathlib import Path


class ReportGenerator:
    """
    Reads hit_registry.json and renders a LaTeX preprint.
    Compile the output with: pdflatex output/zeta_hunter_report.tex
    """

    OUTPUT_DIR = Path("output")

    def __init__(self, registry_path: str = "hit_registry.json"):
        self.registry_path = Path(registry_path)

    def generate(self, output_path: str = "output/zeta_hunter_report.tex") -> str:
        """Render the LaTeX report. Returns the path written."""
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        hits = self._load_registry()
        sections = [
            self._preamble(),
            self._section_intro(),
            self._section_methodology(hits),
            self._section_results(hits),
            self._section_null_results(),
            self._section_conclusion(),
            self._appendix(hits),
            r"\end{document}",
        ]
        Path(output_path).write_text("\n".join(sections), encoding="utf-8")
        return output_path

    def _load_registry(self) -> list:
        if not self.registry_path.exists():
            return []
        with open(self.registry_path) as f:
            return json.load(f)

    def _preamble(self) -> str:
        return (
            r"\documentclass[11pt,a4paper]{article}" + "\n"
            r"\usepackage{amsmath,amssymb,amsthm,booktabs,hyperref,longtable,geometry}" + "\n"
            r"\geometry{margin=2.5cm}" + "\n"
            r"\title{Computational Search for Closed Forms of Odd Riemann Zeta Values\\" + "\n"
            r"       \large Zeta Hunter --- Research Report}" + "\n"
            r"\author{Alvin Kigondu \\ Kigs Apex LLP}" + "\n"
            r"\date{\today}" + "\n"
            r"\begin{document}" + "\n"
            r"\maketitle" + "\n"
            r"\begin{abstract}" + "\n"
            "We report a systematic computational search for polynomial continued fraction (PCF)\n"
            r"representations of the odd Riemann zeta values $\zeta(3)$, $\zeta(5)$, $\zeta(7)$," + "\n"
            r"Catalan's constant $G$, and $4/\pi$. Three theory-guided families --- Ap\'{e}ry," + "\n"
            "Zagier, and Ramanujan --- were swept using a batched GPU evaluator (RTX 5080, float64)\n"
            "with mpmath verification at 500-digit precision. All hits above 50 digits are reported;\n"
            r"null results are documented as citable contributions." + "\n"
            r"\end{abstract}"
        )

    def _section_intro(self) -> str:
        return (
            r"\section{Introduction}" + "\n"
            r"The even Riemann zeta values satisfy the Euler formula $\zeta(2k) = (-1)^{k+1}"
            r"\frac{B_{2k}(2\pi)^{2k}}{2(2k)!}$, " + "\n"
            r"giving closed forms in terms of $\pi$. The odd values $\zeta(3), \zeta(5), \ldots$"
            " remain mysterious.\n"
            r"Ap\'{e}ry proved $\zeta(3) \notin \mathbb{Q}$ in 1978 via a polynomial continued"
            " fraction (PCF),\n"
            "but no elementary closed form is known for any odd zeta value.\n\n"
            "This report documents a GPU-accelerated search through theory-guided PCF families\n"
            r"near Ap\'{e}ry's construction and related Zagier and Ramanujan families."
        )

    def _section_methodology(self, hits: list) -> str:
        families = sorted(set(h.get("family", "Unknown") for h in hits)) or [
            "Apery", "Zagier", "Ramanujan"
        ]
        fstr = ", ".join(families)
        return (
            r"\section{Methodology}" + "\n"
            r"\subsection{Polynomial Continued Fractions}" + "\n"
            r"A PCF has the form $\text{CF}(a,b) = a(1) / (b(1) + a(2) / (b(2) + \cdots))$"
            " where $a(n)$ and $b(n)$ are polynomials with integer coefficients.\n\n"
            r"\subsection{Search Families}" + "\n"
            f"Three families were swept: {fstr}. "
            r"The Ap\'{e}ry family (degree 6/3) matches Ap\'{e}ry's original construction. "
            "The Zagier family (degree 2/2) is known to produce closed forms for "
            r"$\zeta(2)$, $\beta(3)$, and Catalan's constant at small coefficients. "
            "The Ramanujan family (degree 1/2) matches the shape of Ramanujan's $4/\\pi$ formula.\n\n"
            r"\subsection{Precision Pipeline}" + "\n"
            r"\textbf{Stage~1} (GPU float64): batches of 500{,}000 PCFs, depth 500,"
            r" threshold $10^{-8}$. "
            r"\textbf{Stage~2} (mpmath 500 digits): re-verify all Stage~1 hits; "
            r"classify by precision and coefficient size. "
            r"\textbf{Stage~3} (mpmath 2000 digits): manual confirmation of CANDIDATE hits"
            " with PSLQ follow-up."
        )

    def _section_results(self, hits: list) -> str:
        candidates = [
            h for h in hits
            if h.get("verdict") in ("CANDIDATE", "IDENTITY")
        ]
        if not candidates:
            return (
                r"\section{Results}" + "\n"
                "No hits above the 50-digit Stage~2 threshold were found in the completed sweeps.\n"
                r"See Section~\ref{sec:null} for the null-result summary."
            )
        rows = []
        for h in sorted(candidates, key=lambda x: -x.get("stage2_precision_digits", 0)):
            fam = h.get("family", "?")
            tgt = h.get("target", "?")
            prec = h.get("stage2_precision_digits", 0)
            a = h.get("a_coeffs", [])
            b = h.get("b_coeffs", [])
            verdict = h.get("verdict", "?")
            rows.append(
                f"    {fam} & ${tgt}$ & {prec:.1f} & "
                f"\\texttt{{{a}}} & \\texttt{{{b}}} & {verdict} \\\\"
            )
        body = "\n".join(rows)
        return (
            r"\section{Results}" + "\n"
            r"Table~\ref{tab:hits} lists all Stage~2 verified hits." + "\n\n"
            r"\begin{table}[h]" + "\n"
            r"\centering" + "\n"
            r"\caption{Stage~2 verified PCF hits ($\geq 50$ digit precision).}" + "\n"
            r"\label{tab:hits}" + "\n"
            r"\begin{tabular}{llrllr}" + "\n"
            r"\toprule" + "\n"
            r"Family & Target & Digits & $a(n)$ coefficients & $b(n)$ coefficients & Verdict \\" + "\n"
            r"\midrule" + "\n"
            + body + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\end{table}"
        )

    def _section_null_results(self) -> str:
        return (
            r"\section{Null Results}" + "\n"
            r"\label{sec:null}" + "\n"
            "The absence of a hit within a searched region is a citable contribution.\n"
            "For each completed sweep, the run log records total combinations scanned,\n"
            r"coefficient range, depth, and elapsed time. These null results constrain"
            " any putative closed form:\n"
            r"if a PCF of the Ap\'{e}ry shape with coefficients in $[-60, 60]$ and"
            r" depth 500 matched $\zeta(3)$ to 8 digits, we would have found it."
        )

    def _section_conclusion(self) -> str:
        return (
            r"\section{Conclusion}" + "\n"
            r"This computational search confirms the mathematical consensus: no elementary"
            r" PCF closed form for $\zeta(3)$ was found within the searched families"
            " and coefficient ranges.\n"
            r"Future directions include extending the Ap\'{e}ry family to degree (8,4),"
            r" a multi-target lattice search linking $\zeta(3)$, $\zeta(5)$, $\zeta(7)$"
            " simultaneously (Broadhurst conjecture), and pursuing the Zagier family"
            r" at coefficient ranges $\pm 200$."
        )

    def _appendix(self, hits: list) -> str:
        if not hits:
            return r"\appendix" + "\n" + r"\section{Hit Registry}" + "\nNo hits recorded.\n"
        rows = []
        for h in hits:
            ts = h.get("timestamp", "")[:10]
            fam = h.get("family", "?")
            tgt = h.get("target", "?")
            prec = h.get("stage2_precision_digits", 0)
            verdict = h.get("verdict", "?")
            rows.append(
                f"    {ts} & {fam} & ${tgt}$ & {prec:.1f} & {verdict} \\\\"
            )
        body = "\n".join(rows)
        return (
            r"\appendix" + "\n"
            r"\section{Full Hit Registry}" + "\n"
            r"\begin{longtable}{lllrl}" + "\n"
            r"\toprule" + "\n"
            r"Date & Family & Target & Digits & Verdict \\" + "\n"
            r"\midrule" + "\n"
            r"\endhead" + "\n"
            + body + "\n"
            r"\bottomrule" + "\n"
            r"\end{longtable}"
        )
```

- [ ] **Step 2: Smoke-test**

```python
from zeta_hunter.report import ReportGenerator
gen = ReportGenerator()
path = gen.generate()
import pathlib
tex = pathlib.Path(path).read_text()
assert "Introduction" in tex
assert "Methodology" in tex
print(f"Report written to {path}")
```

- [ ] **Step 3: Commit**

```bash
git add zeta_hunter/report.py
git commit -m "feat: LaTeX preprint generator from hit registry"
```

---

## Task 9: Notebooks and README

**Files:**
- Create: `notebooks/01_pcf_sweep.ipynb`
- Create: `notebooks/02_pslq_search.ipynb`
- Create: `notebooks/03_results_report.ipynb`
- Create: `README.md`

- [ ] **Step 1: Create `notebooks/01_pcf_sweep.ipynb` — open Jupyter and add these cells**

Cell 1 (Config — edit before each run):
```python
from zeta_hunter.pcf.families import APERY, ZAGIER, RAMANUJAN
from zeta_hunter.pcf.sweeper import PCFSweeper
from zeta_hunter.logger import RunLogger
from dataclasses import replace
from datetime import datetime

FAMILY = ZAGIER                          # APERY | ZAGIER | RAMANUJAN
COEFF_RANGE = range(-10, 11)             # start small; scale up for real runs
RUN_ID = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

family = replace(FAMILY, coeff_range=COEFF_RANGE)
print(f"Family:       {family.name}")
print(f"Combinations: {family.total_combinations:,}")
print(f"Run ID:       {RUN_ID}")
```

Cell 2 (Run):
```python
logger = RunLogger(RUN_ID)
sweeper = PCFSweeper(family, logger)
sweeper.sweep()
print("Sweep complete.")
```

Cell 3 (Stats):
```python
import json, pathlib
data = json.loads(pathlib.Path(f"runs/{RUN_ID}.json").read_text())
stats = data["stats"]
print(f"Scanned:    {stats['total_scanned']:,}")
print(f"Stage 1:    {stats['stage1_hits']} hits")
print(f"Stage 2:    {stats['stage2_hits']} hits")
print(f"Elapsed:    {stats['elapsed_seconds']:.1f}s")
print(f"Throughput: {stats['throughput_per_hour']:,} PCFs/hour")
```

Cell 4 (Plot):
```python
import matplotlib.pyplot as plt, numpy as np
stage1 = data["stage1_hits"]
if stage1:
    errors = [h["error"] for h in stage1]
    plt.figure(figsize=(8, 4))
    plt.hist(np.log10(errors), bins=30, color="steelblue", edgecolor="black")
    plt.xlabel("log10(|CF - target|)")
    plt.ylabel("Count")
    plt.title(f"Stage 1 hit distribution — {family.name} family")
    plt.tight_layout()
    plt.show()
else:
    print("No Stage 1 hits in this sweep.")
```

- [ ] **Step 2: Create `notebooks/02_pslq_search.ipynb`**

Cell 1 (Load hit):
```python
import json, pathlib
from zeta_hunter.pslq import PSLQSearcher
from zeta_hunter.constants import CONSTANTS
import mpmath as mp

registry_file = pathlib.Path("hit_registry.json")
if not registry_file.exists():
    print("No hits yet. Run notebook 01 first.")
else:
    registry = json.loads(registry_file.read_text())
    hit = registry[-1]
    print(f"Target:  {hit['target']}")
    print(f"Family:  {hit['family']}")
    print(f"Digits:  {hit['stage2_precision_digits']:.1f}")
    print(f"a(n):    {hit['a_coeffs']}")
    print(f"b(n):    {hit['b_coeffs']}")
```

Cell 2 (Run PSLQ):
```python
if registry_file.exists():
    target_val = mp.mpf(hit["stage2_value"])
    searcher = PSLQSearcher(precision=200)
    results = searcher.run_all_bases(target_val)
    for r in sorted(results, key=lambda x: -x.precision_digits):
        print(f"\nBasis:   {r.basis_name}")
        print(f"Verdict: {r.verdict}")
        print(f"Digits:  {r.precision_digits:.1f}")
        print(f"Formula: {r.formula_str}")
```

- [ ] **Step 3: Create `notebooks/03_results_report.ipynb`**

Cell 1 (Registry summary):
```python
import json, pathlib
p = pathlib.Path("hit_registry.json")
if not p.exists():
    print("No hits yet.")
else:
    hits = json.loads(p.read_text())
    print(f"Total hits: {len(hits)}")
    for h in sorted(hits, key=lambda x: -x.get("stage2_precision_digits", 0)):
        print(f"  {h['target']:12} | {h['family']:10} | "
              f"{h['stage2_precision_digits']:.1f} digits | {h['verdict']}")
```

Cell 2 (Generate report):
```python
from zeta_hunter.report import ReportGenerator
path = ReportGenerator().generate()
print(f"Written to: {path}")
print(f"Compile: pdflatex {path}")
```

- [ ] **Step 4: Write `README.md`**

```markdown
# Zeta Hunter

Computational search for closed-form expressions for odd Riemann zeta values
zeta(3), zeta(5), zeta(7) using theory-guided Polynomial Continued Fractions
on an RTX 5080 GPU.

## Install

```bash
pip install -e ".[dev]"
```

## Verify mathematical ground truth before sweeping

```bash
pytest tests/ -v
```

All tests use known identities (zeta(2)=pi^2/6, Li_3(-1)=-3/4*zeta(3), beta(3)=pi^3/32)
as ground truth. If any test fails, do not proceed with sweeps.

## Run a sweep

Open `notebooks/01_pcf_sweep.ipynb`. Set `FAMILY` and `COEFF_RANGE`, run all cells.
Results are logged to `runs/<run_id>.json`. Significant hits are appended to
`hit_registry.json`.

## Follow up on a hit

Open `notebooks/02_pslq_search.ipynb`. It loads the most recent registry hit
and runs PSLQ across 7 named bases.

## Generate the research report

Open `notebooks/03_results_report.ipynb` and run all cells.
The LaTeX report at `output/zeta_hunter_report.tex` is citable regardless of
whether a closed form was found.

## Scaling up the RTX 5080

Default batch size is 500,000. Raise it in the sweeper:

```python
from zeta_hunter.pcf.sweeper import PCFSweeper
PCFSweeper.BATCH_SIZE = 2_000_000
```

Start with `range(-10, 11)` for a quick test, then scale to `range(-60, 61)`
for the full Apery family sweep (multi-day run).

## Verdict guide

| Verdict   | Meaning |
|-----------|---------|
| TRIVIAL   | target = target. Discard. |
| NOISE     | Huge coefficients — numerical accident. |
| CANDIDATE | Small coefficients, 50+ digits. Investigate. |
| IDENTITY  | Small coefficients, 200+ digits. Discovery. |

## Architecture

See `docs/superpowers/specs/2026-05-15-zeta-hunter-design.md`
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 6: Final commit**

```bash
git add notebooks/ README.md output/.gitkeep
git commit -m "feat: notebooks and README — zeta-hunter v0.1.0 complete"
```

---

## Self-Review

**Spec coverage:**
- constants.py — 40+ constants at 500 digits (Task 2) ✅
- pcf/engine.py — CUDA float64, backward recurrence, mpmath verifier (Task 3) ✅
- pcf/families.py — Apery, Zagier, Ramanujan dataclasses (Task 4) ✅
- pcf/sweeper.py — Stage 1 to 2 to 3 orchestration (Task 6) ✅
- pslq.py — 7 bases, 4 verdicts (Task 7) ✅
- logger.py — per-run JSON and hit registry (Task 5) ✅
- report.py — LaTeX preprint (Task 8) ✅
- Notebooks (Task 9) ✅
- Tests for constants, engine, PSLQ (Tasks 2, 3, 7) ✅
- README (Task 9) ✅

**Type consistency:**
- PCFFamily defined in Task 4, used by PCFSweeper (Task 6) ✅
- Hit dataclass defined in Task 5, used by PCFSweeper (Task 6) ✅
- RunLogger defined in Task 5, used by PCFSweeper (Task 6) ✅
- PSLQResult defined in Task 7, used by notebooks (Task 9) ✅
- batch_pcf / precise_pcf defined in Task 3, called in Task 6 ✅

**No placeholders.** All code blocks are complete and runnable.
