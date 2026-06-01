# Zeta Hunter — Design Specification
**Date:** 2026-05-15  
**Status:** Approved  
**Author:** Alvin Kigondu (Kigs Apex LLP)  
**Goal:** Find a closed-form expression for odd Riemann zeta values — ζ(3), ζ(5), ζ(7) — using a theory-guided Polynomial Continued Fraction (PCF) engine backed by an RTX 5080 GPU, with full reproducibility and citation-ready documentation regardless of outcome.

---

## 1. Problem Statement

The even Riemann zeta values have known closed forms (Euler, 1734):
- ζ(2) = π²/6
- ζ(4) = π⁴/90
- ζ(2k) = rational × π^(2k)

The **odd zeta values** — ζ(3), ζ(5), ζ(7), ... — have no known closed form. ζ(3) (Apéry's constant) was proven irrational in 1978 but its transcendence and any closed-form expression remain open after 300+ years of mathematical investigation.

**This project is a moonshot.** If a closed form is found, it is a major mathematical discovery. If not found, the systematic null result is itself a citable contribution.

---

## 2. Scope

### In scope
- Theory-guided PCF sweep targeting ζ(3), ζ(5), ζ(7), Catalan's constant, and 4/π
- PSLQ integer relation search on any PCF hits above 50-digit precision
- Reproducible run logging with enough metadata to re-derive any result
- LaTeX preprint generation from the hit registry
- Three cleaned-up Jupyter notebooks as human-readable companions
- Test suite using known mathematical identities as ground truth

### Out of scope
- Cloud GPU compute (local RTX 5080 only)
- Neural/TGNN approach (deprioritised — PCF is higher expected value)
- Genetic algorithm (kept as a module but not the primary search method)
- Transcendence proofs (computational search only)

---

## 3. Package Architecture

```
zeta_hunter/                      # Root of repository
├── zeta_hunter/                  # Core Python library
│   ├── __init__.py
│   ├── constants.py              # 60+ constants at 500-digit precision
│   ├── pcf/
│   │   ├── __init__.py
│   │   ├── families.py           # PCF family dataclasses
│   │   ├── engine.py             # CUDA float64 batched evaluator
│   │   └── sweeper.py            # Orchestrates Stage 1 → 2 → 3
│   ├── pslq.py                   # PSLQ wrapper with named bases
│   ├── genetic.py                # Genetic formula search (secondary)
│   ├── logger.py                 # Structured JSON logging + hit registry
│   └── report.py                 # LaTeX preprint generator
│
├── notebooks/
│   ├── 01_pcf_sweep.ipynb        # Launch sweep, visualise convergence
│   ├── 02_pslq_search.ipynb      # Follow up on PCF hits
│   └── 03_results_report.ipynb   # Render hit registry, generate .tex
│
├── runs/                         # Auto-created, gitignored
├── docs/
│   └── superpowers/specs/        # This document
├── tests/
│   ├── test_constants.py
│   ├── test_pcf_engine.py
│   └── test_pslq.py
├── pyproject.toml
└── README.md
```

**Boundary rules:**
- `zeta_hunter/` is pure library — no `print()`, no plotting, no hard GPU dependency. Degrades to CPU if CUDA absent.
- `pcf/engine.py` is the only file that imports PyTorch/CUDA.
- `runs/` is gitignored. Significant results are promoted to `docs/` manually.
- Notebooks are thin wrappers — no business logic, only calls to `zeta_hunter`.

---

## 4. Build Order

| Step | Module | Depends on | Validates via |
|------|--------|------------|---------------|
| 1 | `constants.py` | mpmath | `test_constants.py` — known identities at 300 digits |
| 2 | `pcf/engine.py` | PyTorch | `test_pcf_engine.py` — Apéry's known recurrence converges |
| 3 | `pcf/families.py` | nothing | manual inspection |
| 4 | `pcf/sweeper.py` | engine + families + logger | integration test |
| 5 | `pslq.py` | mpmath | `test_pslq.py` — ζ(2) = π²/6 found |
| 6 | `logger.py` | nothing | unit test |
| 7 | `report.py` | logger | manual render check |
| 8 | Notebooks | full library | run top-to-bottom clean |

---

## 5. PCF Engine

### What is a PCF

A Polynomial Continued Fraction has the form:

```
target = b₀ + a₁ / (b₁ + a₂ / (b₂ + a₃ / (...)))
```

where `a(n)` and `b(n)` are polynomials with integer coefficients. Apéry's 1978 irrationality proof for ζ(3) used exactly one such PCF. This engine searches for others.

### Three named families

```python
@dataclass
class PCFFamily:
    name: str
    a_degree: int         # degree of numerator polynomial
    b_degree: int         # degree of denominator polynomial
    coeff_range: range    # integer coefficient sweep range
    depth: int            # continued fraction evaluation depth
    targets: list[str]    # constants to compare against
```

| Family | a(n) degree | b(n) degree | Theoretical basis |
|--------|-------------|-------------|-------------------|
| **Apéry** | 6 | 3 | Apéry's ζ(3) proof uses a(n) = −n⁶, b(n) = 34n³+51n²+27n+5. Adjacent families may hit ζ(5), ζ(7). |
| **Zagier** | 2 | 2 | Zagier's "integer sequences with a closed orbit" — known to produce ζ(2), β(3), Catalan. |
| **Ramanujan** | 1 | 2 | Ramanujan's 4/π formula is this shape. Notebook already found a 6-digit hit for 4/π here. |

### Two-stage precision strategy

```
Stage 1 — GPU (PyTorch float64, RTX 5080)
  Input:     batch of 1,000,000 polynomial coefficient tuples
  Method:    backward recurrence, depth = 500
  Compare:   all targets simultaneously
  Threshold: |result − target| < 1e-8  (8-digit match)
  Output:    promote hits to Stage 2

Stage 2 — CPU (mpmath, 500 digits)
  Input:     exact polynomial coefficients from Stage 1 hit
  Method:    mpmath backward recurrence at 500-digit precision
  Threshold: match > 50 digits  →  log as CANDIDATE
             match > 200 digits →  log as IDENTITY, alert immediately
  Output:    write to hit_registry.json

Stage 3 — CPU (mpmath, 2000 digits)
  Triggered: manually, on any CANDIDATE worth investigating
  Output:    run PSLQ, write to results report
```

**Throughput estimate (RTX 5080, float64):** ~100M PCFs/hour in Stage 1.  
Apéry family at ±60 coefficients = ~5×10¹⁰ combinations → ~3 weeks full sweep.  
Theory-guided pruning (fixing polynomial shape) reduces this to ~10⁸ → hours.

### Hit log entry schema

```json
{
  "run_id": "pcf_apery_20260515_001",
  "family": "Apéry",
  "a_coeffs": [0, 0, 0, 0, 0, 0, -1],
  "b_coeffs": [5, 27, 51, 34],
  "depth": 500,
  "target": "zeta3",
  "stage1_error": 2.4e-11,
  "stage2_precision_digits": 67,
  "stage2_value": "1.20205690315959423664...",
  "verdict": "CANDIDATE",
  "timestamp": "2026-05-15T14:23:01Z"
}
```

---

## 6. PSLQ Module

PSLQ runs after a PCF clears Stage 2. It answers: "can this value be expressed as an integer linear combination of known mathematical constants?"

### Interface

```python
class PSLQSearcher:
    def __init__(self, constants: dict, precision: int = 500)
    def search(self, target: mpf, basis_name: str) -> PSLQResult | None
    def run_all_bases(self, target: mpf) -> list[PSLQResult]
    def generate_bases(self) -> list[Basis]
```

### Seven named bases

| Name | Elements | Rationale |
|------|----------|-----------|
| `pi_powers` | π through π⁶ | Classic Euler hope |
| `pi_ln2` | π^i · ln2^j for i+j ≤ 4 | Appears in Li₃(1/2) identity |
| `apery_polylog` | Li₃(1/2), Li₃(−1), Li₃(1/φ), π³ | Known ζ(3) relatives |
| `clausen_beta` | Cl₂(π/3), β(3), Cl₂(π/4), π³ | Clausen values near ζ(3) |
| `mzv_weight3` | ζ(2,1), ζ(1,2), ζ(3) | Multiple zeta values, weight 3 |
| `gamma_mix` | γ, γ·π², γ², ζ(2) | Euler-Mascheroni combinations |
| `broadhurst` | ζ(3)·ζ(5), ζ(7), π⁸ | Broadhurst conjecture family |

### Verdict categories

| Verdict | Condition | Action |
|---------|-----------|--------|
| `TRIVIAL` | Coefficients include a copy of the target itself | Discard silently |
| `NOISE` | Any coefficient > 10⁶ | Log but do not promote |
| `CANDIDATE` | Small coefficients, 50–200 digit precision | Promote to Stage 3 + human review |
| `IDENTITY` | Small coefficients, 200+ digit precision | Alert immediately, write to report |

---

## 7. Logging

One JSON file per sweep run in `runs/`. A global `hit_registry.json` in the project root aggregates every CANDIDATE and IDENTITY across all runs — append-only, never overwritten.

```
runs/
  pcf_apery_20260515_001.json      ← all Stage 1 hits from this sweep
  pcf_zagier_20260515_002.json
  pslq_followup_20260515_003.json  ← PSLQ run on a Stage 2 hit
hit_registry.json                  ← global, append-only
```

Each run JSON contains:
- Run config (family, coeff range, depth, targets, GPU info, start time)
- All Stage 1 hits
- All Stage 2 verifications
- Total combinations scanned
- Wall-clock time and throughput

---

## 8. LaTeX Report Generator

`report.py` reads `hit_registry.json` and renders a `.tex` file structured as a research preprint.

```
Section 1 — Introduction & background        (static template)
Section 2 — Methodology                      (auto-filled from run configs)
Section 3 — Computational results            (auto-generated table of hits)
Section 4 — Null results                     (search space covered, no hit)
Section 5 — Conclusion                       (static template)
Appendix A — Full hit registry               (every CANDIDATE, sortable)
```

The null results section is treated as a first-class output. "We swept the Apéry family at coefficients ±200, depth 500, covering N combinations, and found no match above 15 digits for ζ(3)" is a citable contribution.

---

## 9. Tests (Mathematical Ground Truth)

| Test file | What it verifies | Pass condition |
|-----------|-----------------|----------------|
| `test_constants.py` | All known identities: ζ(2)=π²/6, ζ(4)=π⁴/90, Li₃(−1)=−3/4·ζ(3), β(3)=π³/32 | Match > 100 digits |
| `test_pcf_engine.py` | Apéry's known recurrence converges to ζ(3) | Stage 2 match > 50 digits |
| `test_pslq.py` | ζ(2) = π²/6 found with small coefficients [−6, 0, 1] | CANDIDATE or IDENTITY verdict |

If any test fails, the engine is wrong. Tests run before every sweep.

---

## 10. Notebooks (Thin Wrappers)

| Notebook | Purpose | Length |
|----------|---------|--------|
| `01_pcf_sweep.ipynb` | Configure and launch a sweep; plot convergence curves and hit distribution | ~15 cells |
| `02_pslq_search.ipynb` | Load a hit from the registry; run all PSLQ bases; display results | ~10 cells |
| `03_results_report.ipynb` | Render hit registry as table; call `report.py` to generate `.tex` | ~8 cells |

No business logic lives in notebooks. Each cell is either a config dict, a single library call, or a plot.

---

## 11. Success Criteria

| Outcome | Definition |
|---------|------------|
| **Major discovery** | A PCF clears Stage 3 (2000-digit match) and PSLQ returns small-coefficient relation → write paper |
| **Minor discovery** | A PCF clears Stage 2 (50+ digits) but PSLQ finds no elementary relation → citable null result with a strong candidate |
| **Null result** | No PCF clears Stage 2 after full sweep → citable null result; update scope to Zagier/Broadhurst families |
| **Infrastructure win** | Engine is reproducible, documented, and open-sourced — usable by other researchers regardless of mathematical outcome |

---

## 12. Known Risks

| Risk | Mitigation |
|------|------------|
| Float64 Stage 1 produces false positives | Stage 2 mpmath verification filters them; every CANDIDATE gets exact re-evaluation |
| Apéry family search space too large | Theory-guided coefficient pruning; start with the immediate neighbourhood of Apéry's known recurrence |
| ζ(3) genuinely has no elementary closed form | Null result is documented and cited; pivot to Zagier/Broadhurst or multi-target lattice search |
| GPU memory limits batch size | Tune batch size to fit within 16GB VRAM; default 1M is conservative |
