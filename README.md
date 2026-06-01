# Zeta Hunter

Computational search for closed-form expressions for odd Riemann zeta values
ζ(3), ζ(5), ζ(7) using theory-guided Polynomial Continued Fractions on an
RTX 5080 GPU.

## Install

```bash
pip install -e ".[dev]"
```

## Verify mathematical ground truth before sweeping

```bash
pytest tests/ -v
```

All tests use known identities (ζ(2)=π²/6, Li₃(-1)=-3/4·ζ(3), β(3)=π³/32)
as ground truth. If any test fails, do not proceed with sweeps.

## Run a sweep

Open `notebooks/01_pcf_sweep.ipynb`. Set `FAMILY` and `COEFF_RANGE`, run all
cells. Results are logged to `runs/<run_id>.json`. Significant hits are
appended to `hit_registry.json`.

## Follow up on a hit

Open `notebooks/02_pslq_search.ipynb`. It loads the most recent registry hit
and runs PSLQ across 7 named bases.

## Generate the research report

Open `notebooks/03_results_report.ipynb` and run all cells. The LaTeX report
at `output/zeta_hunter_report.tex` is citable regardless of whether a closed
form was found.

## Scaling up the RTX 5080

Default batch size is 500,000. Raise it in the sweeper:

```python
from zeta_hunter.pcf.sweeper import PCFSweeper
PCFSweeper.BATCH_SIZE = 2_000_000
```

Start with `range(-10, 11)` for a quick test, then scale to `range(-60, 61)`
for the full Apery family sweep (multi-day run).

## Verdict guide

| Verdict   | Meaning                                          |
|-----------|--------------------------------------------------|
| TRIVIAL   | target = target. Discard.                        |
| NOISE     | Huge coefficients — numerical accident.          |
| CANDIDATE | Small coefficients, 50+ digits. Investigate.     |
| IDENTITY  | Small coefficients, 200+ digits. Discovery.      |

## Architecture

See `docs/superpowers/specs/2026-05-15-zeta-hunter-design.md`
