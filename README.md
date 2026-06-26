# MZV Simplicial Net

Computational research toward closed-form representations of odd Riemann zeta values.

The central objects are the values zeta(3), zeta(5), and zeta(7). Unlike even zeta values, which are rational multiples of powers of pi, these values have no known elementary closed form. Their algebraic independence from pi is certified by the structure of Brown's motivic coaction, and it is precisely that algebraic structure this project encodes computationally.

---

## Approach

The project has two components that run independently and feed the same result pipeline.

**Sheaf neural network over the motivic weight filtration**

The motivic cohomology of mixed Tate motives over the integers carries a graded structure, with graded pieces of dimensions given by the Broadhurst-Kreimer sequence. A sheaf neural network is built over this weight filtration with three node classes: MZV nodes (one per weight level, stalk dimension equal to the Broadhurst-Kreimer dimension), modular form nodes (newforms from LMFDB with their critical L-values), and hypergeometric nodes (families known or suspected to evaluate to MZV values).

Restriction maps on edges implement Brown's motivic coaction. The network learns which combinations of modular forms and hypergeometric series are algebraically compatible with the structure of zeta(5), producing a confidence-ranked decomposition over a fixed basis. That ranked output drives a directed PSLQ integer-relation search, replacing brute-force coefficient sweeping with algebraically motivated tests.

**GPU-accelerated polynomial continued fraction search**

A secondary search runs polynomial continued fraction families (Zagier, Apery, Ramanujan) on an RTX 5080 GPU at approximately three billion evaluations per hour. Each candidate is verified at 500-digit precision using mpmath before logging.

Both components write results through a shared logger with a four-level verdict system: TRIVIAL, NOISE, CANDIDATE, IDENTITY. An IDENTITY verdict triggers immediate high-precision PSLQ verification.

---

## What a positive result looks like

Not a closed form in terms of pi. The motivic coaction makes that impossible for odd zeta values. Instead, either an Apery-class hypergeometric representation for zeta(5), analogous to Beukers' 1979 reinterpretation of Apery's proof for zeta(3), or a direct identity linking zeta(5) to the L-value of a specific modular form, generalising the Gamma_0(6) connection that underlies the Apery constant.

---

## Repository structure

```
zeta_hunter/
  sheaf/          Sheaf NN — nodes, encoders, diffusion, loss, training, PSLQ bridge
  pcf/            GPU polynomial continued fraction engine
  pslq.py         PSLQ integer relation search
  constants.py    500-digit mpmath reference values
  logger.py       Shared result logger

docs/
  specs/          Design documents
  plans/          Implementation plans

notebooks/
  01-03           PCF sweep experiments
  04              Sheaf NN training (stages 1A, 1B, 2)
  05              C-gamma query and PSLQ handoff for zeta(5), zeta(7)
```

---

## Notebooks

The three top-level notebooks are exploratory discovery runs; the three under `notebooks/` are the staged pipeline (sweep, then PSLQ, then report).

| Notebook | What it does | Open in Colab |
|----------|--------------|---------------|
| `deep_discovery_zeta3.ipynb` | Expanded zeta(3) search over 60+ constants (polylogarithms, MZVs, Clausen, Dirichlet beta) with genetic, PySR, and exhaustive PSLQ methods. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iteba15/mzv-simplicial-net/blob/main/deep_discovery_zeta3.ipynb) |
| `mathematical_discovery_BEAST_MODE.ipynb` | Full-scale local-GPU discovery run: larger models, 100+ training identities, extended PySR and exhaustive PSLQ search. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iteba15/mzv-simplicial-net/blob/main/mathematical_discovery_BEAST_MODE.ipynb) |
| `zeta3_discovery_v2_colab.ipynb` | v2 discovery system with simplicial message passing, contrastive learning, grammar-guided search, and anti-memorisation training. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iteba15/mzv-simplicial-net/blob/main/zeta3_discovery_v2_colab.ipynb) |
| `notebooks/01_pcf_sweep.ipynb` | Launch a polynomial continued fraction sweep and visualise hits. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iteba15/mzv-simplicial-net/blob/main/notebooks/01_pcf_sweep.ipynb) |
| `notebooks/02_pslq_search.ipynb` | Load a hit from the registry and run integer relation detection. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iteba15/mzv-simplicial-net/blob/main/notebooks/02_pslq_search.ipynb) |
| `notebooks/03_results_report.ipynb` | Summarise the hit registry and generate the LaTeX preprint. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iteba15/mzv-simplicial-net/blob/main/notebooks/03_results_report.ipynb) |

---

## Requirements

- Python 3.11
- PyTorch 2.x with CUDA 12.8 (tested on RTX 5080)
- mpmath 1.3 or later
- torch-geometric

```
pip install torch mpmath torch-geometric
```

---

## Running the search

Install the package in editable mode from the repository root, then launch the PCF sweep:

```
pip install -e .
python run_sweep.py
```

For the notebook workflow, install the development extras (`pip install -e ".[dev]"`) and run the notebooks under `notebooks/` in order: `01_pcf_sweep`, `02_pslq_search`, `03_results_report`.

---

## Status

The PCF engine is complete and validated. The Zagier family sweep over 51.5 billion combinations is logged. The sheaf NN implementation plan is complete; implementation is in progress.

No identity has been found yet.

---

## Verdict guide

| Verdict | Meaning |
|---------|---------|
| TRIVIAL | Target equals target. Discard. |
| NOISE | Large coefficients, numerical accident. |
| CANDIDATE | Small coefficients, 50 or more digits. Investigate. |
| IDENTITY | Small coefficients, 200 or more digits. Discovery. |

---

## Related project

Companion repository: [neural-mathematical-discovery](https://github.com/iteba15/neural-mathematical-discovery).

---

## Author

**Allan Kiplagat Iteba** (GitHub [@iteba15](https://github.com/iteba15)), BSc Astrophysics & Space Science, University of Nairobi.

- LinkedIn: *(link to be added)*
- ResearchGate: *(link to be added)*

---

This work is dedicated to Professor Carmine Serio.
