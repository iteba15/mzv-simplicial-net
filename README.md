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

## Requirements

- Python 3.11
- PyTorch 2.x with CUDA 12.8 (tested on RTX 5080)
- mpmath 1.3 or later
- torch-geometric

```
pip install torch mpmath torch-geometric
```

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

This work is dedicated to Professor Carmine Serio.
