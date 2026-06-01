# Sheaf Neural Network for Motivic Zeta Value Discovery

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Sheaf Neural Network over the MZV algebra to produce a
confidence-ranked basis decomposition (C-γ vector) for ζ(5) and ζ(7), feeding a
directed PSLQ search that replaces brute-force coefficient sweeping with
algebraically motivated integer-relation tests.

**Architecture:** Three-node-class sheaf (MZV, Modular form, Hypergeometric)
over the motivic weight filtration W₀ ⊂ W₁ ⊂ ... ⊂ W₈. Restriction maps
implement Brown's motivic coaction Δ: gr_n → ⊕_{k+l=n} gr_k ⊗ gr_l. GRT₁
symmetry enforced as an equivariance constraint on all MZV-layer restriction
maps. Two-stage training: algebraic pre-training on MZV Data Mine corpus (no
float values), then numerical fine-tuning with mpmath 500-digit labels.

**Tech Stack:** Python 3.11, PyTorch ≥ 2.0, mpmath ≥ 1.3, toponetx,
torch-geometric, MZV Data Mine (Leipzig), LMFDB API, Zucker-Joyce hypergeometric
evaluation tables.

---

## Background and Motivation

The brute-force PCF sweep (Zagier family, 51.5B combinations at 3.2B PCFs/hour
on RTX 5080) confirms the pipeline works but cannot reach the Apéry family
(121^10 ≈ 6.7 trillion combinations). More critically, the previous TGNN
architecture encoded constants as 154-dimensional float vectors — ζ(3) was just
`1.20205...` — and consistently overfitted without discovering structure. This
happened because the feature space had no algebraic topology: π, e, and ζ(3) are
equidistant floats to the network, even though they live in completely different
motivic cohomology classes.

The Gödelian framing is exact: even zeta values (ζ(2n) = rational × π^{2n}) are
provable within the formal system {π, polynomials}. Odd zeta values are not —
ζ(3)'s motivic coaction Δ(ζ(3)) = ζ(3) ⊗ 1 is the algebraic certificate that
ζ(3) cannot be decomposed into lower-weight periods. The network needs to learn
this geometry, not fit floating-point distances.

**What a breakthrough looks like:** Not a closed form in terms of π (the motivic
coaction makes this impossible). Instead, either (a) an Apéry-class
representation for ζ(5) — a hypergeometric series whose denominators satisfy a
modular-form recurrence, analogous to Beukers' 1979 interpretation of Apéry's
proof — or (b) a direct identity linking ζ(5) to an L-value of a specific
modular form, generalising the Γ₀(6) connection for ζ(3).

---

## Section 1: Data Representation and Graph Construction

### Three Node Classes

**Class A — MZV Nodes** (8 nodes, one per weight level 2–8)

Each node represents the graded motivic cohomology piece gr_n, not an individual
MZV basis element. The stalk at node n is ℝ^{d_n} where d_n is the
Broadhurst-Kreimer dimension.

```python
@dataclass(frozen=True)
class MZVNode:
    weight:        int        # n: node represents gr_n
    motivic_dim:   int        # d_n = dim(gr_n) per Broadhurst-Kreimer
    has_primitive: bool       # True at weights 3, 5, 7, ... (odd-weight primitives)
    depth_profile: list[int]  # [dim at depth 1, depth 2, ...] within gr_n
    lyndon_rank:   int        # rank in the Lyndon basis ordering
```

Stalk dimensions by weight (sequence d_n for n = 2..8: 1, 1, 1, 2, 2, 3, 4):

| Weight | d_n | Primitive | Notes |
|--------|-----|-----------|-------|
| 2 | 1 | ζ(2) | Only primitive with π closed form |
| 3 | 1 | ζ(3) | Apéry; coaction = identity |
| 4 | 1 | — | Expressible as ζ(2)² |
| 5 | 2 | ζ(5) | Primary target; dim-2 basis: {ζ(5), ζ(2)ζ(3)} |
| 6 | 2 | — | Basis: {ζ(6), ζ(3)²} |
| 7 | 3 | ζ(7) | Secondary target |
| 8 | 4 | — | Basis: {ζ(8), ζ(3)ζ(5), ζ(2)ζ(3)², ζ(2)²ζ(4)} |

Node features are **purely algebraic during pre-training** — no float values until
Stage 2. The network cannot shortcut algebraic geometry by memorising numerical
values.

**Class B — Modular Form Nodes** (~12 nodes)

Each node represents a newform f of small level and weight, together with its
critical L-value L(f, k):

```python
@dataclass(frozen=True)
class ModularNode:
    level:          int    # conductor N
    weight:         int    # modular weight k
    dim_space:      int    # dim S_k(Γ₀(N))
    is_cm:          bool   # complex multiplication form
    motivic_weight: int    # k-1 (the relevant period weight)
    beukers_form:   bool   # True for the Γ₀(6) form connecting to ζ(3)
```

Initial modular node set:

| Form | Level | Weight | Connection |
|------|-------|--------|------------|
| η(τ)²η(2τ)²η(3τ)²η(6τ)² | 6 | 4 | L(f,3) ↔ ζ(3) via Beukers (1979) |
| Γ₀(5) weight-6 newform | 5 | 6 | Candidate for ζ(5) |
| Γ₀(7) weight-6 newform | 7 | 6 | Candidate for ζ(5) |
| Γ₀(12) weight-6 newform | 12 | 6 | Candidate for ζ(5) |
| CM forms at imaginary quadratic fields | various | 2–6 | Dirichlet L-function connections |
| Ramanujan Δ | 1 | 12 | L(Δ,11) — benchmark for Eichler-Shimura maps |

Data source: LMFDB API. All L-values tabulated to 50+ digit precision. The
Eichler-Shimura relation provides the restriction map matrix for each
MZV ↔ Modular edge.

**Class C — Hypergeometric Nodes** (~10 nodes)

Each node represents a hypergeometric family ₚFₚ₋₁ at an algebraic argument:

```python
@dataclass(frozen=True)
class HypergeometricNode:
    p:               int          # order
    upper_params:    tuple        # (a₁,...,aₚ) as rationals
    lower_params:    tuple        # (b₁,...,bₚ₋₁) as rationals
    argument:        str          # "1", "1/2", "1/4", etc.
    evaluates_to:    str | None   # known MZV node if evaluation is known
    convergence_exp: float        # log of convergence rate per term
```

Initial hypergeometric node set:

| Family | Parameters | Argument | Known evaluation |
|--------|-----------|----------|-----------------|
| ₄F₃ | (1,1,1,1; 2,2,2) | 1 | = ζ(3) directly |
| ₃F₂ | (1/2,1/2,1/2; 1,1) | 1 | Relates to ζ(3) via Clausen |
| ₅F₄ | (1,1,1,1,1; 2,2,2,2) | 1 | = ζ(4) — even-weight benchmark |
| ₅F₄ | (1/2,1/2,1/2,1/2,1/2; 1,1,1,1) | 1 | Candidate for ζ(5) analogue |
| ₄F₃ | Apéry parameter family | 1 | Training target — Apéry's proof |
| Clausen ₃F₂ | (1/4,1/2,3/4; 1,1) | 1 | π-family benchmark |

For nodes where `evaluates_to` is None, the connecting edge's restriction map
is initialised to zero and learned from scratch — this is where new discoveries
emerge.

Data source: Zucker-Joyce (1987) tables for verified evaluations; mpmath for
500-digit fine-tuning labels.

---

### Edge Types and Restriction Maps

**39 edges total across four types.**

**Type 1 — MZV Product Edges** (9 directed edges)

One directed edge for each (k, l) → n where k + l = n ≤ 8. Carries the motivic
coaction matrix as its restriction map:

```
Δ: gr_n → gr_k ⊗ gr_l
Restriction map shape: d_n × (d_k · d_l)
Largest matrix (weight 8, split 4+4): 4 × (2·2) = 4×4
```

Matrices read directly from MZV Data Mine coaction tables. Coassociativity
`(Δ⊗id)∘Δ = (id⊗Δ)∘Δ` is verified as a mandatory assertion before any training.

**Type 2 — MZV Reduction Edges** (~12 undirected edges)

Between MZV nodes at the same weight, encoding double-shuffle reduction: how
one basis representation maps to another at equal weight. Restriction map is a
fixed change-of-basis matrix (not learnable — exact algebra).

**Type 3 — Eichler-Shimura Edges** (~8 directed edges)

Connecting weight-n MZV nodes to modular nodes of modular weight k = n+1.
Restriction map from LMFDB; the Γ₀(6) ↔ weight-3 edge is the only one fully
determined analytically (Beukers). All others are initialised from LMFDB data
and remain learnable during fine-tuning.

**Type 4 — Hypergeometric Evaluation Edges** (~10 directed edges)

Connecting hypergeometric nodes to MZV nodes they evaluate to (where known),
plus to modular nodes via the theory of hypergeometric motives. Unknown
evaluation edges initialised to zero, fully learned.

---

### GRT₁ Symmetry Constraints

The Grothendieck-Teichmüller group GRT₁ acts on the MZV algebra as
double-shuffle-preserving automorphisms. Each MZV-layer restriction map is
factored as:

```
ρ = P · Δ_canonical · Q
```

where Δ_canonical is fixed from the MZV Data Mine, and P, Q are learnable
matrices constrained to the centraliser of the GRT₁ action (computed from
Schneps 2012 tables at weight ≤ 8). This reduces effective restriction map
parameter count by ~60% and prevents the network from learning GRT₁-violating
coaction values that cannot correspond to real identities.

---

### Data Sources

| Source | Use |
|--------|-----|
| MZV Data Mine (Leipzig, weight ≤ 22) | Coaction tables, shuffle/stuffle identities |
| LMFDB API | Modular form L-values + Eichler-Shimura matrices |
| Zucker-Joyce (1987) | Verified hypergeometric evaluations |
| Schneps (2012) | GRT₁ action matrices up to weight 8 |
| Existing `zeta_hunter/constants.py` | 41 constants at 500-digit mpmath precision |

---

## Section 2: Sheaf Neural Network Architecture

### The Sheaf Laplacian

The sheaf Laplacian L_F replaces the standard graph Laplacian. For each edge
e = (u,v) with restriction maps ρ_{u←e} and ρ_{v←e}:

```
L_F = δᵀδ    where  δ(s)[e] = ρ_{v←e}(s(v)) − ρ_{u←e}(s(u))

Block entries:
  L_F[u,u] = Σ_{e∋u} ρ_{u←e}ᵀ ρ_{u←e}      (diagonal — local tension)
  L_F[u,v] = −ρ_{u←e}ᵀ ρ_{v←e}              (off-diagonal — cross coupling)
```

L_F is a **42×42 block matrix** (sum of all stalk dimensions). Computed as a
dense matrix — 30 nodes and 39 edges make sparse approximations unnecessary.

Key property: ker(L_F) = H⁰(F) — the global sections. Primitive elements
(ζ(3), ζ(5) if primitive) lie in isolated eigenspaces with eigenvalue 0. When
ζ(5) is queried, the C-γ output searches for nodes sharing its eigenspace
structure.

### Node Encoders

Each node class has a class-specific encoder mapping algebraic features to a
stalk embedding:

```python
class MZVNodeEncoder(nn.Module):
    # (weight, motivic_dim, has_primitive, lyndon_rank) → ℝ^{d_n}
    # 2-layer MLP + LayerNorm; output dim = d_n (varies per node)

class ModularNodeEncoder(nn.Module):
    # (level, weight, dim_space, is_cm, beukers_form) → ℝ^{dim_space}
    # Linear projection + GELU

class HypergeometricNodeEncoder(nn.Module):
    # (p, upper_params, lower_params, convergence_exp) → ℝ^p
    # Parameter embedding + positional encoding over rational arguments
```

Float values are **not injected** during Stage 1 pre-training.

### Restriction Map Parameterisation

```
Type 1 (MZV coaction):   ρ = P · Δ_canonical · Q
                          P, Q learnable, constrained to GRT₁ centraliser
                          Learns refinements to canonical coaction

Type 2 (MZV reduction):  ρ = M_basis  (fixed, not learnable)
                          Exact algebra — no learning needed

Type 3 (Eichler-Shimura): ρ = λ · M_ES + ΔM
                           λ learnable scalar, ΔM learnable correction matrix
                           Initialised from LMFDB

Type 4 (Hypergeometric):  ρ = W  (fully learnable, zero-initialised for unknown)
                           Discovery mechanism for novel evaluations
```

Total learnable restriction map parameters: ~800.

### Sheaf Diffusion Layers (×4)

```python
class SheafDiffusionLayer(nn.Module):
    def forward(self, H, L_F_norm):
        # H: [42, hidden_dim]; L_F_norm: normalised sheaf Laplacian [42, 42]
        H_new = H - self.alpha * (L_F_norm @ H)   # alpha: learnable damping
        H_new = self.norm(H_new)
        return self.activation(H_new)
```

After each layer, GRT₁ projection maps MZV stalk embeddings back to the
invariant subspace:

```python
class GRT1Projection(nn.Module):
    # h → h − (I − P_GRT1) h
    # P_GRT1 precomputed from Schneps (2012) at initialisation
    # Applied to MZV nodes only
```

### Stage 2 Float Injection

```python
class FloatInjectionHead(nn.Module):
    # mpmath value downsampled to 64-digit precision vector → ℝ^{d_v}
    # Combined with algebraic embedding via learned gate:
    h_final = gate * h_algebraic + (1 - gate) * h_float
    # gate initialised near 1.0 (trust algebra); decreases during Stage 2
```

### C-γ Output Layer

```python
class CGammaHead(nn.Module):
    def __init__(self, stalk_dim: int, basis_size: int):
        self.proj = nn.Sequential(
            nn.Linear(stalk_dim, stalk_dim * 4),
            nn.GELU(),
            nn.LayerNorm(stalk_dim * 4),
            nn.Linear(stalk_dim * 4, basis_size),
        )
        self.temperature = nn.Parameter(torch.ones(1))  # learned output sharpness

    def forward(self, stalk_embedding):
        return F.softmax(self.proj(stalk_embedding) / self.temperature.clamp(0.1), dim=-1)
```

Weight-5 output basis:

```python
WEIGHT_5_BASIS = [
    ("zeta5",          "ζ(5)"),
    ("zeta2_zeta3",    "ζ(2)·ζ(3)"),
    ("zeta23",         "ζ(2,3)"),
    ("zeta32",         "ζ(3,2)"),
    ("Li5_half",       "Li₅(1/2)"),
    ("pi4_ln2",        "π⁴·ln2"),
    ("L_f6_5",         "L(f_{Γ₀(6)},5)"),
    ("L_f5_5",         "L(f_{Γ₀(5)},5)"),
    ("4F3_apery_like", "₄F₃(1,1,1,1;2,2,2;1)·…"),
    ("5F4_half",       "₅F₄(1/2,…;…;1)"),
]
```

### Full Architecture Flow

```
Algebraic node features (no floats in Stage 1)
        ↓
Class-specific encoders → stalk embeddings H ∈ ℝ^{42 × hidden}
        ↓
Build sheaf Laplacian L_F from restriction maps
        ↓
4 × [SheafDiffusionLayer → GRT1Projection]
        ↓  ← float injection via gated head (Stage 2 only)
C-γ projection + softmax (CGammaHead)
        ↓
Confidence vector over WEIGHT_5_BASIS
        ↓
PSLQ bridge → ranked integer-relation tests
```

---

## Section 3: Training Pipeline

### Loss Function

```
L_total = L₁  +  λ₂(t)·L₂  +  λ₃·L₃  +  λ_GRT·L_GRT1

Stage 1:  λ₂=0,    λ₃=0.1,  λ_GRT=0.5
Stage 2:  λ₂→1.0,  λ₃=0.1,  λ_GRT=0.5
```

λ₂ ramps linearly from 0 to 1 over the first 20% of Stage 2 epochs.

**L₁ — Algebraic Structure Pre-training (masked shuffle completion)**

Randomly mask one term in a known shuffle/stuffle identity. Network predicts
the masked node identity (cross-entropy over 30 nodes) and its rational
coefficient (Huber loss in log-space, coefficients capped at |c| ≤ 1000).

Training data: Leipzig corpus, weight ≤ 8 (~800 identities).
Validation: Leipzig weight 9–12 (~9,200 identities, never seen during training).
Convergence: ≥85% masked-node accuracy on weight-9 holdout.

**Mandatory Pre-training Check — Coassociativity**

Run before training and every 10 epochs. Verifies `(Δ⊗id)∘Δ = (id⊗Δ)∘Δ`
for all weight decompositions up to weight 8. Frobenius error must be < 1e-10.
Training aborts on failure — a coaction parsing error silently corrupts the
entire weight filtration.

```python
def assert_coassociativity(delta: dict, tol: float = 1e-10) -> None:
    for n in range(4, 9):
        for k, l in weight_pairs(n):
            for a, b in weight_pairs(k):
                lhs = kron(delta[(k,a,b)], eye(d[l])) @ delta[(n,k,l)]
                rhs = kron(eye(d[a]), delta[(l,n-k-a,b)]) @ delta[(n,k,l)]
                err = torch.norm(lhs - rhs, p='fro')
                assert err < tol, f"Coassociativity violated: weight {n}→({k},{l})→({a},{b}), err={err:.2e}"
```

**L₂ — Soft-label Basis Prediction**

Ground-truth labels are soft distributions computed from mpmath numerical
proximity at 500 digits, converted via temperature-scaled softmax:

```python
def compute_soft_labels(target, basis, T):
    distances = torch.tensor([
        -float(mp.log10(abs(target - b) + mp.mpf(10)**-500))
        for b in basis
    ])
    return F.softmax(distances / T, dim=0)
```

Temperature schedule: `T(epoch) = 2.0 · exp(−epoch / 50)`.
Loss: KL divergence between predicted distribution and soft labels.

**L₃ — Algebraic Consistency Regulariser**

For each shuffle product identity ζ(a)⋆ζ(b) = Σ cᵢζ(wᵢ) in the corpus, the
predicted decompositions of the factors must satisfy the product structure:

```python
loss_L3 += F.mse_loss(shuffle_bilinear(D_a, D_b), true_product_vector)
```

Prevents the network from predicting decompositions that are numerically
plausible but violate the shuffle algebra.

**L_GRT1 — Equivariance Constraint**

For each GRT₁ generator g, all MZV-layer restriction maps must satisfy
`ρ∘g_source = g_target∘ρ`. Penalised as squared Frobenius norm of the
commutator.

### Training Curriculum

**Stage 1A — Bootstrap (30 epochs)**
- Weights 2–4 only; modular and hypergeometric nodes frozen
- Network learns to represent ζ(2), ζ(3) as orthogonal primitives

**Stage 1B — Full pre-training (150 epochs)**
- All 30 nodes; full Leipzig weight ≤ 8 corpus
- Eichler-Shimura edges activate; modular node begins receiving gradient
- Val target: 85% masked-node accuracy on weight 9–12 holdout
- Key check: ζ(3) C-γ output converges to [1.0] with confidence > 0.95

**Stage 2 — Numerical fine-tuning (100 epochs)**
- FloatInjectionHead activated; mpmath 500-digit values injected
- λ₂ ramps from 0 to 1 over first 20 epochs
- Val target: known weight-5,6,7,8 decompositions correct to 10+ digits

**Stage 3 — Novel prediction (inference only)**
- Query: ζ(5) float value
- Output: C-γ vector → PSLQ bridge

### Optimisation

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, eta_min=1e-5
)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Full graph forward pass: batch size = 1 (30 nodes). Stage 1 estimated 5–10
minutes on RTX 5080. Stage 2 under 30 minutes. Training is not the bottleneck.

### Validation Checkpoints

| Check | When | Pass Criterion |
|-------|------|---------------|
| Coassociativity | Every 10 epochs | Frobenius error < 1e-10 |
| Primitive isolation | End Stage 1B | ζ(3) C-γ confidence > 0.95 |
| Cross-weight generalisation | End Stage 1B | 85% accuracy on weight 9–12 holdout |
| Numerical calibration | End Stage 2 | Weight-5 decompositions to 10+ digits |
| Novel signal | Stage 3 | Any hyp/modular node gets > 5% C-γ confidence |

---

## Section 4: C-γ Output, PSLQ Handoff, and Integration

### C-γ Query Protocol

```
1. Create query node: weight=5, is_primitive=True, all other features
   matching the weight-5 MZV node algebraic features.
2. Inject ζ(5) float value via FloatInjectionHead (gated).
3. Run 4 sheaf diffusion layers — signals propagate from modular and
   hypergeometric nodes into the query node's stalk embedding.
4. Project stalk embedding → WEIGHT_5_BASIS via CGammaHead.
5. Return (label, confidence, mpf_value) triples, sorted by confidence.
```

### Confidence-to-Precision Contract

| Confidence | PSLQ precision | maxcoeff |
|------------|---------------|---------|
| ≥ 0.40 | 500 digits | 10⁶ |
| 0.15–0.40 | 200 digits | 10⁴ |
| 0.05–0.15 | 100 digits | 10³ |
| < 0.05 | Skip | — |

### PSLQ Bridge

```python
class SheavedPSLQBridge:
    MAX_BASIS_SIZE = 6
    JOINT_PROB_FLOOR = 0.003

    def generate_test_vectors(self):
        # 1. Single-element tests (high-confidence elements alone)
        # 2. Two-element combinations, ordered by joint probability c₁·c₂
        # 3. Three-element combinations where joint prob > floor
        # Yields (basis_name, [target, b₁, b₂, ...], joint_probability)

    def run(self):
        for basis_name, basis_vector, joint_prob in self.generate_test_vectors():
            result = PSLQSearcher._run_single(target, basis_name, basis_vector[1:])
            if result.verdict == "IDENTITY":
                return [result]   # stop immediately
        # Fallback: run existing hardcoded PSLQSearcher bases unchanged
        if not results:
            return PSLQSearcher.run_all_bases(target)
```

The fallback ensures the bridge never returns nothing — if the sheaf gives no
signal, the existing PSLQ system runs as before.

### Integration Map

```
constants.py          → 500-digit float labels for Stage 2
sheaf/restriction.py  ← MZV Data Mine coaction tables
sheaf/model.py        → ZetaHunterSheafNN (trained)
sheaf/query.py        → C-γ vector
sheaf/pslq_bridge.py  → prioritised PSLQ basis vectors
pslq.py               → PSLQResult (existing, unchanged)
logger.py             → Hit (existing, unchanged)
report.py             → LaTeX preprint (existing, unchanged)
```

The PCF sweep (pcf/) and the sheaf system are fully independent. Both feed
the same logger and report generator.

### File Structure

```
zeta_hunter/
  sheaf/
    __init__.py
    nodes.py              # MZVNode, ModularNode, HypergeometricNode
    encoders.py           # Class-specific node encoders
    restriction.py        # Restriction map parameterisation + coaction parser
    laplacian.py          # Sheaf Laplacian builder + coassociativity check
    diffusion.py          # SheafDiffusionLayer + GRT1Projection
    injection.py          # FloatInjectionHead + learned gate
    query.py              # C-γ query mechanism + CGammaHead
    model.py              # ZetaHunterSheafNN (full assembly)
    pslq_bridge.py        # SheavedPSLQBridge
    data/
      mzv_coaction.json   # Parsed from MZV Data Mine, weight ≤ 12
      lmfdb_newforms.json # Modular form L-values from LMFDB API
      hyp_evaluations.json # Zucker-Joyce tables
  constants.py            # existing — unchanged
  pcf/                    # existing — unchanged
  pslq.py                 # existing — extended with SheavedPSLQBridge hook
  logger.py               # existing — unchanged
  report.py               # existing — unchanged

notebooks/
  04_sheaf_training.ipynb     # Stages 1A, 1B, 2
  05_sheaf_query.ipynb        # Novel prediction + PSLQ handoff for ζ(5), ζ(7)

tests/
  test_coassociativity.py     # Mandatory pre-training validation
  test_sheaf_laplacian.py     # Laplacian construction + eigenvalue checks
  test_pslq_bridge.py         # Bridge prioritisation + fallback logic
```

### Discovery Signal Definition

A positive result is when any of the following occur:

1. **Hypergeometric identity:** C-γ assigns ≥ 5% confidence to a hypergeometric
   node AND PSLQ finds a relation with ≥ 50 digit precision. This is an
   Apéry-class representation for ζ(5).

2. **Modular connection:** C-γ assigns ≥ 5% confidence to a modular form node
   AND PSLQ finds a relation with ≥ 50 digit precision. This is a Beukers-type
   generalisation for ζ(5).

3. **Novel MZV identity:** PSLQ finds a relation among depth-2 MZVs at weight 5
   that is not already in the Leipzig corpus. Even a CANDIDATE (50–199 digit
   precision) is worth reporting.

All results are logged via the existing `RunLogger` with the standard
TRIVIAL / NOISE / CANDIDATE / IDENTITY verdict system and included in the
LaTeX report automatically.

---

## Open Questions for Implementation

1. **MZV Data Mine parsing:** The coaction tables are distributed as Mathematica
   `.m` files. A parser converting them to `mzv_coaction.json` is needed.
   Alternatively, use the AMZV Python library if it exposes coaction data.

2. **GRT₁ action matrices:** Schneps (2012) gives these as formal power series.
   The weight-≤8 truncation needs to be extracted and verified against known
   MZV images.

3. **LMFDB API access:** Requires network access during `build_full_graph()`.
   Consider caching all L-values to `lmfdb_newforms.json` at build time so
   training runs offline.

4. **Hypergeometric evaluation verification:** The ₅F₄ candidates have no known
   evaluation — their edges are zero-initialised and learned. Periodically run
   mpmath `hyp()` at 500 digits to check if any learned edge has converged to a
   known value.
