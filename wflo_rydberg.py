# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3.11 (qoolqit 1.4.0)
#     language: python
#     name: qoolqit-env
# ---

# %% [markdown]
# # Wind Farm Layout Optimization on simulated Rydberg atoms with QoolQit
#
# **Question.** How much turbine power does a Rydberg-atom solver lose as wind-farm wake physics departs from
# the Rydberg Hamiltonian?
#
# We take the published 4×4 instance of Hancock et al. (arXiv:2312.13123, following Senderovich et al. 2022),
# show that it is exactly maximum independent set on the 4×4 king graph, and then climb a ladder of
# increasingly non-native instances R0–R3. Each rung is solved with a native blockade encoding (A) and an
# embedded-QUBO encoding (B), and compared with exact enumeration and simulated annealing.
#
# **Claim boundaries.**
# - Emulator results only (noiseless state-vector emulation). Published D-Wave numbers are hardware results and
#   are never pooled with emulator numbers.
# - R0 is easy and Rydberg-native; a good result there is a structural observation, not quantum advantage.
# - No speed claims: emulation time measures the classical simulator, not a device.
# - To our knowledge this is the first published neutral-atom WFLO study we could find; we do not claim it is the first.

# %% [markdown]
# ## 0. Setup
# The install cell pins every dependency and is a no-op when the pinned versions are already present.

# %%
import importlib.metadata as _md, subprocess, sys

PINNED = {
    "qoolqit": "1.4.0", "pulser-core": "1.9.1", "pulser-simulation": "1.9.1",
    "emu-sv": "2.9.1", "emu-base": "2.9.1", "torch": "2.14.1",
    "numpy": "2.4.6", "scipy": "1.17.1", "matplotlib": "3.11.2",
}

def _installed(pkg):
    try:
        return _md.version(pkg).split("+")[0]
    except _md.PackageNotFoundError:
        return None

_missing = [f"{p}=={v}" for p, v in PINNED.items() if _installed(p) != v]
if _missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_missing], check=True)

# %%
import itertools, json, os, platform, time

import matplotlib.pyplot as plt
import numpy as np
import torch

MODE = os.environ.get("WFLO_MODE", "demo")  # "demo": R0 at 4x4 on CPU, < ~15 min. "full": the remote GPU sweep.
assert MODE in ("demo", "full")
SEED = 20261018
N_THREADS = 16                              # shared machine: cap CPU threads
GPU = torch.cuda.is_available()

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(N_THREADS)
RNG = np.random.default_rng(SEED)

print(f"MODE={MODE}  SEED={SEED}  GPU={GPU}" + (f" ({torch.cuda.get_device_name(0)})" if GPU else "") + f"  threads={N_THREADS}")
print(f"python {platform.python_version()}")
for p in PINNED:
    print(f"  {p:18s} {_installed(p)}")
assert all(_installed(p) == v for p, v in PINNED.items()), "pinned versions not active; restart the kernel after install"

# %% [markdown]
# ## 1. Instances
#
# ### 1.1 The wake QUBO (Hancock et al., Eqs. 3–5)
# Sites are labelled row-major, site $k$ at (row $k // L$, column $k \% L$), as in the paper's Fig. 2 (0-based here).
# A turbine produces $\frac13 v_d^3$ per wind arrangement $d$; a turbine at $j$ in the wake of $i$ loses
# $\frac13 (v_d^3 - u_{ij}^3)$ with the Jensen deficit
#
# $$u_{ij} = v_d\left(1 - \frac{2a}{(1 + \alpha_T(\delta/r)^2)^2}\right), \qquad \alpha_T = \frac{r - r_t}{x},$$
#
# $a = 0.1$, $r_t = 0.33$ grid units, $\delta = \lVert i - j \rVert_2$; contributions are weighted by $p_d$ and summed
# (linear superposition of wakes).
#
# **Wake membership.** The paper underspecifies which cells lie in a wake ("if a grid location is at all within the
# wake, it is entirely within its wake"). We adopt the rule that reproduces the published optimum set: with $s$ and
# $t$ the downstream and lateral offsets of $j$'s centre from $i$'s, $j$ is in the wake iff
# $0.001 < s \le x + 0.001$ and $t < r\,s$. Wind angles are counted clockwise from west (wind from 0° blows east;
# rows grow southwards).

# %%
A_IND, R_T, V_WIND = 0.1, 0.33, 12.0      # axial induction, turbine radius (grid units), wind speed (m/s)
TOL = 1e-3
P_TURBINE = V_WIND**3 / 3                  # 576 per turbine


def wind_regime(kind):
    """36 directions, 10 deg apart, 12 m/s. 'uniform' = Mosetti case 2 (p = 1/36 each).
    'west75' (our own definition): p = 0.75 from west, the remaining 0.25 spread evenly over the other 35."""
    angles = np.arange(36) * 10.0
    if kind == "uniform":
        p = np.full(36, 1 / 36)
    elif kind == "west75":
        p = np.r_[0.75, np.full(35, 0.25 / 35)]
    else:
        raise ValueError(kind)
    return angles, p


def site_coords(L):
    return np.array([(k // L, k % L) for k in range(L * L)], float)   # (row, col)


def wake_matrix(L, x, r, regime):
    """Symmetric pairwise wake penalty W[i, j] (kW) for turbines at sites i and j, summed over both directions."""
    rc = site_coords(L)
    alpha_t = (r - R_T) / x
    W = np.zeros((L * L, L * L))
    for ang, p_d in zip(*wind_regime(regime)):
        th = np.deg2rad(ang)
        e = np.array([np.sin(th), np.cos(th)])          # direction of travel in (row, col)
        for i, j in itertools.permutations(range(L * L), 2):
            dv = rc[j] - rc[i]
            s, t = dv @ e, abs(dv[0] * e[1] - dv[1] * e[0])
            if TOL < s <= x + TOL and t < r * s:
                u = V_WIND * (1 - 2 * A_IND / (1 + alpha_t * (np.hypot(*dv) / r) ** 2) ** 2)
                W[i, j] += p_d * (V_WIND**3 - u**3) / 3
    return W + W.T


# %% [markdown]
# ### 1.2 The instance ladder
# R0 is the published instance. R1–R3 are our own instances: they change only the wake length $x$, the spread $r$
# and the wind regime, keeping $m = 4$ turbines on the 4×4 grid.

# %%
L_GRID, M_TURB = 4, 4
RUNGS = {   # name: (x, r, regime)
    "R0": (1, 1.5, "uniform"),
    "R1": (3, 1.0, "uniform"),
    "R2": (1, 1.5, "west75"),
    "R3": (3, 1.0, "west75"),
}


def bit_matrix(n):
    """All 2^n bitstrings as a (2^n, n) 0/1 array; row index = int(bitstring, 2), qubit 0 is the leftmost bit."""
    idx = np.arange(2**n)
    return ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.int8)


def layout_power(B, W):
    """Total power (kW) of layouts given as rows of a 0/1 matrix: m * 576 minus summed pairwise wake penalties."""
    B = np.asarray(B, float)
    return P_TURBINE * B.sum(1) - 0.5 * np.einsum("ki,ij,kj->k", B, W, B)


BITS = bit_matrix(L_GRID**2)
WEIGHT = BITS.sum(1)
INST = {}
for name, (x, r, regime) in RUNGS.items():
    W = wake_matrix(L_GRID, x, r, regime)
    power_all = layout_power(BITS, W)                      # power of every one of the 2^16 layouts
    valid = WEIGHT == M_TURB
    best = power_all[valid].max()
    opt_idx = np.flatnonzero(valid & np.isclose(power_all, best, atol=1e-6))
    INST[name] = dict(x=x, r=r, regime=regime, W=W, site_power=np.full(L_GRID**2, P_TURBINE),
                      power_all=power_all, optimum=best, opt_idx=opt_idx)
    print(f"{name}: x={x} r={r} {regime:7s}  optimum {best:8.2f} kW  #optimal layouts {len(opt_idx):3d}"
          f"  e.g. sites {np.flatnonzero(BITS[opt_idx[0]]).tolist()}")

# %% [markdown]
# ### 1.3 Oracle assertions
# Published values for R0: per-turbine power 576, pairwise penalty 83.728 at distance 1 and 17.582 at $\sqrt2$, zero
# beyond, optimum 2304.0 kW. Expected optima for our rungs: R1 2228.6 (unique), R2 2304.0, R3 2238.8 (not a king-graph MIS).

# %%
def king_adjacent(a, b, L=L_GRID):
    return max(abs(a // L - b // L), abs(a % L - b % L)) == 1


W0 = INST["R0"]["W"]
dist = np.hypot(*(site_coords(L_GRID)[:, None, :] - site_coords(L_GRID)[None, :, :]).transpose(2, 0, 1))
assert np.isclose(P_TURBINE, 576.0)
assert np.allclose(W0[np.isclose(dist, 1)], 83.728, atol=1e-3)
assert np.allclose(W0[np.isclose(dist, np.sqrt(2))], 17.582, atol=1e-3)
assert np.all(W0[dist > 1.5] == 0)

king_mis4 = {c for c in itertools.combinations(range(16), 4)
             if not any(king_adjacent(a, b) for a, b in itertools.combinations(c, 2))}
opt_sets = lambda name: {tuple(np.flatnonzero(BITS[k])) for k in INST[name]["opt_idx"]}
assert INST["R0"]["optimum"] == 2304.0 and opt_sets("R0") == king_mis4 and len(king_mis4) == 79

for name, target, n_opt in [("R1", 2228.6, 1), ("R2", 2304.0, None), ("R3", 2238.8, 1)]:
    assert round(INST[name]["optimum"], 1) == target, name
    assert n_opt is None or len(INST[name]["opt_idx"]) == n_opt, name
assert not opt_sets("R3") <= king_mis4                     # R3's optimum is not a king-graph MIS
assert opt_sets("R1") <= king_mis4 and opt_sets("R2") <= king_mis4
assert np.argmax(INST["R0"]["power_all"]) == 2**16 - 1     # without the count penalty, the optimum fills all 16 sites
print("R0 = MIS on the 4x4 king graph: the 79 optimal layouts are exactly its 79 independent 4-sets")
print("R2 optimum set == R0 optimum set:", opt_sets("R2") == opt_sets("R0"))
print("all oracle assertions pass")

# %% [markdown]
# ### 1.4 Full QUBO (Eq. 9 without the proximity term)
# $E(x) = -\sum_i 576\,x_i + \sum_{i<j} W_{ij} x_i x_j + \lambda_1\left(\sum_i x_i - m\right)^2$.
# The paper does not state $\lambda_1$. Any $\lambda_1 > 576$ (the most a fifth turbine could add) makes every
# minimiser have exactly $m$ turbines; we take $\lambda_1 = 1.05 \times 576$, the value that gave bit-flip SA its best
# results in a scan over $\{1.05, 1.25, 1.5, 2\} \times 576$. This $Q$ is only used by bit-flip SA; the Rydberg
# encodings drop the count penalty and post-select (Section 3).

# %%
LAMBDA1 = 1.05 * P_TURBINE


def qubo(W, lam=LAMBDA1, m=M_TURB):
    """Return (h, J, c) with E(x) = h.x + 1/2 x.J.x + c, J symmetric with zero diagonal."""
    n = len(W)
    h = np.full(n, -P_TURBINE + lam * (1 - 2 * m))
    J = W + 2 * lam * (1 - np.eye(n))
    return h, J, lam * m**2


def qubo_energy(X, h, J, c):
    X = np.asarray(X, float)
    return X @ h + 0.5 * np.einsum("ki,ij,kj->k", X, J, X) + c


for name, d in INST.items():
    d["qubo"] = qubo(d["W"])
    E_all = qubo_energy(BITS, *d["qubo"])
    assert np.allclose(-E_all[WEIGHT == M_TURB], d["power_all"][WEIGHT == M_TURB])   # QUBO = -power on 4-turbine layouts
    assert set(np.flatnonzero(np.isclose(E_all, E_all.min()))) == set(d["opt_idx"])  # QUBO minima = optimal layouts
print("QUBO minima coincide with the enumerated optimum sets on all rungs")

# %% [markdown]
# ## 2. Classical baselines
# ### 2.1 Exact optimum
# Enumerating the 1,820 four-turbine layouts (done above over all $2^{16}$ bitstrings) replaces Gurobi at this size.
#
# ### 2.2 Simulated annealing
# Two numpy implementations of Metropolis annealing, 36 independent runs each from random initial states, geometric
# inverse-temperature schedules:
#
# - **SA-QUBO** (the paper's protocol, which used D-Wave's `neal` sampler): single-bit flips on the full QUBO,
#   1,000 sweeps. The schedule runs from $\beta_0 = \ln 2 / (\lambda_1 + 576)$ to
#   $\beta_1 = \ln 100 / \max(\min_{W_{ij}>0} W_{ij}, 1)$. (neal's default takes the cold end from the smallest QUBO
#   coefficient, here a count-penalty coupling $2\lambda_1$, which leaves the final temperature too hot to hold the
#   turbine count.) Every move between two 4-turbine layouts must pass through 3 or 5 turbines, so the count penalty
#   creates barriers of hundreds of kW. On the long-wake rungs (R1, R3) the gap to the second-best layout is only
#   22 kW and 4 kW, and bit-flip SA ends in local minima.
# - **SA-swap** (a stronger classical reference): annealing restricted to exactly $m$ turbines, where each move
#   relocates one turbine to an empty site; 20,000 steps.

# %%
def sa_qubo(h, J, W, lam=LAMBDA1, n_runs=36, n_sweeps=1000, rng=RNG):
    n = len(h)
    betas = np.geomspace(np.log(2) / (lam + P_TURBINE), np.log(100) / max(W[W > 0].min(), 1.0), n_sweeps)
    X = rng.integers(0, 2, size=(n_runs, n)).astype(float)
    field = h + X @ J                                     # local field: dE(flip i) = (1 - 2 x_i) * field_i
    for beta in betas:
        for i in rng.permutation(n):
            dE = (1 - 2 * X[:, i]) * field[:, i]
            flip = (dE <= 0) | (rng.random(n_runs) < np.exp(-beta * np.clip(dE, 0, None)))
            step = np.where(flip, 1 - 2 * X[:, i], 0.0)
            X[:, i] += step
            field += step[:, None] * J[i]
    return X.astype(np.int8)


def sa_swap(W, m=M_TURB, n_runs=36, n_steps=20000, rng=RNG):
    n = len(W)
    betas = np.geomspace(np.log(2) / (m * W.max()), np.log(100) / max(W[W > 0].min(), 1.0), n_steps)
    X = np.zeros((n_runs, n))
    X[np.arange(n_runs)[:, None], np.argsort(rng.random((n_runs, n)), 1)[:, :m]] = 1
    runs = np.arange(n_runs)
    for beta in betas:
        out = np.argmax(rng.random((n_runs, n)) * X, 1)          # a random occupied site
        new = np.argmax(rng.random((n_runs, n)) * (1 - X), 1)    # a random empty site
        dE = (W[new] * X).sum(1) - W[new, out] - (W[out] * X).sum(1)
        acc = (dE <= 0) | (rng.random(n_runs) < np.exp(-beta * np.clip(dE, 0, None)))
        X[runs[acc], out[acc]] = 0
        X[runs[acc], new[acc]] = 1
    return X.astype(np.int8)


def summarise(X, d):
    pct = 100 * layout_power(X, d["W"]) / d["optimum"]
    idx = X.astype(int) @ (1 << np.arange(X.shape[1] - 1, -1, -1))
    return dict(mean_pct=float(pct.mean()), median_pct=float(np.median(pct)), min_pct=float(pct.min()),
                frac_optimal=float(np.isin(idx, d["opt_idx"]).mean()),
                frac_m_turbines=float((X.sum(1) == M_TURB).mean()))


SA_RUNS = 36
sa_results = {}
for name, d in INST.items():
    for method, run in [("SA-QUBO", lambda: sa_qubo(*d["qubo"][:2], d["W"], n_runs=SA_RUNS)),
                        ("SA-swap", lambda: sa_swap(d["W"], n_runs=SA_RUNS))]:
        t0 = time.perf_counter()
        s = summarise(run(), d)
        s["wall_s"] = time.perf_counter() - t0
        sa_results[(name, method)] = s
        print(f"{name} {method}: mean {s['mean_pct']:6.2f}% of optimum (min {s['min_pct']:6.2f}%), "
              f"optimal in {round(s['frac_optimal'] * SA_RUNS):2d}/{SA_RUNS} runs, "
              f"{round(s['frac_m_turbines'] * SA_RUNS)}/{SA_RUNS} with exactly {M_TURB} turbines, {s['wall_s']:.2f} s")

# Phase 1 exit: every rung has W, per-site power, exact optimum and optimum set, and 36-run SA baselines.
assert all(k in INST[n] for n in RUNGS for k in ("W", "site_power", "optimum", "opt_idx", "qubo"))
assert all(r["frac_m_turbines"] == 1 for r in sa_results.values())
assert sa_results[("R0", "SA-swap")]["frac_optimal"] == 1 and sa_results[("R1", "SA-swap")]["frac_optimal"] == 1

assert all(r["frac_m_turbines"] == 1 for r in sa_results.values())

# %% [markdown]
# ## 3. Rydberg encodings
# *(Phase 2–3: Encoding A — native blockade on the grid; Encoding B — QUBO embedded into atom positions.)*
#
# ## 4. Results
# *(Phase 4: comparison table and % of optimum vs rung.)*
#
# ## 5. Limitations
# *(Phase 5.)*

# %% [markdown]
# ## Failure log
# Errors, workarounds and API friction met while building this notebook (source for the second slide).
#
# 1. **No QUBO solver tooling in qoolqit 1.4.0.** `QUBOInstance` and `AdiabaticDriveShaper` (documented under
#    Pasqal's QUBO application tools) are not part of the `qoolqit` package. QUBO→position embedders do ship:
#    `qoolqit.embedding.InteractionEmbedder` (Nelder–Mead fit of $1/r^6$, no scale factor, no minimum-spacing
#    constraint) and `Blade`.
# 2. **`BackendType` is the `pulser.backends` module**, not an enum: `list(BackendType)` fails, and available
#    backends depend on which emulator packages happen to be installed. A default `pip install qoolqit` gives only
#    `QutipBackendV2` (CPU); GPU emulation needs `emu-sv` (listed under qoolqit's `extras`).
# 3. **emu-sv 2.9.1 hard-codes complex128.** Overriding the module-level dtype to complex64 fails inside the
#    backend (`Float did not match Double` in the norm check), so the GPU runs in FP64 (slow on Ada-class cards).
# 4. **Backend cross-check (R0, 16 atoms, 500 shots).** QuTiP and emu-sv agree on the exact probability of the 79
#    optimal layouts to 4e-5. Wall clock: QuTiP 60 s (CPU, 16 threads), emu-sv 33 s (CPU), 8.8 s (GPU).
# 5. **Blockade leak.** A first schedule with final detuning 0.104 (close to $U(\sqrt2 a) = 0.125$) put 26% of shots
#    on 5 excitations, which the 4×4 king graph forbids. The final detuning must sit between $U(2a) \approx 0.0156$
#    and $U(\sqrt2 a)$; at 0.045 the 5-excitation fraction drops to 0.4%.
# 6. **The paper underspecifies wake-cell membership.** Several natural rules (strict vs non-strict inequalities,
#    tolerances, cell overlap) reproduce R0 but give different optima for long wakes (R1: 2187–2238 kW). We adopt the
#    rule that reproduces the expected optimum set on every rung.
# 7. **Bit-flip SA on the penalty QUBO gets trapped.** With the paper's protocol (single-bit flips, count penalty
#    $\lambda_1$) every move between 4-turbine layouts crosses a barrier of hundreds of kW; on R3 it never reaches the
#    optimum in 36 runs, and neal's default cold temperature (from the smallest QUBO coefficient, $2\lambda_1$) is too
#    hot to hold the turbine count at all. A fixed-count swap-move SA is used as the stronger classical reference.
# 8. **25 atoms: emu-sv cannot sample above 24 qubits.** The 25-atom GPU simulation completed (about 22 min at
#    T = 320, complex128) and then crashed in `StateVector.sample` (`torch.multinomial`: "number of categories cannot
#    exceed 2^24"). qoolqit's `LocalEmulator` always appends a `BitStrings` observable, so every emu-sv run above
#    24 atoms through `LocalEmulator` fails. Workaround if needed: call `SVBackend` directly with only a `StateResult`
#    and sample in numpy.
