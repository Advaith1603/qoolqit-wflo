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
# - Emulator results only (noiseless state-vector emulation). Published numbers are reported values, kept in separate
#   columns with their sources; published D-Wave hardware results are cited in prose only.
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
# wake, it is entirely within its wake"). We adopt a rule that reproduces the published R0 optimum set: with $s$ and
# $t$ the downstream and lateral offsets of $j$'s centre from $i$'s, $j$ is in the wake iff
# $0.001 < s \le x + 0.001$ and $t < r\,s - 10^{-9}$ (cells exactly on the cone edge are outside). Wind angles are
# counted clockwise from west (wind from 0° blows east; rows grow southwards).
#
# **One rule for every rung.** The same rule builds R0 and our rungs R1–R3. On R0 (and R2) it gives exactly the same
# wake matrix as a bare test `t < r*s`, so it reproduces the published instance (asserted below). The explicit margin
# only matters for $r = 1$ (R1, R3), our own definitions: there the cone edge is at 45° and diagonal neighbours lie
# exactly on it for some wind directions, so a bare floating-point test includes or excludes them depending on
# rounding (e.g. $\cos 90° \approx 6 \times 10^{-17}$): single entries change by up to 8.3 kW and the grid's mirror
# symmetry breaks by up to 4.2 kW. With the margin, every wake matrix is invariant under the symmetries of its wind regime (asserted below).

# %%
A_IND, R_T, V_WIND = 0.1, 0.33, 12.0      # axial induction, turbine radius (grid units), wind speed (m/s)
TOL = 1e-3
EDGE_EPS = 1e-9                            # cells exactly on the cone edge are outside the wake
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


def wake_matrix(L, x, r, regime, edge_eps=EDGE_EPS):
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
            if TOL < s <= x + TOL and t < r * s - edge_eps:
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
                      power_all=power_all, optimum=best, opt_idx=opt_idx, P_random=power_all[valid].mean())
    print(f"{name}: x={x} r={r} {regime:7s}  optimum {best:8.2f} kW  #optimal layouts {len(opt_idx):3d}"
          f"  e.g. sites {np.flatnonzero(BITS[opt_idx[0]]).tolist()}")

# %% [markdown]
# ### 1.2b Random baseline and normalised score
# A uniformly random 4-turbine layout already scores highly in "% of optimum", because most pairs of sites do not
# interact. $P_{\rm random}$ is the exact mean power over all 1,820 four-turbine layouts. Every raw % of optimum in this
# notebook is accompanied by the normalised score
# $$S = \frac{P - P_{\rm random}}{P_{\rm opt} - P_{\rm random}},$$
# which is 0 for random placement and 1 for an optimal layout (negative = worse than random).

# %%
def norm_score(power, d):
    return (np.asarray(power) - d["P_random"]) / (d["optimum"] - d["P_random"])


for name, d in INST.items():
    print(f"{name}: P_random = {d['P_random']:.2f} kW = {100 * d['P_random'] / d['optimum']:.2f}% of optimum")

# %% [markdown]
# ### 1.3 Oracle assertions
# Published values for R0: per-turbine power 576, pairwise penalty 83.728 at distance 1 and 17.582 at $\sqrt2$, zero
# beyond, optimum 2304.0 kW. Our rungs (with the symmetric edge rule): R1 2229.38 (unique), R2 2304.0 (same 79 optima as
# R0), R3 2242.71 with two mirror-image optima (the diagonal and the anti-diagonal), neither a king-graph MIS.

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

for name, target, n_opt in [("R1", 2229.38, 1), ("R2", 2304.0, 79), ("R3", 2242.71, 2)]:
    assert round(INST[name]["optimum"], 2) == target, name
    assert len(INST[name]["opt_idx"]) == n_opt, name
assert opt_sets("R3") == {(0, 5, 10, 15), (3, 6, 9, 12)}   # the two diagonals
assert not opt_sets("R3") & king_mis4                      # R3's optima are not king-graph MIS

# Symmetry: uniform-wind rungs are invariant under all 8 grid symmetries, west-wind rungs under the north-south mirror.
_rc = site_coords(L_GRID).astype(int)
_perm = lambda f: np.array([f(rr, cc)[0] * L_GRID + f(rr, cc)[1] for rr, cc in _rc])
MIRROR_NS = _perm(lambda rr, cc: (L_GRID - 1 - rr, cc))
D4 = [_perm(f) for f in (lambda rr, cc: (rr, cc), lambda rr, cc: (cc, L_GRID - 1 - rr),
                         lambda rr, cc: (L_GRID - 1 - rr, L_GRID - 1 - cc), lambda rr, cc: (L_GRID - 1 - cc, rr),
                         lambda rr, cc: (L_GRID - 1 - rr, cc), lambda rr, cc: (rr, L_GRID - 1 - cc),
                         lambda rr, cc: (cc, rr), lambda rr, cc: (L_GRID - 1 - cc, L_GRID - 1 - rr))]
for name, d in INST.items():
    for g in (D4 if d["regime"] == "uniform" else [MIRROR_NS]):
        assert np.allclose(d["W"], d["W"][np.ix_(g, g)], atol=1e-9), name

# The edge margin leaves the published instance untouched; it only changes the r = 1 rungs.
for name, (x, r, regime) in RUNGS.items():
    diff = np.abs(wake_matrix(L_GRID, x, r, regime, edge_eps=0.0) - INST[name]["W"]).max()
    print(f"{name}: max |W(bare t < r s) - W(margin)| = {diff:.2f} kW")
    assert (diff == 0) == (name in ("R0", "R2"))
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
# - **SA-swap** (the classical reference used in the comparison table): annealing restricted to exactly $m$
#   turbines, where each move relocates one turbine to an empty site; 20,000 steps.
# - **SA-QUBO** (a note only): bit-flip SA on the full QUBO, 1,000 sweeps, $\lambda_1 = 1.05 \times 576$ chosen by
#   scan (the paper does not state $\lambda_1$). The schedule runs from $\beta_0 = \ln 2 / (\lambda_1 + 576)$ to
#   $\beta_1 = \ln 100 / \max(\min_{W_{ij}>0} W_{ij}, 1)$. Every move between two 4-turbine layouts must pass through
#   3 or 5 turbines, so the count penalty creates barriers of hundreds of kW. On the long-wake rungs (R1, R3) the gap
#   to the second-best layout is only 22 kW and 4 kW, and bit-flip SA ends in local minima.
#
# The paper's SA (pyqubo + D-Wave `neal`, settings and $\lambda_1$ not stated) is **not** reproduced here. Read off
# its Fig. 9, its 36 runs on R0 have a median near 2080 kW (≈90% of optimum) and a minimum near 1730 kW; our SA-QUBO
# reaches 99.8% on average. We therefore make no claim that SA-QUBO reproduces their SA, and quote their numbers
# separately.

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
    pw = layout_power(X, d["W"])
    pct = 100 * pw / d["optimum"]
    idx = X.astype(int) @ (1 << np.arange(X.shape[1] - 1, -1, -1))
    return dict(mean_pct=float(pct.mean()), median_pct=float(np.median(pct)), min_pct=float(pct.min()),
                mean_norm=float(norm_score(pw, d).mean()),
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
        print(f"{name} {method}: mean {s['mean_pct']:6.2f}% of optimum (S = {s['mean_norm']:+.3f}, min {s['min_pct']:6.2f}%), "
              f"optimal in {round(s['frac_optimal'] * SA_RUNS):2d}/{SA_RUNS} runs, "
              f"{round(s['frac_m_turbines'] * SA_RUNS)}/{SA_RUNS} with exactly {M_TURB} turbines, {s['wall_s']:.2f} s")

# Phase 1 exit: every rung has W, per-site power, exact optimum and optimum set, and 36-run SA baselines.
assert all(k in INST[n] for n in RUNGS for k in ("W", "site_power", "optimum", "opt_idx", "qubo"))
assert all(r["frac_m_turbines"] == 1 for r in sa_results.values())
assert sa_results[("R0", "SA-swap")]["frac_optimal"] == 1 and sa_results[("R1", "SA-swap")]["frac_optimal"] == 1

# %% [markdown]
# ### 2.3 Published R0 baselines on the same scale
# Hancock et al. report the mean power of 36 VQE runs as % of optimum (their Table 1, simulator) and show SA as a box
# plot (their Fig. 9); the SA median below is read off that figure by eye (≈2080 kW) and is approximate. These are
# their reported numbers, not reruns; $S$ only re-expresses them against the exact random baseline
# $P_{\rm random}$(R0) = 2187.70 kW printed in Section 1.2b. On this scale the reported VQE means lie between −0.33 and
# +0.11, close to random placement, which suggests that "% of optimum" alone overstates performance on R0.
#
# **Reading uncertainty.** The SA median is read from a figure; we take a reading error of ±1% of optimum (±23 kW).
# Because $P_{\rm opt} - P_{\rm random}$ is only 116 kW on R0, this becomes ±0.2 in $S$: the published SA median is
# $S \approx -0.9 \pm 0.2$, i.e. below random placement for any reading within the error.

# %%
PUBLISHED_R0 = {
    "VQE COBYLA-1.00 (mean)": 95.5, "VQE COBYLA-0.75 (mean)": 94.7, "VQE COBYLA-0.50 (mean)": 94.5,
    "VQE COBYLA-0.25 (mean)": 95.2, "VQE Bayes-1.00 (mean)": 93.3, "VQE Bayes-0.75 (mean)": 94.2,
    "VQE Bayes-0.50 (mean)": 94.6, "VQE Bayes-0.25 (mean)": 93.4,
    "SA, pyqubo/neal (median, read off Fig. 9)": 100 * 2080 / 2304,
}
SA_READ_ERR_PCT = 1.0          # reading error of the Fig. 9 median, in % of optimum
print(f"P_random(R0) = {INST['R0']['P_random']:.2f} kW")
for label, pct in PUBLISHED_R0.items():
    print(f"{label:44s} {pct:6.2f}% of optimum  S = {norm_score(pct / 100 * INST['R0']['optimum'], INST['R0']):+.2f}")
SA_PUB_S = float(norm_score(PUBLISHED_R0["SA, pyqubo/neal (median, read off Fig. 9)"] / 100 * INST["R0"]["optimum"], INST["R0"]))
SA_PUB_S_ERR = SA_READ_ERR_PCT / 100 * INST["R0"]["optimum"] / (INST["R0"]["optimum"] - INST["R0"]["P_random"])
print(f"published SA median: S = {SA_PUB_S:+.1f} +/- {SA_PUB_S_ERR:.1f} (reading error +/-{SA_READ_ERR_PCT:.0f}% of optimum)")
assert round(SA_PUB_S, 1) == -0.9 and round(SA_PUB_S_ERR, 1) == 0.2 and SA_PUB_S + SA_PUB_S_ERR < 0

# %% [markdown]
# ## 3. Rydberg encodings
# ### 3.1 Emulation setup
# Programs are written in QoolQit's dimensionless units (for `AnalogDevice`: distance unit 5 µm, energy unit
# $C_6/(5\,\mu m)^6$, so $U(r) = 1/r^6$) and compiled to `AnalogDevice`. We emulate with emu-sv (exact state vector,
# complex128, GPU if available) and read the **exact final state**, so the primary metrics carry no sampling noise.
# Bitstring convention: character $k$ = qubit $k$ = site $k$; as an integer index, qubit 0 is the most significant bit.

# %%
import logging

from emu_sv import SVConfig
from qoolqit import AnalogDevice, Drive, PiecewiseLinearWaveform, QuantumProgram, Register
from qoolqit.execution import BackendType, LocalEmulator, StateResult

DEVICE = AnalogDevice()
T_MAX = DEVICE.specs["max_duration"]


def adiabatic_drive(delta_final, delta_0, omega, T=T_MAX):
    """Standard MIS schedule: amplitude up / hold / down; detuning held at delta_0, swept to delta_final, held."""
    amp = PiecewiseLinearWaveform([0.1 * T, 0.8 * T, 0.1 * T], [0.0, omega, omega, 0.0])
    det = PiecewiseLinearWaveform([0.1 * T, 0.8 * T, 0.1 * T], [delta_0, delta_0, delta_final, delta_final])
    return Drive(amplitude=amp, detuning=det)


def final_probs(register, drive, gpu=GPU):
    """Compile, emulate with emu-sv and return (exact bitstring probabilities, wall-clock seconds)."""
    prog = QuantumProgram(register, drive)
    prog.compile_to(DEVICE)
    cfg = SVConfig(gpu=gpu, observables=(StateResult(evaluation_times=[1.0]),), log_level=logging.WARNING)
    emu = LocalEmulator(backend_type=BackendType.SVBackend, emulation_config=cfg, num_shots=1)
    t0 = time.perf_counter()
    res = emu.run(prog).results()
    wall = time.perf_counter() - t0
    res = res[0] if isinstance(res, (list, tuple)) else res
    probs = (res.final_state.data.abs() ** 2).double().cpu().numpy()
    return probs / probs.sum(), wall


# %% [markdown]
# ### 3.2 Bitstring-to-site mapping test
# Four atoms: three in a mutually blockaded triangle (spacing 1) and one isolated far away. After an adiabatic
# sweep the isolated atom is excited and exactly one triangle atom is. Placing the isolated atom first and then
# last checks that bit $k$ of the returned index really is qubit $k$.

# %%
TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2)]
for iso in (0, 3):
    others = [q for q in range(4) if q != iso]
    pos = [None] * 4
    pos[iso] = (6.0, 0.0)
    for q, c in zip(others, TRIANGLE):
        pos[q] = c
    pr, _ = final_probs(Register.from_coordinates(pos), adiabatic_drive(0.08, -0.10, 1 / 1.7**6), gpu=False)
    B4 = bit_matrix(4)
    p_iso, p_one = pr[B4[:, iso] == 1].sum(), pr[B4[:, others].sum(1) == 1].sum()
    print(f"isolated qubit {iso}: P(isolated excited) = {p_iso:.3f}, P(exactly one triangle atom) = {p_one:.3f}")
    assert p_iso > 0.9 and p_one > 0.9
print("mapping test passes: bit k of the state index is qubit k")

# %% [markdown]
# ### 3.3 Encoding A (native blockade)
# Atoms sit on the 4×4 grid at spacing $a = 1$ (the device minimum, 5 µm), site $k$ at $(x, y) = (k \% 4, k // 4)$.
# All sites have equal power, so one global detuning sweep suffices.
#
# **Encoding A is rung-blind.** It is one Hamiltonian (same register, same drive) whatever the wake model; the single
# final state is re-scored with each rung's power function. Differences between rungs therefore measure how well the
# blockade physics happens to match each wake model, not any adaptation of the program.
#
# **Rabi amplitude.** $\Omega = 1.7^{-6} \approx 0.0414$ in device units, i.e. $0.0414 \times 55.4 \approx 2.29$ rad/µs
# ($\Omega/2\pi \approx 0.37$ MHz), against a device maximum of 0.2268 (12.6 rad/µs). It was chosen so that the blockade
# radius $R_b = \Omega^{-1/6} = 1.7a$ sits inside $(\sqrt2 a, 2a)$: the blockade then forbids exactly the king-graph
# neighbours, the wake structure of R0. A 3 × 3 check below confirms that larger $\Omega$ (smaller $R_b$) is worse.
#
# **Final detuning.** At the end of the sweep, adding an excitation gains $\Delta_f$ and costs the van der Waals
# tails to atoms already excited: $U(\sqrt2 a) = 0.125$ forbids diagonal neighbours, but a new atom typically sits at
# $2a$ from 2–4 excited atoms, costing $\approx 0.03$–$0.06$. The useful window is therefore $\approx 0.06 < \Delta_f <
# 0.125$. We scan $\Delta_f$ on R0 only and freeze the schedule for every rung (no per-rung tuning); $T$ = device maximum.
#
# **Tuning criterion.** Tuning objectives must not use the optimum. We select $\Delta_f$ by P(exactly $m$ turbines),
# which needs no ground truth. On R0 it peaks at the same $\Delta_f = 0.10$ as the exact P(optimal layout), so the
# two criteria give the same schedule.

# %%
L_SIDE = L_GRID
OMEGA_A, DELTA_0_A = 1 / 1.7**6, -0.10
R_B = OMEGA_A ** (-1 / 6)
assert np.sqrt(2) < R_B < 2
REG_A = Register.from_coordinates([(float(k % L_SIDE), float(k // L_SIDE)) for k in range(L_SIDE**2)])
assert np.isclose(REG_A.min_distance(), 1.0) and REG_A.min_distance() >= DEVICE.specs["min_distance"]


def exact_metrics(probs, d):
    """Sampling-free metrics of a final state for one rung."""
    p_w = np.bincount(WEIGHT, weights=probs, minlength=L_GRID**2 + 1)
    m4 = WEIGHT == M_TURB
    best4 = np.flatnonzero(m4)[np.argmax(probs[m4])]
    return dict(
        P3=float(p_w[3]), P4=float(p_w[4]), P5=float(p_w[5:].sum()),
        P_opt=float(probs[d["opt_idx"]].sum()),
        P_opt_given_4=float(probs[d["opt_idx"]].sum() / p_w[4]),
        exp_pct_given_4=float(100 * (probs[m4] * d["power_all"][m4]).sum() / p_w[4] / d["optimum"]),
        exp_norm_given_4=float(norm_score((probs[m4] * d["power_all"][m4]).sum() / p_w[4], d)),
        mode4_pct=float(100 * d["power_all"][best4] / d["optimum"]),   # paper's rule applied to the exact distribution
        mode4_norm=float(norm_score(d["power_all"][best4], d)),
    )


DELTA_SCAN = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11]
scan = {}
for df in DELTA_SCAN:
    pr, wall = final_probs(REG_A, adiabatic_drive(df, DELTA_0_A, OMEGA_A))
    scan[df] = dict(**exact_metrics(pr, INST["R0"]), wall_s=wall, probs=pr)
print(" Delta_f    P(3)    P(4)    P(5)   P(opt)  P(opt|4)  E[%opt|4]   wall")
for df, m in scan.items():
    print(f"  {df:.2f}   {m['P3']:.4f}  {m['P4']:.4f}  {m['P5']:.4f}  {m['P_opt']:.4f}   {m['P_opt_given_4']:.4f}"
          f"    {m['exp_pct_given_4']:6.2f}  {m['wall_s']:5.1f} s")

# %% [markdown]
# **Ω × Δ_f check.** A 3 × 3 grid, $\Omega \in \{1.7^{-6}, 2 \times 1.7^{-6}, 0.2268\,(\text{device max})\}$ ×
# $\Delta_f \in \{0.09, 0.10, 0.11\}$, ranked by P(4 turbines). Recomputed in full mode (9 emulations); demo mode
# prints the recorded table.

# %%
OMEGA_CHECK_RECORDED = [  # (Omega, Delta_f, P3, P4, P5, P_opt), emu-sv, R0
    (0.0414, 0.09, 0.154, 0.725, 0.109, 0.686), (0.0414, 0.10, 0.099, 0.733, 0.163, 0.696),
    (0.0414, 0.11, 0.067, 0.707, 0.224, 0.672), (0.0829, 0.09, 0.239, 0.635, 0.091, 0.602),
    (0.0829, 0.10, 0.165, 0.656, 0.163, 0.612), (0.0829, 0.11, 0.107, 0.637, 0.250, 0.592),
    (0.2268, 0.09, 0.277, 0.502, 0.153, 0.403), (0.2268, 0.10, 0.238, 0.517, 0.208, 0.380),
    (0.2268, 0.11, 0.249, 0.482, 0.226, 0.326),
]
if MODE == "full":
    omega_check = []
    for om in (OMEGA_A, 2 * OMEGA_A, DEVICE.specs["max_amplitude"]):
        for df in (0.09, 0.10, 0.11):
            m = exact_metrics(final_probs(REG_A, adiabatic_drive(df, DELTA_0_A, om))[0], INST["R0"])
            omega_check.append((om, df, m["P3"], m["P4"], m["P5"], m["P_opt"]))
    assert np.allclose(np.round(np.array(omega_check), 3), np.array(OMEGA_CHECK_RECORDED), atol=2e-3)
else:
    omega_check = OMEGA_CHECK_RECORDED
print(" Omega   R_b   Delta_f  P(3)   P(4)   P(5)  P(opt)")
for om, df, p3, p4, p5, po in omega_check:
    print(f"{om:.4f}  {om ** (-1 / 6):.3f}   {df:.2f}  {p3:.3f}  {p4:.3f}  {p5:.3f}  {po:.3f}")
best_om, best_df = max(omega_check, key=lambda row: row[3])[:2]
assert np.isclose(best_om, OMEGA_A, atol=1e-4) and best_df == 0.10   # the current schedule maximises P(4 turbines)

# %% [markdown]
# **Frozen schedule.** $\Omega = 1.7^{-6}$, $\Delta: -0.10 \to 0.10$, $T = T_{\max}$, 10% / 80% / 10% ramp-hold-ramp,
# used unchanged on every rung.

# %%
DELTA_F_A = 0.10
assert DELTA_F_A == max(scan, key=lambda df: scan[df]["P4"])      # ground-truth-free criterion
assert DELTA_F_A == max(scan, key=lambda df: scan[df]["P_opt"])   # coincides with the exact-optimum criterion on R0
PROBS_A = scan[DELTA_F_A]["probs"]          # Encoding A's final state is the same on every rung (same register, same drive)

# %% [markdown]
# ### 3.4 Selection-rule comparison (36 × 500 resampling)
# The paper selects, per run, the most probable bitstring with exactly $m$ turbines. To reproduce that rule we draw
# 36 independent sets of 500 shots from the exact final state and, in each, take the most frequent 4-turbine
# bitstring (ties broken at random). **The spread across these 36 runs is pure shot noise of one fixed quantum state;
# it is not comparable to the run-to-run spread of the paper's VQE, which comes from 36 separate optimisations.**

# %%
def resample_selection(probs, d, n_runs=36, shots=500, rng=RNG):
    m4 = WEIGHT == M_TURB
    pct, nrm, sel_opt, p_opt_shot, p_valid = [], [], [], [], []
    for _ in range(n_runs):
        counts = np.bincount(rng.choice(len(probs), size=shots, p=probs), minlength=len(probs))
        c4 = np.where(m4, counts, 0)
        p_opt_shot.append(counts[d["opt_idx"]].sum() / shots)
        p_valid.append(c4.sum() / shots)
        if c4.max() == 0:
            pct.append(np.nan); nrm.append(np.nan); sel_opt.append(False); continue
        sel = rng.choice(np.flatnonzero(c4 == c4.max()))
        pct.append(100 * d["power_all"][sel] / d["optimum"])
        nrm.append(norm_score(d["power_all"][sel], d))
        sel_opt.append(sel in set(d["opt_idx"]))
    pct = np.array(pct)
    return dict(mean_pct=float(np.nanmean(pct)), median_pct=float(np.nanmedian(pct)), min_pct=float(np.nanmin(pct)),
                mean_norm=float(np.nanmean(nrm)), frac_selected_optimal=float(np.mean(sel_opt)), mean_P_opt_shot=float(np.mean(p_opt_shot)),
                mean_valid_rate=float(np.mean(p_valid)), n_no_valid=int(np.isnan(pct).sum()))


results_A = {}
for name, d in INST.items():
    results_A[name] = dict(exact=exact_metrics(PROBS_A, d), resampled=resample_selection(PROBS_A, d))
    e, r = results_A[name]["exact"], results_A[name]["resampled"]
    print(f"{name} Encoding A | exact: P(opt) {e['P_opt']:.4f}, P(opt|4) {e['P_opt_given_4']:.4f}, "
          f"E[% opt | 4] {e['exp_pct_given_4']:6.2f} (S {e['exp_norm_given_4']:+.3f}), mode-of-4 {e['mode4_pct']:6.2f}% "
          f"(S {e['mode4_norm']:+.3f}) | 36x500: mean {r['mean_pct']:6.2f}% (S {r['mean_norm']:+.3f}), selected optimal "
          f"{round(r['frac_selected_optimal'] * 36):2d}/36, P(opt shot) {r['mean_P_opt_shot']:.3f}")

# Phase 2 exit (Tier 1): on R0 the most probable 4-turbine layout is optimal, and every resampled run selects an optimal layout.
assert results_A["R0"]["exact"]["mode4_pct"] == 100.0
assert results_A["R0"]["resampled"]["frac_selected_optimal"] == 1.0
print("Tier 1: R0 solved natively (emulator)")

# %% [markdown]
# On R1 the rung-blind state still selects the optimal layout because the van der Waals tail favours spread-out
# layouts (the four corners), which partly mimics the long wake; the exact P(optimal) nevertheless drops from 0.70 to
# 0.32. On R3 the optimum places turbines on diagonal neighbours, which the blockade forbids, so P(optimal) = 0.

# %% [markdown]
# ### 3.5 Encoding B (QUBO embedded into atom positions)
# Among layouts with exactly $m$ turbines, power is $m \cdot 576 - \sum_{i<j} W_{ij} x_i x_j$, so the ranking depends only
# on the wake matrix $W$. We drop the all-to-all count penalty (it is not geometric), use the global detuning as a
# chemical potential to centre the shot distribution on $m$ excitations, and post-select exactly $m$ (the paper's
# VQE used the same post-selection).
#
# Atoms take free 2D positions $r_i$ chosen so that $U_{ij} = C_6/|r_i - r_j|^6$ (with $C_6 = 1$ in device units)
# matches $s\,W_{ij}$ over all pairs:
# $$\epsilon = \frac{\lVert U(r) - sW \rVert_F}{\lVert sW \rVert_F}.$$
# For fixed positions the best scale is $s^* = \lVert U\rVert^2 / \langle U, W\rangle$, which gives
# $\epsilon^2 = 1 - \cos^2(U, W)$; we minimise this with L-BFGS (gradients by torch autograd) plus quadratic penalties
# for pair distances below the device minimum (1) and radial distances above the device maximum (7.6), from 20 random
# starts and one start on the grid. $\epsilon$ is invariant under a uniform dilation, so the fitted layout is finally
# rescaled to minimum spacing 1 (the strongest interactions the device allows), and the register constraints are
# re-checked. We also report the Spearman rank correlation between $U$ and $W$ over pairs, and the same metrics for
# QoolQit's `InteractionEmbedder` (Nelder–Mead fit of $1/r^6$ to $W / \max W$, shipped defaults) as a baseline.

# %%
from scipy.optimize import minimize
from scipy.stats import kendalltau, spearmanr
from qoolqit.embedding import InteractionEmbedder

D_MIN, R_MAX = DEVICE.specs["min_distance"], DEVICE.specs["max_radial_distance"]
IU = np.triu_indices(L_GRID**2, 1)


def pair_U(pos):
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return 1.0 / d[IU] ** 6


def embed_metrics(pos, W):
    U, w = pair_U(pos), W[IU]
    s = (U @ U) / (U @ w)
    eps = np.linalg.norm(U - s * w) / np.linalg.norm(s * w)
    return dict(eps=float(eps), spearman=float(spearmanr(U, w)[0]), s=float(s))


def fit_positions(W, n_random=20, seed=SEED, penalty=1e3):
    n = len(W)
    w = torch.tensor(W[IU], dtype=torch.float64)
    iu = torch.tensor(np.array(IU))

    def objective(flat):
        x = torch.tensor(flat, dtype=torch.float64, requires_grad=True)
        pos = x.view(n, 2)
        d = (pos[iu[0]] - pos[iu[1]]).norm(dim=1)
        U = d ** -6
        loss = 1 - (U @ w) ** 2 / ((U @ U) * (w @ w))
        rad = (pos - pos.mean(0)).norm(dim=1)
        loss = loss + penalty * (torch.relu(D_MIN - d) ** 2).sum() + penalty * (torch.relu(rad - R_MAX) ** 2).sum()
        loss.backward()
        return loss.item(), x.grad.numpy().copy()

    rng = np.random.default_rng(seed)
    grid = site_coords(L_GRID)[:, ::-1] * 1.0                      # (x, y) = (col, row), spacing 1
    starts = [grid] + [rng.uniform(0, 2 * (L_GRID - 1), size=(n, 2)) for _ in range(n_random)]
    fits = []
    for x0 in starts:
        res = minimize(objective, x0.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=5000))
        pos = res.x.reshape(n, 2)
        pos = (pos - pos.mean(0)) / np.linalg.norm(pos[:, None] - pos[None], axis=-1)[IU].min() * D_MIN * (1 + 1e-6)
        fits.append((embed_metrics(pos, W)["eps"], pos))
    eps_all = np.array([f[0] for f in fits])
    best = fits[int(np.argmin(eps_all))][1]
    return best, eps_all


embedding = {}
for name in RUNGS:
    W = INST[name]["W"]
    t0 = time.perf_counter()
    pos, eps_all = fit_positions(W)
    t_fit = time.perf_counter() - t0
    radial = np.linalg.norm(pos - pos.mean(0), axis=1).max()
    ie = InteractionEmbedder().embed(W / W.max())
    pos_ie = np.array([ie.coords[k] for k in sorted(ie.coords)])
    embedding[name] = dict(pos=pos, pos_ie=pos_ie, eps_restarts=eps_all, radial=radial, ours=embed_metrics(pos, W),
                           interaction_embedder=embed_metrics(pos_ie, W),
                           ie_min_distance=float(np.linalg.norm(pos_ie[:, None] - pos_ie[None], axis=-1)[IU].min()))
    e, b = embedding[name]["ours"], embedding[name]["interaction_embedder"]
    print(f"{name}: ours eps {e['eps']:.3f} (restarts: median {np.median(eps_all):.3f}, grid start {eps_all[0]:.3f}), "
          f"Spearman {e['spearman']:+.3f}, max radius {radial:.2f}, fit {t_fit:.1f} s | InteractionEmbedder eps "
          f"{b['eps']:.3f}, Spearman {b['spearman']:+.3f}, min distance {embedding[name]['ie_min_distance']:.3f}")
    assert radial <= R_MAX + 1e-9

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, name in zip(axes, RUNGS):
    pos = embedding[name]["pos"]
    W = INST[name]["W"]
    for i, j in zip(*IU):
        if W[i, j] > 0:
            ax.plot(*pos[[i, j]].T, color="0.6", lw=0.4 + 2.5 * W[i, j] / W.max(), zorder=1)
    ax.scatter(*pos.T, s=60, zorder=2)
    for k, (xx, yy) in enumerate(pos):
        ax.annotate(str(k), (xx, yy), fontsize=7, ha="center", va="center", color="white", zorder=3)
    ax.set_title(f"{name}: ε = {embedding[name]['ours']['eps']:.2f}")
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle("Encoding B atom positions (edge width ∝ wake penalty $W_{ij}$)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.6 Energy ranking: does the programmed Hamiltonian order the layouts correctly?
# What decides the outcome is not how well individual couplings match ($\epsilon$) but how the programmed energy
# $E_{\rm prog}(S) = \sum_{i<j \in S} U_{ij}$ orders the 1,820 four-atom layouts $S$ compared with the true wake
# penalty $E_{\rm true}(S) = \sum_{i<j \in S} W_{ij}$ (lower = more power). We report the Spearman correlation between the
# two over all layouts; Kendall's $\tau_b$ over the lowest-energy 5% by true energy (91 layouts, where the ground state is
# decided); the rank of the best true optimum under $E_{\rm prog}$ (0 = it is the programmed ground state); and the gap
# from the programmed ground state to the next layout. This needs no emulation.
#
# **Ties.** $\tau_b$ is used for the tail because it corrects for ties. On R0 and R2 the tail still holds only two
# distinct true energies (the 79 tied optima and 12 others), so even $\tau_b$ says little there; on those rungs we read
# the rank of the optimum and P(optimal | 4) instead. On R1 and R3 the tail has 15–18 distinct energies.

# %%
M4_IDX = np.flatnonzero(WEIGHT == M_TURB)
B4_ALL = BITS[M4_IDX].astype(float)
PAIR_IN_SET = np.stack([B4_ALL[:, i] * B4_ALL[:, j] for i, j in zip(*IU)], 1)   # (1820, 120)
GRID_POS = site_coords(L_GRID)[:, ::-1].astype(float)
N_TAIL = int(round(0.05 * len(M4_IDX)))


def programmed_energy(pos):
    return PAIR_IN_SET @ pair_U(pos)


def ranking_metrics(pos, d):
    E_true = -d["power_all"][M4_IDX]
    E_prog = programmed_energy(pos)
    tail = np.argsort(E_true)[:N_TAIL]
    order = np.argsort(E_prog)
    rank = min(int(np.flatnonzero(M4_IDX[order] == o)[0]) for o in d["opt_idx"])
    return dict(rho_all=float(spearmanr(E_prog, E_true)[0]),
                tau_tail=float(kendalltau(E_prog[tail], E_true[tail], variant="b")[0]),
                tail_levels=int(len(np.unique(np.round(E_true[tail], 6)))), opt_rank=rank, gap=float(E_prog[order[1]] - E_prog[order[0]]),
                ground_norm=float(norm_score(d["power_all"][M4_IDX[order[0]]], d)))


ranking = {}
print("rung enc  eps    rho(all)  tau_b(low 5%) [levels]  optimum rank  gap")
for name, d in INST.items():
    for enc, pos in (("A", GRID_POS), ("B", embedding[name]["pos"]), ("IE", embedding[name]["pos_ie"])):
        g = ranking[(name, enc)] = dict(eps=embed_metrics(pos, d["W"])["eps"], **ranking_metrics(pos, d))
        print(f" {name}  {enc:2s}  {g['eps']:.3f}   {g['rho_all']:+.3f}    {g['tau_tail']:+.3f}  [{g['tail_levels']:2d}]"
              f"         {g['opt_rank']:4d}     {g['gap']:.1e}")
assert all(ranking[(n, e)]["opt_rank"] == 0 for n in ("R1", "R2") for e in ("A", "B"))
assert ranking[("R3", "A")]["opt_rank"] > 0 and ranking[("R3", "B")]["opt_rank"] > 0

# R3: B's layout differs from the grid, yet both give the optimum the same rank.
disp = np.linalg.norm(embedding["R3"]["pos"] - (GRID_POS - GRID_POS.mean(0)), axis=1)
ahead = {enc: set(M4_IDX[np.argsort(programmed_energy(pos), kind="stable")[:ranking[("R3", enc)]["opt_rank"]]].tolist())
         for enc, pos in (("A", GRID_POS), ("B", embedding["R3"]["pos"]))}
print(f"R3 B displacement from the grid: max {disp.max():.2f}, mean {disp.mean():.2f} (grid spacing 1); "
      f"same {len(ahead['A'])} layouts ranked ahead of the optima under A and B: {ahead['A'] == ahead['B']}")
assert disp.max() > 0.3 and ahead["A"] == ahead["B"]

# %% [markdown]
# **Pairwise $\epsilon$ does not predict solution quality.** Among the embedded registers, R1 has the worst $\epsilon$ yet
# its programmed ground state is the true optimum, while R3 has one of the best $\epsilon$ yet its optimum ranks in the
# hundreds. On R1 and R2 both encodings share the true optimum as programmed ground state, so Encoding B can only differ
# from A through the dynamics (gaps, detuning, how much weight lands on exactly $m$ excitations). On R3 neither does:
# for Encoding A this is structural (the optima put turbines on grid-diagonal neighbours, which the blockade forbids);
# for Encoding B the positions are free, so it is a property of this fit, not a proof that no embedding could work.
# `InteractionEmbedder` (shipped defaults) orders the layouts poorly on every rung.
#
# On R3, B moves atoms by up to about half a grid spacing, yet the optima keep rank 355 because exactly the same 355
# layouts are programmed below them under A and B: the fit, started from the grid, preserves the grid's ordering of the
# low-energy block.

# %%
def tune_and_run(register, d, omegas, deltas, tag):
    """Scan Omega x Delta_f, select by P(exactly m) only (no ground truth), return metrics at the selected point."""
    assert register.min_distance() >= D_MIN
    scan_pts = {}
    for om in omegas:
        for df in deltas:
            pr, _ = final_probs(register, adiabatic_drive(df, DELTA_0_A, om))
            scan_pts[(om, df)] = (exact_metrics(pr, d), pr)
            log_row(tag=tag, omega=om, delta_f=df, **scan_pts[(om, df)][0])
    om, df = max(scan_pts, key=lambda k: scan_pts[k][0]["P4"])
    return dict(omega=om, delta_f=df, exact=scan_pts[(om, df)][0],
                resampled=resample_selection(scan_pts[(om, df)][1], d), probs=scan_pts[(om, df)][1],
                p4_grid={k: v[0]["P4"] for k, v in scan_pts.items()})


RUN_CSV = "runs/results_full.csv"


def log_row(**row):
    """Full mode only: append each emulation result to a CSV on disk as it completes."""
    if MODE != "full":
        return
    os.makedirs(os.path.dirname(RUN_CSV), exist_ok=True)
    new_file = not os.path.exists(RUN_CSV)
    with open(RUN_CSV, "a") as fh:
        if new_file:
            fh.write(",".join(row) + "\n")
        fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")


OMEGA_GRID = (OMEGA_A, 2 * OMEGA_A, DEVICE.specs["max_amplitude"])
DELTA_SCAN_B = [0.02, 0.03, 0.04, 0.06, 0.10, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0]

# %% [markdown]
# ### 3.7 Encoding B programs
# Same pulse shape as Encoding A ($\Delta_0 = -0.10$, $T = T_{\max}$). Per rung, $\Omega \in \{1.7^{-6},
# 2 \times 1.7^{-6}, 0.2268\}$ × $\Delta_f$ over a wide, roughly logarithmic grid (P(4) is not monotonic in $\Delta_f$ for
# embedded registers), selected **only** by P(exactly $m$ excitations). Emulated in full mode; demo mode skips it.

# %%
results_B = {}
if MODE == "full":
    for name in ("R1", "R2", "R3"):
        reg_B = Register.from_coordinates([tuple(map(float, p)) for p in embedding[name]["pos"]])
        results_B[name] = tune_and_run(reg_B, INST[name], OMEGA_GRID, DELTA_SCAN_B, f"B_{name}")
        r = results_B[name]
        print(f"{name} B: selected Omega {r['omega']:.4f}, Delta_f {r['delta_f']}, P(4) {r['exact']['P4']:.3f}")

# %% [markdown]
# ### 3.8 R3 probes (time-boxed)
# Two attempts to make R3's optimum the programmed ground state, both selected by the low-energy-tail ranking metric
# (Kendall $\tau_b$ over the best 5% of layouts), never by the optimum's rank, which is reported only as the outcome.
#
# 1. **Anisotropic lattice.** Atoms on a rectangular lattice with along-wind (west–east) spacing $a_x$ and cross-wind
#    spacing $a_y$, the smaller of the two equal to the device minimum; one scan over the aspect ratio $a_y / a_x$.
#    Over all layouts the correlation prefers cross-wind stretching, over the low-energy tail along-wind stretching;
#    we select by the tail, which decides the ground state. The tail metric is flat (within 0.02) for $a_y/a_x$ between
#    0.25 and 0.42, so we emulate every lattice on that plateau and report the range of results rather than a single
#    tuned value; stretching much beyond 0.25 would exceed the device's maximum radial distance.
# 2. **Ranking-aware fit (non-scalable upper-bound probe).** Positions fitted to maximise a weighted Pearson
#    correlation between $E_{\rm prog}$ and $E_{\rm true}$ over all 1,820 layouts, from the grid, the Encoding B layout
#    and 20 random starts. Tail weightings (rest-of-layouts weight 0.1, 0.01, 0) are compared by the same tail metric.
#    It enumerates all $\binom{n}{m}$ layouts, so it does not scale and serves only to probe what an embedding could
#    achieve at this size.

# %%
ASPECTS = np.geomspace(0.25, 4.0, 17)


def rect_lattice(aspect):
    ax_, ay_ = (1.0, aspect) if aspect >= 1 else (1 / aspect, 1.0)
    pos = np.c_[(np.arange(L_GRID**2) % L_GRID) * ax_, (np.arange(L_GRID**2) // L_GRID) * ay_] * D_MIN * (1 + 1e-6)
    return pos - pos.mean(0)                     # the device measures radial distance from the origin


aspect_scan = {a: ranking_metrics(rect_lattice(a), INST["R3"]) for a in ASPECTS}
print("a_y/a_x  rho(all)  tau_b(low 5%)  optimum rank")
for a, g in aspect_scan.items():
    print(f" {a:5.3f}   {g['rho_all']:+.3f}    {g['tau_tail']:+.3f}       {g['opt_rank']:4d}")
ASPECT_R3 = max(aspect_scan, key=lambda a: aspect_scan[a]["tau_tail"])
print(f"selected a_y/a_x = {ASPECT_R3:.3f}: optimum rank {aspect_scan[ASPECT_R3]['opt_rank']}")
# The tail metric is flat near its maximum, so the lattice is reported as a plateau, not as one tuned aspect ratio.
PLATEAU_TOL = 0.02
PLATEAU = [a for a in ASPECTS if aspect_scan[a]["tau_tail"] >= aspect_scan[ASPECT_R3]["tau_tail"] - PLATEAU_TOL]
assert np.allclose(PLATEAU, ASPECTS[:4]) and np.isclose(PLATEAU[-1], 0.42, atol=1e-3)   # a_y/a_x = 0.25 ... 0.42
print(f"plateau (tau_b within {PLATEAU_TOL} of max): a_y/a_x = {', '.join(f'{a:.3f}' for a in PLATEAU)}; "
      f"optimum rank {min(aspect_scan[a]['opt_rank'] for a in PLATEAU)}-{max(aspect_scan[a]['opt_rank'] for a in PLATEAU)}")

# %%
def ranking_fit(d, w_rest, n_random=20, seed=SEED, penalty=1e3):
    E_true = -d["power_all"][M4_IDX]
    tail = np.zeros(len(M4_IDX), bool)
    tail[np.argsort(E_true)[:N_TAIL]] = True
    w = torch.tensor(np.where(tail, 1.0, w_rest))
    w = w / w.sum()
    Et = torch.tensor(E_true)
    P = torch.tensor(PAIR_IN_SET)
    iu = torch.tensor(np.array(IU))

    def objective(flat):
        x = torch.tensor(flat, dtype=torch.float64, requires_grad=True)
        pos = x.view(-1, 2)
        dist = (pos[iu[0]] - pos[iu[1]]).norm(dim=1)
        Ep = P @ dist**-6
        a, b = Ep - (w * Ep).sum(), Et - (w * Et).sum()
        rho = (w * a * b).sum() / torch.sqrt((w * a * a).sum() * (w * b * b).sum())
        rad = (pos - pos.mean(0)).norm(dim=1)
        loss = 1 - rho + penalty * (torch.relu(D_MIN - dist) ** 2).sum() + penalty * (torch.relu(rad - R_MAX) ** 2).sum()
        loss.backward()
        return loss.item(), x.grad.numpy().copy()

    rng = np.random.default_rng(seed)
    starts = [GRID_POS, embedding[name_probe]["pos"]] + [rng.uniform(0, 6, size=(L_GRID**2, 2)) for _ in range(n_random)]
    fits = []
    for x0 in starts:
        res = minimize(objective, x0.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=5000))
        pos = res.x.reshape(-1, 2)
        pos = (pos - pos.mean(0)) / np.linalg.norm(pos[:, None] - pos[None], axis=-1)[IU].min() * D_MIN * (1 + 1e-6)
        fits.append((ranking_metrics(pos, d), pos))
    return max(fits, key=lambda f: f[0]["tau_tail"])


name_probe = "R3"
rank_fits = {}
for w_rest in (0.1, 0.01, 0.0):
    t0 = time.perf_counter()
    g, pos = ranking_fit(INST["R3"], w_rest)
    rank_fits[w_rest] = (g, pos)
    print(f"weight outside the tail {w_rest}: rho(all) {g['rho_all']:+.3f}, tau_b(low 5%) {g['tau_tail']:+.3f}, "
          f"optimum rank {g['opt_rank']}, {time.perf_counter() - t0:.0f} s")
W_REST = max(rank_fits, key=lambda w: rank_fits[w][0]["tau_tail"])
POS_RANKFIT_R3 = rank_fits[W_REST][1]
assert np.linalg.norm(POS_RANKFIT_R3 - POS_RANKFIT_R3.mean(0), axis=1).max() <= R_MAX
print(f"selected weighting {W_REST}: optimum rank {rank_fits[W_REST][0]['opt_rank']}")

# %%
results_R3_probe = {}
if MODE == "full":
    for label, pos in (("lattice", rect_lattice(ASPECT_R3)), ("rankfit", POS_RANKFIT_R3)):
        reg = Register.from_coordinates([tuple(map(float, p)) for p in pos])
        results_R3_probe[label] = tune_and_run(reg, INST["R3"], OMEGA_GRID, DELTA_SCAN_B, f"R3_{label}")
        r = results_R3_probe[label]
        print(f"R3 {label}: selected Omega {r['omega']:.4f}, Delta_f {r['delta_f']}, P(4) {r['exact']['P4']:.3f}")
lattice_plateau = {}
if MODE == "full":
    for a in PLATEAU:
        reg = Register.from_coordinates([tuple(map(float, p)) for p in rect_lattice(a)])
        r = results_R3_probe["lattice"] if a == ASPECT_R3 else tune_and_run(reg, INST["R3"], OMEGA_GRID, DELTA_SCAN_B, f"R3_lattice_{a:.3f}")
        lattice_plateau[a] = r
        print(f"R3 lattice a_y/a_x = {a:.3f}: Omega {r['omega']:.4f}, Delta_f {r['delta_f']}, P(4) {r['exact']['P4']:.3f}, "
              f"P(opt) {r['exact']['P_opt']:.4f}, E[S|4] {r['exact']['exp_norm_given_4']:+.3f}")

# %% [markdown]
# **Why the R3 probes score below random: dynamics or fit?** From the exact final states already computed (no new
# emulation), take the 10 most probable 4-turbine layouts for each R3 program and look up their rank under that
# program's own energy $E_{\rm prog}$ (0 = programmed ground state, of 1,820). Programmed-*high*-energy layouts carrying
# the weight means the evolution was not adiabatic (diabatic dynamics); programmed-*low*-energy layouts that are truly
# bad means the encoding itself misranks them (fit error).

# %%
def top_shot_diagnosis(probs, pos, d, k=10):
    E_prog = programmed_energy(pos)
    prog_rank = np.empty(len(M4_IDX), int)
    prog_rank[np.argsort(E_prog, kind="stable")] = np.arange(len(M4_IDX))
    top = np.argsort(probs[M4_IDX])[::-1][:k]
    return dict(prog_rank=prog_rank[top].tolist(), S=norm_score(d["power_all"][M4_IDX[top]], d).round(3).tolist(),
                prob=probs[M4_IDX[top]].round(4).tolist())


diagnosis = {}
if MODE == "full":
    for label, probs, pos in (("A", PROBS_A, GRID_POS), ("B", results_B["R3"]["probs"], embedding["R3"]["pos"]),
                              ("B-lattice", results_R3_probe["lattice"]["probs"], rect_lattice(ASPECT_R3)),
                              ("B-rankfit", results_R3_probe["rankfit"]["probs"], POS_RANKFIT_R3)):
        g = diagnosis[label] = top_shot_diagnosis(probs, pos, INST["R3"])
        print(f"R3 {label:9s} top-10 4-turbine shots: programmed rank {g['prog_rank']}")
        print(f"{'':13s} true S {g['S']}")

# %% [markdown]
# **Diagnosis (full run).**
# - **A and B:** the most probable shots are the programmed ground state and its nearest neighbours in energy (ranks
#   0–10 for A, 0–30 for B), so the dynamics roughly follow each program's own low-energy layouts. The loss on R3 is in
#   the encoding: those layouts score $S$ = +0.65 to +0.93 but are not optimal.
# - **Anisotropic lattice ($a_y/a_x$ = 0.25): mainly fit error.** Its two most probable shots *are* its programmed ground and first excited
#   layouts, but they are truly mediocre ($S$ = +0.40); further weight sits on programmed ranks 16–21 and 134–135 with
#   $S$ down to −0.66, so some diabatic spreading adds to it.
# - **Ranking-aware fit: diabatic dynamics.** Its most probable shots sit at programmed ranks 34–309, far above its
#   ground state, so the better ranking it was fitted for is never reached in the finite sweep.

# %% [markdown]
# ### 3.9 Encoding A vs B (exact final states; S = normalised score)

# %%
if MODE == "full":
    rows = [(n, "A", results_A[n], ranking[(n, "A")]) for n in ("R1", "R2", "R3")]
    rows += [(n, "B", results_B[n], ranking[(n, "B")]) for n in ("R1", "R2", "R3")]
    rows += [("R3", f"lattice {a:.2f}", lattice_plateau[a], aspect_scan[a]) for a in PLATEAU]
    rows += [("R3", "B-rankfit*", results_R3_probe["rankfit"], rank_fits[W_REST][0])]
    rows.sort(key=lambda r: r[0])
    print("rung encoding    P(4)   P(opt)  P(opt|4)  E[S|4]  mode-of-4 S  36x500 sel.opt  tau_b(low5%)  opt rank")
    for n, enc, res, g in rows:
        e, rs = res["exact"], res["resampled"]
        print(f" {n}  {enc:13s}{e['P4']:.3f}  {e['P_opt']:.4f}  {e['P_opt_given_4']:.4f}  {e['exp_norm_given_4']:+.3f}"
              f"     {e['mode4_norm']:+.3f}       {round(rs['frac_selected_optimal'] * 36):2d}/36       {g['tau_tail']:+.3f}"
              f"     {g['opt_rank']:4d}")
    print("* non-scalable upper-bound probe (enumerates all C(16, 4) layouts)")

# %% [markdown]
# **Reading the table.**
# - **R1:** with $\Omega \times \Delta_f$ tuned by P(4), Encoding B matches A (P(optimal) 0.317 vs 0.320). Both
#   programmed Hamiltonians have the true optimum as ground state with the same gap, and the fitted layout stays
#   close to the grid.
# - **R2:** both share the true optimum as ground state, so B's lower P(optimal) (0.34 vs 0.70) is a dynamics effect:
#   its gap is about 5× smaller and its best P(4) is 0.48.
# - **R3:** no encoding reaches the optimum (P(optimal) ≤ 0.0005). The two probes improve the tail ranking metric
#   ($\tau_b$ 0.50 for A → 0.56–0.57 across the lattice plateau and 0.66 for the ranking fit) and move the optimum's
#   rank from 355 to 176–309 and 40, but the solutions get *worse*: the expected normalised score of a 4-turbine shot
#   drops from +0.80 (A) to between −0.09 and −0.02 across the four plateau lattices ($a_y/a_x$ = 0.25–0.42) and to
#   −0.18 for the ranking fit, i.e. to about random placement or below. The diagnosis above attributes this mainly to fit error for the
#   lattice and to diabatic dynamics for the ranking fit.

# %% [markdown]
# ## 4. Results
# ### 4.1 Results record
# The full run (`MODE = "full"`, remote GPU, about 30 min) produces the record below, pasted here as a JSON literal so
# the notebook reads no external files. Demo mode recomputes everything that does not need Encoding B or the R3 probes
# (instances, SA, Encoding A on all rungs from the R0 state, embeddings and ranking metrics) and checks it against the
# record; Encoding B, the R3 probes and the Ω check are read from the record.

# %%
FULL_RESULTS = json.loads(r"""{
 "meta": {
  "mode": "full",
  "seed": 20261018,
  "gpu": true,
  "versions": {
   "qoolqit": "1.4.0",
   "pulser-core": "1.9.1",
   "pulser-simulation": "1.9.1",
   "emu-sv": "2.9.1",
   "emu-base": "2.9.1",
   "torch": "2.14.1",
   "numpy": "2.4.6",
   "scipy": "1.17.1",
   "matplotlib": "3.11.2"
  }
 },
 "instances": {
  "R0": {
   "optimum": 2304.0,
   "n_optima": 79,
   "P_random": 2187.702911
  },
  "R1": {
   "optimum": 2229.381159,
   "n_optima": 1,
   "P_random": 2019.340925
  },
  "R2": {
   "optimum": 2304.0,
   "n_optima": 79,
   "P_random": 2160.132048
  },
  "R3": {
   "optimum": 2242.7081,
   "n_optima": 2,
   "P_random": 2064.97052
  }
 },
 "sa": {
  "R0|SA-QUBO": {
   "mean_pct": 99.809226,
   "median_pct": 100.0,
   "min_pct": 98.473806,
   "mean_norm": 0.962205,
   "frac_optimal": 0.777778,
   "frac_m_turbines": 1.0
  },
  "R0|SA-swap": {
   "mean_pct": 100.0,
   "median_pct": 100.0,
   "min_pct": 100.0,
   "mean_norm": 1.0,
   "frac_optimal": 1.0,
   "frac_m_turbines": 1.0
  },
  "R1|SA-QUBO": {
   "mean_pct": 96.772561,
   "median_pct": 96.503341,
   "min_pct": 93.262602,
   "mean_norm": 0.657438,
   "frac_optimal": 0.111111,
   "frac_m_turbines": 1.0
  },
  "R1|SA-swap": {
   "mean_pct": 100.0,
   "median_pct": 100.0,
   "min_pct": 100.0,
   "mean_norm": 1.0,
   "frac_optimal": 1.0,
   "frac_m_turbines": 1.0
  },
  "R2|SA-QUBO": {
   "mean_pct": 99.818299,
   "median_pct": 100.0,
   "min_pct": 99.065537,
   "mean_norm": 0.970901,
   "frac_optimal": 0.805556,
   "frac_m_turbines": 1.0
  },
  "R2|SA-swap": {
   "mean_pct": 99.922128,
   "median_pct": 100.0,
   "min_pct": 99.065537,
   "mean_norm": 0.987529,
   "frac_optimal": 0.916667,
   "frac_m_turbines": 1.0
  },
  "R3|SA-QUBO": {
   "mean_pct": 98.294703,
   "median_pct": 98.564511,
   "min_pct": 95.723125,
   "mean_norm": 0.784824,
   "frac_optimal": 0.027778,
   "frac_m_turbines": 1.0
  },
  "R3|SA-swap": {
   "mean_pct": 100.0,
   "median_pct": 100.0,
   "min_pct": 100.0,
   "mean_norm": 1.0,
   "frac_optimal": 1.0,
   "frac_m_turbines": 1.0
  }
 },
 "A": {
  "R0": {
   "exact": {
    "P3": 0.099173,
    "P4": 0.733293,
    "P5": 0.16309,
    "P_opt": 0.695962,
    "P_opt_given_4": 0.949091,
    "exp_pct_given_4": 99.960939,
    "exp_norm_given_4": 0.992261,
    "mode4_pct": 100.0,
    "mode4_norm": 1.0
   },
   "resampled": {
    "mean_pct": 100.0,
    "median_pct": 100.0,
    "min_pct": 100.0,
    "mean_norm": 1.0,
    "frac_selected_optimal": 1.0,
    "mean_P_opt_shot": 0.700167,
    "mean_valid_rate": 0.737389,
    "n_no_valid": 0
   }
  },
  "R1": {
   "exact": {
    "P3": 0.099173,
    "P4": 0.733293,
    "P5": 0.16309,
    "P_opt": 0.319738,
    "P_opt_given_4": 0.43603,
    "exp_pct_given_4": 98.98623,
    "exp_norm_given_4": 0.892398,
    "mode4_pct": 100.0,
    "mode4_norm": 1.0
   },
   "resampled": {
    "mean_pct": 100.0,
    "median_pct": 100.0,
    "min_pct": 100.0,
    "mean_norm": 1.0,
    "frac_selected_optimal": 1.0,
    "mean_P_opt_shot": 0.313611,
    "mean_valid_rate": 0.7285,
    "n_no_valid": 0
   }
  },
  "R2": {
   "exact": {
    "P3": 0.099173,
    "P4": 0.733293,
    "P5": 0.16309,
    "P_opt": 0.695962,
    "P_opt_given_4": 0.949091,
    "exp_pct_given_4": 99.859388,
    "exp_norm_given_4": 0.977481,
    "mode4_pct": 100.0,
    "mode4_norm": 1.0
   },
   "resampled": {
    "mean_pct": 100.0,
    "median_pct": 100.0,
    "min_pct": 100.0,
    "mean_norm": 1.0,
    "frac_selected_optimal": 1.0,
    "mean_P_opt_shot": 0.695056,
    "mean_valid_rate": 0.732722,
    "n_no_valid": 0
   }
  },
  "R3": {
   "exact": {
    "P3": 0.099173,
    "P4": 0.733293,
    "P5": 0.16309,
    "P_opt": 0.0,
    "P_opt_given_4": 0.0,
    "exp_pct_given_4": 98.390267,
    "exp_norm_given_4": 0.796883,
    "mode4_pct": 99.405766,
    "mode4_norm": 0.925019
   },
   "resampled": {
    "mean_pct": 99.405766,
    "median_pct": 99.405766,
    "min_pct": 99.405766,
    "mean_norm": 0.925019,
    "frac_selected_optimal": 0.0,
    "mean_P_opt_shot": 0.0,
    "mean_valid_rate": 0.729056,
    "n_no_valid": 0
   }
  }
 },
 "A_schedule": {
  "omega": 0.041429,
  "delta_0": -0.1,
  "delta_f": 0.1,
  "T": 332.43764
 },
 "delta_scan_R0": {
  "0.06": {
   "P3": 0.399307,
   "P4": 0.464661,
   "P5": 0.024207,
   "P_opt": 0.436089,
   "P_opt_given_4": 0.93851,
   "exp_pct_given_4": 99.952811,
   "exp_norm_given_4": 0.990651,
   "mode4_pct": 100.0,
   "mode4_norm": 1.0
  },
  "0.07": {
   "P3": 0.320763,
   "P4": 0.583009,
   "P5": 0.038566,
   "P_opt": 0.550158,
   "P_opt_given_4": 0.943651,
   "exp_pct_given_4": 99.95673,
   "exp_norm_given_4": 0.991428,
   "mode4_pct": 100.0,
   "mode4_norm": 1.0
  },
  "0.08": {
   "P3": 0.232204,
   "P4": 0.675501,
   "P5": 0.064963,
   "P_opt": 0.638312,
   "P_opt_given_4": 0.944945,
   "exp_pct_given_4": 99.957734,
   "exp_norm_given_4": 0.991627,
   "mode4_pct": 100.0,
   "mode4_norm": 1.0
  },
  "0.09": {
   "P3": 0.154359,
   "P4": 0.725167,
   "P5": 0.108768,
   "P_opt": 0.685646,
   "P_opt_given_4": 0.945501,
   "exp_pct_given_4": 99.958182,
   "exp_norm_given_4": 0.991715,
   "mode4_pct": 100.0,
   "mode4_norm": 1.0
  },
  "0.1": {
   "P3": 0.099173,
   "P4": 0.733293,
   "P5": 0.16309,
   "P_opt": 0.695962,
   "P_opt_given_4": 0.949091,
   "exp_pct_given_4": 99.960939,
   "exp_norm_given_4": 0.992261,
   "mode4_pct": 100.0,
   "mode4_norm": 1.0
  },
  "0.11": {
   "P3": 0.067208,
   "P4": 0.706725,
   "P5": 0.224191,
   "P_opt": 0.672444,
   "P_opt_given_4": 0.951494,
   "exp_pct_given_4": 99.962707,
   "exp_norm_given_4": 0.992612,
   "mode4_pct": 100.0,
   "mode4_norm": 1.0
  }
 },
 "embedding": {
  "R0": {
   "eps": 0.073497,
   "spearman": 0.845015,
   "eps_ie": 0.911254
  },
  "R1": {
   "eps": 0.552347,
   "spearman": 0.988144,
   "eps_ie": 0.88075
  },
  "R2": {
   "eps": 0.470304,
   "spearman": 0.82204,
   "eps_ie": 0.77167
  },
  "R3": {
   "eps": 0.46885,
   "spearman": 0.689072,
   "eps_ie": 0.810915
  }
 },
 "ranking": {
  "R0|A": {
   "eps": 0.073541,
   "rho_all": 0.991138,
   "tau_tail": 0.491581,
   "tail_levels": 2,
   "opt_rank": 0,
   "gap": 0.014165,
   "ground_norm": 1.0
  },
  "R0|B": {
   "eps": 0.073497,
   "rho_all": 0.991057,
   "tau_tail": 0.481146,
   "tail_levels": 2,
   "opt_rank": 0,
   "gap": 0.014093,
   "ground_norm": 1.0
  },
  "R0|IE": {
   "eps": 0.911254,
   "rho_all": 0.198625,
   "tau_tail": 0.166473,
   "tail_levels": 2,
   "opt_rank": 8,
   "gap": 6e-06,
   "ground_norm": 0.280051
  },
  "R1|A": {
   "eps": 0.552523,
   "rho_all": 0.908652,
   "tau_tail": 0.676336,
   "tail_levels": 15,
   "opt_rank": 0,
   "gap": 0.014165,
   "ground_norm": 1.0
  },
  "R1|B": {
   "eps": 0.552347,
   "rho_all": 0.900567,
   "tau_tail": 0.615475,
   "tail_levels": 15,
   "opt_rank": 0,
   "gap": 0.013717,
   "ground_norm": 1.0
  },
  "R1|IE": {
   "eps": 0.88075,
   "rho_all": 0.297761,
   "tau_tail": 0.171526,
   "tail_levels": 15,
   "opt_rank": 177,
   "gap": 8.3e-05,
   "ground_norm": -0.402215
  },
  "R2|A": {
   "eps": 0.663537,
   "rho_all": 0.719304,
   "tau_tail": 0.491393,
   "tail_levels": 2,
   "opt_rank": 0,
   "gap": 0.014165,
   "ground_norm": 1.0
  },
  "R2|B": {
   "eps": 0.470304,
   "rho_all": 0.826692,
   "tau_tail": 0.481146,
   "tail_levels": 2,
   "opt_rank": 0,
   "gap": 0.002732,
   "ground_norm": 1.0
  },
  "R2|IE": {
   "eps": 0.77167,
   "rho_all": 0.458999,
   "tau_tail": -0.00609,
   "tail_levels": 2,
   "opt_rank": 7,
   "gap": 3.8e-05,
   "ground_norm": 0.120105
  },
  "R3|A": {
   "eps": 0.700706,
   "rho_all": 0.610369,
   "tau_tail": 0.495855,
   "tail_levels": 18,
   "opt_rank": 355,
   "gap": 0.014165,
   "ground_norm": 0.925019
  },
  "R3|B": {
   "eps": 0.46885,
   "rho_all": 0.819674,
   "tau_tail": 0.384931,
   "tail_levels": 18,
   "opt_rank": 355,
   "gap": 0.001854,
   "ground_norm": 0.925019
  },
  "R3|IE": {
   "eps": 0.810915,
   "rho_all": 0.410135,
   "tau_tail": -0.047299,
   "tail_levels": 18,
   "opt_rank": 297,
   "gap": 3.9e-05,
   "ground_norm": 0.824326
  }
 },
 "R3_probe_selection": {
  "aspect": 0.25,
  "aspect_metrics": {
   "rho_all": 0.005853,
   "tau_tail": 0.569326,
   "tail_levels": 18,
   "opt_rank": 176,
   "gap": 0.0,
   "ground_norm": 0.404321
  },
  "w_rest": 0.0,
  "rankfit_metrics": {
   "rho_all": -0.125499,
   "tau_tail": 0.657654,
   "tail_levels": 18,
   "opt_rank": 40,
   "gap": 0.000193,
   "ground_norm": -0.741594
  }
 },
 "omega_check_R0": [
  [
   0.041429,
   0.09,
   0.154359,
   0.725167,
   0.108768,
   0.685646
  ],
  [
   0.041429,
   0.1,
   0.099173,
   0.733293,
   0.16309,
   0.695962
  ],
  [
   0.041429,
   0.11,
   0.067208,
   0.706725,
   0.224191,
   0.672444
  ],
  [
   0.082858,
   0.09,
   0.23948,
   0.635001,
   0.090896,
   0.601979
  ],
  [
   0.082858,
   0.1,
   0.16494,
   0.655709,
   0.162814,
   0.612362
  ],
  [
   0.082858,
   0.11,
   0.106552,
   0.636747,
   0.249892,
   0.592405
  ],
  [
   0.226804,
   0.09,
   0.277445,
   0.502196,
   0.153124,
   0.403042
  ],
  [
   0.226804,
   0.1,
   0.238214,
   0.51699,
   0.2082,
   0.379896
  ],
  [
   0.226804,
   0.11,
   0.249437,
   0.482392,
   0.225984,
   0.326175
  ]
 ],
 "B": {
  "R1": {
   "omega": 0.041429,
   "delta_f": 0.1,
   "exact": {
    "P3": 0.095287,
    "P4": 0.715929,
    "P5": 0.184548,
    "P_opt": 0.317424,
    "P_opt_given_4": 0.443374,
    "exp_pct_given_4": 98.974401,
    "exp_norm_given_4": 0.891142,
    "mode4_pct": 100.0,
    "mode4_norm": 1.0
   },
   "resampled": {
    "mean_pct": 100.0,
    "median_pct": 100.0,
    "min_pct": 100.0,
    "mean_norm": 1.0,
    "frac_selected_optimal": 1.0,
    "mean_P_opt_shot": 0.318222,
    "mean_valid_rate": 0.718722,
    "n_no_valid": 0
   }
  },
  "R2": {
   "omega": 0.082858,
   "delta_f": 0.1,
   "exact": {
    "P3": 0.134564,
    "P4": 0.479697,
    "P5": 0.370689,
    "P_opt": 0.337222,
    "P_opt_given_4": 0.702991,
    "exp_pct_given_4": 99.044213,
    "exp_norm_given_4": 0.846934,
    "mode4_pct": 100.0,
    "mode4_norm": 1.0
   },
   "resampled": {
    "mean_pct": 100.0,
    "median_pct": 100.0,
    "min_pct": 100.0,
    "mean_norm": 1.0,
    "frac_selected_optimal": 1.0,
    "mean_P_opt_shot": 0.334056,
    "mean_valid_rate": 0.474944,
    "n_no_valid": 0
   }
  },
  "R3": {
   "omega": 0.082858,
   "delta_f": 0.06,
   "exact": {
    "P3": 0.289653,
    "P4": 0.40115,
    "P5": 0.208683,
    "P_opt": 0.000481,
    "P_opt_given_4": 0.001199,
    "exp_pct_given_4": 97.024711,
    "exp_norm_given_4": 0.624576,
    "mode4_pct": 97.220072,
    "mode4_norm": 0.649226
   },
   "resampled": {
    "mean_pct": 98.093443,
    "median_pct": 98.564511,
    "min_pct": 97.220072,
    "mean_norm": 0.759429,
    "frac_selected_optimal": 0.0,
    "mean_P_opt_shot": 0.000556,
    "mean_valid_rate": 0.401056,
    "n_no_valid": 0
   }
  }
 },
 "R3_probes": {
  "lattice": {
   "omega": 0.041429,
   "delta_f": 0.02,
   "exact": {
    "P3": 0.138737,
    "P4": 0.263005,
    "P5": 0.548955,
    "P_opt": 0.000425,
    "P_opt_given_4": 0.001615,
    "exp_pct_given_4": 91.354433,
    "exp_norm_given_4": -0.090905,
    "mode4_pct": 95.279162,
    "mode4_norm": 0.404321
   },
   "resampled": {
    "mean_pct": 88.099331,
    "median_pct": 89.658242,
    "min_pct": 70.719224,
    "mean_norm": -0.501637,
    "frac_selected_optimal": 0.0,
    "mean_P_opt_shot": 0.000611,
    "mean_valid_rate": 0.263167,
    "n_no_valid": 0
   }
  },
  "rankfit": {
   "omega": 0.226804,
   "delta_f": 0.06,
   "exact": {
    "P3": 0.246739,
    "P4": 0.355643,
    "P5": 0.311827,
    "P_opt": 0.00027,
    "P_opt_given_4": 0.00076,
    "exp_pct_given_4": 90.625967,
    "exp_norm_given_4": -0.182824,
    "mode4_pct": 86.197641,
    "mode4_norm": -0.741594
   },
   "resampled": {
    "mean_pct": 90.148795,
    "median_pct": 89.141473,
    "min_pct": 82.371041,
    "mean_norm": -0.243033,
    "frac_selected_optimal": 0.0,
    "mean_P_opt_shot": 0.000278,
    "mean_valid_rate": 0.358778,
    "n_no_valid": 0
   }
  }
 },
 "R3_diagnosis": {
  "A": {
   "prog_rank": [
    0,
    1,
    3,
    2,
    4,
    6,
    7,
    8,
    5,
    10
   ],
   "S": [
    0.925,
    0.819,
    0.819,
    0.819,
    0.819,
    0.667,
    0.667,
    0.667,
    0.667,
    0.692
   ],
   "prob": [
    0.3197,
    0.032,
    0.032,
    0.032,
    0.032,
    0.032,
    0.032,
    0.032,
    0.032,
    0.0127
   ]
  },
  "B": {
   "prog_rank": [
    8,
    7,
    4,
    3,
    1,
    2,
    0,
    29,
    30,
    27
   ],
   "S": [
    0.649,
    0.649,
    0.819,
    0.819,
    0.819,
    0.819,
    0.925,
    0.667,
    0.667,
    0.667
   ],
   "prob": [
    0.0119,
    0.0119,
    0.0109,
    0.0109,
    0.0109,
    0.0109,
    0.0098,
    0.0076,
    0.0076,
    0.0076
   ]
  },
  "B-lattice": {
   "prog_rank": [
    0,
    1,
    17,
    16,
    18,
    20,
    21,
    19,
    134,
    135
   ],
   "S": [
    0.404,
    0.404,
    0.132,
    0.132,
    -0.309,
    -0.309,
    -0.309,
    -0.309,
    -0.657,
    -0.657
   ],
   "prob": [
    0.0016,
    0.0016,
    0.0016,
    0.0016,
    0.0016,
    0.0016,
    0.0016,
    0.0016,
    0.0016,
    0.0016
   ]
  },
  "B-rankfit": {
   "prog_rank": [
    65,
    213,
    214,
    35,
    34,
    38,
    39,
    309,
    135,
    137
   ],
   "S": [
    -0.742,
    -0.582,
    -0.582,
    0.191,
    0.191,
    -0.324,
    -0.324,
    -0.869,
    0.515,
    0.515
   ],
   "prob": [
    0.0066,
    0.005,
    0.005,
    0.0043,
    0.0042,
    0.0041,
    0.0041,
    0.0038,
    0.0033,
    0.0033
   ]
  }
 },
 "R3_lattice_plateau": {
  "0.250": {
   "omega": 0.041429,
   "delta_f": 0.02,
   "exact": {
    "P3": 0.138737,
    "P4": 0.263005,
    "P5": 0.548955,
    "P_opt": 0.000425,
    "P_opt_given_4": 0.001615,
    "exp_pct_given_4": 91.354433,
    "exp_norm_given_4": -0.090905,
    "mode4_pct": 95.279162,
    "mode4_norm": 0.404321
   },
   "resampled": {
    "mean_pct": 88.099331,
    "median_pct": 89.658242,
    "min_pct": 70.719224,
    "mean_norm": -0.501637,
    "frac_selected_optimal": 0.0,
    "mean_P_opt_shot": 0.000611,
    "mean_valid_rate": 0.263167,
    "n_no_valid": 0
   },
   "opt_rank": 176,
   "tau_tail": 0.569326
  },
  "0.297": {
   "omega": 0.041429,
   "delta_f": 0.02,
   "exact": {
    "P3": 0.142679,
    "P4": 0.266109,
    "P5": 0.539622,
    "P_opt": 0.000431,
    "P_opt_given_4": 0.001619,
    "exp_pct_given_4": 91.40685,
    "exp_norm_given_4": -0.084291,
    "mode4_pct": 95.279162,
    "mode4_norm": 0.404321
   },
   "resampled": {
    "mean_pct": 89.688416,
    "median_pct": 89.955375,
    "min_pct": 70.719224,
    "mean_norm": -0.301125,
    "frac_selected_optimal": 0.0,
    "mean_P_opt_shot": 0.0005,
    "mean_valid_rate": 0.264889,
    "n_no_valid": 0
   },
   "opt_rank": 180,
   "tau_tail": 0.569112
  },
  "0.354": {
   "omega": 0.041429,
   "delta_f": 0.02,
   "exact": {
    "P3": 0.151007,
    "P4": 0.272118,
    "P5": 0.52018,
    "P_opt": 0.000443,
    "P_opt_given_4": 0.001629,
    "exp_pct_given_4": 91.554493,
    "exp_norm_given_4": -0.065661,
    "mode4_pct": 95.279162,
    "mode4_norm": 0.404321
   },
   "resampled": {
    "mean_pct": 87.80414,
    "median_pct": 89.579274,
    "min_pct": 70.719224,
    "mean_norm": -0.538884,
    "frac_selected_optimal": 0.0,
    "mean_P_opt_shot": 0.000667,
    "mean_valid_rate": 0.2725,
    "n_no_valid": 0
   },
   "opt_rank": 299,
   "tau_tail": 0.559413
  },
  "0.420": {
   "omega": 0.041429,
   "delta_f": 0.02,
   "exact": {
    "P3": 0.161159,
    "P4": 0.278413,
    "P5": 0.49689,
    "P_opt": 0.000452,
    "P_opt_given_4": 0.001624,
    "exp_pct_given_4": 91.91522,
    "exp_norm_given_4": -0.020145,
    "mode4_pct": 95.279162,
    "mode4_norm": 0.404321
   },
   "resampled": {
    "mean_pct": 91.204349,
    "median_pct": 90.922027,
    "min_pct": 70.719224,
    "mean_norm": -0.109843,
    "frac_selected_optimal": 0.0,
    "mean_P_opt_shot": 0.000167,
    "mean_valid_rate": 0.280222,
    "n_no_valid": 0
   },
   "opt_rank": 309,
   "tau_tail": 0.567834
  }
 }
}""")


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if k not in ("probs", "p4_grid")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def collect_results():
    rec = {
        "meta": {"mode": MODE, "seed": SEED, "gpu": GPU, "versions": {p: _installed(p) for p in PINNED}},
        "instances": {n: dict(optimum=d["optimum"], n_optima=len(d["opt_idx"]), P_random=d["P_random"]) for n, d in INST.items()},
        "sa": {f"{n}|{m}": dict(v) for (n, m), v in sa_results.items()},
        "A": {n: dict(exact=v["exact"], resampled=v["resampled"]) for n, v in results_A.items()},
        "A_schedule": dict(omega=OMEGA_A, delta_0=DELTA_0_A, delta_f=DELTA_F_A, T=T_MAX),
        "delta_scan_R0": {str(df): {k: v for k, v in m.items() if k != "probs"} for df, m in scan.items()},
        "embedding": {n: dict(eps=v["ours"]["eps"], spearman=v["ours"]["spearman"], eps_ie=v["interaction_embedder"]["eps"])
                      for n, v in embedding.items()},
        "ranking": {f"{n}|{e}": v for (n, e), v in ranking.items()},
        "R3_probe_selection": dict(aspect=ASPECT_R3, aspect_metrics=aspect_scan[ASPECT_R3], w_rest=W_REST,
                                   rankfit_metrics=rank_fits[W_REST][0]),
    }
    for g in rec["sa"].values():
        g.pop("wall_s", None)
    for m in rec["delta_scan_R0"].values():
        m.pop("wall_s", None)
    if MODE == "full":
        rec["omega_check_R0"] = [list(row) for row in omega_check]
        rec["B"] = {n: dict(omega=v["omega"], delta_f=v["delta_f"], exact=v["exact"], resampled=v["resampled"],
                            p4_grid=[[om, df, p4] for (om, df), p4 in v["p4_grid"].items()]) for n, v in results_B.items()}
        rec["R3_probes"] = {k: dict(omega=v["omega"], delta_f=v["delta_f"], exact=v["exact"], resampled=v["resampled"])
                            for k, v in results_R3_probe.items()}
        rec["R3_diagnosis"] = diagnosis
        rec["R3_lattice_plateau"] = {f"{a:.3f}": dict(omega=v["omega"], delta_f=v["delta_f"], exact=v["exact"],
                                                       resampled=v["resampled"], opt_rank=aspect_scan[a]["opt_rank"],
                                                       tau_tail=aspect_scan[a]["tau_tail"])
                                     for a, v in lattice_plateau.items()}
    return json.loads(json.dumps(_clean(rec)))


def compare(new, ref, path="", rtol=1e-4, atol=1e-4):
    """Numeric leaves present in both records must agree; returns (n_checked, mismatches)."""
    if isinstance(new, dict) and isinstance(ref, dict):
        out = [compare(new[k], ref[k], f"{path}/{k}", rtol, atol) for k in new if k in ref and k != "meta"]
        return sum(o[0] for o in out), [m for o in out for m in o[1]]
    if isinstance(new, list) and isinstance(ref, list) and len(new) == len(ref):
        out = [compare(a, b, f"{path}[{i}]", rtol, atol) for i, (a, b) in enumerate(zip(new, ref))]
        return sum(o[0] for o in out), [m for o in out for m in o[1]]
    if isinstance(new, (int, float)) and isinstance(ref, (int, float)) and not isinstance(new, bool):
        return 1, ([] if np.isclose(new, ref, rtol=rtol, atol=atol) else [(path, new, ref)])
    return 0, []


RECORD = collect_results()
if MODE == "full":
    os.makedirs("runs", exist_ok=True)
    with open("runs/results_full.json", "w") as fh:
        json.dump(RECORD, fh, indent=1)
    print("full-run record written to runs/results_full.json (paste it into FULL_RESULTS above)")
if FULL_RESULTS:
    n_checked, mismatches = compare(RECORD, FULL_RESULTS)
    for m in mismatches[:20]:
        print("MISMATCH", m)
    assert not mismatches, f"{len(mismatches)} recomputed values differ from the full-run record"
    print(f"{MODE} mode: {n_checked} recomputed values match the full-run record")
    RES = {**FULL_RESULTS, **{k: v for k, v in RECORD.items() if k != "meta"}}
else:
    assert MODE == "full", "demo mode needs the pasted full-run record"
    RES = RECORD

# %% [markdown]
# ### 4.2 Comparison tables
# Every cell gives the raw % of optimum and the normalised score $S$ (0 = random 4-turbine layout, 1 = optimal), and
# every column states its statistic, because they differ:
# - **Expected $S$** (emulator columns of the first table and the plot): the exact expected power of one post-selected
#   4-turbine shot, from the exact final state. It has no published counterpart and is used only to compare our own
#   emulations and classical runs across rungs.
# - **Selection rule** (second table): the paper's protocol, one layout per run (the most probable 4-turbine bitstring),
#   averaged over 36 runs. This is the statistic the published VQE means use, so it is the like-for-like comparison.
#
# All emulator results are noiseless state-vector emulations, not hardware. Published numbers are reported values,
# not reruns, and sit in their own columns.
#
# **D-Wave (prose only).** Nigro et al. (Milan; *Adv. Quantum Technol.* 8, e00358, 2025, doi:10.1002/qute.202500358)
# that their D-Wave quantum annealer "provides solutions one order of magnitude faster than Gurobi with at most 3% lower
# power output". We do not tabulate this: a 3% shortfall cannot be interpreted without that instance's random baseline, and
# on R0 a random 4-turbine layout is already only about 5% below optimum.

# %%
try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown, display = None, print


def cell(pct, S):
    return f"{pct:.2f}% (S {S:+.2f})"


def p_txt(p):
    return "<0.001" if 0 < p < 1e-3 else f"{p:.3f}"


lines = ["| Rung | Exact optimum (kW) | Random layout (exact mean) | SA-swap, classical: mean of 36 runs | "
         "Rydberg A, emulator: expected S of a 4-turbine shot | Rydberg B, emulator: expected S of a 4-turbine shot |",
         "|---|---|---|---|---|---|"]
for n in RUNGS:
    inst = RES["instances"][n]
    sa = RES["sa"][f"{n}|SA-swap"]
    a = RES["A"][n]["exact"]
    a_txt = f"{cell(a['exp_pct_given_4'], a['exp_norm_given_4'])}<br>P(opt) {p_txt(a['P_opt'])}"
    if n in RES.get("B", {}):
        b = RES["B"][n]["exact"]
        b_txt = f"{cell(b['exp_pct_given_4'], b['exp_norm_given_4'])}<br>P(opt) {p_txt(b['P_opt'])}"
    else:
        b_txt = "not run (embedding recovers the grid, ε = %.2f)" % RES["embedding"][n]["eps"]
    rand_pct = 100 * inst["P_random"] / inst["optimum"]
    lines.append(f"| {n} | {inst['optimum']:.2f} ({inst['n_optima']} optimal) | {cell(rand_pct, 0.0)} | "
                 f"{cell(sa['mean_pct'], sa['mean_norm'])} | {a_txt} | {b_txt} |")
sa_q = "; ".join(f"{n} {cell(RES['sa'][f'{n}|SA-QUBO']['mean_pct'], RES['sa'][f'{n}|SA-QUBO']['mean_norm'])}" for n in RUNGS)
plat = RES["R3_lattice_plateau"]
plat_S = [v["exact"]["exp_norm_given_4"] for v in plat.values()]
plat_pct = [v["exact"]["exp_pct_given_4"] for v in plat.values()]
plat_Popt = max(v["exact"]["P_opt"] for v in plat.values())
lines += ["", f"SA-QUBO (bit-flip SA on the full QUBO, λ₁ = 1.05 × 576 chosen by scan; the paper does not state λ₁): {sa_q}.",
          "", f"R3 anisotropic lattice (emulator, expected S), over its τ_b plateau a_y/a_x = {min(map(float, plat)):.2f}–"
          f"{max(map(float, plat)):.2f} ({len(plat)} lattices): {min(plat_pct):.2f}–{max(plat_pct):.2f}% "
          f"(S {min(plat_S):+.2f} to {max(plat_S):+.2f}), P(opt) ≤ {plat_Popt:.4f}."]
display(Markdown("\n".join(lines)) if Markdown else "\n".join(lines))

# Like-for-like R0 comparison under the paper's selection rule. Published and our numbers stay in separate columns.
vqe = [v for k, v in PUBLISHED_R0.items() if k.startswith("VQE")]
S_of = lambda pct: float(norm_score(pct / 100 * INST["R0"]["optimum"], INST["R0"]))
sa_pub = PUBLISHED_R0["SA, pyqubo/neal (median, read off Fig. 9)"]
a_sel = RES["A"]["R0"]["resampled"]
sa_ours = RES["sa"]["R0|SA-swap"]
pub = ["**R0, like for like** (one layout per run, 36 runs). Published: Hancock et al., arXiv:2312.13123 (reported "
       "numbers, not reruns). This work: this notebook.", "",
       "| Method (platform) | Statistic | Published raw % | Published S | This work raw % | This work S |",
       "|---|---|---|---|---|---|",
       f"| VQE, 8 variants (Qiskit simulator) | selection rule, mean of 36 runs (Table 1) | {min(vqe):.1f}–{max(vqe):.1f}% | "
       f"{S_of(min(vqe)):+.2f} to {S_of(max(vqe)):+.2f} | | |",
       f"| SA, pyqubo/neal (classical) | median of 36 runs, read from Fig. 9 (±{SA_READ_ERR_PCT:.0f}% reading error) | "
       f"≈{sa_pub:.1f} ± {SA_READ_ERR_PCT:.0f}% | ≈{SA_PUB_S:+.1f} ± {SA_PUB_S_ERR:.1f} | | |",
       "| Gurobi (classical) | all 36 runs (Fig. 9) | 100% | +1.00 | | |",
       f"| Rydberg A (noiseless emulator) | selection rule, mean of 36 runs of 500 shots; "
       f"{round(a_sel['frac_selected_optimal'] * 36)}/36 optimal | | | {a_sel['mean_pct']:.2f}% | {a_sel['mean_norm']:+.2f} |",
       f"| SA-swap (classical) | mean of 36 runs; {round(sa_ours['frac_optimal'] * 36)}/36 optimal | | | "
       f"{sa_ours['mean_pct']:.2f}% | {sa_ours['mean_norm']:+.2f} |",
       "", f"Rydberg A's expected S of a single post-selected shot on R0 is {RES['A']['R0']['exact']['exp_norm_given_4']:+.2f} "
       "(first table); it is an emulator-only statistic and is not compared with the published numbers.",
       ]
display(Markdown("\n".join(pub)) if Markdown else "\n".join(pub))
assert a_sel["frac_selected_optimal"] == 1.0 and round(a_sel["mean_norm"], 2) == 1.0     # like-for-like: 36/36, S = 1.00
assert len(plat) == len(PLATEAU) and max(plat_S) < RES["A"]["R3"]["exact"]["exp_norm_given_4"]   # no lattice beats A on R3
assert not any("D-Wave" in l for l in lines + pub)                                       # D-Wave is cited in prose only
assert round(100 - 100 * RES["instances"]["R0"]["P_random"] / RES["instances"]["R0"]["optimum"]) == 5   # "about 5%" below

# %% [markdown]
# ### 4.3 Normalised score against rung

# %%
INK, INK2, MUTED, GRIDC, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"Rydberg A (emulator)": ("#2a78d6", "o"), "Rydberg B (emulator)": ("#eb6834", "s"), "SA-swap (classical)": ("#1baf7a", "^")}
rungs = list(RUNGS)
xs = np.arange(len(rungs))
vals = {
    "Rydberg A (emulator)": [RES["A"][n]["exact"]["exp_norm_given_4"] for n in rungs],
    "Rydberg B (emulator)": [RES["B"][n]["exact"]["exp_norm_given_4"] if n in RES.get("B", {}) else np.nan for n in rungs],
    "SA-swap (classical)": [RES["sa"][f"{n}|SA-swap"]["mean_norm"] for n in rungs],
}
fig, ax = plt.subplots(figsize=(8, 5.4), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.axhline(0, color=BASE, lw=1.5, zorder=1)
ax.text(0.55, 0.03, "random layout (S = 0)", color=MUTED, fontsize=9, va="bottom")
STAT = {"Rydberg A (emulator)": "expected S of one shot", "Rydberg B (emulator)": "expected S of one shot",
        "SA-swap (classical)": "mean of 36 runs"}
# R0, like for like (one selected layout per run, 36 runs): published VQE means and SA median next to our selection rule.
X_LL = -0.36
ax.axvline(-0.17, color=GRIDC, lw=1, ls=(0, (2, 3)), zorder=1)
lo, hi = S_of(min(vqe)), S_of(max(vqe))
ax.add_patch(plt.Rectangle((X_LL - 0.07, lo), 0.14, hi - lo, facecolor="none", edgecolor=INK2, hatch="////", lw=1, zorder=2))
ax.text(X_LL + 0.12, (lo + hi) / 2, "published VQE means\n(simulator)", color=INK2, fontsize=8, va="center")
ax.errorbar([X_LL], [SA_PUB_S], yerr=[SA_PUB_S_ERR], fmt="s", ms=7, mfc="#fcfcfb", mec=INK2, ecolor=INK2, mew=1.5,
            capsize=3, zorder=3)
ax.text(X_LL + 0.12, SA_PUB_S, "published SA median\n(read from figure, ±0.2)", color=INK2, fontsize=8, va="center")
ax.plot([X_LL], [RES["A"]["R0"]["resampled"]["mean_norm"]], marker="D", ms=8, mfc="#fcfcfb", mec=SERIES["Rydberg A (emulator)"][0],
        mew=2, ls="none", zorder=4)
ax.text(X_LL, 0.90, "A, selection\nrule (36/36)", color=INK2, fontsize=8, va="top", ha="center")
ax.text(X_LL, SA_PUB_S - SA_PUB_S_ERR - 0.06, "R0, one layout\nper run, 36 runs", color=MUTED,
        fontsize=8, va="top", ha="center")
for label, ys in vals.items():
    color, marker = SERIES[label]
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec="#fcfcfb", mew=2, zorder=3, label=f"{label}: {STAT[label]}")
    last = max(i for i, v in enumerate(ys) if np.isfinite(v))
    ax.text(xs[last] + 0.08, ys[last], label.split(" (")[0], color=INK2, fontsize=9, va="center")
ax.set_xticks(xs, [f"{n}\nx={RUNGS[n][0]}, r={RUNGS[n][1]:g}\n{RUNGS[n][2]}" for n in rungs], color=INK2, fontsize=9)
ax.set_xlim(-0.75, len(rungs) - 1 + 1.1)
ax.set_ylim(SA_PUB_S - SA_PUB_S_ERR - 0.35, 1.08)
ax.set_ylabel("normalised score S\n(0 = random layout, 1 = optimal)", color=INK2, fontsize=9)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color=GRIDC, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(BASE)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
ax.set_title("Solution quality as wake physics departs from the Rydberg Hamiltonian", color=INK, fontsize=11, loc="left")
plt.tight_layout()
plt.show()

# Phase 4 exit: every rung has exact, random, SA-swap and Encoding A entries; B on R1-R3.
assert all(n in RES["A"] and f"{n}|SA-swap" in RES["sa"] for n in RUNGS)
assert all(n in RES["B"] for n in ("R1", "R2", "R3"))

# %% [markdown]
# ### 4.4 Conclusions
# - **R0 (published instance) is Rydberg-native.** Its 79 optima are the 79 independent 4-sets of the 4×4 king graph,
#   so a blockade register solves it directly: exact P(optimal) 0.70 and the paper's selection rule picks an optimal
#   layout in 36/36 resampled runs. This is a structural observation about the instance, not a quantum advantage.
# - **R1 (long wake, uniform wind):** the rung-blind Encoding A still has the optimum as its programmed ground state,
#   because the van der Waals tails favour spread-out layouts; P(optimal) halves to 0.32. Encoding B matches A.
# - **R2 (short wake, prevailing west wind):** the optimum set is unchanged, so A is unaffected; B is worse through
#   its dynamics (a gap about 5× smaller), not its ranking.
# - **R3 (long wake, prevailing west wind): both bottlenecks bind.** Every scalable encoding (A, B and the anisotropic
#   lattice) misranks the optimum: 355, 355 and 176–309 of the 1,820 four-turbine layouts are programmed below it, and
#   their most probable shots are their own programmed low-energy layouts, which are good but not optimal. That is an
#   encoding failure. The one encoding that ranks the optimum well, the ranking-aware fit (rank 40; non-scalable, it
#   enumerates every layout), then fails on finite-time dynamics: its most probable shots sit at programmed ranks
#   34–309, and its expected $S$ falls below random. On every R3 program P(optimal) stays below 0.001.

# %%
# Checks behind the R3 conclusion (ranks: 0 = programmed ground state, of 1,820 layouts).
_diag = RES["R3_diagnosis"]
_plateau_ranks = [aspect_scan[a]["opt_rank"] for a in PLATEAU]
assert RES["ranking"]["R3|A"]["opt_rank"] == RES["ranking"]["R3|B"]["opt_rank"] == 355
assert min(_plateau_ranks) == 176 and max(_plateau_ranks) == 309                  # lattice plateau: still misranked
assert RES["R3_probe_selection"]["rankfit_metrics"]["opt_rank"] == 40              # ranking-aware fit ranks it well
# Encoding failure: A, B and the lattice put their top weight on their own programmed low-energy layouts, none optimal.
assert max(_diag["A"]["prog_rank"]) <= 10 and max(_diag["B"]["prog_rank"]) <= 30
assert _diag["B-lattice"]["prog_rank"][:2] == [0, 1]
assert all(s < 1 for k in ("A", "B", "B-lattice") for s in _diag[k]["S"])
# Dynamics failure: the ranking-aware fit's most probable shots are far above its programmed ground state.
assert min(_diag["B-rankfit"]["prog_rank"]) == 34 and max(_diag["B-rankfit"]["prog_rank"]) == 309
assert RES["R3_probes"]["rankfit"]["exact"]["exp_norm_given_4"] < 0
assert max(RES["A"]["R3"]["exact"]["P_opt"], RES["B"]["R3"]["exact"]["P_opt"],
           *(v["exact"]["P_opt"] for v in RES["R3_probes"].values())) < 1e-3
print("R3 conclusion checks pass: scalable encodings misrank the optimum; the ranking-aware fit fails on dynamics")

# %% [markdown]
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
# 5. **Choosing the final detuning.** A first schedule ending at $\Delta_f = 0.104$ put 26% of shots on 5 excitations;
#    lowering it to 0.045 (counting only the $\sqrt2 a$ neighbour) removed those but left 3- and 4-excitation states
#    nearly degenerate (P(3) = 0.45, P(optimal) = 0.24), because a new excitation typically sits at $2a$ from 2–4
#    excited atoms and pays $\approx 0.03$–$0.06$ in van der Waals tails. Scanning $\Delta_f \in [0.06, 0.11]$ on R0 at
#    $T_{\max}$ gives the best exact P(optimal) = 0.696 at $\Delta_f = 0.10$ (P(5) = 0.16), frozen for every rung.
# 6. **The paper underspecifies wake-cell membership.** Several natural rules (strict vs non-strict inequalities,
#    tolerances, cell overlap) reproduce R0 but give different optima for long wakes (R1: 2187–2238 kW). A first rule,
#    chosen because it matched a set of expected R1/R3 optima (2228.6, 2238.8, unique R3 optimum), turned out to depend
#    on floating-point rounding at the 45° cone edge and broke the grid's mirror symmetry by up to 4.2 kW; those
#    expected values were rounding artifacts. We now exclude cells exactly on the edge ($t < r s - 10^{-9}$), which is
#    symmetric: R0 and R2 are unchanged, R1 becomes 2229.38, and R3 has two mirror-image non-MIS optima (2242.71). With
#    an inclusive edge instead, R3's optimum would be the corner MIS and the ladder would lose its non-native end.
# 7. **Bit-flip SA on the penalty QUBO gets trapped.** With the paper's protocol (single-bit flips, count penalty
#    $\lambda_1$) every move between 4-turbine layouts crosses a barrier of hundreds of kW; on R3 it never reaches the
#    optimum in 36 runs, and neal's default cold temperature (from the smallest QUBO coefficient, $2\lambda_1$) is too
#    hot to hold the turbine count at all. A fixed-count swap-move SA is used as the stronger classical reference.
# 8. **25 atoms: emu-sv cannot sample above 24 qubits.** The 25-atom GPU simulation completed (about 22 min at
#    T = 320, complex128) and then crashed in `StateVector.sample` (`torch.multinomial`: "number of categories cannot
#    exceed 2^24"). qoolqit's `LocalEmulator` always appends a `BitStrings` observable, so every emu-sv run above
#    24 atoms through `LocalEmulator` fails. Workaround if needed: call `SVBackend` directly with only a `StateResult`
#    and sample in numpy.
# 9. **`InteractionEmbedder` as a baseline.** With shipped defaults (Nelder–Mead from random positions in the unit
#    square, no scale factor, no spacing constraint) it reaches $\epsilon$ = 0.77–0.88 on R1–R3 and a pairwise Spearman
#    correlation of only 0.10–0.20 with $W$, against $\epsilon$ = 0.47–0.55 and 0.69–0.99 for our L-BFGS fit; on R1 its
#    minimum spacing (0.993) is below the device minimum, so its layout would not compile on `AnalogDevice`.
# 10. **Minimum-distance check is exact.** A layout rescaled to minimum spacing exactly 1.0 failed `compile_to` with
#     "minimum distance 1.0000 goes below the minimum allowed", from floating-point rounding (0.99999…). We rescale to
#     $1 + 10^{-6}$.
# 11. **P(4) versus $\Delta_f$ is not monotonic for embedded registers.** On R1 (Encoding B, $\Omega = 1.7^{-6}$) it
#     peaks at 0.72 for $\Delta_f = 0.10$, falls to 0.03 at 0.3 and rises again to 0.15 at 0.7, so the per-rung scan
#     needs a wide, roughly logarithmic grid.
# 12. **Radial distance is measured from the origin.** `AnalogDevice`'s `max_radial_distance` (7.6) is checked
#     against the distance from the coordinate origin, not from the register's centre, so an uncentred register that
#     would fit comfortably fails `compile_to` ("maximum radial distance 7.7403"). Registers are centred before compiling.
# 13. **Comparing different statistics.** A first version of the comparison set the published VQE means (one selected
#     layout per run, averaged over runs) against our expected $S$ of a single post-selected shot (+0.99 on R0). The
#     like-for-like number is our selection-rule result (36/36 optimal, $S$ = 1.00); every column now names its
#     statistic, and expected $S$ is used only between our own runs. Likewise, a single tuned aspect ratio for the R3
#     lattice overstated the precision of a choice made on a flat metric; it is now reported over the whole plateau.
