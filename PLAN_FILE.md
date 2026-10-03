# Wind Farm Layout on Rydberg Atoms — QoolQit Contest Project Plan

Oct 2, 2026 · @Advaith

## Overview

Question: how much turbine power does a Rydberg-atom solver lose as wind-farm wake physics departs from the Rydberg Hamiltonian, and how does that compare with D-Wave, QAOA, VQE and simulated annealing on the same instances? Deadline 18 Oct 2026, 23:59 CEST; budget about 27 working hours.

**Tier 2 (main track):** a ladder of 4×4 WFLO instances, from Rydberg-native to fully non-native, each solved with two encodings (native blockade and embedded QUBO) and compared with exact enumeration, simulated annealing and published quantum results.

**Tier 1 (fallback, banked by hour \~12):** the published Plymouth instance alone, solved natively and compared head-to-head with their VQE, SA and Gurobi numbers. Tier 1 is also the first rung of Tier 2, so it is built on the way, not as separate work.

**Claim boundaries:**

- Emulator results only; published D-Wave numbers are hardware and are labelled as such, never pooled with emulator numbers.
- The published instance is easy and Rydberg-native; a good result there is presented as a structural observation, not as quantum advantage.
- No speed claims: emulation time measures the classical simulator, not the device.
- 4×4 grids (16 atoms) are the core; 5×5 (25 atoms, on the remote GPU) is a gated extension (Phase 4b).

## Positioning

WFLO has been solved on D-Wave, QAOA and simulated VQE, but no published neutral-atom result was found (5 targeted searches on 2 Oct 2026, 4 of them neutral-atom specific, plus an independent manual search). Pasqal does list windfarm layout optimization as a use case on its [Energy & Utilities page](https://www.pasqal.com/industries/energy-utilities/), with no linked paper, so unpublished internal or client work may exist. Claim "first published neutral-atom study we could find", never "first", and ask on the Pasqal Community Portal before submitting.

| Study | Platform | Instance | Result |
| --- | --- | --- | --- |
| [Hancock et al., arXiv:2312.13123](https://arxiv.org/abs/2312.13123) | Qiskit simulator (VQE), SA, Gurobi | 4×4, Mosetti case 2, x = 1, r = 1.5, m = 4 | Gurobi always optimal (2304.0 kW); VQE mean 93.3–95.5% of optimum; VQE \~11,500 s vs Gurobi 1.1 s |
| [Milan group, Advanced Quantum Technologies](https://expertise.unimi.it/resource/item/1540710) | D-Wave hardware, QAOA, Gurobi | Not yet read (open question) | D-Wave \~10× faster than Gurobi, at most 3% lower power |
| [Plymouth follow-up, arXiv:2512.17582](https://arxiv.org/abs/2512.17582) | Circuit-based, fewer qubits | Larger farms (to check) | To extract |
| [Senderovich et al. 2022](https://doi.org/10.1109/ACCESS.2022.3193143) | Hybrid and classical | Original QUBO formulation | Source of the model |

**Finding from rebuilding the Plymouth QUBO** ([their code](https://github.com/jamesshancock/WindfarmLayoutOptimization)): the wake terms couple only nearest neighbours (83.7) and diagonal neighbours (17.6), identically in every direction, and nothing beyond. All 79 optimal layouts are exactly the 79 independent 4-sets of the 4×4 king graph, the graph family used in Ebadi et al. 2022. The Q matrix is dense only because of the turbine-count penalty; without that penalty the optimum fills all 16 sites.

| Rung | Wake | Wind | Max anisotropy | Optimum is king-graph MIS? |
| --- | --- | --- | --- | --- |
| R0 (published) | x = 1, r = 1.5 | Uniform, 36 directions | 0% | Yes, 79 optima |
| R1 | x = 3, r = 1 | Uniform | 11% | Yes, unique optimum |
| R2 | x = 1, r = 1.5 | 75% from west | 83% | Yes |
| R3 | x = 3, r = 1 | 75% from west | 99% | No |

The contribution is the curve across this ladder: native at R0, increasingly mismatched at R1–R3 (long range, wrong decay rate, then direction-dependent).

## Phase 0 — Environment and onboarding (3–4 h)

Exit: a 16-atom QuantumProgram on a 4×4 square register compiles and returns bitstrings on the local emulator, and you know the wall-clock cost per run.

1. Fresh venv; pip install qoolqit; pin versions in requirements.txt immediately.
2. Run the Register → Drive → QuantumProgram → compile → run tutorial unmodified.
3. Read the QUBO tooling docs: QUBOInstance, AdiabaticDriveShaper, and whether an embedder (QUBO → atom positions) ships with it.
4. On the remote machine, time a 16-atom run (CPU) and a 25-atom run (GPU) at 500 shots. These numbers set the Phase 4 sweep size and Gate G4b.

| Failure | Detection | Fix / alternative |
| --- | --- | --- |
| Install conflicts | Import or resolver errors | Python 3.11 clean venv; versions from the QoolQit pyproject; last resort Colab with pinned installs |
| 16 atoms too slow on the state-vector emulator | Over 5 min per run | Use the MPS / tensor-network backend (state the bond dimension); cut shots to 200; if still too slow, use 3×3 rungs (9 atoms) and say so |
| No QUBO embedder in QoolQit | Not in docs or API | Write your own in Phase 3 (scipy minimisation over positions, \~30 lines) |
| Remote access needs credentials | Auth failure | Everything runs locally; remote is optional |

**Gate G0 (hour 5):** if no 16-atom run completes, move to Phase 1 (no quantum dependency) and post the error on the Pasqal Community Portal.

## Phase 1 — Instance ladder and classical baselines (2–3 h)

Exit: Q matrices for R0–R3 saved, each with its exact optimum, full optimum set, and an SA baseline over 36 runs.

1. Reimplement the QUBO from the paper's equations (the Plymouth repo has no license, so its code is not copied). Use their repo only as a test oracle: assert per-turbine power 576, penalty 83.728 at distance 1 and 17.582 at √2, zero beyond, R0 optimum 2304.0 kW with exactly 79 optima.
2. Build R1–R3 by changing only x, r and the wind regime (table in Positioning). Keep m = 4.
3. Exact optimum by enumeration over all 1,820 four-turbine layouts (instant). This replaces Gurobi; Gurobi needs a licence and adds nothing at this size.
4. SA: your own numpy implementation (fewer dependencies), 36 runs, the same protocol as the paper.
5. Store per rung: wake matrix W (pairwise penalties only), per-site power, optimum set.

| Failure | Detection | Fix / alternative |
| --- | --- | --- |
| Your reimplementation disagrees with the oracle numbers | Oracle assertions fail | Compare the wake cone rule cell by cell (rotation convention, the 0.001 tolerance, strict vs non-strict inequality) against their code run separately, outside the notebook |
| A rung is degenerate in a way that makes it trivial (e.g. R2 behaves exactly like R0) | Same optimum set as R0 | Keep it as a control (it isolates anisotropy at short range) or replace with a stronger prevailing wind (90%) |
| SA solves everything perfectly | 100% optimum on all rungs | Expected at 16 variables; it stays as the reference, not a competitor to beat |

## Phase 2 — Native encoding and the R0 run (3–4 h)

Exit: on R0, the emulator returns an optimal 4-turbine layout (2304.0 kW) as its most frequent valid bitstring, reproducibly. This result is Tier 1.

**Encoding A (native):** atoms on the 4×4 square lattice, spacing a, with blockade radius R\_b between √2·a and 2·a. The blockade then forbids exactly the king-graph neighbours, which is the wake structure of R0. All sites have equal power, so one global detuning sweep suffices; no DMM needed at R0. The king graph's maximum independent set has size 4 = m, so the blockade enforces the turbine count for free.

1. Register.from\_coordinates on the grid; check min\_distance against the device limit.
2. Drive: amplitude ramp up, hold, down; global detuning swept from negative to positive (the standard MIS adiabatic schedule).
3. Compute R\_b from the final Ω and Δ and assert √2·a < R\_b < 2·a.
4. Run 500 shots. Per run, take the most frequent bitstring with exactly 4 turbines and no adjacency, matching the paper's selection rule. Score it with the Plymouth Energy() function.
5. Repeat 36 runs with different sampling seeds; report mean % of optimum and P(optimal shot).

| Failure | Detection | Fix / alternative |
| --- | --- | --- |
| R\_b not inside (√2·a, 2·a) after compilation | Computed R\_b off target | Adjust final Δ or Ω; or rescale spacing with the MIN\_DISTANCE compiler profile and re-check |
| Many shots with 3 turbines, few with 4 | Count histogram | Sweep too fast or final Δ too small: lengthen the sweep, raise final Δ |
| Bitstring order mismatched with grid labels | Wrong layout drawn on a known test | Unit-test the mapping on 2 atoms, then on the grid labelling of the Plymouth code |
| C6 tails between next-nearest atoms favour some of the 79 optima | Skewed distribution over optimal layouts | Not a problem: all 79 score 2304.0 kW. Report it as an observation |

**Gate G2 (hour \~12):** R0 solved = Tier 1 banked. From here every extra hour only adds Tier 2 rungs.

## Phase 3 — QUBO-to-geometry embedding (4–5 h)

Exit: for R1–R3, atom positions whose Rydberg interactions reproduce the wake matrix, with the embedding error computed and the programs compiling.

**Why the count penalty can be dropped:** among layouts with exactly m turbines, total power is m·576 minus the summed wake penalties, so the ranking depends only on the wake matrix W. Drop the all-to-all count penalty (not geometric), use the global detuning as a chemical potential to centre the shot distribution on 4 turbines, and post-select exactly 4. The Plymouth VQE used the same post-selection, so the comparison stays fair.

**Encoding B (embedded):** choose positions r\_i and a scale s to minimise the squared mismatch between C6/|r\_i − r\_j|^6 and s·W\_ij over all pairs, subject to the device's minimum spacing.

```latex
\epsilon = \frac{\lVert U(r) - sW \rVert_F}{\lVert sW \rVert_F}, \qquad U_{ij}(r) = \frac{C_6}{\lvert r_i - r_j \rvert^6}
```

1. Use QoolQit's embedder if Phase 0 found one; otherwise scipy.optimize over 32 coordinates + s, with 20 random restarts plus one start from the grid.
2. Report ε per rung. Expect it to grow from R1 to R3: wake penalties decay \~2× per lattice step where C6/r^6 decays 64×, and R2–R3 are direction-dependent while van der Waals is not.
3. Also run Encoding A (grid + blockade, long-range wake ignored) on every rung. A vs B on the same rung is the core comparison.

| Failure | Detection | Fix / alternative |
| --- | --- | --- |
| ε very large (above \~50%) on all rungs | Embedding report | Fit only the strongest couplings (top-k per atom) and let weak ones go; report ε for both full and truncated W |
| Optimiser collapses atoms below minimum spacing | Compile error or min\_distance check | Add a hard penalty below d\_min; start from the grid |
| Embedded positions give interactions far above the drive scale | Effective blockade covers most of the register | Rescale s, or accept a partial hard-core and record it |
| Embedding works but dynamics ignore weak couplings | B performs no better than A | That is a result: the drive cannot resolve energy differences that small. Report it with the energy gaps |

**Gate G3 (hour \~17):** if no rung has a compiled Encoding B program, drop B and run Encoding A across the whole ladder (Tier 2 lite). That still yields the degradation curve.

## Phase 4 — Tier 2 sweep and comparison (4–6 h)

Exit: the comparison table filled for every rung reached, plus one plot of % of optimum against rung for Encoding A, Encoding B and SA.

| Metric | Definition | Why |
| --- | --- | --- |
| Power, % of optimum | Selected layout (most frequent valid 4-turbine bitstring) scored by the Plymouth Energy(), over 36 runs | Same figure of merit as the published VQE table |
| P(optimal shot) | Fraction of shots that are an optimal layout | Distribution quality, not one lucky selection |
| Valid-shot rate | Shots with exactly 4 turbines | Cost of dropping the count penalty |
| Embedding error ε | Phase 3 definition | Explains the degradation, rung by rung |
| Wall clock | Per run | Reported for completeness, never as a speed claim |

**Comparison table to fill** (emulator and hardware kept in separate columns):

| Rung | Exact (kW) | SA | Rydberg A | Rydberg B | VQE (published, simulator) | D-Wave (published, hardware) |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 2304.0 |  |  |  | 93.3–95.5% | Pending Milan paper |
| R1 | 2228.6 |  |  |  | n/a | n/a |
| R2 | 2304.0 |  |  |  | n/a | n/a |
| R3 | 2238.8 |  |  |  | n/a | n/a |

Published quantum baselines exist only for R0 (and for the Milan instances, if their parameters can be rebuilt). For R1–R3 the reference is exact plus SA. Optional, only if ahead of schedule: a numpy state-vector QAOA (p ≤ 3) on all rungs as a same-class gate-model reference.

| Failure | Detection | Fix / alternative |
| --- | --- | --- |
| Sweep will not fit the remaining time | Phase 0 timing × runs | Cut to 12 runs per cell, then drop R2 (the short-range control) before R1 or R3 |
| Both encodings collapse on R3 | Near-random % of optimum | Report it with ε and the energy gaps; it is the negative end of the curve |
| Results noisy across runs | Wide spread | Report median and IQR, as box plots like the paper |
| Milan parameters not recoverable | Paper unread or unspecified | Drop that column; cite their headline numbers in prose only |

### Phase 4b — 5×5 extension (optional, 2–3 h)

Exit: the same ladder at 25 atoms, Encoding A on every rung and Encoding B where it worked at 4×4, with exact optima, so the degradation curve has a second system size.

Hardware: shared remote machine, Ubuntu 20.04, 2× Xeon Silver 4314 (32 cores, 64 threads), 251 GB RAM (\~173 GB free at last check), RTX 5000 Ada 32 GB (driver 580). Ada cards are slow at FP64, so prefer complex64 on the GPU or compare against the CPU; cap jobs at 16–24 workers because the box is shared. A 25-atom state vector is \~512 MB (complex128), so it fits on the GPU with solver workspace. 30 atoms (16 GB per vector) is the practical ceiling and not planned; 36 atoms cannot be emulated exactly.

1. Rebuild R0–R3 with len\_grid = 5 and m = 9, the maximum independent set size of the 5×5 king graph. Assert that the R0 optimum is the king-graph MIS.
2. Exact optimum by enumeration over all 2,042,975 nine-turbine layouts, parallelised across up to 24 cores (minutes).
3. Encoding A on all rungs; Encoding B only on rungs where it compiled at 4×4; 12 runs per cell.
4. Add a 5×5 row group to the comparison table and a second line on the degradation plot.

| Failure | Detection | Fix / alternative |
| --- | --- | --- |
| Emulator backend has no GPU support | Runs on CPU only | Run on 16 CPU threads if under \~10 min per run; otherwise the MPS backend with a stated bond dimension |
| Enumeration too slow | Over 30 min per rung | Exact ILP (PuLP + CBC) on the QUBO, or numba-compiled enumeration |
| Embedding 50 coordinates converges poorly | Large ε or wide spread over restarts | Warm-start from the 4×4 solution tiled onto the larger grid; fewer, longer restarts |

**Gate G4b (hour 21):** start only if Tier 1 is banked, the 4×4 table has R0, R1 and R3 filled for Encoding A, and a 25-atom run takes under 5 minutes. Otherwise skip and list 5×5 under limitations. The judges' demo notebook stays at 4×4; 5×5 results load from CSV.

## Tier 1 fallback

Ship Tier 1 if any of these hold: Phase 2 finishes after hour 17; Gate G3 fails and Encoding A on R1–R3 cannot finish by hour 23; or it is 14 Oct and the Phase 4 table has no Tier 2 cell filled.

What Tier 1 contains:

- R0 solved natively on QoolQit, 36 runs, % of optimum and P(optimal shot), next to the published VQE (93.3–95.5%), SA and Gurobi numbers.
- The king-graph finding: the published benchmark is exactly MIS on the 4×4 king graph, so it is native to Rydberg hardware and easy for every method.
- One extra rung if any is finished (R1 with Encoding A is cheapest: same register, one rerun).
- Slide 2 built from the failure log (what blocked Tier 2 is itself QoolQit feedback).

Intermediate outcomes are fine to ship as they stand: Tier 1 plus any subset of rungs, with unfinished cells marked as not run rather than dropped.

## Phase 5 — Notebook and two-slide deck (2–3 h)

Exit: the notebook runs top to bottom in a fresh environment in under \~15 minutes (reduced default sweep, full results loaded from CSV), and the deck is two slides.

**Self-contained notebook rules:** one pinned install cell at the top; a MODE flag (demo by default: R0 at 4×4 on CPU in under 15 minutes; full: the remote-machine sweep); no external files, with final full-run results pasted into a cell as a JSON literal and demo mode checking its recomputed subset against them; GPU auto-detected, CPU fallback; seeds and versions printed first. Develop in a private git repo with jupytext pairing and nbstripout; the .ipynb alone is the deliverable.

1. Header: QoolQit and Python versions, the question, claim boundaries.
2. Instances: the Plymouth QUBO builder, the R0–R3 ladder, and the king-graph check (79 optima = 79 independent 4-sets).
3. Baselines: enumeration and SA.
4. Encoding A and B: registers, drives, ε per rung, compiled sequence plots.
5. Results: the comparison table and the % of optimum vs rung plot.
6. Limitations: emulator only, 16 atoms, published baselines only for R0, the count handled by post-selection.

Slide 1: problem → the ladder → the degradation plot → why it matters (layout of real farms has long, directional wakes; R0-type benchmarks overstate how native the problem is). Slide 2: QoolQit positives (dimensionless programming, register / drive abstraction, compile profiles) and challenges (embedding tooling, count constraints, emulator cost at 16 atoms), taken from the running failure log.

| Failure | Detection | Fix / alternative |
| --- | --- | --- |
| Fails in a clean environment | Fresh venv or Colab test | Pin everything in the install cell; no imports beyond qoolqit, numpy, scipy, matplotlib |
| Too slow for a judge to rerun | Over 15 min | Demo flag runs R0 with 100 shots; full sweep from saved CSV |
| Deck overflows | Count | One plot per slide; detail lives in the notebook |

## Schedule and gates

Tier 1 is complete by about hour 12, so the remaining 13–15 hours only add Tier 2 rungs; nothing after G2 can leave you with nothing to submit.

&#91;embedded content: phase sequence · 4 gates, Tier 1 banked at G2\]

A missed gate means taking its fallback the same day, not spending another evening on the same bug.

| Dates | Phases | Done when |
| --- | --- | --- |
| 2–5 Oct | 0 and 1 | 16-atom run timed; R0–R3 optima and SA saved |
| 6–9 Oct | 2 | R0 solved natively: Tier 1 banked |
| 10–13 Oct | 3 | ε reported for R1–R3; Encoding B compiles |
| 14–16 Oct | 4 | Comparison table filled (14 Oct: Tier 1 fallback check) |
| 17 Oct | 5 | Notebook passes a fresh run; deck done |
| 18 Oct | Buffer | Submitted |

23:59 CEST on 18 Oct is 03:29 IST on 19 Oct; treat the evening of 18 Oct IST as the real deadline.

## References

- [QoolQit contest page](https://www.pasqal.com/resources/the-qoolqit-contest-is-live/) — rules, submission format, deadline
- [QoolQit docs: Quantum Program, Register, Drive](https://docs.pasqal.com/qoolqit/fundamentals/quantum_program/)
- [QoolQit QUBO adiabatic drive shaper](https://docs.pasqal.com/applicationsolvingtools/qubo/content/driveshaping/adiabatic)
- [Hancock et al., Investigating techniques to optimise the layout of turbines in a windfarm using a quantum computer, arXiv:2312.13123](https://arxiv.org/abs/2312.13123) — R0 instance and VQE/SA/Gurobi baselines
- [Plymouth WFLO code (GitHub)](https://github.com/jamesshancock/WindfarmLayoutOptimization)
- [Plymouth follow-up on larger farms, arXiv:2512.17582](https://arxiv.org/abs/2512.17582)
- [Leveraging Quantum Annealing for Layout Optimization, Advanced Quantum Technologies (UNIMI record)](https://expertise.unimi.it/resource/item/1540710) — D-Wave, QAOA, Gurobi; [APS abstract](https://archive.aps.org/smt/2025/mar-x22/14)
- [Senderovich et al. 2022, IEEE Access](https://doi.org/10.1109/ACCESS.2022.3193143) — original WFLO QUBO
- [Mosetti et al. 1994](<https://doi.org/10.1016/0167-6105(94)90080-9>) — wind-regime benchmarks
- [Pichler et al. 2018, MIS with Rydberg atom arrays](https://arxiv.org/abs/1808.10816)
- [Ebadi et al. 2022, MIS on king graphs with 289 atoms](https://www.arxiv.org/abs/2202.09372)
- [Nguyen et al., Quantum optimization with arbitrary connectivity using Rydberg atom arrays](https://arxiv.org/pdf/2209.03965) — background for non-native embeddings
