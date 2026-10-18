# Project handoff: QoolQit contest, wind farm layout optimisation on simulated Rydberg atoms

Handoff written 18 Oct 2026, when the project moved from the Linux server (`math`, user M25IQT003) to the user's Mac.
Read this first; the notebook is the source of truth for every number.

## Deadline and what is left

- **Contest deadline: 18 Oct 2026, 23:59 CEST.** Contest page (rules, submission format):
  https://www.pasqal.com/resources/the-qoolqit-contest-is-live/
- Deliverables: the notebook `submission/wflo_rydberg.ipynb` (done, executed, outputs kept) and a **two-slide deck**
  (in progress; the user is making the PPT on the Mac).
- Remaining: (1) finish the two-slide deck, (2) final review, (3) submit. Do not rerun or change the notebook unless
  the user asks; it is finished and its results are verified.

## The project in brief

Question: how much turbine power does a Rydberg-atom solver lose as wind-farm wake physics departs from the Rydberg
Hamiltonian? Start from the published 4×4 instance of Hancock et al. (arXiv:2312.13123), show it is exactly MIS on the
4×4 king graph, then climb a ladder of our own instances R0–R3. Each rung uses Encoding A (native blockade, one fixed
program for every rung) and Encoding B (QUBO embedded into fitted atom positions), emulated with QoolQit 1.4.0 + emu-sv,
and compared with exact enumeration, simulated annealing and the published VQE/SA numbers. Score: normalised
S = (P − P_random)/(P_opt − P_random), 0 = random layout, 1 = optimal.

| Rung | Wake | Wind | Optimum (kW) | A: P(opt) | A: expected S | B: expected S | SA-swap S |
|---|---|---|---|---|---|---|---|
| R0 (published) | x=1, r=1.5 | uniform | 2304.00 (79 optima = king-graph MIS) | 0.70 | +0.99 | not run | +1.00 |
| R1 | x=3, r=1 | uniform | 2229.38 (unique: four corners) | 0.32 | +0.89 | +0.89 | +1.00 |
| R2 | x=1, r=1.5 | 75% from west | 2304.00 (same 79) | 0.70 | +0.98 | +0.85 | +0.99 |
| R3 | x=3, r=1 | 75% from west | 2242.71 (2 diagonals, not MIS) | 0.000 | +0.80 | +0.62 | +1.00 |

Headline claims (wording already checked in the notebook; keep it):
- R0: the paper's selection rule picks an optimal layout in 36/36 runs for each of 11 resampling seeds (S = 1.00);
  published VQE means are S = −0.33 to +0.11, published SA median S ≈ −0.9 ± 0.2 (read off their Fig. 9).
- R1–R2: Encoding A keeps the optimum as its programmed ground state; selection rule 36/36 for all 11 seeds.
- R3: no encoding reaches the optimum (P(opt) < 0.001 everywhere). Scalable encodings misrank it (rank 355 of 1,820
  for A and B, 176–309 for the anisotropic lattice); the non-scalable ranking-aware fit ranks it 40th but fails on
  finite-time dynamics (expected S below 0).
- SPAM errors (Pulser's default literature values, De Léséleuc 2018: η 0.005, ε 0.01, ε′ 0.05): P(opt) R0 0.70 → 0.52,
  R1 0.32 → 0.23; selection rule still 36/36.
- QPU proposal: one Encoding A program scored for R0 and R1, 2 jobs × 500 shots ≈ 67 min at 0.25 Hz (Pasqal Orion
  Beta technical overview, March 2025).

**Claim boundaries (never break these on the slides):** emulator results only; R0 being easy is a structural
observation, not quantum advantage; no speed claims; D-Wave results (Nigro et al. 2025) in prose only, never in a table
next to emulator numbers; "first published neutral-atom WFLO study we could find", never "first"; R1–R3 are our own
instances; label published numbers as reported values, not reruns.

## Files

This folder (`qoolqit-wflo/`, private git repo, branch `master`, no remote):
- `submission/wflo_rydberg.ipynb`: **the submitted notebook**, executed in demo mode, outputs kept.
- `wflo_rydberg.ipynb` + `wflo_rydberg.py`: the development copy (jupytext pair; outputs stripped by nbstripout).
- `PLAN_FILE.md`: full project plan and status. `README.md`: repo notes, including the nbstripout exemption.
- `dev/`: Phase 0 scripts, wake-rule checks, the 24-atom issue write-up, `check_placeholders.py` (run before any
  submission build: `python dev/check_placeholders.py wflo_rydberg.ipynb submission/wflo_rydberg.ipynb`).
- `runs/` (gitignored): logs and `results_full.csv` from the full GPU runs.

Deck draft (`qoolqit-deck-draft/`, shipped next to this folder, deliberately kept out of the repo):
- `qoolqit_wflo.pptx`, `qoolqit_wflo.pdf`, `slide-1.png`, `slide-2.png`: draft of 16 Oct.
  Slide 1: "Emulated Rydberg atoms solve native wakes, not long directional ones" (the ladder, the S-vs-rung plot,
  36/36, S = 0.89, P(opt) ≈ 0). Slide 2: "QoolQit in practice, and a two-job QPU ask" (what helped / what we hit,
  QPU ask).
- `build_deck.py`: python-pptx script that pulls every number from the JSON record in `wflo_rydberg.py`. It expects
  to live at `qoolqit-wflo/deck/build_deck.py` (ROOT = two levels up) and writes `deck/s_vs_rung.png`; usage
  `python deck/build_deck.py submission/wflo_rydberg.ipynb`. Copy it there (and keep `deck/` out of git) or edit ROOT.
  python-pptx is a dev-only dependency, not a notebook one.
- The draft predates the final commit (11-seed ranges); check slide wording against the notebook's header and §4.4.

Public showcase repo (separate): https://github.com/Advaith1603/wflo-rydberg-qoolqit (README, the executed notebook,
summary plot, the 24-atom issue, MIT). Independent of this private repo.

## Working on the Mac

- Notebook was only tested on Linux, Python 3.11.17. For the deck you do not need to run it: the executed notebook
  already holds every output and figure.
- If you do run it: Python ≥ 3.11 venv, `pip install -r` the pins from the install cell (qoolqit 1.4.0, pulser-core
  1.9.1, pulser-simulation 1.9.1, emu-sv 2.9.1, emu-base 2.9.1, torch 2.14.1, numpy 2.4.6, scipy 1.17.1,
  matplotlib 3.11.2), then demo mode (default, ~1 min compute). Full mode (`WFLO_MODE=full`) needs a GPU (~65 min) and
  was run on the server; there is no CUDA on a Mac.
- **nbstripout:** `.git/info/attributes` (copied with `.git`) applies an nbstripout filter to `*.ipynb` and exempts
  `submission/*.ipynb`. If nbstripout is not installed on the Mac, git operations on notebooks error. Install it
  (`pip install nbstripout`) before committing, and verify `git check-attr filter -- submission/wflo_rydberg.ipynb`
  prints `filter: unset` so the submitted copy keeps its outputs.

## User preferences

- **Never add `Co-Authored-By: Claude` (or any Claude trailer) to commits.** Credit Claude in a README
  Acknowledgements section instead. Commit identity: `Advaith1603 <88267748+Advaith1603@users.noreply.github.com>`.
- Confirm before anything outward-facing (pushing, publishing, submitting).
- Write plainly; every number must trace to the notebook's results record.

## Technical facts worth knowing (verified on the server)

- QoolQit 1.4.0: `BackendType` is the `pulser.backends` module, not an enum. No `QUBOInstance` /
  `AdiabaticDriveShaper`; `InteractionEmbedder` and `Blade` do ship. `AnalogDevice` dimensionless specs:
  max_duration 332.4, max_amplitude 0.2268, min_distance 1.0, max_radial 7.6; units 5 µm, 55.4 rad/µs, 18.05 ns; C6 = 1.
  `max_radial_distance` is measured from the origin, so registers must be centred; a spacing of exactly 1.0 fails
  `compile_to` after rounding (rescale to 1 + 1e-6).
- emu-sv 2.9.1 hard-codes complex128. 16 atoms: 8.8 s GPU, 33 s CPU per emulation; agrees with QuTiP to 4e-5.
  `LocalEmulator` + emu-sv cannot sample above 24 atoms (`torch.multinomial` 2^24 cap), so 5×5 was dropped.
- Wake rule (user decision 15 Oct): j is in the wake of i iff 0.001 < s ≤ x + 0.001 and t < r·s − 1e-9 (cone edge
  excluded); angles clockwise from west, rows grow south; "75% from west" = 0.75 at 0°, 0.25 spread over the other 35.
  Never reintroduce the bare `t < r*s` test (floating-point rounding at the 45° edge broke symmetry).
