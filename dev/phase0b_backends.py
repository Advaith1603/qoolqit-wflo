"""Phase 0b: cross-check QuTiP vs emu-sv on the R0 native program and time CPU / GPU.

usage: python phase0b_backends.py L {qutip|sv-cpu|sv-gpu} [c64]
Prints one JSON line with wall time, exact P(optimal layouts) and the exact excitation-count distribution.
"""
import itertools, json, sys, time
import numpy as np
import torch
from qoolqit import AnalogDevice, Drive, PiecewiseLinearWaveform, QuantumProgram, Register
from qoolqit.execution import BackendType, EmulationConfig, LocalEmulator, StateResult

L, backend = int(sys.argv[1]), sys.argv[2]
c64 = len(sys.argv) > 3 and sys.argv[3] == "c64"
N = L * L
torch.set_num_threads(16)

# Retuned schedule: R_b = Omega^(-1/6) = 1.7 a; final detuning inside (U(2a), U(sqrt2 a)) = (0.0156, 0.125).
OMEGA = 1 / 1.7**6            # 0.0414
D0, D1 = -0.10, 0.045
T = 320.0                     # device max 332.4
reg = Register.from_coordinates([(float(c), float(r)) for r in range(L) for c in range(L)])  # qubit k = (row k//L, col k%L)
amp = PiecewiseLinearWaveform([0.1 * T, 0.8 * T, 0.1 * T], [0.0, OMEGA, OMEGA, 0.0])
det = PiecewiseLinearWaveform([0.1 * T, 0.8 * T, 0.1 * T], [D0, D0, D1, D1])
prog = QuantumProgram(reg, Drive(amplitude=amp, detuning=det))
prog.compile_to(AnalogDevice())

if backend == "qutip":
    emu = LocalEmulator(backend_type=BackendType.QutipBackendV2,
                        emulation_config=EmulationConfig(observables=(StateResult(evaluation_times=[1.0]),)),
                        num_shots=500)
else:
    from emu_sv import SVConfig
    if c64:  # experiment only: emu-sv 2.9.1 hard-codes complex128 as module constants
        import emu_sv.state_vector, emu_sv.sparse_operator, emu_sv.dense_operator
        for mod in (emu_sv.state_vector, emu_sv.sparse_operator, emu_sv.dense_operator):
            mod.dtype = torch.complex64
    cfg = SVConfig(gpu=(backend == "sv-gpu"), observables=(StateResult(evaluation_times=[1.0]),), log_level=30)
    emu = LocalEmulator(backend_type=BackendType.SVBackend, emulation_config=cfg, num_shots=500)

t0 = time.perf_counter()
res = emu.run(prog).results()
wall = time.perf_counter() - t0
r = res[0] if isinstance(res, (list, tuple)) else res
state = r.final_state

if backend == "qutip":
    bp = state.bitstring_probabilities(cutoff=0.0)
    idx = np.array([int(b, 2) for b in bp]); probs = np.zeros(2**N); probs[idx] = list(bp.values())
else:
    probs = (state.data.abs() ** 2).double().cpu().numpy()   # index -> bitstring is plain binary, qubit 0 = MSB
weights = np.array([bin(i).count("1") for i in range(2**N)])
wdist = np.bincount(weights, weights=probs, minlength=N + 1)

out = dict(L=L, backend=backend, c64=c64, wall_s=round(wall, 2), norm=float(probs.sum()),
           weight_dist={int(k): round(float(v), 5) for k, v in enumerate(wdist) if v > 1e-6})
if L == 4:
    king = lambda a, b: max(abs(a // 4 - b // 4), abs(a % 4 - b % 4)) == 1
    opt = [c for c in itertools.combinations(range(16), 4) if not any(king(a, b) for a, b in itertools.combinations(c, 2))]
    opt_idx = [sum(1 << (N - 1 - q) for q in c) for c in opt]
    out["p_opt"] = float(probs[opt_idx].sum())
    out["p_opt_given_4"] = out["p_opt"] / float(wdist[4])
shots = r.final_bitstrings
top = max(shots, key=shots.get)
out["top_shot"] = top
out["top_shot_is_opt"] = (L == 4 and int(top, 2) in set(out.get("p_opt") is not None and opt_idx or []))
out["max_prob_bitstring"] = format(int(np.argmax(probs)), f"0{N}b")
print("RESULT", json.dumps(out))
