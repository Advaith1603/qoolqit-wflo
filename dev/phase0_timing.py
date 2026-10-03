"""Phase 0: build, compile and time an L x L adiabatic MIS-style program at 500 shots."""
import sys, time, inspect
import numpy as np
from qoolqit import Register, Drive, QuantumProgram, AnalogDevice, PiecewiseLinearWaveform
from qoolqit.execution import LocalEmulator, BackendType

L = int(sys.argv[1]); backend_name = sys.argv[2] if len(sys.argv) > 2 else "QutipBackendV2"
SHOTS = 500
a = 1.0                                   # lattice spacing = device min_distance
coords = [(a * i, a * j) for j in range(L) for i in range(L)]
reg = Register.from_coordinates(coords)
print("n_qubits", reg.n_qubits, "min_distance", reg.min_distance())

# Standard MIS schedule: Omega up/hold/down, Delta swept negative -> positive.
omega = 1 / 1.7**6                        # R_b = Omega^(-1/6) = 1.7 a, inside (sqrt2 a, 2a)
T = 300.0                                 # dimensionless; device max 332.4
d0, d1 = -2.5 * omega, 2.5 * omega
amp = PiecewiseLinearWaveform([0.1 * T, 0.8 * T, 0.1 * T], [0.0, omega, omega, 0.0])
det = PiecewiseLinearWaveform([0.1 * T, 0.8 * T, 0.1 * T], [d0, d0, d1, d1])
prog = QuantumProgram(reg, Drive(amplitude=amp, detuning=det))
prog.compile_to(AnalogDevice())
print("compiled:", prog.is_compiled)

bt = getattr(BackendType, backend_name)
emu = LocalEmulator(backend_type=bt, num_shots=SHOTS)
t0 = time.perf_counter(); job = emu.run(prog); t1 = time.perf_counter()
print(f"WALL {L*L} atoms {backend_name}: {t1 - t0:.1f} s")
res = job.results() if callable(getattr(job, "results", None)) else job
print("job type", type(job).__name__, "| result type", type(res).__name__)
r = res[0] if isinstance(res, (list, tuple)) else res
bs = r.final_bitstrings if hasattr(r, "final_bitstrings") else None
print("result attrs", [m for m in dir(r) if not m.startswith("_")][:40])
if bs is not None:
    top = sorted(bs.items(), key=lambda kv: -kv[1])[:5]
    print("top bitstrings", top)
    print("hamming weight histogram", np.bincount([k.count("1") for k, v in bs.items() for _ in range(v)]))
