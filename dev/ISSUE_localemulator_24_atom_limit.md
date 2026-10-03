**Title:** `LocalEmulator` with `SVBackend` fails for more than 24 atoms: `torch.multinomial` category limit in `StateVector.sample`

**Summary**

Running a QoolQit `QuantumProgram` with more than 24 atoms through `LocalEmulator(backend_type=BackendType.SVBackend)`
completes the time evolution, then crashes while sampling bitstrings:

```
RuntimeError: number of categories cannot exceed 2^24
```

`emu_sv.StateVector.sample` draws shots with `torch.multinomial` over all $2^N$ basis states, and torch caps
`multinomial` at $2^{24}$ categories. `LocalEmulator.validate_emulation_config` always adds a `BitStrings`
observable, so the user cannot avoid sampling, and every run above 24 atoms fails after the full simulation.
On a 25-atom adiabatic program (T ≈ 6 µs) on GPU, this happened after about 22 minutes of emulation.

**Versions**

- qoolqit 1.4.0, emu-sv 2.9.1, emu-base 2.9.1, pulser-core 1.9.1, torch 2.14.1+cu130
- Python 3.11.17, Linux 5.15 (Ubuntu 20.04), x86_64; reproduced on CPU (`gpu=False`) and on GPU (RTX 5000 Ada)

**Minimal reproducer** (about 4.5 minutes on 16 CPU threads)

```python
import torch
torch.set_num_threads(16)

from qoolqit import AnalogDevice, ConstantWaveform, Drive, QuantumProgram, Register
from qoolqit.execution import BackendType, LocalEmulator
from emu_sv import SVConfig

register = Register.from_coordinates([(float(i), float(j)) for j in range(5) for i in range(5)])  # 25 atoms
program = QuantumProgram(register, Drive(amplitude=ConstantWaveform(10.0, 0.05)))
program.compile_to(AnalogDevice())

emulator = LocalEmulator(backend_type=BackendType.SVBackend, emulation_config=SVConfig(gpu=False), num_shots=100)
results = emulator.run(program).results()
```

**Traceback** (end)

```
  File ".../qoolqit/execution/backends.py", line 116, in run
    results = self._backend.run()
  File ".../emu_sv/sv_backend.py", line 35, in run
    results.append(self._run_from_sequence_data(sequence_data, self._config))
  File ".../emu_sv/sv_backend.py", line 41, in _run_from_sequence_data
    return impl._run()
  File ".../emu_sv/sv_backend_impl.py", line 274, in _run
    self.step(step)
  File ".../emu_sv/sv_backend_impl.py", line 181, in step
    self._apply_observables(step_idx)
  File ".../emu_sv/sv_backend_impl.py", line 250, in _apply_observables
    callback(
  File ".../pulser/backend/observable.py", line 190, in __call__
    value_to_store = self.apply(
  File ".../pulser/backend/default_observables.py", line 173, in apply
    return state.sample(
  File ".../emu_sv/state_vector.py", line 166, in sample
    outcomes = torch.multinomial(probabilities, num_shots, replacement=True)
RuntimeError: number of categories cannot exceed 2^24
```

The limit comes from torch itself: `torch.multinomial(torch.ones(2**24 + 1), 1)` raises the same error, while
`2**24` categories work.

**Expected behaviour**

Either sampling works for state vectors above $2^{24}$ entries, or the limit is reported before the
(potentially long) simulation starts.

**Possible fixes**

1. emu-sv: sample without `torch.multinomial`, for example with inverse-CDF sampling
   (`torch.searchsorted(torch.cumsum(p, 0), torch.rand(num_shots))`), which has no category limit.
2. qoolqit: let users opt out of the automatic `BitStrings` observable in `LocalEmulator` (for example when the
   config already requests a `StateResult`), so the final state can be retrieved and sampled elsewhere.
3. At minimum, validate `n_qubits <= 24` up front when a `BitStrings` observable is attached to `SVBackend`.

**Workaround**

Call `emu_sv.SVBackend` directly with only a `StateResult` observable and sample the returned state in numpy.
