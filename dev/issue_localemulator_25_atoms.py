import torch
torch.set_num_threads(16)  # shared machine

from qoolqit import AnalogDevice, ConstantWaveform, Drive, QuantumProgram, Register
from qoolqit.execution import BackendType, LocalEmulator
from emu_sv import SVConfig

register = Register.from_coordinates([(float(i), float(j)) for j in range(5) for i in range(5)])  # 25 atoms
program = QuantumProgram(register, Drive(amplitude=ConstantWaveform(10.0, 0.05)))
program.compile_to(AnalogDevice())

emulator = LocalEmulator(backend_type=BackendType.SVBackend, emulation_config=SVConfig(gpu=False), num_shots=100)
results = emulator.run(program).results()
