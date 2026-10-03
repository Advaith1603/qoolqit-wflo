import inspect, qoolqit
from qoolqit import Register, Drive, QuantumProgram, AnalogDevice, MockDevice, DigitalAnalogDevice, available_default_devices
from qoolqit.embedding import Blade, BladeConfig
for obj in [Register, Register.from_coordinates, Drive, QuantumProgram, QuantumProgram.compile_to, AnalogDevice, BladeConfig, available_default_devices]:
    try: print("==", obj.__qualname__, inspect.signature(obj))
    except Exception as e: print("==", obj, e)
    d = inspect.getdoc(obj); print((d or "")[:900]); print()
print("Register methods:", [m for m in dir(Register) if not m.startswith("_")])
print("QuantumProgram methods:", [m for m in dir(QuantumProgram) if not m.startswith("_")])
print(available_default_devices())
for D in [AnalogDevice, MockDevice, DigitalAnalogDevice]:
    d = D(); print(D.__name__, [m for m in dir(d) if not m.startswith("_")])
