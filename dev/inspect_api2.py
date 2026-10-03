import inspect
from qoolqit import QuantumProgram, RampWaveform, ConstantWaveform, PiecewiseLinearWaveform, InterpolatedWaveform, AnalogDevice
print(inspect.getdoc(QuantumProgram.compile_to)[800:])
for W in [RampWaveform, ConstantWaveform, PiecewiseLinearWaveform, InterpolatedWaveform]:
    print("==", W.__name__, inspect.signature(W)); print((inspect.getdoc(W) or "")[:500])
print(AnalogDevice().specs() if callable(AnalogDevice().specs) else AnalogDevice().specs)
print(inspect.getdoc(AnalogDevice().converter.__class__)[:1500])
c = AnalogDevice().converter; print([m for m in dir(c) if not m.startswith("_")]); 
for m in ["factors","energy","time","distance"]:
    if hasattr(c, m): print(m, getattr(c, m))
