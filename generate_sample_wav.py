import numpy as np
from scipy.io.wavfile import write

sr = 22050  # sample rate
duration = 2  # seconds
t = np.linspace(0., duration, int(sr * duration))
signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # 440Hz tone (A4)

write("tests/sample.wav", sr, signal.astype(np.float32))