from math import pi, log2, sin
from typing import Iterable

# basic waveforms
def square(x: float, freq: float, amp: float = 1) -> float:
	return amp*(2*(x/freq//1%2)-1)

def triangle(x: float, freq: float, amp: float = 1) -> float:
	return amp*(abs(4*((x/freq)%1)-2)-1)

def sine(x: float, freq: float, amp: float = 1) -> float:
	return amp*sin(2*pi*x/freq)

def sawtooth(x: float, freq: float, amp: float = 1) -> float:
	return amp*(2*((x/freq)%1)-1)

# note to freq conversions
note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
c0 = 55/2**1.75

def f2n(f: float) -> str:
	o=int(log2(f/c0))
	n=round(12*(log2(f/c0)-o))
	return note_names[n]+str(o)

def n2f(n: str) -> float:
	o=int(n[-1:])
	n=int(note_names.index(n[:-1]))
	return c0*2**(o+n/12)

# wave class for manipulating data
class Wave:
	def __init__(self, amplitude_data: Iterable[float], sample_rate: int = 44100) -> None:
		self.amplitude_data = amplitude_data
		self.sample_rate = sample_rate

	@property
	def max(self) -> float:
		return max(abs(i) for i in self.amplitude_data)

	@property
	def normalized(self):
		# type (Wave) -> Wave
		m = self.max
		return Wave((i/m for i in self.amplitude_data), self.sample_rate)

	@property
	def pcm(self) -> bytes:
		"""transform into 16-bit mono pcm"""
		import numpy as np
		return (0x7FFF * np.array(self.normalized)).astype(np.int16).tobytes()

	def wav(self) -> None:
		"""output as wav file"""
		from mochaaudio import pcm_to_wav
		pcm_to_wav(self.pcm, sample_rate=self.sample_rate)

	@staticmethod
	def square(freq: int = 440, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, int) -> Wave
		return Wave(square(i/sample_rate, freq) for i in range(int(t*sample_rate)))
