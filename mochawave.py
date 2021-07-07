from math import pi, log2, sin
from typing import Callable, Iterable

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

	# properties

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

	# double underscore methods

	def __add__(self, other):
		# type: (Wave, Wave) -> Wave
		"""concatenate two waves"""
		assert self.sample_rate == other.sample_rate
		return Wave(list(self.amplitude_data) + list(other.amplitude_data), self.sample_rate)

	def __len__(self) -> int:
		"""sample points"""
		return len(self.amplitude_data)

	def __matmul__(self, other):
		# type: (Wave, Wave) -> Wave
		"""overlay two waves"""
		assert self.sample_rate == other.sample_rate and len(self) == len(other)
		return Wave((x + other.amplitude_data[i] for i, x in enumerate(self.amplitude_data)), self.sample_rate)

	def __mul__(self, other: float):
		# type: (Wave, float) -> Wave
		"""change amplitude of wave"""
		return Wave((i*other for i in self.amplitude_data), self.sample_rate)

	def __neg__(self):
		# type: (Wave) -> Wave
		"""negate amplitude"""
		return Wave((-x for x in self.amplitude_data), self.sample_rate)

	# methods

	def set_speed(self, mul: int = 1, div: int = 1):
		# type: (Wave, int, int) -> Wave
		"""mul increases length by the multiplier and div decreases it by the same"""
		return Wave([self.amplitude_data[i//mul] for i in range(mul*len(self))][::div], self.sample_rate)

	def wav(self) -> None:
		"""output as wav file"""
		from mochaaudio import pcm_to_wav
		pcm_to_wav(self.pcm, sample_rate=self.sample_rate)

	# static methods

	@staticmethod
	def from_function(f: Callable[[float, float, float], float], freq: int = 440, t: float = 1, sample_rate: int = 44100):
		# type: (Callable[[float, float, float], float], int, float, int) -> Wave
		return Wave(f(i/sample_rate, freq) for i in range(int(t*sample_rate)))

	@staticmethod
	def sawtooth(freq: int = 440, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, int) -> Wave
		return Wave.from_function(sawtooth, freq, t, sample_rate)

	@staticmethod
	def sine(freq: int = 440, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, int) -> Wave
		return Wave.from_function(sine, freq, t, sample_rate)

	@staticmethod
	def square(freq: int = 440, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, int) -> Wave
		return Wave.from_function(square, freq, t, sample_rate)

	@staticmethod
	def triangle(freq: int = 440, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, int) -> Wave
		return Wave.from_function(triangle, freq, t, sample_rate)
