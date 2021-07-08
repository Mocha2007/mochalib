from math import acos, copysign, pi, log2, sin
from random import uniform
from typing import Callable, Iterable, List

def _test() -> None:
	from mochaaudio import pcm_to_wav, play_file
	# w = Wave.from_function(brown_noise, amp=0.1).oscillate_amplitude(10, 0.5)
	w = Wave.sine(amp=0.1).binary_harmonic(10)
	#  w = Wave.from_function(brown_noise, amp=0.05)
	pcm_to_wav(w.pcm)
	play_file('output.wav')

# basic waveforms
def square(x: float, freq: float, amp: float = 1) -> float:
	return amp*(2*(x*freq//1%2)-1)

def triangle(x: float, freq: float, amp: float = 1) -> float:
	return amp*(abs(4*((x*freq)%1)-2)-1)

def sine(x: float, freq: float, amp: float = 1) -> float:
	return amp*sin(2*pi*x*freq)

def sawtooth(x: float, freq: float, amp: float = 1) -> float:
	return amp*(2*((x*freq)%1)-1)

# momomomomomomomo
def brown_noise(_: float, __: float, amp: float = 1) -> float:
	brown_noise.prev += 0.1*uniform(-amp-brown_noise.prev, amp-brown_noise.prev)
	return brown_noise.prev
brown_noise.prev = 0

def circle(x: float, freq: float, amp: float = 1) -> float:
	x *= 4*freq
	return amp*sin(acos(x%2 - 1))*(-1)**(x//2 % 2)

def white_noise(_: float, __: float, amp: float = 1) -> float:
	return amp*uniform(-1, 1)

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
		self.amplitude_data = amplitude_data if isinstance(amplitude_data, list) else list(amplitude_data) # type: List[float]
		self.sample_rate = sample_rate

	# properties

	@property
	def copy(self):
		# type (Wave) -> Wave
		return Wave(self.amplitude_data, self.sample_rate)

	@property
	def cut_off(self):
		"""force amplitude to be in [-1, 1]"""
		# type (Wave) -> Wave
		return Wave((i if abs(i) < 1 else copysign(1, i) for i in self.amplitude_data), self.sample_rate)

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
		return (0x7FFF * np.array(self.cut_off.amplitude_data)).astype(np.int16).tobytes()

	@property
	def reverse(self):
		# type: (Wave) -> Wave
		return Wave(self.amplitude_data[::-1], self.sample_rate)

	# double underscore methods

	def __add__(self, other):
		# type: (Wave, Wave) -> Wave
		"""concatenate two waves"""
		assert self.sample_rate == other.sample_rate
		return Wave(self.amplitude_data + other.amplitude_data, self.sample_rate)

	def __len__(self) -> int:
		"""sample points"""
		return len(self.amplitude_data)

	def __matmul__(self, other):
		# type: (Wave, Wave) -> Wave
		"""overlay two waves"""
		assert self.sample_rate == other.sample_rate
		a, b = self, other
		if len(a) < len(b):
			a = a.pad(len(b) - len(a))
		elif len(b) < len(a):
			b = b.pad(len(a) - len(b))
		return Wave((x + b.amplitude_data[i] for i, x in enumerate(a.amplitude_data)), self.sample_rate)

	def __mul__(self, other: float):
		# type: (Wave, float) -> Wave
		"""change amplitude of wave"""
		return Wave((i*other for i in self.amplitude_data), self.sample_rate)

	def __neg__(self):
		# type: (Wave) -> Wave
		"""negate amplitude"""
		return Wave((-x for x in self.amplitude_data), self.sample_rate)

	# methods

	def binary_harmonic(self, n: int = 3):
		# type (Wave, int) -> Wave
		"""bear in mind this function could as much as double the amplitude of the original wave"""
		o = self.copy
		for i in range(1, n+1):
			o @= self.set_speed(div=2**i).repeat(2**i) * 0.5**i
		return o

	def linear_falloff(self, t: float = 0.1):
		# type: (Wave, float) -> Wave
		"""to soften the end"""
		return self.reverse.linear_onset(t).reverse

	def linear_onset(self, t: float = 0.1):
		# type: (Wave, float) -> Wave
		"""to soften the beginning"""
		return Wave((x*(i/self.sample_rate if i/self.sample_rate < t else 1) for i, x in enumerate(self.amplitude_data)), self.sample_rate)

	def oscillate_amplitude(self, freq: int, min_amp: float = 0):
		# type: (Wave, int, float) -> Wave
		o = lambda x: (sin(2*pi*x*freq/self.sample_rate)+1+min_amp)/(2+min_amp)
		return Wave((x*o(i) for i, x in enumerate(self.amplitude_data)), self.sample_rate)

	def pad(self, n: int):
		# type: (Wave, int) -> Wave
		return Wave(self.amplitude_data+[0]*n, self.sample_rate)

	def repeat(self, count: int):
		# type: (Wave, int) -> Wave
		return Wave(self.amplitude_data*count, self.sample_rate)

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
	def from_function(f: Callable[[float, float, float], float], freq: int = 440,
			amp: float = 1, t: float = 1, sample_rate: int = 44100):
		# type: (Callable[[float, float, float], float], int, float, float, int) -> Wave
		return Wave((f(i/sample_rate, freq, amp) for i in range(int(t*sample_rate))), sample_rate)

	@staticmethod
	def sawtooth(freq: int = 440, amp: float = 1, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, float, int) -> Wave
		return Wave.from_function(sawtooth, freq, amp, t, sample_rate)

	@staticmethod
	def sine(freq: int = 440, amp: float = 1, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, float, int) -> Wave
		return Wave.from_function(sine, freq, amp, t, sample_rate)

	@staticmethod
	def square(freq: int = 440, amp: float = 1, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, float, int) -> Wave
		return Wave.from_function(square, freq, amp, t, sample_rate)

	@staticmethod
	def triangle(freq: int = 440, amp: float = 1, t: float = 1, sample_rate: int = 44100):
		# type: (int, float, float, int) -> Wave
		return Wave.from_function(triangle, freq, amp, t, sample_rate)
