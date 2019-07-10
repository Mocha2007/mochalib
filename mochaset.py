from mochamath import sgn
# set and sequence library

function = type(lambda:1)
inf = float('inf')


def n_generator(n: int) -> int:
	"""Natural numbers
	https://oeis.org/A000027"""
	i = 1
	while i <= n:
		yield i
		i += 1


def z_generator(n: int) -> int:
	"""Integers
	https://oeis.org/A001057"""
	i = 0
	count = 0
	while count <= n:
		count += 1
		yield i
		if sgn(i) == 1:
			i = -i
		elif sgn(i) == -1:
			i = 1-i
		else:
			i = 1


def fib_generator(n: int) -> int:
	"""Natural numbers
	https://oeis.org/A000045"""
	yield 0
	a, b = 0, 1
	i = 1
	while i < n:
		yield b
		a, b = b, a+b
		i += 1


def fib_inclusion(n: int) -> bool:
	if n % 1 or n < 0:
		return False
	i = 1
	while 1:
		f_n = list(fib_generator(i))
		if n in f_n:
			return True
		if n < max(f_n):
			return False
		i += 1


class Function:
	pass
	# methods
	# def __and__(self, other):
	# 	"""Intersection"""


class Sequence(Function):
	"""Functions N -> R"""

	def __init__(self, generator: function, **kwargs):
		self.generator = generator
		self.kwargs = kwargs
	
	@property
	def inclusion(self) -> function: # -> function
		"""Optimized function to determine inclusion of an element in the sequence"""
		return self.kwargs['inclusion']

	def __str__(self) -> str:
		return '{' + str(list(self.generator(10)))[1:-1] + ', ...}'
	
	# methods
	def contains(self, other) -> bool:
		return self.inclusion(other)


N = Sequence(n_generator, **{
	'inclusion': lambda x: 0 < x and x % 1 == 0
})
Z = Sequence(z_generator, **{
	'inclusion': lambda x: x % 1 == 0
})
Fib = Sequence(fib_generator, **{
	'inclusion': fib_inclusion
})
# todo Q