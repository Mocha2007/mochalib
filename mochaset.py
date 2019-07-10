from mochamath import is_prime, sgn
# set and sequence library

function = type(lambda:1)
inf = float('inf')


def p_generator(quantity: int) -> int:
	"""Primes
	https://oeis.org/A000040"""
	i = 0
	n = 2
	while i < quantity:
		if is_prime(n):
			i += 1
			yield n
		n += 1


def n_generator(n: int) -> int:
	"""Natural numbers
	https://oeis.org/A000027"""
	i = 1
	while i < n:
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


def square_generator(n: int) -> int:
	"""Perfect Squares
	https://oeis.org/A000290"""
	for i in range(n):
		yield i**2


def square_inclusion(n: int) -> bool:
	r = n**.5
	return int(r) == r


class Function:
	"""Function X -> X"""
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		# todo memory

	# properties
	@property
	def f(self) -> function:
		return self.kwargs['f']

	# methods
	def domain_has(self, other) -> bool:
		"""Optimized function to determine inclusion of an element in the domain"""
		return self.kwargs['domain_has'](other)

	def range_has(self, other) -> bool:
		"""Optimized function to determine inclusion of an element in the range"""
		return self.kwargs['range_has'](other)


class Set(Function):
	"""Function X -> B"""
	# methods
	def contains(self, other) -> bool:
		return self.f(other)


class Interval(Set):
	# properties
	@property
	def bound_closedness(self) -> (bool, bool):
		if 'bound_closedness' in self.kwargs:
			return self.kwargs['bound_closedness']
		return 'min' in self.kwargs, 'max' in self.kwargs

	@property
	def inf(self):
		"""Infimum"""
		return self.kwargs['min'] if 'min' in self.kwargs else self.kwargs['inf']

	@property
	def max(self):
		return self.kwargs['max']

	@property
	def min(self):
		return self.kwargs['min']

	@property
	def sup(self):
		"""Supremum"""
		return self.kwargs['max'] if 'max' in self.kwargs else self.kwargs['sup']

	# double underscore methods
	def __contains__(self, other) -> bool:
		b, B = self.bound_closedness
		try:
			if b:
				if other < self.min:
					return False
			elif other <= self.min:
				return False
			if B:
				if other > self.max:
					return False
			elif other >= self.max:
				return False
		except TypeError:
			return False
		return True

	def __str__(self) -> str:
		b, B = self.bound_closedness
		m, M = '(['[b], ')]'[B]
		return m + str(self.inf) + ', ' + str(self.sup) + M


class Sequence(Function):
	"""Function N -> X"""
	# properties	
	@property
	def generator(self) -> function:
		return self.kwargs['generator']

	@property
	def is_finite(self) -> bool:
		"""Finite number of elements in sequence?"""
		if 'is_finite' in self.kwargs:
			return self.kwargs['is_finite']
		return False
	
	@property
	def set(self) -> Set:
		return Set(f=self.range_has)

	# double underscore methods
	def __contains__(self, other) -> bool:
		return self.range_has(other)

	def __eq__(self, other) -> bool:
		# not perfect, but still...
		return self.generator == other.generator

	def __getitem__(self, key: int) -> float:
		return list(self.generator(key))[-1]

	def __len__(self) -> int:
		return len(list(self.generator(inf))) if self.is_finite else 0

	def __str__(self) -> str:
		return '{' + str(list(self.generator(10)))[1:-1] + ', ...}'


P = Sequence(**{
	'generator': p_generator,
	'range_has': is_prime,
})
N = Sequence(**{
	'generator': n_generator,
	'range_has': lambda x: 0 < x and x % 1 == 0,
})
Z = Sequence(**{
	'generator': z_generator,
	'range_has': lambda x: x % 1 == 0,
})
R = Interval(**{
	'inf': -inf,
	'sup': inf,
})
Fib = Sequence(**{
	'generator': fib_generator,
	'range_has': fib_inclusion,
})
Squares = Sequence(**{
	'generator': square_generator,
	'range_has': square_inclusion,
})
unit_interval = Interval(**{
	'min': 0,
	'max': 1,
})
# todo Q