from mochamath import is_prime, sgn
# set and sequence library

function = type(lambda:1)
inf = float('inf')


def p_generator(quantity: int) -> int:
	"""Primes
	https://oeis.org/A000040"""
	i = 0
	n = 2
	while i <= quantity:
		if is_prime(n):
			i += 1
			yield n
		n += 1


def n_generator(n: int) -> int:
	"""Natural numbers
	https://oeis.org/A000027"""
	i = 1
	while i <= n:
		yield i
		i += 1


def q_generator(quantity: int) -> int:
	"""Rationals (spiral method)"""
	from fractions import Fraction
	from math import gcd
	yield Fraction(0, 1)
	i = 1
	y, x = 1, 1
	while i <= quantity:
		q = Fraction(y, x)
		i += 1
		yield q
		if quantity < i:
			break
		i += 1
		yield -q
		while 1: # can't use break check b/c needs to run through at least once
			if y < x:
				y += 1
			else:
				x -= 1
			if x == 0:
				x, y = y+1, 1
			if gcd(x, y) == 1:
				break


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

	@property
	def inf(self):
		"""Infimum"""
		return self.kwargs['min'] if 'min' in self.kwargs else self.kwargs['inf']

	@property
	def limit_points(self): # -> Set
		"""Set of all limit points"""
		return self.kwargs['limit_points']

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

	# methods
	def domain_has(self, other) -> bool:
		"""Optimized function to determine inclusion of an element in the domain"""
		return self.kwargs['domain_has'](other)

	def range_has(self, other) -> bool:
		"""Optimized function to determine inclusion of an element in the range"""
		return self.kwargs['range_has'](other)


class Set(Function):
	"""Function X -> B"""
	# properties
	@property
	def is_bounded(self) -> bool:
		return self.range_mag < inf

	@property
	def is_closed(self):
		return self.kwargs['is_closed']

	@property
	def is_compact(self) -> bool:
		return self.is_bounded and self.is_closed

	@property
	def is_open(self):
		return self.kwargs['is_open']

	@property
	def is_perfect(self) -> bool:
		return self.isolated_points == Empty and self.is_closed

	@property
	def isolated_points(self):
		return self.kwargs['isolated_points']

	@property
	def range_mag(self) -> float:
		return abs(self.sup - self.min)

	# double underscore methods
	def __str__(self) -> str:
		if self == Empty:
			return 'Empty'
		raise NotImplementedError

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
	def isolated_points(self) -> Set:
		return Empty

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

	def __len__(self) -> float:
		return self.sup - self.inf

	def __str__(self) -> str:
		b, B = self.bound_closedness
		m, M = '(['[b], ')]'[B]
		return m + str(self.inf) + ', ' + str(self.sup) + M

	# methods
	def is_closed(self, other) -> bool:
		return all(self.bound_closedness)

	def is_open(self, other) -> bool:
		return not is_closed if len(self) else True


class Sequence(Function):
	"""Function N -> X"""
	# properties	
	@property
	def generator(self) -> function:
		return self.kwargs['generator']

	@property
	def max(self) -> bool:
		if 'max' in self.kwargs:
			return self.kwargs['max']
		if self.monotone_decreasing:
			return self[0]
		raise NotImplementedError

	@property
	def min(self) -> bool:
		if 'min' in self.kwargs:
			return self.kwargs['min']
		if self.monotone_increasing:
			return self[0]
		raise NotImplementedError

	@property
	def monotone(self) -> bool:
		return self.kwargs['monotone']

	@property
	def monotone_decreasing(self) -> bool:
		return self.monotone and self[0] > self[9]

	@property
	def monotone_increasing(self) -> bool:
		return self.monotone and self[0] < self[9]

	@property
	def is_finite(self) -> bool:
		"""Finite number of elements in sequence?"""
		if 'is_finite' in self.kwargs:
			return self.kwargs['is_finite']
		return False

	@property
	def set(self) -> Set:
		kwargs = self.kwargs.copy()
		kwargs['f'] = self.range_has
		try:
			kwargs['min'] = self.min
		except NotImplementedError:
			pass
		try:
			kwargs['max'] = self.max
		except NotImplementedError:
			pass
		out = Set(**kwargs)
		out.kwargs['isolated_points'] = out
		return out

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
		return '{' + ', '.join(map(str, self.generator(10))) + ', ...}'


Empty = Set(**{
	'generator': lambda n: [],
	'is_open': True,
	'range_has': lambda x: False,
})
Empty.kwargs['limit_points'] = Empty
Empty.kwargs['isolated_points'] = Empty
P = Sequence(**{
	'generator': p_generator,
	'is_open': False,
	'range_has': is_prime,
	'limit_points': Empty,
	'monotone': True,
	'sup': inf,
})
N = Sequence(**{
	'generator': n_generator,
	'is_open': False,
	'range_has': lambda x: 0 < x and x % 1 == 0,
	'limit_points': Empty,
	'monotone': True,
	'sup': inf,
})
Z = Sequence(**{
	'generator': z_generator,
	'is_open': False,
	'range_has': lambda x: x % 1 == 0,
	'limit_points': Empty,
	'inf': -inf,
	'sup': inf,
})
R = Interval(**{
	'inf': -inf,
	'is_open': True,
	'sup': inf,
})
R.kwargs['limit_points'] = R
R.kwargs['isolated_points'] = Empty
Q = Sequence(**{
	'generator': q_generator,
	'is_open': False,
	'range_has': lambda x: hasattr(x, 'numerator'),
	'limit_points': R,
	'inf': -inf,
	'sup': inf,
})
Fib = Sequence(**{
	'generator': fib_generator,
	'is_open': False,
	'range_has': fib_inclusion,
	'limit_points': Empty,
	'monotone': True,
	'sup': inf,
})
Squares = Sequence(**{
	'generator': square_generator,
	'is_open': False,
	'range_has': square_inclusion,
	'limit_points': Empty,
	'monotone': True,
	'sup': inf,
})
unit_interval = Interval(**{
	'min': 0,
	'max': 1,
})
unit_interval.kwargs['limit_points'] = unit_interval
# todo Q