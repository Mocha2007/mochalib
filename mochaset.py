from mochamath import is_prime
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


def lucas_like(a0: int, a1: int) -> function:
	def output(n: int) -> int:
		a, b = a0, a1
		i = 0
		while i < n:
			yield a
			a, b = b, a+b
			i += 1
	return output


def lucas_like_inclusion(a0: int, a1: int) -> function:
	def output(n: int) -> bool:
		if n in list(lucas_like(a0, a1)(2)): # first 2 elements
			return True
		if n % 1 or n < 0:
			return False
		i = 1
		while 1:
			f_n = list(lucas_like(a0, a1)(i))
			if n in f_n:
				return True
			if n < max(f_n):
				return False
			i += 1
	return output


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
	def is_closed(self) -> bool:
		return all(self.bound_closedness)

	@property
	def is_open(self) -> bool:
		return not self.is_closed if len(self) else True

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
		kwargs['is_open'] = False
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

	# methods
	def index(self, other) -> int:
		"""Returns the index of the first instance"""
		if other not in self: # idiot-proofing
			raise ValueError('{} is not in sequence'.format(other))
		i = 0
		while self[i] != other:
			i += 1
		return i

	def list(self, n: int) -> tuple:
		"""First n items in sequence"""
		return tuple(self.generator(n))


Empty = Set(**{
	'generator': lambda n: [],
	'is_open': True,
	'range_has': lambda x: False,
})
Empty.kwargs['limit_points'] = Empty
Empty.kwargs['isolated_points'] = Empty
P = Sequence(**{
	'generator': p_generator,
	'range_has': is_prime,
	'limit_points': Empty,
	'monotone': True,
	'sup': inf,
})
N = Sequence(**{
	'generator': lambda n: [(yield i+1) for i in range(n)],
	'range_has': lambda x: 0 < x and x % 1 == 0,
	'limit_points': Empty,
	'monotone': True,
	'sup': inf,
})
Z = Sequence(**{
	'generator': lambda n: [(yield (1-(-1)**i*(2*i+1))//4) for i in range(n)],
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
	'range_has': lambda x: hasattr(x, 'numerator'),
	'limit_points': R,
	'inf': -inf,
	'sup': inf,
})
unit_interval = Interval(**{
	'min': 0,
	'max': 1,
})
unit_interval.kwargs['limit_points'] = unit_interval

# less-useful sequences from OEIS
from math import e, factorial
from mochamath import divisors, totient

# all zeroes
A000004 = Sequence(**{
	'generator': lambda n: [(yield 0) for _ in range(n)],
	'range_has': lambda n: n == 0,
	'limit_points': Empty,
	'monotone': True,
	'min': 0,
	'max': 0,
})

# d(n) (also called tau(n) or sigma_0(n)), the number of divisors of n. 
A000005 = Sequence(**{
	'generator': lambda n: [(yield len(divisors(i))) for i in range(1, n+1)],
	'range_has': lambda n: n in N,
	'limit_points': Empty,
	'monotone': False,
	'min': 1,
	'sup': inf,
})

# Integer part of square root of n-th prime. 
A000006 = Sequence(**{
	'generator': lambda n: [(yield int(P[i]**.5)) for i in range(n)],
	'range_has': lambda n: n in N,
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'sup': inf,
})

# Euler totient function phi(n): count numbers <= n and prime to n. 
A000010 = Sequence(**{
	'generator': lambda n: [(yield totient(i)) for i in range(1, n+1)],
	'range_has': lambda n: 0 < n and (n == 1 or n % 2 == 0),
	'limit_points': Empty,
	'monotone': False,
	'min': 1,
	'sup': inf,
})

# The simplest sequence of positive numbers: the all 1's sequence. 
A000012 = Sequence(**{
	'generator': lambda n: [(yield 1) for _ in range(n)],
	'range_has': lambda n: n == 1,
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'max': 1,
})

A000027 = N

# Initial digit of n. 
A000030 = Sequence(**{
	'generator': lambda n: [(yield int(str(i)[0])) for i in range(n)],
	'range_has': lambda n: 0 <= n <= 9 and n % 1 == 0,
	'limit_points': Empty,
	'monotone': False,
	'min': 0,
	'max': 9,
})

# Lucas numbers beginning at 2: L(n) = L(n-1) + L(n-2), L(0) = 2, L(1) = 1. 
A000032 = Sequence(**{
	'generator': lucas_like(2, 1),
	'range_has': lucas_like_inclusion(2, 1),
	'limit_points': Empty,
	'monotone': False,
	'sup': inf,
})

# Period 2: repeat [1, 2]; a(n) = 1 + (n mod 2). 
A000034 = Sequence(**{
	'generator': lambda n: [(yield 1 + (i % 2)) for i in range(n)],
	'range_has': lambda n: n in (1, 2),
	'limit_points': Empty,
	'monotone': False,
	'min': 1,
	'max': 2,
})

# Period 2: repeat [0, 1]; a(n) = n mod 2; parity of n. 
A000035 = Sequence(**{
	'generator': lambda n: [(yield i % 2) for i in range(n)],
	'range_has': lambda n: n in (0, 1),
	'limit_points': Empty,
	'monotone': False,
	'min': 0,
	'max': 1,
})

# A000037, ..., 

# Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1. 
A000045 = Sequence(**{
	'generator': lucas_like(0, 1),
	'range_has': lucas_like_inclusion(0, 1),
	'limit_points': Empty,
	'monotone': True,
	'sup': inf,
})

# Powers of 2: a(n) = 2^n. 
A000079 = Sequence(**{
	'generator': lambda n: [(yield 2**i) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'sup': inf,
})
A000079.kwargs['range_has'] = lambda n: n == 1 or (1 < n and n % 2 == 0 and n//2 in A000079)

# Catalan numbers: C(n) = binomial(2n,n)/(n+1) = (2n)!/(n!(n+1)!). Also called Segner numbers. 
A000108 = Sequence(**{
	'generator': lambda n: [(yield factorial(2*i) // (factorial(i) * factorial(i+1))) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'sup': inf,
})

# Factorial numbers: n! = 1*2*3*4*...*n (order of symmetric group S_n, number of permutations of n letters). 
A000142 = Sequence(**{
	'generator': lambda n: [(yield factorial(i)) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'sup': inf,
})

# a(n) = floor(e^n). 
A000149 = Sequence(**{
	'generator': lambda n: [(yield int(e**i)) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'sup': inf,
})

# Triangular numbers: a(n) = binomial(n+1,2) = n(n+1)/2 = 0 + 1 + 2 + ... + n. 
A000217 = Sequence(**{
	'generator': lambda n: [(yield i * (i+1) // 2) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 0,
	'sup': inf,
})

# a(n) = 2^n - 1. (Sometimes called Mersenne numbers, although that name is usually reserved for A001348.) 
A000225 = Sequence(**{
	'generator': lambda n: [(yield 2**i - 1) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 0,
	'sup': inf,
})

# Powers of 3. 
A000244 = Sequence(**{
	'generator': lambda n: [(yield 3**i) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'sup': inf,
})
A000244.kwargs['range_has'] = lambda n: n == 1 or (1 < n and n % 3 == 0 and n//3 in A000244)

# The squares: a(n) = n^2. 
A000290 = Sequence(**{
	'generator': lambda n: [(yield i**2) for i in range(n)],
	'range_has': lambda x: x**.5 % 1 == 0,
	'limit_points': Empty,
	'monotone': True,
	'sup': inf,
})

# Tetrahedral (or triangular pyramidal) numbers: a(n) = C(n+2,3) = n*(n+1)*(n+2)/6. 
A000292 = Sequence(**{
	'generator': lambda n: [(yield (i*(i+1)*(i+2))//6) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 0,
	'sup': inf,
})

# Square pyramidal numbers: a(n) = 0^2 + 1^2 + 2^2 + ... + n^2 = n*(n+1)*(2*n+1)/6. 
A000330 = Sequence(**{
	'generator': lambda n: [(yield (i*(i+1)*(2*i+1))//6) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 0,
	'sup': inf,
})

def perfect_generator(quantity: int) -> int:
	if quantity:
		yield 6
	n = 28
	i = 1
	while i < quantity:
		if sum(divisors(n) - {n}) == n:
			i += 1
			yield n
		n += 18 # perfect numbers mod 18 == 10 for all > 6

# Perfect numbers n: n is equal to the sum of the proper divisors of n. 
A000396 = Sequence(**{
	'generator': perfect_generator,
	'limit_points': Empty,
	'monotone': True,
	'min': 6,
	'sup': inf,
})

# The cubes: a(n) = n^3. 
A000578 = Sequence(**{
	'generator': lambda n: [(yield i**3) for i in range(n)],
	'range_has': lambda n: n**(1/3) % 1 == 0,
	'limit_points': Empty,
	'monotone': True,
	'min': 0,
	'sup': inf,
})

# a(n) = 2^(2^n). 
A001146 = Sequence(**{
	'generator': lambda n: [(yield 2**2**i) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'sup': inf,
})
A001146.kwargs['range_has'] = lambda n: n == 2 or (1 < n and n**.5 % 1 == 0 and int(n**.5) in A001146)

def abundant_generator(quantity: int) -> int:
	n = 12
	i = 0
	while i < quantity:
		if n < sum(divisors(n) - {n}):
			i += 1
			yield n
		n += 1

# Abundant numbers (sum of divisors of n exceeds 2n). 
A005101 = Sequence(**{
	'generator': abundant_generator,
	'limit_points': Empty,
	'monotone': True,
	'min': 12,
	'sup': inf,
})

# The nonnegative even numbers: a(n) = 2n. 
A005408 = Sequence(**{
	'generator': lambda n: [(yield 2*i+1) for i in range(n)],
	'range_has': lambda n: 0 < n and n % 2 == 1,
	'limit_points': Empty,
	'monotone': True,
	'min': 1,
	'sup': inf,
})

# The nonnegative even numbers: a(n) = 2n. 
A005843 = Sequence(**{
	'generator': lambda n: [(yield 2*i) for i in range(n)],
	'range_has': lambda n: 0 <= n and n % 2 == 0,
	'limit_points': Empty,
	'monotone': True,
	'min': 0,
	'sup': inf,
})

# Fermat primes: primes of the form 2^(2^k) + 1, for some k >= 0.
A019434 = Sequence(**{
	'generator': lambda n: [(yield 2**(2**i) + 1) for i in range(n)],
	'limit_points': Empty,
	'monotone': True,
	'min': 3,
	'sup': inf,
})

# Powers of 2: a(n) = 2^n. 
A033999 = Sequence(**{
	'generator': lambda n: [(yield (-1)**i) for i in range(n)],
	'range_has': lambda n: n in {-1, 1},
	'limit_points': Empty,
	'monotone': False,
	'min': -1,
	'max': 1,
})