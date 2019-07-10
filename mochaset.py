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


# N = n_generator(inf)
# Z = z_generator(inf)


class Set:
	pass
	# methods
	# def __and__(self, other):
	# 	"""Intersection"""


class Sequence(Set):
	def __init__(self, generator: function, **kwargs):
		self.generator = generator
		self.kwargs = kwargs
	
	@property
	def inclusion(self) -> function: # -> function
		"""Optimized function to determine inclusion of an element in the sequence"""
		return self.kwargs['inclusion']
	
	# double underscore methods
	def __add__(self, other):
		assert not isinstance(other, Set)

		def new_generator(n: int):
			yield list(z_generator(n)) + other
		
		kwargs = {
			'inclusion': lambda x: self.inclusion(x - other)
		}

		return Sequence(new_generator, **kwargs)

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
# todo Q