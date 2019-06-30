def is_number(x) -> bool:
	return type(x) in {int, float, complex}


class Dimension:
	def __init__(self, value, *tags):
		self.value = value
		self.tags = set(tags)

	# properties
	@property
	def multi(self):
		return Multidimension(self.value, {type(self): 1}, *self.tags)

	# double underscore methods
	def __abs__(self):
		return type(self)(abs(self.value), *self.tags)

	def __add__(self, other):
		assert type(self) == type(other)
		return type(self)(self.value+other.value, *self.tags)

	def __bool__(self) -> bool:
		return bool(self.value)

	def __complex__(self) -> complex:
		return complex(self.value)

	def __eq__(self, other) -> bool:
		assert type(self) == type(other)
		return self.value == other.value

	def __float__(self) -> float:
		return float(self.value)

	def __hash__(self) -> int:
		return hash(self.value)

	def __int__(self) -> int:
		return int(self.value)

	def __le__(self, other) -> bool:
		assert type(self) == type(other)
		return self.value <= other.value

	def __lt__(self, other) -> bool:
		assert type(self) == type(other)
		return self.value < other.value

	def __mul__(self, other): # returns either type(self) or Multidimension
		if isinstance(other, Dimension):
			return self.multi * other.multi
		if isinstance(other, Multidimension):
			return other * self
		return type(self)(self.value*other, *self.tags)

	def __neg__(self):
		return type(self)(-self.value, *self.tags)

	def __pos__(self):
		return self

	def __repr__(self) -> str:
		return '{0}({1}, *{2})'.format(type(self).__name__, self.value, self.tags)

	def __rmul__(self, other):
		return self * other

	def __rtruediv__(self, other):
		return other / self.multi

	def __sub__(self, other):
		assert type(self) == type(other)
		return type(self)(self.value-other.value, *self.tags)

	def __truediv__(self, other): # returns either type(self) or Multidimension
		if isinstance(other, Dimension):
			if type(self) == type(other):
				return self.value / other.value
			return self.multi / other.multi
		if isinstance(other, Multidimension): # call rtruediv of multidimension
			return self.multi / multi
		return type(self)(self.value/other, *self.tags)


class Length(Dimension):
	# properties
	@property
	def astro(self) -> str:
		x = self.value
		LD = 3.84402e8
		au = 1.495978707e11
		ly = 9.4607304725808e15
		if self.value < au:
			return str(x/LD) + ' LD'
		if self.value < ly:
			return str(x/au) + ' au'
		return str(x/ly) + ' ly'

	@property
	def imperial(self) -> str:
		x = self.value
		inch = 2.54e-2
		ft = 0.3048
		yd = 0.9144
		mi = 1609.344
		if self.value < ft:
			return str(x/inch) + ' in'
		if self.value < yd:
			return str(x/ft) + ' ft'
		if self.value < mi:
			return str(x/yd) + ' yd'
		return str(x/mi) + ' mi'

	# double underscore methods
	def __str__(self) -> str:
		if 'imperial' in self.tags:
			return self.imperial
		x = self.value
		if x == 0:
			return '0 m'
		if x < 1e-6:
			return str(x*1e9) + ' nm'
		if x < 1e-3:
			return str(x*1e6) + ' μm'
		if x < 1:
			return str(x*1e3) + ' mm'
		if x < 1e3:
			return str(x) + ' m'
		if x < 1e6:
			return str(x/1e3) + ' km'
		if x < 1e9:
			return str(x/1e6) + ' Mm'
		if 'astro' in self.tags:
			return self.astro
		if x < 1e12:
			return str(x/1e9) + ' Gm'
		return str(x/1e12) + ' Tm'


class Multidimension:
	def __init__(self, value: float, dimensions: dict, *tags):
		self.value = value
		self.dimensions = dimensions # type dict Class -> int
		self.tags = set(tags)

	# double underscore methods
	def __add__(self, other):
		assert self.dimensions == other.dimensions
		return Multidimension(self.value + other.value, self.dimensions, *self.tags)

	def __mul__(self, other):
		if isinstance(other, Dimension):
			return self * other.multi
		if isinstance(other, Multidimension):
			dimensions = self.dimensions.copy()
			for dimension, i in other.dimensions.items():
				if dimension in dimensions:
					dimensions[dimension] += i
				else:
					dimensions[dimension] = i
			return Multidimension(self.value * other.value, dimensions, *self.tags)
		return Multidimension(self.value*other, self.dimensions, *self.tags)

	def __neg__(self):
		return Multidimension(-self.value, self.dimensions, *self.tags)

	def __pos__(self):
		return self

	def __repr__(self) -> str:
		return 'Multivalue({0}, {1}, *{2})'.format(self.value, self.dimensions, self.tags)

	def __rtruediv__(self, other):
		return Multidimension(other, {}, *self.tags) / self

	def __sub__(self, other):
		return self + -other

	def __truediv__(self, other): # possibilities: other is number or dimension
		dimensions = self.dimensions.copy()
		if isinstance(other, Dimension):
			return self / other.multi
		if isinstance(other, Multidimension):
			for dimension in other.dimensions:
				if dimension in dimensions:
					dimensions[dimension] -= 1
				else:
					dimensions[dimension] = -1
			return Multidimension(other.value / self.value, dimensions, *self.tags)
		return Multidimension(other / self.value, dimensions, *self.tags)
