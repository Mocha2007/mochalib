from math import floor, log10
from typing import Dict

si_prefix = ' kMGTPEZY'
si_prefix2 = ' mµnpfazy'


def add_dims(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
	new_dict = {}
	for key in set(a) | set(b):
		new_dict[key] = 0
		if key in a:
			new_dict[key] += a[key]
		if key in b:
			new_dict[key] += b[key]
		if new_dict[key] == 0:
			del new_dict[key]
	return new_dict


class Dimension:
	def __init__(self, **dimensions):
		self.dimensions = dimensions # type: Dict[str, int]

	def __repr__(self) -> str:
		dimensions = [key.upper()+'^'+str(value) for key, value in sorted(self.dimensions.items(), key=lambda x: x[0])]
		return ' '.join(dimensions)

	def __bool__(self) -> bool:
		return bool(self.dimensions)

	def __invert__(self):
		return Dimension(**{key: -value for key, value in self.dimensions.items()})

	def __mul__(self, other):
		a, b = self.dimensions, other.dimensions
		new_dict = {}
		for key in set(a) | set(b):
			new_dict[key] = 0
			if key in a:
				new_dict[key] += a[key]
			if key in b:
				new_dict[key] += b[key]
			if new_dict[key] == 0:
				del new_dict[key]
		return Dimension(**new_dict)

	def __truediv__(self, other):
		return self * ~other


class Unit:
	def __init__(self, symbol: str = '', si_multiple: float = 1, dimension: Dimension = Dimension(**{})):
		self.symbol = symbol
		self.si_multiple = si_multiple
		self.dimension = dimension

	def __repr__(self) -> str:
		return self.symbol

	def __invert__(self):
		return Unit('/'+self.symbol, 1/self.si_multiple, ~self.dimension)

	def __mul__(self, other):
		dimension = self.dimension * other.dimension
		# if dimension:
		return Unit(self.symbol+other.symbol, self.si_multiple*other.si_multiple, dimension)
		# return self.si_multiple*other.si_multiple

	def __truediv__(self, other):
		return self * ~other

	def __floordiv__(self, other):
		assert self.dimension == other.dimension
		return (self/other).si_multiple

	def __pow__(self, power: int):
		assert isinstance(power, int)
		if power == 0:
			return Unit()
		if power == 1:
			return self
		if power < 0:
			return Unit() / self ** -power
		return self * self ** (power-1)


class Value:
	def __init__(self, value: float = 0, unit: Unit = Unit(), uncertainty: float = 0):
		self.value = value
		self.unit = unit
		self.uncertainty = abs(uncertainty)

	def __repr__(self) -> str:
		if self.uncertainty:
			return '{0} +/- {1} {2}'.format(self.value, self.uncertainty, self.unit.symbol)
		return '{0} {1}'.format(self.value, self.unit.symbol)

	def si_pretty(self) -> str:
		index = floor(log10(abs(self.value)))//3
		if index == 0:
			return self.__repr__()
		prefix = [si_prefix2, si_prefix][int(0 < index)][abs(index)]
		multiple = 10**(-3*index)
		return Value(self.value*multiple, Unit(prefix+self.unit.symbol), self.uncertainty*multiple).__repr__()

	def convert(self, desired_unit: Unit):
		assert self.unit.dimension == desired_unit.dimension
		scalar = self.unit // desired_unit
		return Value(self.value*scalar, desired_unit, self.uncertainty*scalar)

	def __neg__(self):
		return Value(-self.value, self.unit, self.uncertainty)

	def __add__(self, other):
		assert self.unit.dimension == other.unit.dimension
		scalar = other.unit // self.unit
		return Value(self.value+other.value*scalar, self.unit, self.uncertainty+other.uncertainty)

	def __sub__(self, other):
		return self + -other

	def __mul__(self, other):
		uncertainty = self.value * other.uncertainty + other.value * self.uncertainty
		return Value(self.value*other.value, self.unit*other.unit, uncertainty)

	def __truediv__(self, other):
		uncertainty = self.value*other.uncertainty-other.value*self.uncertainty
		uncertainty /= other.value**2 - other.uncertainty**2
		return Value(self.value/other.value, self.unit/other.unit, uncertainty)

	# def __pow__(self, power: int):
	# 	uncertainty = (self.value+self.uncertainty)**power # todo issues with uncertainty calculation
	# 	uncertainty -= (self.value-self.uncertainty)**power
	# 	return Value(self.value**power, self.unit**power, uncertainty/2)


mass = Dimension(m=1)
length = Dimension(l=1)
time = Dimension(t=1)


kilogram = Unit('kg', 1, mass)
meter = Unit('m', 1, length)
second = Unit('s', 1, time)
newton = kilogram * meter / second / second
newton.symbol = 'N'
joule = newton * meter
joule.symbol = 'J'
watt = joule / second
watt.symbol = 'W'
foot = Unit('ft', 0.3048, length)

# tests
# print(meter) # m
# print(second) # s
# print(meter*second) # ms
# print(meter/second) # m/s
# print(meter/second/second) # m/s/s
# print(meter/second**2) # m/ss
# print((meter/second/second).dimension) # m 1, s -2
# print((meter/second**2).dimension) # m 1, s -2
