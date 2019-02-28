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


class Unit:
	def __init__(self, symbol: str = '', si_multiple: float = 1, dimension: Dict[str, int] = {}):
		self.symbol = symbol
		self.si_multiple = si_multiple
		self.dimension = dimension

	def __repr__(self) -> str:
		return self.symbol

	def __mul__(self, other):
		dimension = add_dims(self.dimension, other.dimension)
		return Unit(self.symbol+other.symbol, self.si_multiple*other.si_multiple, dimension)

	def __truediv__(self, other):
		return self * Unit('/'+other.symbol, other.si_multiple, {key: -val for key, val in other.dimension.items()})

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

	def __add__(self, other):
		assert self.unit.dimension == other.unit.dimension
		return Value(self.value+other.value, self.unit, self.uncertainty+other.uncertainty)

	def __sub__(self, other):
		assert self.unit.dimension == other.unit.dimension
		return Value(self.value-other.value, self.unit, self.uncertainty+other.uncertainty)

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


kilogram = Unit('kg', 1, {'kg': 1})
meter = Unit('m', 1, {'m': 1})
second = Unit('s', 1, {'s': 1})
newton = kilogram * meter / second / second
newton.symbol = 'N'
joule = newton * meter
joule.symbol = 'J'
watt = joule / second
watt.symbol = 'W'

# tests
# print(meter) # m
# print(second) # s
# print(meter*second) # ms
# print(meter/second) # m/s
# print(meter/second/second) # m/s/s
# print(meter/second**2) # m/ss
# print((meter/second/second).dimension) # m 1, s -2
# print((meter/second**2).dimension) # m 1, s -2
