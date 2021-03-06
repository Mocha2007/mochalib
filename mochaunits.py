from copy import deepcopy
from math import floor, log10, pi

prefixes = {key-8: value for key, value in enumerate('yzafpnµm kMGTPEZY')}
prefixes[0] = '' # set up prefix dict
temperatures = {
	'celsius': ('C', 1, -273.15),
	'delisle': ('D', -3/2, 559.725),
	'fahrenheit': ('F', 9/5, -459.67),
	'kelvin': ('K', 1, 0),
	'newton': ('N', 1/3, -91.05),
	'planck': ('T_P', 7.058119e-33, 0),
	'rankine': ('R', 9/5, 0),
	'reaumur': ('Ré', 4/5, -218.52),
	'romer': ('Rø', 21/40, -135.90375),
	'urist': ('U', 9/5, 9508.33),
}


def get_si(value: float) -> (float, str):
	if value == 0:
		return 0, prefixes[0]
	index = floor(log10(value)/3)
	index = max(min(prefixes), min(max(prefixes), index))
	new_value = value / 10**(3*index)
	return new_value, prefixes[index]


# https://stackoverflow.com/a/10854034/2579798
def round_time(dt = None, round_to: int = 1):
	"""Round a datetime object to any time lapse in seconds
	dt : datetime.datetime object, default now.
	round_to : Closest number of seconds to round to, default 1 second.
	Author: Thierry Husson 2012 - Use it as you want but don't blame me.
	"""
	import datetime
	if dt == None:
		dt = datetime.datetime.now()
	seconds = (dt.replace(tzinfo=None) - dt.min).seconds
	rounding = (seconds+round_to/2) // round_to * round_to
	return dt + datetime.timedelta(0, rounding-seconds, -dt.microsecond)


def pretty_dim(multidim, rounding: int=3) -> str:
	"""Prettify dim"""
	val, unit = str(multidim).split(' ')
	val = round(float(val), rounding)
	if val % 1 == 0:
		val = int(val)
	return '{} {}'.format(val, unit)


class Dimension:
	def __init__(self, value, *tags):
		self.value = value
		self.tags = set(tags)

	# properties
	@property
	def copy(self):
		return deepcopy(self)

	@property
	def multi(self):
		return Multidimension(self.value, {type(self): 1}, *self.tags)

	# double underscore methods
	def __abs__(self):
		return type(self)(abs(self.value), *self.tags)

	def __add__(self, other):
		assert isinstance(self, type(other))
		return type(self)(self.value+other.value, *self.tags)

	def __bool__(self) -> bool:
		return bool(self.value)

	def __complex__(self) -> complex:
		return complex(self.value)

	def __eq__(self, other) -> bool:
		assert isinstance(self, type(other))
		return self.value == other.value

	def __float__(self) -> float:
		return float(self.value)

	def __hash__(self) -> int:
		return hash(self.value)

	def __int__(self) -> int:
		return int(self.value)

	def __le__(self, other) -> bool:
		assert isinstance(self, type(other))
		return self.value <= other.value

	def __lt__(self, other) -> bool:
		assert isinstance(self, type(other))
		return self.value < other.value

	def __mul__(self, other): # returns either type(self) or Multidimension
		if isinstance(other, Dimension):
			return self.multi * other.multi
		if isinstance(other, Multidimension):
			return self.multi * other
		return type(self)(self.value*other, *self.tags)

	def __neg__(self):
		return type(self)(-self.value, *self.tags)

	def __pos__(self):
		return self

	def __pow__(self, other):
		return self.multi ** other

	def __repr__(self) -> str:
		return '{}({}, *{})'.format(type(self).__name__, self.value, self.tags)

	def __rmul__(self, other):
		return self * other

	def __rtruediv__(self, other):
		return other / self.multi

	def __sub__(self, other):
		assert isinstance(self, type(other))
		return type(self)(self.value-other.value, *self.tags)

	def __truediv__(self, other): # returns either type(self) or Multidimension
		if isinstance(other, Dimension):
			if isinstance(self, type(other)):
				return self.value / other.value
			return self.multi / other.multi
		if isinstance(other, Multidimension): # call rtruediv of Multidimension
			return self.multi / other
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
		return '{} {}ly'.format(*get_si(x/ly))

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
		x = self.value
		if x < 0:
			return '-' + str(-self)
		if 'imperial' in self.tags:
			return self.imperial
		if 'astro' in self.tags and 3.84402e8 < x:
			return self.astro
		return '{} {}m'.format(*get_si(x))


class Mass(Dimension):
	# properties
	@property
	def astro(self) -> str:
		x = self.value
		m_m = 7.342e22
		m_e = 5.97237e24
		m_j = 1.8982e27
		m_s = 1.98847e30
		if self.value < m_e:
			return str(x/m_m) + ' Lunar Masses'
		if self.value < m_j:
			return str(x/m_e) + ' Earth Masses'
		if self.value < m_s:
			return str(x/m_j) + ' Jupiter Masses'
		return str(x/m_s) + ' Solar Masses'

	@property
	def imperial(self) -> str:
		x = self.value
		lb = .45359237
		oz = lb / 12
		if self.value < lb:
			return str(x/oz) + ' oz'
		return str(x/lb) + ' lb'

	# double underscore methods
	def __str__(self) -> str:
		x = self.value
		if x < 0:
			return '-' + str(-self)
		if 'imperial' in self.tags:
			return self.imperial
		if 1e23 < x and 'astro' in self.tags:
			return self.astro
		return '{} {}g'.format(*get_si(x*1000))


class Time(Dimension):
	# properties
	@property
	def imperial(self) -> str:
		x = self.value
		minute = 60
		h = 60*minute
		d = 24*h
		wk = 7*d
		yr = 365.2425*d
		mo = yr / 12
		if self.value < h:
			return str(x/minute) + ' min'
		if self.value < d:
			return str(x/h) + ' h'
		if self.value < wk:
			return str(x/d) + ' d'
		if self.value < mo:
			return str(x/wk) + ' wk'
		if self.value < yr:
			return str(x/mo) + ' mo'
		return '{} {}yr'.format(*get_si(x/yr))

	# double underscore methods
	def __str__(self) -> str:
		x = self.value
		if x < 0:
			return '-' + str(-self)
		if 'imperial' in self.tags and 60 <= x:
			return self.imperial
		return '{} {}s'.format(*get_si(x))


class Temperature(Dimension):
	# double underscore methods
	def __str__(self) -> str:
		for name, (sym, scalar, offset) in temperatures.items():
			if name in self.tags:
				return '{} {}{}'.format(scalar * self.value + offset, '°' if offset else '', sym)
		return '{} K'.format(self.value)


class Current(Dimension):
	# double underscore methods
	def __str__(self) -> str:
		x = self.value
		if x < 0:
			return '-' + str(-self)
		return '{} {}A'.format(*get_si(x))


class Angle(Dimension):
	# properties
	@property
	def degrees(self) -> str:
		x = self.value
		deg = pi/180
		arcmin = deg / 60
		arcsec = arcmin / 60
		if deg < self.value:
			return str(x/deg) + '°'
		if arcmin < self.value:
			return str(x/arcmin) + '′'
		if arcsec < self.value:
			return str(x/arcsec) + '″'
		return '{} {}as'.format(*get_si(x/arcsec))

	# double underscore methods
	def __str__(self) -> str:
		x = self.value
		if x < 0:
			return '-' + str(-self)
		if 'deg' in self.tags:
			return self.degrees
		return '{} {}rad'.format(*get_si(x))


quantities = [
	({Length: 1, Time: -2}, 'Acceleration', 'm/s^2'),
	({Length: 2, Mass: 1, Time: -1}, 'Angular Momentum', 'N*m*s'),
	({Length: 2}, 'Area', 'm^2'),
	({Length: -3, Mass: 1}, 'Density', 'kg/m^3'),
	({Length: 2, Mass: 1, Time: -2}, 'Energy', 'J'),
	({Length: 1, Mass: 1, Time: -2}, 'Force', 'N'),
	({Length: 1, Mass: 1, Time: -1}, 'Momentum', 'N*s'),
	({Time: -1}, 'Frequency', 'Hz'),
	({Length: 2, Mass: 1, Time: -3}, 'Power', 'W'),
	({Length: -1, Mass: 1, Time: -2}, 'Pressure', 'Pa'),
	({Length: 1, Time: -1}, 'Speed', 'm/s'),
	({Length: 3}, 'Volume', 'm^3'),
	# w/ temperature
	({Length: 2, Mass: 1, Temperature: -1, Time: -2}, 'Entropy', 'J/K'),
	# w/ current
	({Current: 1, Time: 1}, 'Electric Charge', 'C'),
	({Current: 2, Length: -2, Mass: -1, Time: 4}, 'Electrical Capacitance', 'F'),
	({Current: 2, Length: -2, Mass: -1, Time: 3}, 'Electrical Conductance', 'S'),
	({Current: -2, Length: 2, Mass: 1, Time: -2}, 'Electrical Inductance', 'H'),
	({Current: -2, Length: 2, Mass: 1, Time: -3}, 'Electrical Resistance', 'Ω'),
	({Current: -1, Length: 2, Mass: 1, Time: -2}, 'Magnetic Flux', 'Wb'),
	({Current: -1, Mass: 1, Time: -2}, 'Magnetic Induction', 'T'),
	({Current: -1, Length: 2, Mass: 1, Time: -3}, 'Voltage', 'V'),
]


class Multidimension:
	def __init__(self, value: float, dimensions: dict, *tags):
		self.value = value
		self.dimensions = dimensions # type dict Class -> int
		self.tags = set(tags)

	# properties
	@property
	def clean(self):
		"""Delete units with 0"""
		new = self.copy
		new.dimensions = {key: value for key, value in self.dimensions.items() if value}
		return new

	@property
	def copy(self):
		return deepcopy(self)

	@property
	def inverse(self):
		new = self.copy
		new.value = 1/self.value
		new.dimensions = {key: -value for key, value in self.dimensions.items()}
		return new

	# properties
	@property
	def quantity(self) -> str:
		"""Attempt to fetch the name"""
		for dim, name, _ in quantities:
			if dim == self.clean.dimensions:
				return name
		raise KeyError

	@property
	def unit(self) -> str:
		"""Attempt to fetch the unit"""
		for dim, _, unit in quantities:
			if dim == self.clean.dimensions:
				return unit
		raise KeyError(self.dimensions)

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

	def __pow__(self, other):
		assert isinstance(other, int)
		return Multidimension(self.value**other, {t: other*i for t, i in self.dimensions.items()}, *self.tags)

	def __repr__(self) -> str:
		return 'Multivalue({0}, {1}, *{2})'.format(self.value, self.dimensions, self.tags)

	def __rtruediv__(self, other):
		return Multidimension(other, {}, *self.tags) / self

	def __str__(self) -> str:
		x = self.value
		if x < 0:
			return '-' + str(-self)
		val, prefix = get_si(x)
		return '{} {}{}'.format(val, prefix, self.unit)

	def __sub__(self, other):
		return self + -other

	def __truediv__(self, other): # possibilities: other is number or dimension
		dimensions = self.dimensions.copy()
		if isinstance(other, Dimension):
			return self / other.multi
		if isinstance(other, Multidimension):
			return self * other.inverse
		return Multidimension(other / self.value, dimensions, *self.tags)
