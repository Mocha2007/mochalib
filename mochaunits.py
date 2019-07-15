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

	def __pow__(self, other):
		return self.multi ** other

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
		x = self.value
		if x < 0:
			return '-' + str(-self)
		if 'imperial' in self.tags:
			return self.imperial
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
		if x == 0:
			return '0 kg'
		if x < 1e-6:
			return str(x*1e9) + ' μg'
		if x < 1e-3:
			return str(x*1e6) + ' mg'
		if x < 1:
			return str(x*1e3) + ' g'
		if x < 1e3:
			return str(x) + ' kg'
		if x < 1e6:
			return str(x/1e3) + ' Mg'
		if x < 1e9:
			return str(x/1e6) + ' Gg'
		if x < 1e12:
			return str(x/1e9) + ' Tg'
		if x < 1e15:
			return str(x/1e12) + ' Pg'
		if x < 1e18:
			return str(x/1e15) + ' Eg'
		if x < 1e21:
			return str(x/1e18) + ' Zg'
		return str(x/1e21) + ' Yg'


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
		if self.value < minute:
			return str(x) + ' s'
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
		if self.value < 1e3*yr:
			return str(x/yr) + ' yr'
		if self.value < 1e6*yr:
			return str(x/1e3/yr) + ' kyr'
		if self.value < 1e9*yr:
			return str(x/1e6/yr) + ' Myr'
		return str(x/1e9/yr) + ' Gyr'

	# double underscore methods
	def __str__(self) -> str:
		x = self.value
		if x < 0:
			return '-' + str(-self)
		if x == 0:
			return '0 s'
		if 'imperial' in self.tags:
			return self.imperial
		if x < 1e-6:
			return str(x*1e9) + ' ns'
		if x < 1e-3:
			return str(x*1e6) + ' μg'
		if x < 1:
			return str(x*1e3) + ' ms'
		if x < 1e3:
			return str(x) + ' s'
		if x < 1e6:
			return str(x/1e3) + ' ks'
		if x < 1e9:
			return str(x/1e6) + ' Ms'
		if x < 1e12:
			return str(x/1e9) + ' Gs'
		if x < 1e15:
			return str(x/1e12) + ' Ts'
		if x < 1e18:
			return str(x/1e15) + ' Ps'
		if x < 1e21:
			return str(x/1e18) + ' Es'
		if x < 1e24:
			return str(x/1e21) + ' Zs'
		return str(x/1e24) + ' Ys'


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

	def __pow__(self, other):
		assert isinstance(other, int)
		return Multidimension(self.value**other, {t: other*i for t, i in self.dimensions.items()}, *self.tags)

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
