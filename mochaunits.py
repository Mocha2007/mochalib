class Length:
	def __init__(self, value, *tags):
		self.value = value
		self.tags = set(tags)

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
	def __add__(self, other):
		return Length(self.value+other.value)

	def __eq__(self, other) -> bool:
		return self.value == other.value

	def __le__(self, other) -> bool:
		return self.value <= other.value

	def __lt__(self, other) -> bool:
		return self.value < other.value

	def __neg__(self):
		return Length(-self.value)

	def __pos__(self):
		return self

	def __repr__(self) -> str:
		return 'Length({0})'.format(self.value)

	def __str__(self) -> str:
		if 'imperial' in self.tags:
			return self.imperial
		x = self.value
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

	def __sub__(self, other):
		return Length(self.value-other.value)
