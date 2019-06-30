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

	# double underscore methods
	def __add__(self, other):
		return Length(self.value+other.value)

	def __neg__(self):
		return Length(-self.value)

	def __pos__(self):
		return self

	def __repr__(self) -> str:
		return 'Length({0})'.format(self.value)

	def __str__(self) -> str:
		x = self.value
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
