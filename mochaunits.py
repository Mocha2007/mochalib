class Length:
	def __init__(self, value):
		self.value = value

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
		if x < 1e3:
			return str(x) + ' m'
		if x < 1e6:
			return str(x/1e3) + ' km'
		if x < 1e9:
			return str(x/1e6) + ' Mm'
		return str(x/1e9) + ' Gm'

	def __sub__(self, other):
		return Length(self.value-other.value)
