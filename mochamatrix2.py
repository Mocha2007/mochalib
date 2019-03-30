from math import acos, cos, sin
from random import randint
# Note to self:
# http://www.siafoo.net/article/57
# https://docs.python.org/3/library/operator.html


class Matrix:
	def __init__(self, data: list):
		self.data = data

	def __repr__(self) -> str:
		# todo align
		return '\n'.join(map(lambda x: str(x).replace(',', ''), self.data))

	# math ops

	def __abs__(self) -> float:
		# determinant
		assert len(self.data) == len(self[0])
		if len(self.data) == 1:
			return self[0][0]
		s = 0
		for i, datum in enumerate(self[0]):
			uh = self.delete_row(0).delete_col(i)
			s += datum * abs(uh) * (-1)**i
		return s

	def __add__(self, other):
		return Matrix([[datum + other[i][j] for j, datum in enumerate(row)] for i, row in enumerate(self.data)])

	def __invert__(self): # 1/A
		return 1/abs(self) * self.adjugate()

	def __neg__(self):
		return -1 * self

	def __mul__(self, other):
		assert len(self.data[0]) == len(other.data)
		newmatrix = []
		for row in self.data:
			newrow = []
			for col_id, _ in enumerate(other[0]):
				newrow.append(sum(datum * other[i][col_id] for i, datum in enumerate(row)))
			newmatrix.append(newrow)
		return Matrix(newmatrix)

	def __pow__(self, power, modulo=None):
		if power == 1:
			return self
		assert len(self.data) == len(self[0])
		if power == 0:
			return identity(len(self.data))
		if power < 0:
			return ~(self**-power)
		return self * self ** (power-1)

	def __rmul__(self, other):
		return Matrix([[other*datum for datum in row] for row in self.data])

	def __sub__(self, other):
		return self + -other

	def __truediv__(self, other):
		return self * ~other

	def adjugate(self):
		return self.cofactor().transpose()

	def cofactor(self):
		return Matrix([[abs(self.delete_row(i).delete_col(j)) * (-1)**(i+j) for j, _ in enumerate(row)] for i, row in enumerate(self.data)])

	def row_echelon(self):
		new_matrix = self.copy()
		# if 0,0 is 0, swap with first row that isn't 0
		zero_counter = 0
		while new_matrix.data[0][0] == 0:
			zero_counter += 1
			new_matrix.data.append(new_matrix.data.pop())
			if zero_counter == len(new_matrix.data):
				raise ValueError
		# divide row by lead
		new_matrix = new_matrix.multiply_row(0, 1/new_matrix[0][0])
		# if last row, return
		if len(new_matrix.data) == 1:
			return new_matrix
		# make later rows' zeroth zero
		for i, row in enumerate(new_matrix.data):
			if i == 0:
				continue
			new_matrix[i] = (Matrix([row]) - row[0] * Matrix([new_matrix.data[0]])).data[0]
		# we are left with something like
		# 1 2 3
		# 0 5 6
		# 0 8 9
		bottom_right = new_matrix.delete_col(0).delete_row(0).row_echelon()
		for i, row in enumerate(new_matrix.data):
			if i == 0:
				continue
			for j, _ in enumerate(new_matrix.data):
				if j == 0:
					continue
				new_matrix[i][j] = bottom_right[i-1][j-1]
		return new_matrix # todo test

	def transpose(self):
		newmatrix = zero(*self.size())
		for i, row in enumerate(self.data):
			for j, datum in enumerate(row):
				newmatrix[j][i] = datum
		return newmatrix

	# other ops
	def __and__(self, other):
		assert self.size() == other.size()
		return Matrix([[int(datum and other[i][j]) for j, datum in enumerate(row)] for i, row in enumerate(self.data)])

	def __bool__(self) -> bool:
		return self != zero(*self.size())

	def __eq__(self, other) -> bool:
		if not isinstance(other, Matrix):
			return False
		return self.data == other.data

	def __getitem__(self, item):
		return self.data[item]

	def __hash__(self) -> int:
		return hash(tuple(tuple(datum for datum in row) for row in self.data))

	def __or__(self, other):
		assert self.size() == other.size()
		return Matrix([[int(datum or other[i][j]) for j, datum in enumerate(row)] for i, row in enumerate(self.data)])

	def __pos__(self):
		return self

	def __setitem__(self, key, value):
		self.data[key] = value

	def __xor__(self, other):
		assert self.size() == other.size()
		return Matrix([[int(datum != other[i][j]) for j, datum in enumerate(row)] for i, row in enumerate(self.data)])

	def copy(self):
		return Matrix([[datum for datum in row] for row in self.data])

	def flip_cols(self):
		return self.flip_rows().transpose()

	def flip_rows(self):
		return Matrix(self.data[::-1])

	def get_col(self, n: int) -> list:
		return [row[n] for row in self.data]

	def multiply_row(self, n: int, multiple: float):
		return Matrix([[datum*[1, multiple][i == n] for datum in row] for i, row in enumerate(self.data)])

	def delete_col(self, n: int):
		return Matrix([[datum for i, datum in enumerate(row) if i != n] for row in self.data])

	def delete_row(self, n: int):
		return Matrix([row for i, row in enumerate(self.data) if i != n])

	def rre(self): # warning: taken from the old code, may not work or be pretty!
		matrix = self.row_echelon() # type: Matrix
		m, n = self.size()
		# check each row and check the column with the 1st 1
		for i in range(m):
			# check each column for the nonzero
			firstnonzero = -1
			for j in range(n):
				if matrix[i][j]:
					firstnonzero = j
					break
			# check each row above for other nonzeroes
			for j in range(i):
				if matrix[j][firstnonzero]:
					# make it zero!!!
					c = matrix[j][firstnonzero]
					# for each row element, subtract c*that one
					for k in range(n):
						matrix[j][k] -= c * matrix[i][k]
		return matrix

	def size(self) -> (int, int):
		return len(self.data), len(self.data[0])

	def solve_augment(self, augment):
		return augment / self

	def solve_markov(self):
		assert False not in [abs(sum(row)) <= 1 for row in self.data]
		old = 2 * self
		new = self.copy()
		while 2e-15 < abs((new - old)[0][0]):
			old = new.copy()
			new **= 2
		return Matrix([new.data[0]])

	def trace(self) -> float:
		assert len({self.size()}) == 1 # must be a square matrix
		return sum(self[i][i] for i in range(self.size()[0]))


class Vector:
	def __init__(self, data: list):
		self.data = data

	def __abs__(self) -> float:
		return sum(i**2 for i in self.data)

	def __add__(self, other):
		return Vector([dim + other[i] for i, dim in enumerate(self.data)])

	def __eq__(self, other) -> bool:
		return self.data == other.data

	def __getitem__(self, item) -> float:
		return self.data[item]

	def __hash__(self) -> int:
		return hash(tuple(self.data))

	def __mul__(self, other) -> float:
		return sum(dim * other[i] for i, dim in enumerate(self.data))

	def __neg__(self):
		return -1 * self

	def __repr__(self) -> str:
		return str(self.data).replace(",", "")

	def __rmul__(self, other: float):
		return Vector([i*other for i in self.data])

	def __setitem__(self, key, value):
		self.data[key] = value

	def __sub__(self, other):
		return self + -other

	def angle(self, other) -> float:
		return acos(self*other/abs(self)/abs(other))

	# def cross_product(self, other):
	# 	assert self.size() == other.size()
	# 	a1, a2, a3 = self.data
	# 	b1, b2, b3 = other.data
	# 	return Vector([a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1])

	def cross(self, *others):
		assert {i.size() for i in others} <= {self.size()} # all same size
		assert self.size() == len(others) + 2 # dimension - 1 matrices total
		adj = Matrix([i.data for i in list(others) + [self, Vector([0]*self.size())]]).adjugate()
		return -Vector(adj.get_col(-1))

	def normalize(self):
		return 1/abs(self) * self

	def size(self) -> int:
		return len(self.data)


def zero(m: int, n: int) -> Matrix:
	"""Produce the m x n zero matrix."""
	return Matrix([[0 for _ in range(n)] for _ in range(m)])


def identity(n: int) -> Matrix:
	"""Produce the n x n identity matrix."""
	return Matrix([[int(row == col) for col in range(n)] for row in range(n)])


def random(n: int) -> Matrix:
	"""Produce an n x n matrix containing random integers from -n to n."""
	return Matrix([[randint(-n, n) for _ in range(n)] for _ in range(n)])


def rotation(theta: float) -> Matrix:
	"""Produce the rotation matrix for the angle theta, given in radians."""
	return Matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


e333 = Matrix([[2, 1, 3], [-1, 4, -2], [3, 1, 5]])
markov_test = Matrix([
	[.9, .075, .025],
	[.15, .8, .05],
	[.25, .25, .5],
])
vector_a = Vector([1, 2, 3])
vector_b = Vector([4, 5, 6])
