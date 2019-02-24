from math import log, sin, cos, tan, asin, acos, atan, gcd

inf = float('inf')


def center(string: str, width: int) -> str:
	if width <= len(string):
		return string
	return ' '*((width-len(string))//2) + string


def can_apply_function(x) -> bool:
	for i in x.variables:
		if type(i) in evaluable:
			i = i.evaluate()
			if type(i) in evaluable:
				return False
		elif type(i) == Variable:
			return False
	return True


class Variable:
	def __init__(self, name: str):
		self.name = name
	
	def __repr__(self) -> str:
		return self.name


def function(**kwargs): # needs repr and f
	global evaluable

	class Function:
		def __init__(self, *variables):
			self.variables = list(variables)
	
		def __repr__(self) -> str:
			string = kwargs['repr'].format(*self.variables)
			if len(self.variables) == 1:
				return string
			return '('+string+')'

		def evaluate(self):
			can_apply = can_apply_function(self)
			variables = [i.evaluate() if type(i) in evaluable else i for i in self.variables]
			out = kwargs['f'](*variables) if can_apply else Function(*variables)
			return out.simplify() if out in evaluable else out

		def derivative(self, with_respect_to: Variable, *args):
			n = args[0] if args else 1
			# print('f =', self)
			out = kwargs['d'](self, with_respect_to)
			# print('f\' =', out)
			if n == 1:
				return out.simplify() if is_function(out) else out
			else:
				return out.derivative(with_respect_to, n-1) if is_function(out) else 0

		def integral(self, with_respect_to: Variable, *args):
			# is definite?
			if args:
				from_x, to_x = args
				return self.integral(with_respect_to).evaluated_from(with_respect_to, from_x, to_x)
			# thus indefinite
			if not contains_variable(self, with_respect_to):
				return Product(self, with_respect_to)
			# must contain x
			if type(self) in (Sum, Difference):
				a, b = self.variables
				return type(self)(get_integral(a, with_respect_to), get_integral(b, with_respect_to))
			if type(self) == Product:
				a, b = self.variables
				if {type(a), type(b)} == {Csc, Cot} and a.variables == b.variables:
					s = a.variables[0]
					c = get_linear(s, with_respect_to)[0]
					return Quotient(Difference(0, Csc(s)), c)
				if {type(a), type(b)} == {Sec, Tan} and a.variables == b.variables:
					s = a.variables[0]
					c = get_linear(s, with_respect_to)[0]
					return Quotient(Sec(s), c)
				if not contains_variable(a, with_respect_to): # a(...)
					return Product(a, b.integral(with_respect_to))
				if not contains_variable(a, with_respect_to): # (...)b
					return Product(b, a.integral(with_respect_to))
			elif type(self) == Quotient:
				a, b = self.variables
				if not contains_variable(a, with_respect_to): # a/...
					if is_linear(b, with_respect_to):  # c/(ax+b)
						s = b.variables
						c = get_linear(s, with_respect_to)[0]
						return Quotient(Product(a, Log(Abs(b))), c)
					elif type(b) == Sum: # a/(...+...)
						d1, d2 = b.variables
						if type(d1) == Power and not contains_variable(d2, with_respect_to): # a/(...^+u^2)
							if tuple(d1.variables) == (with_respect_to, 2):
								u = Power(d2, Quotient(1, 2))
								return Product(Quotient(a, u), Arctan(Quotient(with_respect_to, u)))
						if type(d2) == Power and not contains_variable(d1, with_respect_to): # a/(u^2+...^...)
							if tuple(d2.variables) == (with_respect_to, 2):
								u = Power(d1, Quotient(1, 2))
								return Product(Quotient(a, u), Arctan(Quotient(with_respect_to, u)))
					elif type(b) == Power and b.variables[1] == Quotient(1, 2): # a/sqrt(...)
						d1 = b.variables[0]
						if type(d1) == Difference and d1.variables[1] == Power(with_respect_to, 2): # a/sqrt(...-x^2)
							if not contains_variable(d1.variables[0], with_respect_to): # a/sqrt(u^2-x^2)
								u = Power(d1.variables[0], Quotient(1, 2))
								return Product(a, Arcsin(Quotient(with_respect_to, u)))
			elif type(self) == Power:
				a, b = self.variables
				if a == with_respect_to and not contains_variable(b, a):
					return Quotient(Power(a, Sum(b, 1)), Sum(b, 1))
				a, b = self.variables
				if type(a) == Csc and b == 2:
					s = a.variables[0]
					c = get_linear(s, with_respect_to)[0]
					return Quotient(Difference(0, Cot(s)), c)
				if type(a) == Sec and b == 2:
					s = a.variables[0]
					c = get_linear(s, with_respect_to)[0]
					return Quotient(Tan(s), c)
				if a == Euler and is_linear(b, with_respect_to): # e^(ax+b)
					s = b.variables[0]
					c = get_linear(s, with_respect_to)[0]
					return Quotient(self, c)
				if not contains_variable(a, with_respect_to) and is_linear(b, with_respect_to): # c^(ax+b)
					s = b.variables[0]
					c = get_linear(s, with_respect_to)[0]
					return Quotient(self, Product(c, Log(a)))
			# log
			elif type(self) == Log:
				s = self.variables[0]
				if is_linear(s, with_respect_to):
					a = get_linear(s, with_respect_to)[0]
					return Quotient(Difference(Product(s, self), s), a)
			# trig functions
			elif type(self) == Sin:
				s = self.variables[0]
				if is_linear(s, with_respect_to):
					a = get_linear(s, with_respect_to)[0]
					return Quotient(Difference(0, Cos(s)), a)
			elif type(self) == Cos:
				s = self.variables[0]
				if is_linear(s, with_respect_to):
					a = get_linear(s, with_respect_to)[0]
					return Quotient(Sin(s), a)
			elif type(self) == Tan:
				s = self.variables[0]
				if is_linear(s, with_respect_to):
					a = get_linear(s, with_respect_to)[0]
					return Quotient(Log(Abs(Sec(s))), a)
			elif type(self) == Sec:
				s = self.variables[0]
				if is_linear(s, with_respect_to):
					a = get_linear(s, with_respect_to)[0]
					return Quotient(Log(Abs(Sum(Sec(s), Tan(s)))), a)
			# todo https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
			# todo Section Indefinite integrals of inverse trigonometric functions
			raise ValueError('Unsolvable Integral')

		def let(self, **variables):
			new_self = type(self)(*self.variables)
			for i, variable in enumerate(new_self.variables):
				if type(variable) in evaluable:
					new_self.variables[i] = variable.let(**variables)
				elif type(variable) == Variable and variable.name in variables:
					new_self.variables[i] = variables[variable.name]
			return new_self.simplify()
		
		def simplify(self):
			# print('before:', self)
			a = self.variables[0]
			a = a.simplify() if is_function(a) else a
			# unary identities
			# log identity
			if type(self) == Log: # ln(1) -> 0
				if a == 1:
					return 0
				return self # otherwise, stay the same
			elif type(self) == Abs: # ln(1) -> 0
				if type(a) == int:
					return abs(a)
				return self # otherwise, stay the same
			elif type(self) in trig_functions:
				new = self
				if type(a) in trig_functions: # trig(trig(...))
					inner = a.variables[0]
					t = type(self), type(a)
					if t in {(Sin, Arcsin), (Cos, Arccos), (Tan, Arctan)}:
						new = inner
					elif t in {(Cos, Arccsc), (Sin, Arcsec), (Tan, Arccot)}:
						new = Quotient(Power(Difference(Power(inner, 2), 1), Quotient(1, 2)), inner)
					elif t in {(Cos, Arcsin), (Sin, Arccos)}:
						new = Power(Difference(1, Power(inner, 2)), Quotient(1, 2))
					elif t in {(Sin, Arctan), (Cos, Arccot)}:
						new = Quotient(inner, Power(Sum(1, Power(inner, 2)), Quotient(1, 2)))
					elif t in {(Cos, Arctan), (Sin, Arccot)}:
						new = Quotient(1, Power(Sum(1, Power(inner, 2)), Quotient(1, 2)))
					elif t in {(Sin, Arccsc), (Cos, Arcsec)}:
						new = Quotient(1, inner)
					elif t == (Tan, Arcsin):
						new = Quotient(inner, Power(Difference(1, Power(inner, 2)), Quotient(1, 2)))
					elif t == (Tan, Arccos):
						new = Quotient(Power(Difference(1, Power(inner, 2)), Quotient(1, 2)), inner)
					elif t == (Tan, Arccsc):
						new = Quotient(1, Power(Difference(Power(inner, 2), 1), Quotient(1, 2)))
					elif t == (Tan, Arcsec):
						new = Power(Difference(Power(inner, 2), 1), Quotient(1, 2))
				if type(a) == Quotient and a.variables[0] == 1:
					inner = a.variables[1]
					ta = type(self)
					if ta == Arcsin:
						new = Arccsc(inner)
					elif ta == Arccos:
						new = Arcsec(inner)
					elif ta == Arcsec:
						new = Arccos(inner)
					elif ta == Arccsc:
						new = Arcsin(inner)
				if new in evaluable:
					return new.simplify()
				return new
			# binary identities
			b = self.variables[1]
			b = b.simplify() if is_function(b) else b
			# additive identity
			if type(self) == Sum:
				if a == 0: # 0+b -> b
					return b
				if b == 0: # a+0 -> a
					return a
				if type(a) == Difference and a.variables[1] == b: # (m-b)+b
					return a.variables[0]
				if type(b) == Difference and b.variables[1] == a: # a+(m-a)
					return b.variables[0]
				if {type(a), type(b)} <= {float, int}:
					return a+b
				if type(a) == Power == type(b): # sin^2 + cos^2 = 1
					a_, ap = a.variables
					b_, bp = b.variables
					powers_are_two = ap == bp == 2
					if (type(a_), type(b_)) in ((Sin, Cos), (Cos, Sin)) and powers_are_two:
						a_interior = a_.variables[0]
						b_interior = b_.variables[0]
						if a_interior == b_interior:
							return 1
			# subtractive identity
			elif type(self) == Difference:
				if b == 0: # a-0 -> a
					return a
				if type(b) == Difference: # a-(m-n) -> a+(n-m)
					m, n = b.variables
					return Sum(a, Difference(n, m)).simplify()
				if type(a) == int == type(b):
					return a-b
			# multiplicative identity
			elif type(self) == Product:
				if a == 0 or b == 0: # 0x -> 0
					return 0
				if a == 1: # 1b -> b
					return b
				if b == 1: # 1a -> a
					return a
				if type(a) == Quotient == type(b): # (m/n)*(o/p)
					m, n = a.variables
					o, p = b.variables
					return Quotient(Product(m, o), Product(n, p)).simplify()
				if type(a) == Quotient: # (m/n)*b
					m, n = a.variables
					return Quotient(Product(b, m), n).simplify()
				if type(b) == Quotient: # a*(m/n)
					m, n = b.variables
					return Quotient(Product(a, m), n).simplify()
				if type(a) == int == type(b):
					return a*b
				# trig products
				t = {type(a), type(b)}
				if t <= trig_functions:
					new = self
					inner = b.variables[0]
					if t == {Sin, Csc}:
						return 1
					if t == {Cos, Sec}:
						return 1
					if t == {Tan, Cot}:
						return 1
					if t == {Sin, Cot}:
						new = Cos(inner)
					elif t == {Sin, Sec}:
						new = Tan(inner)
					elif t == {Cos, Tan}:
						new = Sin(inner)
					elif t == {Cos, Csc}:
						new = Cot(inner)
					elif t == {Tan, Csc}:
						new = Sec(inner)
					elif t == {Cot, Sec}:
						new = Csc(inner)
					return new.simplify()
			# division identity
			elif type(self) == Quotient:
				if a == 0: # 0/b -> 0
					if b == 0:
						return self # don't simplify
					return 0
				if b == 1: # a/1 -> a
					return a
				if a == b: # a/a -> 1
					return 1
				if type(a) == Product and a.variables[0] == b: # (m*n)/m
					return a.variables[1]
				if type(a) == Product and a.variables[1] == b: # (m*n)/n
					return a.variables[0]
				if type(a) == Quotient == type(b): # (m/n)/(o/p)
					m, n = a.variables
					o, p = b.variables
					return Quotient(Product(m, p), Product(n, o)).simplify()
				if type(a) == Quotient: # (m/n)/b
					m, n = a.variables
					return Quotient(m, Product(b, n)).simplify()
				if type(b) == Quotient: # a/(m/n)
					m, n = b.variables
					return Quotient(Product(a, n), m).simplify()
				if type(a) == int == type(b) and 1 < gcd(a, b):
					g = gcd(a, b)
					return Quotient(a//g, b//g).simplify()
				# trig division
				tb = type(b)
				if tb in trig_functions:
					inner = b.variables[0]
					if tb == Sin:
						return Product(a, Csc(inner))
					if tb == Cos:
						return Product(a, Sec(inner))
					if tb == Tan:
						return Product(a, Cot(inner))
					if tb == Cot:
						return Product(a, Tan(inner))
					if tb == Sec:
						return Product(a, Cos(inner))
					if tb == Csc:
						return Product(a, Sin(inner))
			# power identity
			elif type(self) == Power:
				if a == 0:
					return 0 if b else self # don't simplify
				if a == 1:
					return 1
				if b == 0: # a^0 = 1
					return 1
				if b == 1: # a^1 = a
					return a
				if type(a) == int == type(b):
					return a**b
				if type(a) == Power: # (m^n)^b
					m, n = a.variables
					return Power(m, Product(n, b)).simplify()
			elif type(self) == Equality:
				if type(a) == type(b) and is_function(a):
					# both are the same function, let's see if they share the same arguments
					if a.variables[0] == b.variables[0]:
						# first arguments are identical
						return Equality(a.variables[1], b.variables[1]).simplify()
					if a.variables[1] == b.variables[1]:
						# second arguments are identical
						return Equality(a.variables[0], b.variables[0]).simplify()
					if type(a) == Difference:
						a1, a2 = a.variables
						return Equality(a1, Sum(a2, b)).simplify()
					if type(b) == Difference:
						b1, b2 = b.variables
						return Equality(Sum(a, b2), b1).simplify()
					if type(a) == Quotient:
						a1, a2 = a.variables
						return Equality(a1, Product(a2, b)).simplify()
					if type(b) == Quotient:
						b1, b2 = b.variables
						return Equality(Product(a, b2), b1).simplify()
			# otherwise, stay the same
			# print('after:', self)
			return type(self)(a, b)
		
		def solve_for(self, x: Variable) -> set:
			if type(self) != Equality:
				raise AttributeError
			s = self.simplify()
			# check which side contains it
			a, b = s.variables
			# if already solved for, return
			if a == x or b == x:
				return {s}
			# alright, then solve!
			a_contains = contains_variable(a, x)
			b_contains = contains_variable(b, x)
			assert a_contains or b_contains
			if b_contains and not a_contains: # variable on RHS
				return Equality(b, a).solve_for(x)
			# LINEAR
			if is_linear(a, x) and is_linear(b, x):
				f = get_linear(a, x)
				g = get_linear(b, x)
				a, b = f[0]-g[0], f[1]-g[1]
				return {Quotient(Difference(0, b), Product(2, a)).simplify()}
			# QUADRATIC
			if is_quadratic(a, x) and is_quadratic(b, x):
				f = get_quadratic(a, x)
				g = get_quadratic(b, x)
				a, b, c = f[0]-g[0], f[1]-g[1], f[2]-g[2]
				return quadratic(a, b, c)
			# for now, we pretend as if a is only on the LHS
			# not accounted for: TT (Variable on both sides)
			assert not (b_contains and a_contains) # todo
			# general forms
			if type(a) == Sum:
				m, n = a.variables
				if contains_variable(m, x):
					a = m
					b = Difference(b, n)
				else:
					a = n
					b = Difference(b, m)
			elif type(a) == Difference:
				m, n = a.variables
				if contains_variable(m, x):
					a = m
					b = Sum(b, n)
				else:
					a = n
					b = Difference(m, b)
			elif type(a) == Product:
				m, n = a.variables
				if contains_variable(m, x):
					a = m
					b = Quotient(b, n)
				else:
					a = n
					b = Quotient(b, m)
			elif type(a) == Quotient:
				m, n = a.variables
				if contains_variable(m, x):
					a = m
					b = Product(b, n)
				else:
					a = n
					b = Quotient(m, b)
			elif type(a) == Power:
				m, n = a.variables
				if contains_variable(m, x):
					a = m
					b = Power(b, Quotient(1, n))
				else:
					a = n
					b = Quotient(Log(b), Log(m))
			elif type(a) == Log:
				a = a.variables[0]
				b = Power(Euler, b)
			# print(self)
			return Equality(a, b).solve_for(x)

		def tangent(self, with_respect_to: Variable, at):
			"""Finds the tangent line of an expression at the given point"""
			x, y = at, self.let(**{with_respect_to.name: at})
			slope = self.derivative(with_respect_to).let(**{with_respect_to.name: at})
			y_intercept = Difference(y, Product(slope, x))
			# print(x, y, slope, y_intercept)
			return Sum(Product(slope, with_respect_to), y_intercept).simplify()

		def limit(self, with_respect_to, at):
			"""Finds the limit of an expression, may return None, another expression, an int, or if LHS!=RHS a tuple"""
			# is it already possible anyway?
			try:
				lhs = self.let(**{with_respect_to.name: at})
				return lhs.evaluate() if type(lhs) in evaluable else lhs
			except ZeroDivisionError:
				pass
			# guess not

			# try to l'Hopital it
			if type(self) == Quotient:
				f, g = self.variables
				# get limit of f
				if type(f) not in evaluable:
					fl = f
				else:
					fl = f.limit(with_respect_to, at)
				# get limit of g
				if type(g) not in evaluable:
					gl = g
				else:
					gl = g.limit(with_respect_to, at)
				if fl == gl and {fl, gl} < {0, inf, -inf}: # f(x)=g(x)=0 or inf or -inf
					g_ = g.derivative(with_respect_to)
					if type(g_) == int:
						if g_:
							return Quotient(f.derivative(with_respect_to), g_).let(**{with_respect_to.name: at})
					elif g_.let(**{with_respect_to.name: at}): # g'(x) != 0
						return Quotient(f.derivative(with_respect_to), g_.limit(with_respect_to, at)).let(**{with_respect_to.name: at})

			h = 1
			lhs, rhs = None, None
			while h != 0:
				try:
					lhs = self.let(**{with_respect_to.name: at-h})
					rhs = self.let(**{with_respect_to.name: at+h})
				except ZeroDivisionError:
					pass
				h /= 2
			if type(lhs) in evaluable:
				lhs = lhs.evaluate()
			if type(rhs) in evaluable:
				rhs = rhs.evaluate()
			if lhs == rhs:
				return lhs
			return lhs, rhs

		def evaluated_from(self, with_respect_to: Variable, from_x: float, to_x: float):
			a = self.let(**{with_respect_to.name: to_x})
			b = self.let(**{with_respect_to.name: from_x})
			return Difference(a, b).simplify()

		def graph(self, with_respect_to: Variable, from_x: float, to_x: float):
			resolution_x = 33
			resolution_y = 22
			points = [[' ']*resolution_x for _ in range(resolution_y)]
			h = (to_x-from_x)/resolution_x
			x = from_x
			y = []
			slope_function = self.derivative(with_respect_to)
			slopes = []
			for i in range(resolution_x):
				# record y
				new_y = self.let(**{with_respect_to.name: x})
				if type(new_y) in evaluable:
					new_y = new_y.evaluate()
				y.append(new_y)
				# record slopes
				if type(slope_function) in evaluable:
					slope = slope_function.let(**{with_respect_to.name: x})
					if type(slope) in evaluable:
						slopes.append(slope.evaluate())
					else:
						slopes.append(slope)
				else:
					slopes.append(slope_function)
				# next step
				x += h
			ymin, ymax = min(y), max(y)
			# print(slope_function)
			# print(slopes)
			for i, point in enumerate(y):
				points_y = min(round(resolution_y*(point-ymin)/(ymax-ymin)), resolution_y-1)
				# print(points_y, i)
				if 0 < slopes[i]:
					char = '/'
				elif slopes[i] == 0:
					char = '-'
				elif slopes[i] < 1:
					char = '\\'
				else:
					char = '*'
				points[points_y][i] = char
			print(center('{0} for {1} in [{2}, {3}]'.format(self, with_respect_to, from_x, to_x), resolution_x))
			# todo x- and y-axis labels
			print('\n'.join([''.join(i) for i in points][::-1]))

		def critical_points(self, with_respect_to: Variable) -> set:
			f_ = self.derivative(with_respect_to)
			return Equality(f_, 0).solve_for(with_respect_to)

		def inflection_points(self, with_respect_to: Variable) -> set:
			f_ = self.derivative(with_respect_to, 2)
			return Equality(f_, 0).solve_for(with_respect_to)

		def local_extrema(self, with_respect_to: Variable) -> set:
			"""Returns set of local extrema, which may or may not also be global extrema."""
			f__ = self.derivative(with_respect_to, 2)
			cp = self.critical_points(with_respect_to)
			if type(f__) in evaluable:
				return set(filter(lambda x: f__.let(**{with_respect_to.name: x}), cp))
			return cp if f__ else set()

		def converges(self, with_respect_to: Variable) -> bool:
			s = self.simplify()
			if type(s) in {float, int, Variable}:
				return s == 0
			# limit of the summand
			if s.limit(with_respect_to, inf):
				return False
			# ratio test
			numerator = s.let(**{with_respect_to.name: Sum(with_respect_to, 1)}) # x -> x+1 for all x in f
			print(Quotient(numerator, s).limit(with_respect_to, inf))
			r = abs(Quotient(numerator, s).limit(with_respect_to, inf))
			if r < 1:
				return True
			if 1 < r:
				return False
			# else, inconclusive

		def estimate_integral(self, with_respect_to: Variable, from_x: float, to_x: float) -> float:
			resolution = 10000
			c = (from_x - to_x)/resolution
			boxes = []
			for i in range(resolution):
				value = self.let(**{with_respect_to.name: c*i + from_x})
				if type(value) in evaluable:
					boxes.append(value.evaluate())
				else:
					boxes.append(value)
			return sum(boxes)/resolution

		def average_value(self, with_respect_to: Variable, from_x: float, to_x: float):
			return Quotient(self.integral(with_respect_to, from_x, to_x), Difference(to_x, from_x)).simplify()
	evaluable.add(Function)
	return Function


def is_function(expression) -> bool:
	return expression.__class__.__name__ == 'Function'


def contains_variable(expression, variable: Variable) -> bool:
	if type(expression) in evaluable:
		for i in expression.variables:
			if contains_variable(i, variable):
				return True
	return expression == variable


def is_constant(expression, variable: Variable) -> bool:
	"""Returns True if the expression is a constant function of a variable, else False."""
	return not contains_variable(expression, variable)


def is_linear(expression, variable: Variable) -> bool:
	"""Returns True if the expression is a linear function of a variable (ie. ax+b), else False."""
	return is_constant(get_derivative(expression, variable), variable)


def is_quadratic(expression, variable: Variable) -> bool:
	"""Returns True if the expression is a quadratic function of a variable (ie. ax^2+bx+c), else False."""
	return is_constant(get_derivative(expression, variable, 2), variable)


def get_linear(expression, variable: Variable) -> tuple:
	assert is_linear(expression, variable)
	if not contains_variable(expression, variable):
		return 0, expression
	# else, it MUST contain the variable
	# sum/diff
	if type(expression) in (Sum, Difference):
		a, b = expression.variables
		gla, glb = get_linear(a, variable), get_linear(b, variable)
		return type(expression)(gla[0], glb[0]), type(expression)(gla[1], glb[1])
	# product
	if type(expression) == Product:
		a, b = expression.variables
		gla, glb = get_linear(a, variable), get_linear(b, variable)
		if contains_variable(a, variable):
			return Product(gla[0], glb[1]), Product(gla[1], glb[1])
		return Product(glb[0], gla[1]), Product(glb[1], gla[1])
	# quotient
	if type(expression) == Quotient:
		a, b = expression.variables
		assert type(b) == int # todo, complicated shit unimplemented
		gla, glb = get_linear(a, variable), get_linear(b, variable)
		return Quotient(gla[0], glb[1]), Quotient(gla[1], glb[1])
	# constant
	return 1, 0


def get_quadratic(expression, variable: Variable) -> tuple:
	assert is_quadratic(expression, variable)
	if not contains_variable(expression, variable):
		return 0, 0, expression
	# else, it MUST contain the variable
	# sum/diff
	if type(expression) in (Sum, Difference):
		a, b = expression.variables
		gla, glb = get_quadratic(a, variable), get_quadratic(b, variable)
		return tuple(type(expression)(gla[i], glb[i]) for i in range(len(gla)))
	# product
	if type(expression) == Product:
		a, b = expression.variables
		a_has, b_has = contains_variable(a, variable), contains_variable(b, variable)
		if a_has and b_has: # (ax+b)(cx+d)
			aa, bb = get_linear(a, variable)
			cc, dd = get_linear(b, variable)
			return Product(aa, cc), Sum(Product(aa, dd), Product(bb, cc)), Product(bb, dd)
		if a_has: # (ax^2+bx+c)(d)
			aa, bb, cc = get_quadratic(a, variable)
			return Product(aa, b), Product(bb, b), Product(cc, b)
		# (d)(ax^2+bx+c)
		aa, bb, cc = get_quadratic(b, variable)
		return Product(aa, a), Product(bb, a), Product(cc, a)
	# quotient
	if type(expression) == Quotient:
		a, b = expression.variables
		assert type(b) == int # todo, complicated shit unimplemented
		# (ax^2+bx+c)/(d)
		aa, bb, cc = get_quadratic(a, variable)
		return Quotient(aa, b), Quotient(bb, b), Quotient(cc, b)
	# power
	if type(expression) == Power: # MUST be EXACTLY x^2 by this point
		return 1, 0, 0
	# the constant x
	return 0, 1, 0


def quadratic(a, b, c) -> set:
	divisor = Product(2, a).simplify()
	left = Difference(0, b).simplify()
	discriminant = Power(Difference(Power(b, 2), Product(Product(4, a), c)), Quotient(1, 2)).simplify()
	return {Quotient(Difference(left, discriminant), divisor).simplify(),
			Quotient(Sum(left, discriminant), divisor).simplify()}


def get_derivative(x, with_respect_to: Variable, *args):
	if args:
		n = args[0]
	else:
		n = 1
	if n == 1:
		if is_function(x):
			return x.derivative(with_respect_to)
		return int(x == with_respect_to)
	return get_derivative(get_derivative(x, with_respect_to, n-1), with_respect_to)


def get_integral(x, with_respect_to: Variable):
	if is_function(x):
		return x.integral(with_respect_to)
	if x == with_respect_to:
		return Product(Quotient(1, 2), Power(x, 2))
	return Product(x, with_respect_to)


def sum_d(sum_object, with_respect_to): # Sum -> Sum; Difference -> Difference; chain rule unnecessary
	a, b = sum_object.variables
	# print('g =', a)
	# print('h =', b)
	a_ = get_derivative(a, with_respect_to)
	b_ = get_derivative(b, with_respect_to)
	# print('g\' =', a_)
	# print('h\' =', b_)
	return type(sum_object)(a_, b_)


def product_d(product_object, with_respect_to): # Product -> Sum; chain rule unnecessary
	a, b = product_object.variables
	# print('g =', a)
	# print('h =', b)
	a_ = get_derivative(a, with_respect_to)
	b_ = get_derivative(b, with_respect_to)
	# print('g\' =', a_)
	# print('h\' =', b_)
	return Sum(Product(a, b_), Product(a_, b))


def quotient_d(quotient_object, with_respect_to): # Quotient -> Quotient; chain rule unnecessary
	a, b = quotient_object.variables
	# print('g =', a)
	# print('h =', b)
	a_ = get_derivative(a, with_respect_to)
	b_ = get_derivative(b, with_respect_to)
	# print('g\' =', a_)
	# print('h\' =', b_)
	return Quotient(Difference(Product(a_, b), Product(a, b_)), Power(b, 2))


def power_d(power_object, with_respect_to): # Accounts for the chain rule now, don't worry
	a, b = power_object.variables
	if not (contains_variable(a, with_respect_to) or contains_variable(b, with_respect_to)):
		# neither are wrt; chain rule irrelevant
		return 0
	a_ = get_derivative(a, with_respect_to)
	if contains_variable(a, with_respect_to) and not contains_variable(b, with_respect_to):
		# base is wrt; accounts for chain rule
		if b == 0:
			return 0
		if b == 1:
			return a
		return Product(Product(b, Power(a, Difference(b, 1))), a_)
	if not contains_variable(a, with_respect_to) and contains_variable(b, with_respect_to):
		# power is wrt; accounts for chain rule
		if a in (0, 1):
			return 0
		b_ = get_derivative(b, with_respect_to)
		return Product(Product(Log(a), Power(a, b)), b_)
	b_ = get_derivative(b, with_respect_to)
	return Product(Power(a, Difference(b_, 1)), Sum(Product(b, a_), Product(Product(a, Log(a)), b_)))


def abs_d(abs_object, with_respect_to):
	# accounts for chain rule
	a = abs_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Quotient(Product(a, a_), Abs(a))


def ln_d(ln_object, with_respect_to):
	# accounts for chain rule
	a = ln_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Quotient(a_, a)


def eq_d(eq_object, with_respect_to):
	# accounts for chain rule
	a, b = eq_object.variables
	a_ = get_derivative(a, with_respect_to)
	b_ = get_derivative(b, with_respect_to)
	return Equality(a_, b_)


def sin_d(sin_object, with_respect_to):
	# accounts for chain rule
	a = sin_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Product(a_, Cos(a))


def cos_d(cos_object, with_respect_to):
	# accounts for chain rule
	a = cos_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Product(Difference(0, a_), Sin(a))


def tan_d(tan_object, with_respect_to):
	# accounts for chain rule
	a = tan_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Product(a_, Power(Sec(a), 2))


def cot_d(cot_object, with_respect_to):
	# accounts for chain rule
	a = cot_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Product(Difference(0, a_), Power(Csc(a), 2))


def sec_d(sec_object, with_respect_to):
	# accounts for chain rule
	a = sec_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Product(a_, Product(Tan(a), Sec(a)))


def csc_d(csc_object, with_respect_to):
	# accounts for chain rule
	a = csc_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Product(Difference(0, a_), Product(Cot(a), Csc(a)))


def arcsin_d(arcsin_object, with_respect_to):
	# accounts for chain rule
	a = arcsin_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Quotient(a_, Power(Difference(1, Power(a, 2)), Quotient(1, 2)))


def arccos_d(arccos_object, with_respect_to):
	# accounts for chain rule
	a = arccos_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Quotient(Difference(0, a_), Power(Difference(1, Power(a, 2)), Quotient(1, 2)))


def arctan_d(arctan_object, with_respect_to):
	# accounts for chain rule
	a = arctan_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Quotient(a_, Sum(Power(a, 2), 1))


def arccot_d(arccot_object, with_respect_to):
	# accounts for chain rule
	a = arccot_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Quotient(Difference(0, a_), Sum(Power(a, 2), 1))


def arcsec_d(arcsec_object, with_respect_to):
	# accounts for chain rule
	a = arcsec_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Quotient(a_, Product(Abs(a), Power(Difference(Power(a, 2), 1), Quotient(1, 2))))


def arccsc_d(arccsc_object, with_respect_to):
	# accounts for chain rule
	a = arccsc_object.variables[0]
	a_ = get_derivative(a, with_respect_to)
	return Quotient(Difference(0, a_), Product(Abs(a), Power(Difference(Power(a, 2), 1), Quotient(1, 2))))


evaluable = set()

# not sure if this accounts for chain rule yet... check after power rules are completed
Sum = function(f=(lambda a, b: a+b), repr='{0}+{1}', d=sum_d)
Difference = function(f=(lambda a, b: a-b), repr='{0}-{1}', d=sum_d)
Product = function(f=(lambda a, b: a*b), repr='{0}*{1}', d=product_d)
Quotient = function(f=(lambda a, b: a/b), repr='{0}/{1}', d=quotient_d)
Power = function(f=(lambda a, b: a**b), repr='{0}^{1}', d=power_d)
Abs = function(f=(lambda a: abs(a)), repr='|{0}|', d=abs_d)
Log = function(f=(lambda a: log(a)), repr='ln({0})', d=ln_d)
Sin = function(f=(lambda a: sin(a)), repr='sin({0})', d=sin_d)
Cos = function(f=(lambda a: cos(a)), repr='cos({0})', d=cos_d)
Tan = function(f=(lambda a: tan(a)), repr='tan({0})', d=tan_d)
Cot = function(f=(lambda a: 1/tan(a)), repr='cot({0})', d=cot_d)
Sec = function(f=(lambda a: 1/cos(a)), repr='sec({0})', d=sec_d)
Csc = function(f=(lambda a: 1/sin(a)), repr='csc({0})', d=csc_d)
Arcsin = function(f=(lambda a: asin(a)), repr='arcsin({0})', d=arcsin_d)
Arccos = function(f=(lambda a: acos(a)), repr='arccos({0})', d=arccos_d)
Arctan = function(f=(lambda a: atan(a)), repr='arctan({0})', d=arctan_d)
Arccot = function(f=(lambda a: atan(1/a)), repr='arccot({0})', d=arccot_d)
Arcsec = function(f=(lambda a: acos(1/a)), repr='arcsec({0})', d=arcsec_d)
Arccsc = function(f=(lambda a: asin(1/a)), repr='arccsc({0})', d=arccsc_d)
# todo hyperbolic functions
Equality = function(f=(lambda a, b: a == b), repr='{0} = {1}', d=eq_d)
Euler = Variable('e')
trig_functions = {
	Sin,
	Cos,
	Tan,
	Cot,
	Sec,
	Csc,
	Arcsin,
	Arccos,
	Arctan,
	Arccot,
	Arcsec,
	Arccsc,
}

# testing

# sum_test = Sum(1, 2)
# print(sum_test)
# print(sum_test.evaluate())
# variable_test = Variable('x')
# variable_test2 = Sum(variable_test, 1)
# print(variable_test2) # should be x+1
# print(variable_test2.evaluate()) # should be x+1
# print(variable_test2.let(x=1)) # should be 1+1
# # print(variable_test2.let(x=1).evaluate()) # should be 2
qa, qb, qc, qx, qy = [Variable(i) for i in 'abcxy']
# quadratic = Quotient(Sum(Difference(0, qb),
			# Power(Difference(Power(qb, 2), Product(Product(4, qa), qc)), Quotient(1, 2))), Product(2, qa))
# print(quadratic.let(a=0)) # should be 0/0
# print(quadratic.let(b=0)) # should be sqrt(-4ac)/(2a)
# print(quadratic.let(c=0)) # should be 0
# equality_test = Equality(Product(3, qa), 5)
# print(equality_test.solve_for(qa))
# equality_test_2 = Equality(quadratic, qx)
# print(equality_test_2.solve_for(qc))
# print('bing bing bong bing bong')
# print(quadratic.derivative(qc))
# print(Arctan(qx))
# print(Arctan(qx).derivative(qx))
# print(Arctan(qx).derivative(qx, 2))
# test = Power(Euler, Difference(Difference(2, Product(4, qx)), Product(8, Power(qx, 2))))
# test.limit(qx, inf).let(e=2.718).evaluate()
# test = Sum(Difference(Power(qx, 3), Product(3, Power(qx, 2))), Product(2, qx))
# test.graph(qx, -1/2, 5/2)
# test = Arctan(qx).derivative(qx)
# test.graph(qx, -4, 4)
# input()
# print(Difference(Power(qx, 2), 1).critical_points(qx))
# integrand = Quotient(1, Sum(1, Power(qx, 2)))
# print(integrand.integral(qx).simplify())
# WHY DOESNT THIS SIMPLIFY TO ARCTAN(X)?! SPENT AN HOUR DEBUGGING AND EVERYTHING INDICATES THE PROGRAM DOES SIMPLIFY, BUT REFUSES TO USE THE SIMPLIFIED VALUE?!
