from math import log, sin, cos, tan, asin, acos, atan
from fractions import gcd

def can_apply_function(x): # -> (bool, variables)
	can_apply = True
	for i in x.variables:
		if type(i) in evaluable:
			i = i.evaluate()
			if type(i) in evaluable:
				can_apply = False
		elif type(i) == Variable:
			can_apply = False
	return can_apply, x.variables


class Variable:
	def __init__(self, name: str):
		self.name = name
	
	def __repr__(self) -> str:
		return self.name

# class Sum:
	# def __init__(self, a, b):
		# self.a = a
		# self.b = b
	
	# def __repr__(self):
		# return str(self.a) + '+' + str(self.b)

	# def evaluate(self):
		# can_add, variables = can_apply_function(self)
		# a, b = variables
		# return a+b if can_add else Sum(a, b)

	# def let(self, **variables):
		# if type(self.a) in evaluable:
			# self.a = self.a.let(**variables)
		# elif type(self.a) == Variable and self.a.name in variables:
			# self.a = variables[self.a.name]

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
			can_apply, variables = can_apply_function(self)
			out = kwargs['f'](*variables) if can_apply else Function(*variables)
			return out.simplify() if out in evaluable else out

		def derivative(self, with_respect_to, *args):
			if type(with_respect_to) == Variable:
				with_respect_to = with_respect_to.name
			if len(args):
				n = args[0]
			else:
				n = 1
			# print('f =', self)
			out = kwargs['d'](self, with_respect_to)
			# print('f\' =', out)
			if n == 1:
				return out.simplify() if is_function(out) else out
			else:
				return out.derivative(with_respect_to, n-1) if is_function(out) else 0

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
				# todo https://en.wikipedia.org/wiki/Inverse_trigonometric_functions#Relationships_between_trigonometric_functions_and_inverse_trigonometric_functions
				return self
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
				if type(a) == int == type(b):
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
			# power identity
			elif type(self) == Power:
				if a in (0, 1):
					if a == b == 0:
						return self # don't simplify
					return a
				if b == 0: # a^0 = 1
					return 1
				if b == 1: # a^1 = a
					return a
				if type(a) == Power: # (m^n)^b
					m, n = a.variables
					return Power(m, Product(n, b)).simplify()
			# otherwise, stay the same
			# print('after:', self)
			return type(self)(a, b)
		
		def solve_for(self, x: str):
			if type(self) != Equality:
				raise AttributeError
			self = self.simplify()
			# check which side contains it
			a, b = self.variables
			# if already solved for, return
			if type(a) == Variable and a.name == x:
				return self
			if type(b) == Variable and b.name == x:
				return self
			# alright, then solve!
			a_contains = contains_variable(a, x)
			b_contains = contains_variable(b, x)
			if a_contains == b_contains == False: # variable absent from both sides
				return self # can't simplify
			if b_contains and not a_contains: # variable on RHS
				return type(self)(b, a).solve_for(x)
			# not accounted for: TT (Variable on both sides)
			if b_contains and a_contains:
				return self.simplify() # todo
			# for now, we pretend as if a is only on the LHS
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
			
	evaluable.add(Function)
	return Function

def is_function(expression) -> bool:
	return expression.__class__.__name__ == 'Function'

def contains_variable(expression, variable: str) -> bool:
	if type(expression) in evaluable:
		for i in expression.variables:
			if contains_variable(i, variable):
				return True
	elif type(expression) == Variable:
		if expression.name == variable:
			return True
	return False

def get_derivative(x, with_respect_to):
	return x.derivative(with_respect_to) if is_function(x) else (int(x.name == with_respect_to) if type(x) == Variable else 0)

def sum_d(sum_object, with_respect_to): # Sum -> Sum; Difference -> Difference; chain rule unnecessary
	a, b = sum_object.variables
	# print('g =', a)
	# print('h =', b)
	a_= get_derivative(a, with_respect_to)
	b_= get_derivative(b, with_respect_to)
	# print('g\' =', a_)
	# print('h\' =', b_)
	return type(sum_object)(a_, b_)

def product_d(product_object, with_respect_to): # Product -> Sum; chain rule unnecessary
	a, b = product_object.variables
	# print('g =', a)
	# print('h =', b)
	a_= get_derivative(a, with_respect_to)
	b_= get_derivative(b, with_respect_to)
	# print('g\' =', a_)
	# print('h\' =', b_)
	return Sum(Product(a, b_), Product(a_, b))

def quotient_d(quotient_object, with_respect_to): # Quotient -> Quotient; chain rule unnecessary
	a, b = quotient_object.variables
	# print('g =', a)
	# print('h =', b)
	a_= get_derivative(a, with_respect_to)
	b_= get_derivative(b, with_respect_to)
	# print('g\' =', a_)
	# print('h\' =', b_)
	return Quotient(Difference(Product(a_, b), Product(a, b_)), Power(b, 2))

def power_d(power_object, with_respect_to): # Accounts for the chain rule now, don't worry
	a, b = power_object.variables
	if not (contains_variable(a, with_respect_to) or contains_variable(b, with_respect_to)): # neither are wrt; chain rule irrelevant
		return 0
	a_= get_derivative(a, with_respect_to)
	if contains_variable(a, with_respect_to) and not contains_variable(b, with_respect_to): # base is wrt; accounts for chain rule
		if b == 0:
			return 0
		if b == 1:
			return a
		return Product(Product(b, Power(a, Difference(b, 1))), a_)
	if not contains_variable(a, with_respect_to) and contains_variable(b, with_respect_to): # power is wrt; accounts for chain rule
		if a in (0, 1):
			return 0
		b_= get_derivative(b, with_respect_to)
		return Product(Product(Log(a), Power(a, b)), b_)
	b_= get_derivative(b, with_respect_to)
	return Product(Power(a, Difference(b_, 1)), Sum(Product(b, a_), Product(Product(a, Log(a)), b_)))

def abs_d(abs_object, with_respect_to):
	# accounts for chain rule
	a = abs_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Quotient(Product(a, a_), Abs(a))

def ln_d(ln_object, with_respect_to):
	# accounts for chain rule
	a = ln_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Quotient(a_, a)

def eq_d(eq_object, with_respect_to):
	# accounts for chain rule
	a, b = ln_object.variables
	a_= get_derivative(a, with_respect_to)
	b_= get_derivative(b, with_respect_to)
	return Equality(a_, b_)

def sin_d(sin_object, with_respect_to):
	# accounts for chain rule
	a = sin_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Product(a_, Cos(a))

def cos_d(cos_object, with_respect_to):
	# accounts for chain rule
	a = cos_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Product(Difference(0, a_), Sin(a))

def tan_d(tan_object, with_respect_to):
	# accounts for chain rule
	a = tan_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Product(a_, Power(Sec(a), 2))

def cot_d(cot_object, with_respect_to):
	# accounts for chain rule
	a = cot_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Product(Difference(0, a_), Power(Csc(a), 2))

def sec_d(sec_object, with_respect_to):
	# accounts for chain rule
	a = sec_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Product(a_, Product(Tan(a), Sec(a)))

def csc_d(csc_object, with_respect_to):
	# accounts for chain rule
	a = csc_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Product(Difference(0, a_), Product(Cot(a), Csc(a)))

def arcsin_d(arcsin_object, with_respect_to):
	# accounts for chain rule
	a = arcsin_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Quotient(a_, Power(Difference(1, Power(a, 2)), Quotient(1, 2)))

def arccos_d(arccos_object, with_respect_to):
	# accounts for chain rule
	a = arccos_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Quotient(Difference(0, a_), Power(Difference(1, Power(a, 2)), Quotient(1, 2)))

def arctan_d(arctan_object, with_respect_to):
	# accounts for chain rule
	a = arctan_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Quotient(a_, Sum(Power(a, 2), 1))

def arccot_d(arccot_object, with_respect_to):
	# accounts for chain rule
	a = arctan_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Quotient(Difference(0, a_), Sum(Power(a, 2), 1))

def arcsec_d(arcsec_object, with_respect_to):
	# accounts for chain rule
	a = arcsec_object.variables[0]
	a_= get_derivative(a, with_respect_to)
	return Quotient(a_, Product(Abs(a), Power(Difference(Power(a, 2), 1), Quotient(1, 2))))

def arccsc_d(arccsc_object, with_respect_to):
	# accounts for chain rule
	a = arccsc_object.variables[0]
	a_= get_derivative(a, with_respect_to)
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
Arctan = function(f=(lambda a: atan(a)), repr='arctan({0})', d=arctan_d) # todo other inverse trig functions
Arccot = function(f=(lambda a: atan(1/a)), repr='arccot({0})', d=arccot_d)
Arcsec = function(f=(lambda a: acos(1/a)), repr='arcsec({0})', d=arcsec_d)
Arccsc = function(f=(lambda a: asin(1/a)), repr='arccsc({0})', d=arccsc_d)
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

sum_test = Sum(1, 2)
print(sum_test)
print(sum_test.evaluate())
variable_test = Variable('x')
variable_test2 = Sum(variable_test, 1)
print(variable_test2) # should be x+1
print(variable_test2.evaluate()) # should be x+1
print(variable_test2.let(x=1)) # should be 1+1
# print(variable_test2.let(x=1).evaluate()) # should be 2
qa, qb, qc, qx = Variable('a'), Variable('b'), Variable('c'), Variable('x')
quadratic = Quotient(Sum(Difference(0, qb), Power(Difference(Power(qb, 2), Product(Product(4, qa), qc)), Quotient(1, 2))), Product(2, qa))
print(quadratic.let(a=0)) # should be 0/0
print(quadratic.let(b=0)) # should be sqrt(-4ac)/(2a)
print(quadratic.let(c=0)) # should be 0
equality_test = Equality(Product(3, qa), 5)
print(equality_test.solve_for('a'))
equality_test_2 = Equality(quadratic, qx)
print(equality_test_2.solve_for('c'))
print('bing bing bong bing bong')
print(quadratic.derivative('c'))
print(Arctan(qx))
print(Arctan(qx).derivative('x'))
print(Arctan(qx).derivative('x', 2))