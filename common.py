"""For super basic functions missing from vanilla python I expect to use in multiple places..."""
from math import ceil, copysign, cos, inf, log2, sin, tan
from typing import Any, Callable, Iterable
# try https://www.csestack.org/calling-c-functions-from-python/ for newton-raphson

cbrt: Callable[[float], float] = lambda x: x**(1/3) if 0 <= x else -(-x)**(1/3)
cot: Callable[[float], float]  = lambda x: 1/tan(x)
sec: Callable[[float], float]  = lambda x: 1/cos(x)
sign: Callable[[float], float]  = lambda x: copysign(1, x)
sinc: Callable[[float], float]  = lambda x: sin(x)/x if x else 1

def average(values: Iterable[float]) -> float:
	"""arithmetic mean of the given values"""
	try:
		return sum(values)/len(values)
	except TypeError: # tried to use a generator or something
		return average(list(values))

def bisection_method(x_min: float, x_max: float, f, max_iter = 20, threshold = 1e-10) -> float:
	"""root-finding method"""
	while 0 < max_iter:
		x = (x_min + x_max)/2
		fx = f(x)
		# check threshold
		if -threshold < fx < threshold:
			return x
		# otherwise, iterate
		if sign(fx) == sign(f(x_min)):
			x_min = x
		else:
			x_max = x
		max_iter -= 1
	return x

def ITP_method(a: float, b: float, f, threshold = 1e-10, k_1: float = 0.1, k_2: float = 2, n_0: float = 1) -> float:
	"""Root-finding method

	k1 > 0

	k2 in [1, 1+phi]
	
	n_0 non-negative"""
	# https://en.wikipedia.org/wiki/ITP_method
	# make sure a and b are ordered right
	ya = f(a)
	yb = f(b)
	if not (ya < 0 < yb):
		_ya = ya
		ya = yb
		yb = _ya
		_a = a
		a = b
		b = _a
	# Preprocessing
	n_12 = ceil(log2((b-a)/(2*threshold)))
	n_max = n_12 + n_0
	j = 0
	while 2*threshold < b-a:
		# input(f"{a}, {b}, {ya}, {yb}")
		# Calculating Parameters
		x_12 = (a+b)/2
		r = threshold * 2**(n_max-j) - (b-a)/2
		delta = k_1*(b-a)**k_2
		# Interpolation
		x_f = (yb*a - ya*b) / (yb - ya)
		# Truncation
		sigma = sign(x_12 - x_f)
		if delta <= abs(x_12 - x_f):
			x_t = x_f + sigma*delta
		else:
			x_t = x_12
		# Projection
		if abs(x_t - x_12) <= r:
			x_itp = x_t
		else:
			x_itp = x_12 - sigma*r
		# Updating Interval
		y_itp = f(x_itp)
		if y_itp > 0:
			b = x_itp
			yb = f(b)
		elif y_itp < 0:
			a = x_itp
			ya = f(a)
		else:
			return x_itp # functionally equivalent but simpler
	return (a+b)/2

def brents_method(a: float, b: float, f, max_iter = 20, threshold = 1e-10) -> float:
	"""Root-finding method. The root must be bracketed by a and b"""
	# https://en.wikipedia.org/wiki/Brent%27s_method
	# assert f(a)*f(b) < 0
	if abs(f(a)) < abs(fb := f(b)): # make sure |f(a)| is bigger
		_a = a
		a = b
		b = _a
	c = d = s = a
	mflag = True
	while 0 <= max_iter and threshold < abs(b-a) and (fb := f(b)) and f(s):
		if (fa := f(a)) != (fc := f(c)) and fb != fc:
			s = a*fb*fc / ((fa-fb)*(fa-fc)) \
				+ b*fa*fc / ((fb-fa)*(fb-fc)) \
				+ c*fa*fb / ((fc-fa)*(fc-fb))
		else:
			s = b - fb * (b-a) / (fb - fa)
		if not ((3*a+b)/4 <= s <= b or b <= s <= (3*a+b)/4) \
			or (mflag and abs(s-b) >= abs(b-c)/2) \
			or (not mflag and abs(s-b) >= abs(c-d)/2) \
			or (mflag and abs(b-c) < threshold) \
			or (not mflag and abs(c-d) < threshold):
			s = (a+b)/2
			mflag = True
		else:
			mflag = False
		d = c
		c = b
		if fa*f(s) < 0:
			b = s
		else:
			a = s
		if abs(f(a)) < abs(f(b)):
			_a = a
			a = b
			b = _a
		max_iter -= 1
	return s if fb else b

def clamp(x: float, min: float, max: float) -> float:
	"""Force x to be in a specific range - uses min or max if x is outside"""
	return min if x < min else max if x < max else x

# todo: annotate the types for f and such here and in newton_raphson
def halleys_method(x: float, f, f_, f__, max_iter = 20, threshold = 1e-10) -> float:
	"""root-finding method

	measured as taking 85 microseconds avg. on my laptop
	to find the root of a random quadratic with random start
	"""
	try:
		while 0 < max_iter and threshold < abs(x -
				(x_ := x - 2*f(x)*f_(x)/(2*f_(x)**2 - f(x)*f__(x)))):
			x = x_
			max_iter -= 1
	except ZeroDivisionError:
		return x
	return x

def linear_interpolation(x: float, xx: Iterable[float], yy: Iterable[float]) -> float:
	"""given xy coords, and an x value, use linear interpolation to find the estimated y value"""
	for i, x_tick in enumerate(xx[1:]):
		if x <= x_tick:
			return remap(x, xx[i], x_tick, yy[i], yy[i+1])
	raise ValueError(f"{x} not in [{xx[0]}, {xx[-1]}]")

def newton_raphson(x: float, f, f_, max_iter = 20, threshold = 1e-10) -> float:
	"""root-finding method

	measured as taking 53 microseconds avg. on my laptop
	to find the root of a random quadratic with random start
	"""
	try:
		while 0 < max_iter and threshold < abs(x - (x_ := x - f(x)/f_(x))):
			x = x_
			max_iter -= 1
	except ZeroDivisionError: # stuck in a flat zone of the function
		# try to move x by a miniscule amount
		return newton_raphson(x + threshold, f, f_, max_iter, threshold) \
			if f_(x + threshold) else x
	return x

def remap(val: float, min1: float, max1: float, min2: float = 0, max2: float = 1) -> float:
	"""applies a conformal map from one linear range to another"""
	range1, range2 = max1-min1, max2-min2
	return (val-min1)/range1 * range2 + min2

# debugging functions

_debug_range_max = -inf
_debug_range_min = inf

def debug_range(f: Callable[[Any], float]) -> Callable[[Any], float]:
	"""use this as a decorator to get min/max values for a function"""
	def inner(*args, **kwargs) -> float:
		global _debug_range_max, _debug_range_min
		x = f(*args, **kwargs)
		if _debug_range_max < x:
			_debug_range_max = x
		if x < _debug_range_min:
			_debug_range_min = x
		return x
	return inner

_debug_mag_range_max = -inf
_debug_mag_range_min = inf

def debug_mag_range(f: Callable[[Any], complex]) -> Callable[[Any], complex]:
	"""use this as a decorator to get min/max magnitudes for a function"""
	def inner(*args, **kwargs) -> float:
		global _debug_mag_range_max, _debug_mag_range_min
		x = f(*args, **kwargs)
		if _debug_mag_range_max < abs(x):
			_debug_mag_range_max = abs(x)
		if abs(x) < _debug_mag_range_min:
			_debug_mag_range_min = abs(x)
		return x
	return inner

_debug_timer_times: list[int] = [] # in ns

def debug_timer(f: Callable) -> Callable:
	from time import perf_counter_ns
	"""use this as a decorator to time a function"""
	def inner(*args, **kwargs) -> Any:
		global _debug_timer_times
		start = perf_counter_ns()
		x = f(*args, **kwargs)
		_debug_timer_times.append(perf_counter_ns() - start)
		return x
	return inner

def _test() -> None:
	"""Used for testing various functions in common.py"""
	from random import uniform
	from math import pi
	for _ in range(100000):
		# finding the theta (for mollweide) of a random latitude
		lat = uniform(-pi/2, pi/2)
		f = lambda x: 2*x + sin(2*x) - pi*sin(lat)
		f_ = lambda x: 2 + 2*cos(2*x)
		# f__ = lambda x: -4*sin(2*x)
		initial_guess = 0.071374*lat**3 + 0.756175*lat
		@debug_timer
		# @debug_range
		def test_helper() -> float:
			return newton_raphson(initial_guess, f, f_, threshold=1e-4)
		theta = test_helper()
		# print(f"{lat}\t{theta}")
	print(f"{average(_debug_timer_times)/1e3} μs")
