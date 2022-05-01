"""For super basic functions missing from vanilla python I expect to use in multiple places..."""
from math import copysign, cos, inf, sin, tan
from typing import Any, Callable, Iterable

cbrt = lambda x: x**(1/3) if 0 <= x else -(-x)**(1/3)
cot = lambda x: 1/tan(x)
sec = lambda x: 1/cos(x)
sign = lambda x: copysign(1, x)
sinc = lambda x: sin(x)/x if x else 1

def average(*values: Iterable[float]) -> float:
	"""arithmetic mean of the given values"""
	return sum(values)/len(values)

def clamp(x: float, min: float, max: float) -> float:
	"""Force x to be in a specific range - uses min or max if x is outside"""
	return min if x < min else max if x < max else x

def linear_interpolation(x: float, xx: Iterable[float], yy: Iterable[float]) -> float:
	"""given xy coords, and an x value, use linear interpolation to find the estimated y value"""
	for i, x_tick in enumerate(xx[1:]):
		if x <= x_tick:
			return remap(x, xx[i], x_tick, yy[i], yy[i+1])
	raise ValueError(f"{x} not in [{xx[0]}, {xx[-1]}]")

def newton_raphson(x: float, f, f_, max_iter = 100) -> float:
	"""root-finding method

	measured as taking 100 microseconds avg. on my laptop
	to find the root of a random quadratic with random start
	"""
	try:
		while ((x_ := x - f(x)/f_(x)) != x and 0 < max_iter):
			x = x_
			max_iter -= 1
	except ZeroDivisionError: # todo fix this
		return x
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

_debug_timer_times: list[float] = []

def debug_timer(f: Callable) -> Callable:
	from time import time
	"""use this as a decorator to time a function"""
	def inner(*args, **kwargs) -> Any:
		global _debug_timer_times
		start = time()
		x = f(*args, **kwargs)
		_debug_timer_times.append(time() - start)
		return x
	return inner

def _test() -> None:
	"""Used for testing various functions in common.py"""
	from random import random
	for _ in range(1000):
		# finding the root of a random quadratic
		a = random()
		b = random()
		c = random()
		f = lambda x: a*x**2 + b*x + c
		f_ = lambda x: 2*a*x + b
		# @debug_timer
		@debug_range
		def test_helper() -> float:
			return newton_raphson(random(), f, f_)
		test_helper()
