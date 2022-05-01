"""For super basic functions missing from vanilla python I expect to use in multiple places..."""
from math import copysign, cos, sin, tan
from typing import Iterable

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
	"""root-finding method"""
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
