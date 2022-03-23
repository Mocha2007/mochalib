from math import acos, asin, atan, copysign, cos, floor, log, pi, radians, sin, tan
from PIL import Image
from typing import Iterable, Tuple

def clamp(x: float, min: float, max: float) -> float:
	if x < min:
		return min
	if max < x:
		return max
	return x

debug_max_x = 0
debug_max_y = 0

def debug_max(f):
	"""use this as a decorator to get max x/y"""
	def inner(lat: float, lon: float) -> Tuple[float, float]:
		global debug_max_x, debug_max_y
		y, x = f(lat, lon)
		if debug_max_x < abs(x):
			debug_max_x = abs(x)
		if debug_max_y < abs(y):
			debug_max_y = abs(y)
		return y, x
	return inner

def linear_interpolation(x: float, xx: Iterable[float], yy: Iterable[float]) -> float:
	for i, x_tick in enumerate(xx[1:]):
		if x <= x_tick:
			return remap(x, xx[i], x_tick, yy[i], yy[i+1])
	raise ValueError(f"{x} not in [{xx[0]}, {xx[-1]}]")

def newton_raphson(x: float, f, f_, max_iter = 100) -> float:
	while ((x_ := x - f(x)/f_(x)) != x and 0 < max_iter):
		x = x_
		max_iter -= 1
	return x

cbrt = lambda x: x**(1/3) if 0 <= x else -(-x)**(1/3)
cot = lambda x: 1/tan(x)
sec = lambda x: 1/cos(x)
sign = lambda x: copysign(1, x)
sinc = lambda x: sin(x)/x if x else 1

def remap(val: float, min1: float, max1: float, min2: float = 0, max2: float = 1) -> float:
	range1, range2 = max1-min1, max2-min2
	return (val-min1)/range1 * range2 + min2

# projections

def azimuthal_equidistant(lat: float, lon: float) -> Tuple[float, float]:
	rho = pi/2 - lat
	x = rho * sin(lon)
	y = -rho * cos(lon)
	# remap to [-1, 1] for both
	x /= pi
	y /= pi
	return y, x

def equal_earth(lat: float, lon: float) -> Tuple[float, float]:
	a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.0003796
	theta = asin(3**.5 / 2 * sin(lat))
	x = 2*3**.5 * lon * cos(theta)
	x /= 3 * (9*a4*theta**8 + 7*a3*theta**6 + 3*a2*theta**2 + a1)
	y = a4*theta**9 + a3*theta**7 + a2*theta**3 + a1*theta
	# remap to [-1, 1] for both
	x /= 2.7066297319215575 # found experimentally
	y /= 1.312188760937488 # found experimentally
	return y, x

def eu4(lat: float, lon: float) -> Tuple[float, float]:
	# the eu4 map is mostly miller, but:
	# (1) the poles are trimmed
	if not radians(-56) < lat < radians(72):
		return 1, 1
	# (2) east siberia is stretched
	if radians(50) < lat and radians(154) < lon:
		lat += (lon - radians(154)) / 3
	# (3) australia is shunken and moved northward
	if lat < radians(-11) and radians(112) < lon < radians(154):
		lat = remap(lat, radians(-39), radians(-11), radians(-35), radians(-11))
		lon = remap(lon, radians(112), radians(154), radians(116), radians(151))
	# (4) new zealand is moved northward
	if lat < radians(-33) and radians(165) < lon:
		lat += radians(8)
	# (5) greenland and iceland and jan mayen are moved northward
	if radians(-57) < lon < radians(-8) and radians(59) < lat:
		lat += radians(5)
	# (6) the americas are moved northward
	elif lon < radians(-34):
		lat += radians(13)
		# (7) in addition, the bottom of south america is squished
		if lat < radians(-31):
			lat = remap(lat, radians(-45), radians(-31), radians(-37), radians(-31))
	y, x = miller(clamp(lat, -pi/2, pi/2), lon)
	y = remap(y, -28/90, 61/90, -1, 1)
	return y, x

def gnomonic(lat0: float, lon0: float):
	# https://mathworld.wolfram.com/GnomonicProjection.html
	def function(lat: float, lon: float) -> Tuple[float, float]:
		c = sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon - lon0)
		# determine clipping
		if not -pi/2 < acos(c) < pi/2:
			return 1, 1
		x = cos(lat)*sin(lon - lon0)
		y = cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(lon-lon0)
		# remap to [-1, 1] for both
		x /= 2*c
		y /= 2*c
		return y, x
	return function

def imperator(lat: float, lon: float) -> Tuple[float, float]:
	# https://steamcommunity.com/sharedfiles/filedetails/?id=2333851309
	y, x = lambert_conformal_conic(radians(5), radians(60))(lat, lon)
	x -= 0.57
	y -= 0.66
	x *= 1.4
	y *= 2.8
	return y, x

def lambert_conformal_conic(lat1: float, lat2: float, lat0: float = 0, lon0: float = 0):
	if lat1 == lat2:
		n = sin(lat1)
	else:
		n = log(cos(lat1) * sec(lat2)) \
			/ log(tan(pi/4 + lat2/2) * cot(pi/4 + lat1/2))
	F = cos(lat1) * tan(pi/4 + lat1/2)**n / n
	rho0 = F * cot(pi/4 + lat0/2)**n
	def function(lat: float, lon: float) -> Tuple[float, float]:
		rho = F * cot(pi/4 + lat/2)**n
		x = rho*sin(n*(lon-lon0))
		y = rho0 - rho*cos(n*(lon-lon0))
		# todo: scale...
		return y, x
	return function

def mercator(lat: float, lon: float) -> Tuple[float, float]:
	"""truncated near poles"""
	x = lon
	y = log(tan(pi/4 + lat/2))
	# remap to [-1, 1] for both
	x /= pi
	y /= pi
	return y, x

def miller(lat: float, lon: float) -> Tuple[float, float]:
	x = lon
	y = 5/4 * log(tan(pi/4 + 2*lat/5))
	# remap to [-1, 1] for both
	x /= pi
	y /= 1498/2044 * pi # appx
	return y, x

def mocha_eu_us(lat: float, lon: float) -> Tuple[float, float]:
	# a map that gets europe and the US really clear
	y, x = lambert_conformal_conic(
		radians(30), radians(65), radians(40), radians(-55)
		)(lat, lon)
	#x += 0.2
	y -= 0.27
	x /= 4/3
	y *= 1.5
	return y, x

def mocha_eu(lat: float, lon: float) -> Tuple[float, float]:
	# a map that gets europe and the US really clear
	y, x = lambert_conformal_conic(
		radians(40), radians(70), radians(50), radians(15)
		)(lat, lon)
	#x += 0.2
	y -= 0.05
	x *= 1.5
	y *= 3
	return y, x

@debug_max
def mocha3(lat: float, lon: float) -> Tuple[float, float]:
	"""a projection that mimics the stereotype of how people see the world"""
	def s(x: float) -> float:
		"""(-inf, 0) -> (-1, 1/4) -> (0, 1/2) -> (1, 3/4) -> (inf, 1)"""
		return 1/pi * atan(x) + 1/2
	# stretches certain latitudes out
	stretch_factor = s(2*(lat-pi/5)) - s(2*(lat+pi/5)) + 4/5
	x = lon * cos(lat)**.5 # the cos is to preserve area
	x *= stretch_factor**.15 # take a root so the horizontal compensation is not as extreme
	# artificially raise the americas north
	lat -= 0.3 * sin(lon + 0.2)
	y = lat*stretch_factor
	x /= 2.5272094703960315 # found experimentally
	y /= 1.25
	return y, x

def mollweide(lat: float, lon: float) -> Tuple[float, float]:
	if abs(lat) == pi/2:
		theta = lat
	else:
		theta = newton_raphson(lat, 
			lambda x: 2*x + sin(2*x) - pi*sin(lat),
			lambda x: 2 + 2*cos(2*x))
	x = lon * cos(theta) / pi
	y = sin(theta)
	# todo: remap to [-1, 1] for both
	return y, x

def orthographic(lat0: float, lon0: float):
	def function(lat: float, lon: float) -> Tuple[float, float]:
		# determine clipping
		c = acos(sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0))
		if not -pi/2 < c < pi/2:
			return 1, 1
		# main
		x = cos(lat) * sin(lon - lon0)
		y = cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(lon-lon0)
		# remap to [-1, 1] for both
		# x /= pi
		# y /= pi
		return y, x
	return function

def orthographic2(lat0: float, lon0: float):
	"""double orthographic - shows both sides"""
	def function(lat: float, lon: float) -> Tuple[float, float]:
		# determine clipping
		c = acos(sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0))
		if not -pi/2 < c < pi/2:
			try:
				y, x = orthographic2(-lat0, lon0 + pi)(lat, lon)
			except RecursionError:
				return 1, 1
			x += 1
			return y, x
		# main
		x = cos(lat) * sin(lon - lon0)
		y = cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(lon-lon0)
		# remap to [-1, 0] so the other half can be on the right
		x -= 1
		x /= 2
		return y, x
	return function

def robinson_helper(lat: float) -> Tuple[float, float]:
	lat = abs(lat)
	xx = [radians(5*i) for i in range(19)] # every 5 deg from 0 to 90
	# https://en.wikipedia.org/wiki/Robinson_projection#Formulation
	X = linear_interpolation(lat, xx,
		[1, 0.9986, 0.9954, 0.99, 0.9822, 0.973, 0.96, 0.9427, 0.9216, 0.8962,
		0.8679, 0.835, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322]
	)
	Y = linear_interpolation(lat, xx, 
		[0, 0.062, 0.124, 0.186, 0.248, 0.31, 0.372, 0.434, 0.4958, 0.5571,
		0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1]
	)
	return X, Y

def robinson(lat: float, lon: float) -> Tuple[float, float]:
	X, Y = robinson_helper(lat)
	x = 0.8487 * X * lon
	y = 1.3523 * Y * sign(lat)
	# remap to [-1, 1] for both
	x /= 0.8487 * pi
	y /= 1.3523
	return y, x

def sinusoidal(lat: float, lon: float) -> Tuple[float, float]:
	x = lon * cos(lat)
	y = lat
	x /= pi
	y /= pi/2
	return y, x

def stereographic(lat0: float, lon0: float):
	# todo account for lat shift
	def function(lat: float, lon: float) -> Tuple[float, float]:
		r = 2*tan(pi/4 - lat/2)
		x = r*cos(lon-lon0)
		y = r*sin(lon-lon0)
		# remap to [-1, 1] for both
		x /= pi
		y /= pi
		return y, x
	return function

def victoria2(lat: float, lon: float) -> Tuple[float, float]:
	# the victoria 2 map is identical to the eu4 map, except AUS and south america are left unchanged.
	# (1) the poles are trimmed
	if not radians(-56) < lat < radians(72):
		return 1, 1
	# (2) east siberia is stretched
	if radians(50) < lat and radians(154) < lon:
		lat += (lon - radians(154)) / 3
	# (3) new zealand is moved northward, but less than eu4
	if lat < radians(-33) and radians(165) < lon:
		lat += radians(4)
	# (4) greenland and iceland and jan mayen are moved northward
	if radians(-57) < lon < radians(-8) and radians(59) < lat:
		lat += radians(5)
	# (5) the americas are moved northward
	elif lon < radians(-34):
		lat += radians(13)
	y, x = miller(clamp(lat, -pi/2, pi/2), lon)
	y = remap(y, -32/90, 61/90, -1, 1)
	return y, x

def winkel_tripel(lat: float, lon: float) -> Tuple[float, float]:
	alpha = acos(cos(lat) * cos(lon/2))
	x = lon + 2*cos(lat)*sin(lon/2)/sinc(alpha)
	y = lat + sin(lat)/sinc(alpha)
	x /= 2*pi
	y /= pi
	return y, x

# function to convert EQ to Zompist's Sinusoidal
def zomp(lat: float, lon: float) -> Tuple[float, float]:
	# convert zompist map from interrupted sinusoidal to equirectangular
	# (1) find where each pixel maps to in the EQ -> SIN projection
	# (2) using that map, take each pixel in zompist's image and remap it to the EQ image
	# interruptions in the northern hemisphere: -45, +90
	# interruptions in the southern hemisphere: +22.5
	# account for interruptions
	delta_x = 0
	# southern hemi
	if lat < 0:
		# left part
		if lon < pi/8:
			lon -= -pi/4
			delta_x += -pi/4
		# right part
		else:
			lon -= pi/2
			delta_x += pi/2
	# northern hemi
	else:
		# left part
		if lon < -pi/4:
			lon -= -pi/2
			delta_x += -pi/2
		# middle part
		elif lon < pi/2:
			lon -= pi/8
			delta_x += pi/8
		# right part
		else:
			lon -= 3*pi/4
			delta_x += 3*pi/4
	# main
	x = lon * cos(lat) + delta_x
	y = lat
	# [-pi/2, pi/2] -> [-1, 1]
	x /= pi
	y /= pi/2
	# both in [-1, 1]
	return y, x

def xy_to_lat_lon(size: Tuple[int, int], x: int, y: int) -> Tuple[float, float]:
	w, h = size
	lon = x/w * 2*pi - pi
	lat = pi/2 - y/h * pi
	return lat, lon

def sinu_scaled_to_xy(size: Tuple[int, int],
		rel_lat: float, rel_lon: float) -> Tuple[int, int]:
	# map [-1, 1] -> [h, 0] for lat
	# map [-1, 1] -> [0, w] for lon
	w, h = size
	x = (rel_lon+1)/2 * w
	y = h - h * (rel_lat+1)/2
	return clamp(floor(x), 0, w-1), clamp(floor(y), 0, h-1)

formats = {
	'azimuthal equidistant': azimuthal_equidistant, # equidistant
	'equal earth': equal_earth, # equal-area
	'eu4': eu4, # compromise
	'gnomonic': gnomonic(pi/2, 0), # gnomonic
	'imperator': imperator, # conformal
	'mercator': mercator, # conformal
	'miller': miller, # compromise
	'mocha 1': mocha_eu_us, # debug/testing
	'mocha 2': mocha_eu, # debug/testing
	'mocha 3': mocha3, # debug/testing
	'mollweide': mollweide, # equal-area
	'orthographic': orthographic(0, 0), # perspective
	'orthographic2': orthographic2(0, 0), # perspective
	'orthographic africa': orthographic(radians(5), radians(15)),
	'orthographic antarctica': orthographic(-pi/2, 0),
	'orthographic asia': orthographic(pi/4, pi/2),
	'orthographic australia': orthographic(radians(-25), radians(135)),
	'orthographic north america': orthographic(radians(35), radians(-100)),
	'orthographic europe': orthographic(radians(50), radians(15)),
	'orthographic south america': orthographic(radians(-15), radians(-60)),
	'robinson': robinson, # compromise
	'sinusoidal': sinusoidal, # equal-area
	'stereographic': stereographic(0, 0), # conformal
	'victoria2': victoria2, # compromise
	'winkel tripel': winkel_tripel, # compromise
	'zompist': zomp, # equal-area
}

def convert_to_equirectangular(source_filename: str, projection: str, destination_filename: str = 'output.png') -> None:
	if isinstance(projection, str):
		proj = formats[projection]
	else:
		proj = projection
	with Image.open(f"maps/{source_filename}") as im:
		w, h = im.width, im.height
		output = Image.new('RGB', (w, h))
		for x in range(w):
			for y in range(h):
				# turn x, y of EQ map to lat, lon
				lat, lon = xy_to_lat_lon((w, h), x, y)
				# turn lat, lon into [-1, 1] range coords, then shift those to match ZOMP image
				x_, y_ = sinu_scaled_to_xy((w, h), *proj(lat, lon))
				# print(x_, y_)
				# take that pixel of ZOMP and associate it with the coords of EQ
				output.putpixel((x, y), im.getpixel((x_, y_)))
	output.save(f"maps/{destination_filename}", "PNG")

def convert_from_equirectangular(source_filename: str, projection: str, destination_filename: str = 'output.png') -> None:
	if isinstance(projection, str):
		proj = formats[projection]
	else:
		proj = projection
	with Image.open(f"maps/{source_filename}") as im:
		w, h = im.width, im.height
		output = Image.new('RGB', (w, h))
		for x in range(w):
			for y in range(h):
				lat, lon = xy_to_lat_lon((w, h), x, y)
				x_, y_ = sinu_scaled_to_xy((w, h), *proj(lat, lon))
				output.putpixel((x_, y_), im.getpixel((x, y)))
	output.save(f"maps/{destination_filename}", "PNG")

def invert_zomp() -> None:
	convert_to_equirectangular("zomp.png", "zompist", "zomp_out.png")

# invert_zomp()
# convert_to_equirectangular("ib.png", "robinson")
# blue marble is...
# orthographic(radians(-28), radians(38))