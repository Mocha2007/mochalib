from math import asin, copysign, cos, floor, log, pi, radians, sin, tan
from PIL import Image
from typing import Tuple

def clamp(x: float, min: float, max: float) -> float:
	if x < min:
		return min
	if max < x:
		return max
	return x

sign = lambda x: copysign(1, x)

def remap(val: float, min1: float, max1: float, min2: float = 0, max2: float = 1) -> float:
	range1, range2 = max1-min1, max2-min2
	return (val-min1)/range1 * range2 + min2

# projections

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
	y = remap(y, -28/90, 62/90, -1, 1)
	return y, x

def miller(lat: float, lon: float) -> Tuple[float, float]:
	x = lon
	y = 5/4 * log(tan(pi/4 + 2*lat/5))
	# remap to [-1, 1] for both
	x /= pi
	y /= 1498/2044 * pi # appx
	return y, x

def robinson_helper(lat: float) -> Tuple[float, float]:
	lat = abs(lat)
	# appx...
	# https://en.wikipedia.org/wiki/Robinson_projection#Formulation
	X = cos(0.642 * lat)
	Y = 2/pi * lat
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

# convert zompist map from interrupted sinusoidal to equirectangular
# (1) find where each pixel maps to in the EQ -> SIN projection
# (2) using that map, take each pixel in zompist's image and remap it to the EQ image

# function to convert EQ to Zompist's Sinusoidal
def eq_to_zomp(lat: float, lon: float) -> Tuple[float, float]:
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
	'equal earth': equal_earth,
	'eu4': eu4,
	'miller': miller,
	'robinson': robinson,
	'sinusoidal': sinusoidal,
	'zompist': eq_to_zomp,
}

def convert_to_equirectangular(source_filename: str, projection: str, destination_filename: str = 'output.png') -> None:
	proj = formats[projection]
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
	proj = formats[projection]
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