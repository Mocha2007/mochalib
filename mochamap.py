from math import copysign, cos, floor, pi
from PIL import Image
from typing import Tuple

def clamp(x: float, min: float, max: float) -> float:
	if x < min:
		return min
	if max < x:
		return max
	return x

sign = lambda x: copysign(1, x)

# projections

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
	'robinson': robinson,
	'sinusoidal': sinusoidal,
	'zompist': eq_to_zomp,
}

def convert_to_equirectangular(source_filename: str, projection: str, destination_filename: str = 'output.png') -> None:
	proj = formats[projection]
	zomp_map = {}
	with Image.open(f"maps/{source_filename}") as im:
		w, h = im.width, im.height
		for x in range(w):
			for y in range(h):
				# turn x, y of EQ map to lat, lon
				lat, lon = xy_to_lat_lon((w, h), x, y)
				# turn lat, lon into [-1, 1] range coords, then shift those to match ZOMP image
				x_, y_ = sinu_scaled_to_xy((w, h), *proj(lat, lon))
				# print(x_, y_)
				# take that pixel of ZOMP and associate it with the coords of EQ
				zomp_map[(x, y)] = im.getpixel((x_, y_))
	output = Image.new('RGB', (w, h))
	for x in range(w):
		for y in range(h):
			output.putpixel((x, y), zomp_map[(x, y)])
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