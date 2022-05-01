
from math import acos, log, tan
from common import clamp, cot, linear_interpolation, newton_raphson, radians, remap, sec, sign, sinc
from mochamap2 import *

def aitoff(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	alpha = acos(cos(lat) * cos(lon/2))
	x = 2*cos(lat)*sin(lon/2)/sinc(alpha)
	y = sin(lat)/sinc(alpha)
	x /= pi
	y /= pi/2
	return MapCoord(x, y)

def azimuthal_equidistant(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	rho = pi/2 - lat
	x = rho * sin(lon)
	y = -rho * cos(lon)
	# remap to [-1, 1] for both
	x /= pi
	y /= pi
	return MapCoord(x, y)

def equal_earth(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.0003796
	theta = asin(3**.5 / 2 * sin(lat))
	x = 2*3**.5 * lon * cos(theta)
	x /= 3 * (9*a4*theta**8 + 7*a3*theta**6 + 3*a2*theta**2 + a1)
	y = a4*theta**9 + a3*theta**7 + a2*theta**3 + a1*theta
	# remap to [-1, 1] for both
	x /= 2.7066297319215575 # found experimentally
	y /= 1.312188760937488 # found experimentally
	return MapCoord(x, y)

def equirectangular(coord: GeoCoord) -> MapCoord:
	return MapCoord(coord.lon/pi, coord.lat/(pi/2))

def eu4(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
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
	return MapCoord(x, y)

def gnomonic(coord0: GeoCoord):
	# https://mathworld.wolfram.com/GnomonicProjection.html
	lat0, lon0 = coord0.lat, coord0.lon
	def function(coord: GeoCoord) -> MapCoord:
		lat, lon = coord.lat, coord.lon
		c = sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon - lon0)
		# determine clipping
		if not -pi/2 < acos(c) < pi/2:
			return 1, 1
		x = cos(lat)*sin(lon - lon0)
		y = cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(lon-lon0)
		# remap to [-1, 1] for both
		x /= 2*c
		y /= 2*c
		return MapCoord(x, y)
	return function

def imperator(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	# https://steamcommunity.com/sharedfiles/filedetails/?id=2333851309
	y, x = lambert_conformal_conic(radians(5), radians(60))(lat, lon)
	x -= 0.57
	y -= 0.66
	x *= 1.4
	y *= 2.8
	return MapCoord(x, y)

def lambert_conformal_conic(coord0: GeoCoord, coord1: GeoCoord):
	lat0, lon0 = coord0.lat, coord0.lon
	lat1, lon1 = coord1.lat, coord1.lon
	if lat0 == lat1:
		n = sin(lat0)
	else:
		n = log(cos(lat0) * sec(lat1)) \
			/ log(tan(pi/4 + lat1/2) * cot(pi/4 + lat0/2))
	F = cos(lat0) * tan(pi/4 + lat0/2)**n / n
	rho0 = F * cot(pi/4 + lat0/2)**n
	def function(coord: GeoCoord) -> MapCoord:
		lat, lon = coord.lat, coord.lon
		rho = F * cot(pi/4 + lat/2)**n
		x = rho*sin(n*(lon-lon0))
		y = rho0 - rho*cos(n*(lon-lon0))
		# todo: scale...
		return MapCoord(x, y)
	return function

def mercator(coord: GeoCoord) -> MapCoord:
	"""truncated near poles"""
	x = coord.lon
	y = log(tan(pi/4 + coord.lat/2))
	return MapCoord(x/pi, y/pi)

def miller(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	x = lon
	y = 5/4 * log(tan(pi/4 + 2*lat/5))
	# remap to [-1, 1] for both
	x /= pi
	y /= 1498/2044 * pi # appx
	return MapCoord(x, y)

def mollweide(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	if abs(lat) == pi/2:
		theta = lat
	else:
		start_guess = 0.0322574 * lat**5 + -0.0157267 * lat**3 + 0.801411*lat
		# found experimentally via regressionon desmos.com
		# start_guess = lat 							-> 1.4355945587158203 s for test.png
		# start_guess = 0.8615*lat						-> 1.3415701389312744 s for test.png
		# start_guess = 1.35075 * asin(0.571506 * lat)	-> 1.2727408409118652 s for test.png
		# start_guess = [cubic]							-> 1.2726495265960693 s for test.png
		# start_guess = [quintic]						-> 1.272195816040039 s for test.png
		theta = newton_raphson(start_guess,
			lambda x: 2*x + sin(2*x) - pi*sin(lat),
			lambda x: 2 + 2*cos(2*x),
			threshold=1e-4)
			# this should be sufficient for getting the pixel in the correct location;
			# it results in a worst-case 0.2 px error if the map size is 2048 px
			# lat is a very good first guess for x_0; theta(lat) looks like arcsin(0.6x) but flatter
			# this takes 10 iterations at MOST, but usually takes 2
	x = lon * cos(theta) / pi
	y = sin(theta)
	return MapCoord(x, y)

def orthographic(coord0: GeoCoord):
	lat0, lon0 = coord0.lat, coord0.lon
	def function(coord: GeoCoord) -> MapCoord:
		lat, lon = coord.lat, coord.lon
		# determine clipping
		try:
			c = acos(sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon-lon0))
		except ValueError: # unfortunate floating point errors for 45 degree lat0
			c = pi if lat < 0 else 0
		if not -pi/2 < c < pi/2:
			return MapCoord(1, 1)
		# main
		x = cos(lat) * sin(lon - lon0)
		y = cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(lon-lon0)
		return MapCoord(x, y)
	return function


def orthographic2(coord0: GeoCoord):
	"""double orthographic - shows both sides"""
	def function(coord: GeoCoord) -> MapCoord:
		# normal map
		point1 = orthographic(coord0)(coord)
		if point1.x != 1 and point1.y != 1:
			point1.x -= 1
			point1.x /= 2
			return point1
		# find antipode
		coord_ = coord
		coord_.lon += pi
		coord_.lat *= -1
		# opposite map
		point2 = orthographic(coord0)(coord_)
		point2.x += 1
		point2.x /= 2
		# fix inversion
		point2.y *= -1
		return point2
	return function

def robinson(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	def robinson_helper(lat: float) -> tuple[float, float]:
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
	X, Y = robinson_helper(lat)
	x = 0.8487 * X * lon
	y = 1.3523 * Y * sign(lat)
	# remap to [-1, 1] for both
	x /= 0.8487 * pi
	y /= 1.3523
	return MapCoord(x, y)


def sinusoidal(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	x = lon * cos(lat)
	y = lat
	x /= pi
	y /= pi/2
	return MapCoord(x, y)

def stereographic(coord0: GeoCoord):
	lat0, lon0 = coord0.lat, coord0.lon
	# todo account for lat shift
	def function(coord: GeoCoord) -> MapCoord:
		lat, lon = coord.lat, coord.lon
		r = 2*tan(pi/4 - lat/2)
		x = r*cos(lon-lon0)
		y = r*sin(lon-lon0)
		# remap to [-1, 1] for both
		x /= pi
		y /= pi
		return MapCoord(x, y)
	return function

def victoria2(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
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
	return MapCoord(x, y)

winkel_tripel = blend_proj(aitoff, equirectangular)

# function to convert EQ to Zompist's Sinusoidal
def zomp(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
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
	return MapCoord(x, y)
