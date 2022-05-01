
from math import acos, log, tan
from mochamap2 import *

def equirectangular(coord: GeoCoord) -> MapCoord:
	return MapCoord(coord.lon/pi, coord.lat/(pi/2))

def mercator(coord: GeoCoord) -> MapCoord:
	"""truncated near poles"""
	x = coord.lon
	y = log(tan(pi/4 + coord.lat/2))
	return MapCoord(x/pi, y/pi)

def mollweide(coord: GeoCoord) -> MapCoord:
	lat, lon = coord.lat, coord.lon
	if abs(lat) == pi/2:
		theta = lat
	else:
		theta = newton_raphson(lat, 
			lambda x: 2*x + sin(2*x) - pi*sin(lat),
			lambda x: 2 + 2*cos(2*x))
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

debug_proj = blend_proj(mollweide, equirectangular)