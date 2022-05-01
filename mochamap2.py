from __future__ import annotations
from math import asin, atan2, cos, pi, sin
from PIL import Image

class ImageCoord:
	"""pixel coords"""
	def __init__(self, x: int, y: int) -> None:
		self.x = x
		self.y = y
	def geocoord_from_eq(self, image: Image) -> GeoCoord:
		lon = self.x/image.width * 2*pi - pi
		lat = pi/2 - self.y/image.height * pi
		return GeoCoord(lat, lon)

class MapCoord:
	"""[-1, 1] floats"""
	def __init__(self, x: float, y: float) -> None:
		self.x = x
		self.y = y
	def imagecoord(self, image: Image) -> ImageCoord:
		x = int((self.x + 1)/2 * image.width)
		y = int((1 - self.y)/2 * image.height)
		return ImageCoord(x, y)

class GeoCoord:
	"""The latitude and longitude"""
	def __init__(self, lat: float, lon: float) -> None:
		self.lat = lat
		self.lon = lon
	@property
	def cartesiancoord(self) -> SpatialCoord:
		return SpatialCoord(cos(self.lat)*cos(self.lon), cos(self.lat)*sin(self.lon), sin(self.lat))
	# methods
	def project(self, projection) -> MapCoord:
		return projection(self)

class SpatialCoord:
	"""x, y, z of a lat/lon pair"""
	def __init__(self, x: float, y: float, z: float) -> None:
		self.x = x
		self.y = y
		self.z = z
	@property
	def geocoords(self) -> GeoCoord:
		return GeoCoord(atan2(self.y, self.x), asin(self.z))

class Map:
	def __init__(self, width: int, height: int) -> None:
		self.width = width
		self.height = height
		self.image = Image.new('RGB', (width, height))
	def from_eq(source_filename: str, projection, destination_filename: str = 'output.png') -> None:
		with Image.open(f"maps/{source_filename}") as im:
			w, h = im.width, im.height
			output = Image.new('RGB', (w, h))
			for x in range(w):
				for y in range(h):
					coord_ = ImageCoord(x, y).geocoord_from_eq(output).project(projection).imagecoord(output)
					x_, y_ = coord_.x, coord_.y
					try:
						output.putpixel((x_, y_), im.getpixel((x, y)))
					except IndexError:
						pass
		output.save(f"maps/{destination_filename}", "PNG")

# utility functions
def newton_raphson(x: float, f, f_, max_iter = 100) -> float:
	while ((x_ := x - f(x)/f_(x)) != x and 0 < max_iter):
		x = x_
		max_iter -= 1
	return x
# projections
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
	# todo: remap to [-1, 1] for both
	return MapCoord(x, y)
# I need like... a map DATA object... ugh...
# is this how it works??? I forgot tbh
# (1) Take input image
# (2) Turn image into matrix of data (lat/lon) -> color
# (3) Turn use data matrix
def test() -> None:
	Map.from_eq('test.png', mollweide)
