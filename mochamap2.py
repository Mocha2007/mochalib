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
		x = (self.x + 1)/2 * image.width
		y = (1 - self.y)/2 * image.height
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
					output.putpixel((x_, y_), im.getpixel((x, y)))
		output.save(f"maps/{destination_filename}", "PNG")

# I need like... a map DATA object... ugh...