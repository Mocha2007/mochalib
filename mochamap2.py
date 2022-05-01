from __future__ import annotations
from math import asin, atan2, cos, pi, sin
from PIL import Image
from subprocess import check_output
from typing import Iterable
from common import average

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
	# double underscore methods
	def __repr__(self) -> str:
		return f"GeoCoord({self.lat}, {self.lon})"
	# methods
	def project(self, projection) -> MapCoord:
		return projection(self)
	def normalize(self) -> None:
		"""Forces lat in [-pi/2, pi/2] and lon in [-pi, pi]"""
		# first fix lat
		self.lat += pi/2
		self.lat %= 2*pi
		self.lat -= pi/2
		# now we know lat is in [-pi/2, 3*pi/2]
		if pi/2 < self.lat: # outside range
			self.lat = pi - self.lat
			self.lon += pi
		# then fix lon
		self.lon += pi
		self.lon %= 2*pi
		self.lon -= pi

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

	def from_eq(source_filename: str, projection, destination_filename: str = 'output.png',
			interpolate: bool = False, output_resolution: tuple[int, int] = None) -> None:
		with Image.open(f"maps/{source_filename}") as im:
			w, h = im.width, im.height
			if not output_resolution:
				output_resolution = w, h
			output = Image.new('RGB', output_resolution)
			def get_coord_(xx, yy) -> tuple[int, int]:
				coord_ = ImageCoord(xx, yy).geocoord_from_eq(im).project(projection).imagecoord(output)
				return coord_.x, coord_.y
			for x in range(w):
				for y in range(h):
					# interpolation
					if interpolate:
						x_, y_ = get_coord_(x + 0.5, y + 0.5)
						try:
							color = average_colors(
								im.getpixel((x, y)),
								im.getpixel((x+1, y)),
								im.getpixel((x, y+1)),
								im.getpixel((x+1, y+1))
							)
							output.putpixel((x_, y_), color)
						except IndexError:
							pass
					# regular
					x_, y_ = get_coord_(x, y)
					try:
						output.putpixel((x_, y_), im.getpixel((x, y)))
					except IndexError:
						pass
		output.save(f"maps/{destination_filename}", "PNG")

	def to_eq(source_filename: str, projection, destination_filename: str = 'output.png',
			output_resolution: tuple[int, int] = None) -> None:
		with Image.open(f"maps/{source_filename}") as im:
			w, h = im.width, im.height
			if not output_resolution:
				output_resolution = w, h
			output = Image.new('RGB', output_resolution)
			for x in range(w):
				for y in range(h):
					coord_ = ImageCoord(x, y).geocoord_from_eq(im).project(projection).imagecoord(output)
					x_, y_ = coord_.x, coord_.y
					try:
						output.putpixel((x, y), im.getpixel((x_, y_)))
					except IndexError:
						pass
		output.save(f"maps/{destination_filename}", "PNG")

	def sequence_from_eq(source_filename: str, projection_series,
			interpolate: bool = False, output_resolution: tuple[int, int] = None) -> None:
		series = list(projection_series)
		print(f"Running Map Sequence[{len(series)}]")
		for i, projection in enumerate(series):
			print(f"Rendering frame {i+1}/{len(series)}")
			Map.from_eq(source_filename, projection, "sequence/" + "{0}".format(str(i).zfill(5)) + ".png", interpolate, output_resolution)
		print(f"Attempting to stitch using ffmpeg...")
		print(check_output('ffmpeg -framerate 30 -i "maps/sequence/%05d.png" -c:v libx264 -pix_fmt yuv420p "maps/output.mp4"')) # shell=True
		print(check_output('ffmpeg -i "maps/output.mp4" -pix_fmt rgb24 "maps/output.gif"'))

# utility functions

def average_colors(*colors: Iterable[tuple[int, int, int]]) -> tuple[int, int, int]:
	r = int(average(*(c[0] for c in colors)))
	g = int(average(*(c[1] for c in colors)))
	b = int(average(*(c[2] for c in colors)))
	return r, g, b

def blend_proj(proj1, proj2):
	def output(coord: GeoCoord) -> MapCoord:
		map1 = proj1(coord)
		map2 = proj2(coord)
		return MapCoord((map1.x + map2.x)/2, (map1.y + map2.y)/2)
	return output

# testing/debug

debug_max_x = 0
debug_max_y = 0
# debug_proj = blend_proj(mollweide, equirectangular)

def debug_max(f):
	"""use this as a decorator to get max x/y"""
	def inner(lat: float, lon: float) -> tuple[float, float]:
		global debug_max_x, debug_max_y
		y, x = f(lat, lon)
		if debug_max_x < abs(x):
			debug_max_x = abs(x)
		if debug_max_y < abs(y):
			debug_max_y = abs(y)
		return y, x
	return inner

def _test() -> None:
	from mochamap2projections import mollweide
	from time import time
	start = time()
	Map.from_eq('test.png', mollweide)
	print(time() - start)
	# Map.to_eq('test.png', mollweide, interpolate=True)
	# Map.from_eq('test.png', mollweide, output_resolution=(300, 300))
	#Map.sequence_from_eq('almea2.png',
	#	(orthographic(GeoCoord(0, radians(12*i))) for i in range(30)),
	#False, (512, 512))
