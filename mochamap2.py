from __future__ import annotations
from math import asin, atan2, cos, hypot, pi, sin
from PIL import Image
from subprocess import check_output
from typing import Callable, Iterable
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
	# todo __neg__
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
	def rotate(self, delta: GeoCoord) -> GeoCoord:
		"""rotate center of map by lat, lon"""
		lat, lon = self.lat, self.lon
		lon += delta.lon # shift longitude
		out = GeoCoord(lat, lon)
		# time to shift latitude
		spatial = out.cartesiancoord
		# y coord will be held constant, x/z will be rotated about y-axis
		# convert the x, z to r, theta about the y-axis
		r = hypot(spatial.x, spatial.z)
		theta = atan2(spatial.z, spatial.x)
		theta += delta.lat
		spatial.x = r*cos(theta)
		spatial.z = r*sin(theta)
		# now convert back to 3d spherical
		out = spatial.geocoords
		out.normalize()
		return out
	def rotate_x(self, delta: float) -> GeoCoord:
		"""rotate about the X-axis (causes world to look upside-down)"""
		lat, lon = self.lat, self.lon
		out = GeoCoord(lat, lon)
		# time to shift latitude
		spatial = out.cartesiancoord
		# x coord will be held constant, y/z will be rotated about x-axis
		# convert the y, z to r, theta about the x-axis
		r = hypot(spatial.y, spatial.z)
		theta = atan2(spatial.z, spatial.y)
		theta += delta
		spatial.y = r*cos(theta)
		spatial.z = r*sin(theta)
		# now convert back to 3d spherical
		out = spatial.geocoords
		out.normalize()
		return out

class Interruption:
	def __init__(self, left_bound: float, right_bound: float, center: float, in_northern_hemisphere: bool):
		self.left_bound = left_bound
		self.right_bound = right_bound
		self.center = center
		self.in_northern_hemisphere = in_northern_hemisphere
	@staticmethod
	def wrap_projection(*i: Iterable[Interruption]) -> Callable[[Callable[[GeoCoord], MapCoord]], Callable[[GeoCoord], MapCoord]]:
		def inner(proj: Callable[[GeoCoord], MapCoord]) -> Callable[[GeoCoord], MapCoord]:
			def innest(coord: GeoCoord) -> MapCoord:
				delta_x = 0
				for interruption in i:
					# correct hemisphere
					if (0 < coord.lat) == interruption.in_northern_hemisphere:
						# correct part
						if interruption.left_bound <= coord.lon < interruption.right_bound:
							coord.lon -= interruption.center
							delta_x += interruption.center
							break
				# now apply projection as normal
				mc = proj(coord)
				# correct for interruption
				mc.x += delta_x/pi
				return mc
			return innest
		return inner

class SpatialCoord:
	"""x, y, z of a lat/lon pair"""
	def __init__(self, x: float, y: float, z: float) -> None:
		self.x = x
		self.y = y
		self.z = z
	@property
	def geocoords(self) -> GeoCoord:
		return GeoCoord(asin(self.z), atan2(self.y, self.x))
	# double underscore methods
	def __repr__(self) -> str:
		return f"SpatialCoord({self.x}, {self.y}, {self.z})"

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
					if interpolate and x+1 < w and y+1 < h:
						x_, y_ = get_coord_(x + 0.5, y + 0.5)
						color = average_colors(
							im.getpixel((x, y)),
							im.getpixel((x+1, y)),
							im.getpixel((x, y+1)),
							im.getpixel((x+1, y+1))
						)
						try:
							output.putpixel((x_, y_), color)
						except IndexError: # can happen with interruptions + oblique center
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
			interpolate: bool = False, output_resolution: tuple[int, int] = None, fps: int = 30) -> None:
		series = list(projection_series)
		print(f"Running Map Sequence[{len(series)}]")
		for i, projection in enumerate(series):
			print(f"Rendering frame {i+1}/{len(series)}")
			Map.from_eq(source_filename, projection, "sequence/" + "{0}".format(str(i).zfill(5)) + ".png", interpolate, output_resolution)
		print(f"Attempting to stitch using ffmpeg...")
		print(check_output(f'ffmpeg -framerate {fps} -i "maps/sequence/%05d.png" -c:v libx264 -pix_fmt yuv420p "maps/output.mp4"')) # shell=True
		print(check_output('ffmpeg -i "maps/output.mp4" -pix_fmt rgb24 "maps/output.gif"'))

# utility functions

def average_colors(*colors: Iterable[tuple[int, int, int]]) -> tuple[int, int, int]:
	return tuple(map(int, map(average, zip(*colors))))

def blend_proj(proj1, proj2):
	def output(coord: GeoCoord) -> MapCoord:
		map1 = proj1(coord)
		map2 = proj2(coord)
		return MapCoord((map1.x + map2.x)/2, (map1.y + map2.y)/2)
	return output

def center(c: GeoCoord, rot: float = 0) -> Callable[[Callable[[GeoCoord], MapCoord]], Callable[[GeoCoord], MapCoord]]:
	"""A decorator to recenter a projection
	
	Must be added BEFORE any interruptions"""
	def inner(proj: Callable[[GeoCoord], MapCoord]) -> Callable[[GeoCoord], MapCoord]:
		def innest(coord: GeoCoord) -> MapCoord:
			return proj(coord.rotate(c).rotate_x(rot))
		return innest
	return inner
