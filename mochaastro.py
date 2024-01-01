"""MochaAstro
An astronomy library used to compute various things for my conworlds.
"""
# pylint: disable=unused-import, unused-variable
# pylint bugs out with pygame, and I want var names for some unused vars,
# in case i need them in the future
from typing import Callable, Iterable, Set
from mochaastro_common import *
from mochaastro_data import *
from mochaastro_body import *
from mochaastro_orbit import Orbit
from mochaastro_system import System
from common import staticproperty

# classes

class Tools:
	"""Miscelleneous tools to filter and manage bodies."""

	@staticproperty
	def namei_system(self) -> System:
		"""Precompute the namei system"""
		return System(search('namei'), *(search(s) for s in \
			"ara falto tata bau oneia don eisen neuve dicito mannu".split(" ")))

	@staticproperty
	def proto_sol(self) -> System:
		"""Precompute the nascent solar system"""
		return System(search('Proto-Sol'), search('Theia'), *(search("Proto-"+s) for s in \
			"mercury venus earth mars jupiter saturn p9 uranus neptune".split(" ")))

	@staticmethod
	def filter(*filters: Callable[[Iterable[Body]], Iterable[Body]]) -> Set[Body]:
		"""eg. Tools.filter(Tools.TNOs)"""
		# wanna find the average mass of a TNO?
		# (lambda *x: sum(x)/len(x)) \
		# (*(x.mass for x in Tools.filter(Tools.TNOs) if 'mass' in x.properties))
		bodies = universe.values()
		for f in filters:
			bodies = filter(f, bodies)
		return set(bodies)

	@staticmethod
	def TNOs(b: Body) -> bool:
		"""A filter to extract TNOs"""
		return 'orbit' in b.properties and neptune.orbit.a < b.orbit.a and \
			'parent' in b.orbit.properties and b.orbit.parent == sun


# planet_nine.orbit.plot
# distance_audio(earth.orbit, mars.orbit)
# solar_system.sim()
# burn = earth.orbit.transfer(mars.orbit)
# universe_sim(sun)
# solar_system_object.grav_sim()
