from math import acos, atan2, cos, exp, inf, log10, pi, sin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Patch
from mpl_toolkits.mplot3d import Axes3D

# constants
g = 6.674e-11 # standard gravitational constant
c = 299792458 # m/s
L_0 = 3.0128e28 # W

pc = 3.0857e16 # m
ly = 9.4607e15 # m
au = 149597870700 # m
mi = 1609.344 # m

lb = 0.45359237 # kg
minute = 60 # s
hour = 3600 # s
day = 86400 # s
year = 31556952 # s
jyear = 31536000 # s
deg = 2*pi/360 # rad
arcmin = deg/60 # rad
arcsec = arcmin/60 # rad


# functions
def axisEqual3D(ax):
	import numpy as np
	extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
	sz = extents[:,1] - extents[:,0]
	centers = np.mean(extents, axis=1)
	maxsize = max(abs(sz))
	r = maxsize/2
	for ctr, dim in zip(centers, 'xyz'):
		getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


# classes
class Orbit:
	def __init__(self, **properties):
		self.properties = properties

	@property
	def a(self) -> float:
		"""Semimajor Axis (m)"""
		return self.properties['sma']

	@property
	def animate(self):
		"""Animate orbit with pyplot"""
		n = 1000
		ts = [i*self.p/n for i in range(n)]
		cs = [self.cartesian(t) for t in ts]
		xs, ys, zs, vxs, vys, vzs = zip(*cs)

		fig = plt.figure(figsize=(7, 7))
		ax = Axes3D(fig)
		def update(i: int):
			i *= 5
			i %= n
			plt.cla()
			# ax.axis('scaled') this feature has apparently been "in progress" for 7+ years... yeah, guess that'll never happen...
			# https://github.com/matplotlib/matplotlib/issues/1077/
			# https://stackoverflow.com/a/19248731/2579798
			ax.set_title('Orbit')
			ax.set_xlabel('x (m)')
			ax.set_ylabel('y (m)')
			ax.set_zlabel('z (m)')
			ax.plot(xs+(xs[0],), ys+(ys[0],), zs+(zs[0],), color='k', zorder=1)
			ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
			ax.scatter(xs[i], ys[i], zs[i], marker='o', color='b', s=15, zorder=3)
			axisEqual3D(ax)

		xyanimation = FuncAnimation(fig, update, interval=50) # 20 fps
		plt.show()

	@property
	def aop(self) -> float:
		"""Argument of periapsis (radians)"""
		return self.properties['aop']

	@property
	def apo(self) -> float:
		"""Apoapsis (m)"""
		return (1+self.e)*self.a

	@property
	def e(self) -> float:
		"""Eccentricity (dimensionless)"""
		return self.properties['e']

	@property
	def i(self) -> float:
		"""Inclination (radians)"""
		return self.properties['i']

	@property
	def lan(self) -> float:
		"""Longitude of the ascending node (radians)"""
		return self.properties['lan']

	@property
	def L_z(self) -> float:
		"""L_z Kozai constant (dimensionless)"""
		return (1-self.e**2)**.5*cos(self.i)

	@property
	def man(self) -> float:
		"""Mean Anomaly (radians)"""
		return self.properties['man']

	@property
	def mean_longitude(self) -> float:
		"""Mean Longitude (radians)"""
		return self.lan + self.aop + self.man

	@property
	def orbital_energy(self) -> float:
		"""Specific orbital energy (J)"""
		return -self.parent.mu/(2*self.a)

	@property
	def p(self) -> float:
		"""Period (seconds)"""
		return 2*pi*(self.a**3/self.parent.mu)**.5

	@property
	def parent(self):
		"""Parent body"""
		return self.properties['parent'] # type: Body

	@property
	def peri(self) -> float:
		"""Periapsis (m)"""
		return (1-self.e)*self.a

	@property
	def plot(self):
		"""Plot orbit with pyplot"""
		n = 1000
		ts = [i*self.p/n for i in range(n)]
		cs = [self.cartesian(t) for t in ts]
		xs, ys, zs, vxs, vys, vzs = zip(*cs)

		fig = plt.figure(figsize=(7, 7))
		ax = Axes3D(fig)
		plt.cla()
		ax.set_title('Orbit')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		ax.set_zlabel('z (m)')
		ax.plot(xs+(xs[0],), ys+(ys[0],), zs+(zs[0],), color='k', zorder=1)
		ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
		ax.scatter(xs[0], ys[0], zs[0], marker='o', color='b', s=15, zorder=3)
		axisEqual3D(ax)

		plt.show()

	@property
	def v(self) -> float:
		"""Mean orbital velocity (m/s)"""
		return (self.a/self.parent.mu)**-.5

	@property
	def v_apo(self) -> float:
		"""Orbital velocity at apoapsis (m/s)"""
		if e == 0:
			return self.v
		e = self.e
		return ((1-e)*self.parent.mu/(1+e)/self.a)**.5

	@property
	def v_peri(self) -> float:
		"""Orbital velocity at periapsis (m/s)"""
		if e == 0:
			return self.v
		e = self.e
		return ((1+e)*self.parent.mu/(1-e)/self.a)**.5
	
	# double underscore methods
	def __gt__(self, other) -> bool:
		return other.apo < self.peri

	def __lt__(self, other) -> bool:
		return self.apo < other.peri

	# methods
	def cartesian(self, t: float=0) -> (float, float, float, float, float, float):
		"""Get cartesian orbital parameters (m, m, m, m/s, m/s, m/s)"""
		# https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
		# todo 1 set mean anomaly at new epoch
		# 2 GOOD eccentric anomaly
		E = self.eccentric_anomaly(t)
		# 3 true anomaly
		nu = self.true_anomaly(t)
		# 4 distance to central body
		a, e = self.a, self.e
		r_c = a*(1-e*cos(E))
		# 5 get pos and vel vectors o and o_
		# FIXME something in the O-s is wrong
		mu = self.parent.mu
		o = tuple(r_c*i for i in (cos(nu), sin(nu), 0))
		o_ = tuple(((mu*a)**.5/r_c)*i for i in (-sin(E), (1-e**2)**.5*cos(E), 0))
		# transform o, o_ into inertial frame
		# todo fix whatever the fuck is broken here
		i = self.i
		omega, Omega = self.aop, self.lan
		c, C, s, S = cos(omega), cos(Omega), sin(omega), sin(Omega)
		R = lambda x: (
			x[0]*(c*C - s*cos(i)*S) - x[1]*(s*C + c*cos(i)*S),
			x[0]*(c*S + s*cos(i)*C) + x[1]*(c*cos(i)*C - s*S),
			x[0]*(s*sin(i))         + x[1]*(c*sin(i))
		)
		r, r_ = R(o), R(o_)
		# print([i/au for i in o], [i/au for i in r])
		return r + r_

	def distance(self, other) -> (float, float):
		# other is of type Orbit
		"""Min and max distances (rad)"""
		ds = abs(self.peri - other.apo), abs(self.apo - other.peri), \
			 abs(self.peri + other.apo), abs(self.apo + other.peri)
		dmin, dmax = min(ds), max(ds)
		return dmin, dmax

	def distance_to(self, other, t: float) -> float:
		# other is of type Orbit
		"""Distance between orbits at time t (m)"""
		a, b = self.cartesian(t)[:3], other.cartesian(t)[:3]
		return sum((i-j)**2 for i, j in zip(a, b))**.5

	def eccentric_anomaly(self, t: float=0) -> float:
		"""Eccentric anomaly (radians)"""
		# get new anomaly
		tau = 2*pi
		tol = 1e-10
		e, p = self.e, self.p
		# dt = day * t
		M = (self.man + tau*t/p) % tau
		# E = M + e*sin(E)
		E = M
		while 1: # ~2 digits per loop
			E_ = M + e*sin(E)
			if abs(E-E_) < tol:
				return E
			E = E_
	
	def relative_inclination(self, other) -> float:
		"""Relative inclination between two orbital planes (rad)"""
		import numpy as np
		t, p , T, P = self.i, self.lan, other.i, other.lan
		# vectors perpendicular to orbital planes
		v_self = np.array([sin(t)*cos(p), sin(t)*sin(p), cos(t)])
		v_other = np.array([sin(T)*cos(P), sin(T)*sin(P), cos(T)])
		return acos(np.dot(v_self, v_other))

	def resonant(self, ratio: float):
		"""Get a resonant orbit from this one"""
		p = {key: value for key, value in self.properties.items()}
		p['sma'] = self.a * ratio**(2/3)
		return Orbit(**p)

	def synodic(self, other) -> float:
		"""Synodic period of two orbits (s)"""
		p1, p2 = self.p, other.p
		if p2-p1:
			return p1*p2/(p2-p1)
		return inf

	def T_kozai(self, other) -> float:
		"""Kozai oscillation timescale (s)"""
		# other is type Body
		e, M, m_2, p, p_2 = other.orbit.e, self.parent.mass, other.mass, self.p, other.orbit.p
		return M/m_2*p_2**2/p*(1-e**2)**1.5
	
	def tisserand(self, other) -> float:
		"""Tisserand's parameter (dimensionless)"""
		a, a_P, e, i = self.a, other.a, self.e, self.relative_inclination(other)
		return a_P/a + 2*cos(i) * (a/a_P * (1-e**2))**.5

	def true_anomaly(self, t: float=0) -> float:
		"""True anomaly (rad)"""
		E, e = self.eccentric_anomaly(t), self.e
		return 2 * atan2((1+e)**.5 * sin(E/2), (1-e)**.5 * cos(E/2))

	def v_at(self, r: float) -> float:
		"""Orbital velocity at radius (m/s)"""
		return (self.parent.mu*(2/r-1/self.a))**.5


class Rotation:
	def __init__(self, **properties):
		self.properties = properties

	@property
	def p(self) -> float:
		"""Period (s)"""
		return self.properties['period']

	@property
	def tilt(self) -> float:
		"""Axial tilt (rad)"""
		return self.properties['tilt']


class Atmosphere:
	def __init__(self, **properties):
		self.properties = properties

	@property
	def composition(self) -> dict:
		"""Composition (element: fraction)"""
		return self.properties['composition']

	@property
	def greenhouse(self) -> float:
		"""Estimate greenhouse factor (dimensionless)"""
		# based on trial and error
		# desired results:
		# venus.atmosphere.greenhouse -> 3.01
		# earth.atmosphere.greenhouse -> 1.16
		# mars.atmosphere.greenhouse  -> .975
		pp_ratio = self.partial_pressure('CO2') / earth.atmosphere.partial_pressure('CO2')
		c = 288/earth.temp # ratio for earth
		# return c * pp_ratio ** .08
		x, y = 0.05466933153152969, 0.06302583949080053
		s_ratio = self.surface_pressure / earth.atmosphere.surface_pressure
		# print(c, pp_ratio, 'x', s_ratio, 'y')
		return c * pp_ratio ** x * s_ratio ** y

	@property
	def scale_height(self) -> float:
		"""Scale height (m)"""
		return self.properties['scale_height']

	@property
	def surface_pressure(self) -> float:
		"""Surface pressure (Pa)"""
		return self.properties['surface_pressure']

	# methods
	def partial_pressure(self, molecule: str) -> float:
		"""Partial pressure of a molecule on the surface (Pa)"""
		return self.surface_pressure * self.composition[molecule]

	def pressure(self, altitude: float) -> float:
		"""Pressure at altitude (Pa)"""
		return self.surface_pressure * exp(-altitude / self.scale_height)


class Body:
	def __init__(self, **properties):
		self.properties = properties

	# orbital properties
	@property
	def orbit(self) -> Orbit:
		return self.properties['orbit']

	@property
	def esi(r: float, m: float, T: float) -> float:
		# Radius, Density, Escape Velocity, Temperature
		"""Earth similarity index (dimensionless)"""
		r, m, T = self.radius, self.mass, self.temp
		r_e, m_e = earth.radius, earth.mass
		esi1 = 1-abs((r-r_e)/(r+r_e))
		esi2 = 1-abs((self.density-earth.density)/(self.density+earth.density))
		esi3 = 1-abs((self.v_e-earth.v_e)/(self.v_e+earth.v_e))
		esi4 = 1-abs((T-255)/(T+255))
		return esi1**(.57/4)*esi2**(1.07/4)*esi3**(.7/4)*esi4**(5.58/4)

	@property
	def hill(self) -> float:
		"""Hill Sphere (m)"""
		a, e, m, M = self.orbit.a, self.orbit.e, self.mass, self.orbit.parent.mass
		return a*(1-e)*(m/3/M)**(1/3)

	@property
	def soi(self) -> float:
		"""Sphere of influence (m)"""
		return self.orbit.a*(self.mass/self.orbit.parent.mass)**.4

	@property
	def star(self): # -> Star
		"""Get the nearest star in the hierarchy"""
		p = self.orbit.parent
		return p if isinstance(p, Star) else p.star

	@property
	def temp(self) -> float:
		"""Planetary equilibrium temperature (K)"""
		a, R, sma, T = self.albedo, self.orbit.parent.radius, self.orbit.a, self.star.temperature
		return T*(1-a)**.25*(R/2/sma)**.5

	@property
	def tidal_locking(self) -> float:
		"""Tidal locking timeframe (s)"""
		return 5e28 * self.orbit.a**6 * self.radius / (self.mass * self.orbit.parent.mass**2)

	# rotation properties
	@property
	def rotation(self) -> Rotation:
		return self.properties['rotation']

	@property
	def solar_day(self) -> float:
		"""True solar day (s)"""
		t, T = self.rotation.p, self.orbit.p
		return (t*T)/(T-t)

	@property
	def v_tan(self) -> float:
		"""Equatorial rotation velocity (m/s)"""
		return 2*pi*self.radius/self.rotation.p

	@property
	def atmosphere_mass(self) -> float:
		"""Mass of the atmosphere (kg)"""
		return self.atmosphere.surface_pressure * self.area / self.surface_gravity

	# atmospheric properties
	@property
	def atmosphere(self) -> Atmosphere:
		return self.properties['atmosphere']

	# physical properties
	@property
	def albedo(self) -> float:
		"""Albedo (dimensionless)"""
		return self.properties['albedo']

	@property
	def area(self) -> float:
		"""Surface area (m^2)"""
		return 4*pi*self.radius**2
	
	@property
	def category(self) -> str:
		"""Attempt to categorize the body"""
		if isinstance(self, Star):
			return 'star'
		isround = 2e5 < self.radius
		if isinstance(self.orbit.parent, Star):
			# heliocentric
			if not isround:
				return 'asteroid'
			# planets and dwarf planets
			if self.planetary_discriminant < 1:
				return 'dwarf planet'
			# planets
			if self.radius < mercury.radius:
				return 'mesoplanet'
			if self.radius < 4/5 * earth.radius:
				return 'subearth'
			if self.radius < 5/4 * earth.radius:
				return 'earthlike'
			if self.mass < 10 * earth.mass:
				return 'superearth'
			if self.mass < 3e26: # SWAG
				return 'ice giant'
			if self.mass < 13*jupiter.mass:
				return 'gas giant'
			return 'brown dwarf'
		# moons
		if isround:
			return 'major moon'
		return 'minor moon'

	@property
	def circumference(self) -> float:
		"""Circumference (m)"""
		return 2*pi*self.radius

	@property
	def composition(self) -> dict:
		"""Composition (element: fraction)"""
		return self.properties['composition']

	@property
	def density(self) -> float:
		"""Density (kg/m^3)"""
		return self.mass/self.volume

	@property
	def diameter(self) -> float:
		"""Diameter (m)"""
		return 2*self.radius

	@property
	def gravbinding(self) -> float:
		"""gravitational binding energy (J)"""
		return 3*g*self.mass**2/(5*self.radius)

	@property
	def lunar_eclipse(self) -> float:
		"""Draw maximum eclipsing radii"""
		moon, planet = self, self.orbit.parent
		a = moon.orbit.peri
		r = planet.radius
		moon_radius = moon.angular_diameter_at(a-r)
		umbra_radius = atan2(planet.umbra_at(a), a-r)
		penumbra_radius = atan2(planet.penumbra_at(a), a-r)

		fig, ax = plt.subplots()
		ax.axis('scaled')
		plt.title('Apparent Diameters from Surface')
		plt.xlabel('x (rad)')
		plt.ylabel('y (rad)')
		# umbra
		umbra_circle = Circle((0, 0), radius=umbra_radius, color='k')
		umbra_ring = Circle((0, 0), radius=umbra_radius, color='k', linestyle='-', fill=False)
		# penumbra
		penumbra_circle = Circle((0, 0), radius=penumbra_radius, color='grey')
		penumbra_ring = Circle((0, 0), radius=penumbra_radius, color='grey', linestyle='-', fill=False)
		# moon
		moon_circle = Circle((0, 0), radius=moon_radius, color='orange')

		ax.add_artist(penumbra_circle)
		ax.add_artist(umbra_circle)
		ax.add_artist(moon_circle)
		ax.add_artist(penumbra_ring)
		ax.add_artist(umbra_ring)

		# legend
		plt.legend(handles=[
			Patch(color='orange', label='Moon'),
			Patch(color='black', label='Umbra'),
			Patch(color='grey', label='Penumbra'),
		])

		plt.show()

	@property
	def mass(self) -> float:
		"""Mass (kg)"""
		return self.properties['mass']

	@property
	def mu(self) -> float:
		"""Gravitational parameter (m^3/s^2)"""
		return g * self.mass

	@property
	def planetary_discriminant(self) -> float:
		"""Margot's planetary discriminant (dimensionless)"""
		a, m, M = self.orbit.a/au, self.mass/earth.mass, self.orbit.parent.mass/sun.mass
		# everything above is confirmed correct
		# https://en.wikipedia.org/wiki/Clearing_the_neighbourhood#cite_note-5
		# C, m_earth, m_sun, t_sun = 2*3**.5, earth.mass, sun.mass, sun.lifespan/year
		# k = 3**.5 * C**(-3/2) * (100*t_sun)**(3/4) * m_earth/m_sun
		# everything below is confirmed correct
		# print(807, '~', k)
		k = 807
		return k*m/(M**(5/2)*a**(9/8))

	@property
	def radius(self) -> float:
		"""Radius (m)"""
		return self.properties['radius']

	@property
	def rest_energy(self) -> float:
		"""rest energy"""
		return self.mass*c**2

	@property
	def surface_gravity(self) -> float:
		"""Surface gravity (m/s^2)"""
		return self.mu/self.radius**2

	@property
	def schwarzschild(self) -> float:
		"""Schwarzschild radius (m)"""
		return 2*self.mu/c**2

	@property
	def solar_eclipse(self) -> float:
		"""Draw maximum eclipsing radii"""
		moon, planet, star = self, self.orbit.parent, self.orbit.parent.orbit.parent
		moon_radius = moon.angular_diameter_at(moon.orbit.peri - planet.radius)
		star_radius = star.angular_diameter_at(planet.orbit.apo - planet.radius)

		fig, ax = plt.subplots()
		ax.axis('scaled')
		plt.title('Apparent Diameters from Surface')
		plt.xlabel('x (rad)')
		plt.ylabel('y (rad)')
		# star
		star_circle = Circle((0, 0), radius=star_radius, color='y')
		star_ring = Circle((0, 0), radius=star_radius, color='y', linestyle='-', fill=False)
		# moon
		moon_circle = Circle((0, 0), radius=moon_radius, color='k')

		ax.add_artist(star_circle)
		ax.add_artist(moon_circle)
		ax.add_artist(star_ring)

		# legend
		plt.legend(handles=[
			Patch(color='y', label='Star'),
			Patch(color='k', label='Moon'),
		])

		plt.show()

	@property
	def v_e(self) -> float:
		"""Surface escape velocity (m/s)"""
		return (2*self.mu/self.radius)**.5

	@property
	def volume(self) -> float:
		"""Volume (m^3)"""
		return 4/3*pi*self.radius**3

	# satellite properties
	@property
	def synchronous(self) -> float:
		"""SMA of synchronous orbit (m)"""
		mu = g*self.mass
		return (mu*self.rotation.p**2/4/pi**2)**(1/3)
	
	# double underscore methods
	def __gt__(self, other) -> bool:
		# WA uses radius, so despite my better judgement, so will I
		return other.radius < self.radius

	def __lt__(self, other) -> bool:
		return self.radius < other.radius
	
	# methods
	def acc_towards(self, other, t: float) -> float:
		"""Acceleration of self towards other body at time t (m/s^2)"""
		return self.force_between(other, t) / self.mass

	def acc_vector_towards(self, other, t: float) -> (float, float, float):
		"""Acceleration vector of self towards other body at time t (m/s^2)"""
		a, b = self.orbit.cartesian(t)[:3], other.orbit.cartesian(t)[:3]
		dx, dy, dz = [i-j for i, j in zip(a, b)]
		scale_factor = (dx**2 + dy**2 + dz**2)**.5
		acc = self.acc_towards(other, t)
		ax, ay, az = [acc*i/scale_factor for i in (dx, dy, dz)] # scale_factor*i is in [0, 1]
		# print(self.acc_towards(other, t))
		# print(ax, ay, az)
		# print('plot_grav_acc_vector(sedna, planet_nine)')
		return ax, ay, az

	def angular_diameter(self, other: Orbit) -> (float, float):
		"""Angular diameter, min and max (rad)"""
		dmin, dmax = self.orbit.distance(other)
		return self.angular_diameter_at(dmax) , self.angular_diameter_at(dmin)

	def angular_diameter_at(self, dist: float) -> float:
		"""Angular diameter at distance (rad)"""
		return 2*atan2(self.radius, dist)

	def app_mag(self, other: Orbit) -> (float, float):
		"""Apparent magnitude, min and max (dimensionless)"""
		dmin, dmax = self.orbit.distance(other)
		return self.app_mag_at(dmax) , self.app_mag_at(dmin)

	def app_mag_at(self, dist: float) -> float:
		"""Apparent magnitude at distance (dimensionless)"""
		# https://astronomy.stackexchange.com/a/5983
		correction = 8 # solution given in SE gives high results
		a_p, r_p, d_s, v_sun = self.albedo, self.radius, self.orbit.a, self.orbit.parent.abs_mag
		v_planet = -2.5*log10(a_p * r_p**2 / (4*d_s**2)) - v_sun + correction
		return v_planet + 5*log10(dist / (10*pc))

	def bielliptic(self, inner: Orbit, mid: Orbit, outer: Orbit) -> float:
		"""Bielliptic transfer delta-v (m/s)"""
		i, m, o = inner.a, mid.a, outer.a
		mu = self.mu
		a1 = (i+m)/2
		a2 = (m+o)/2
		dv1 = (2*mu/i-mu/a1)**.5-(mu/i)**.5
		dv2 = (2*mu/m-mu/a2)**.5-(2*mu/m-mu/a1)**.5
		dv3 = (2*mu/o-mu/a2)**.5-(mu/o)**.5
		return dv1 + dv2 + dv3

	def force_between(self, other, t: float) -> float:
		"""Force between two bodies at time t (N)"""
		m1, m2 = self.mass, other.mass
		r = self.orbit.distance_to(other.orbit, t)
		return g*m1*m2/r**2

	def hohmann(self, inner: Orbit, outer: Orbit) -> float:
		"""Hohmann transfer delta-v (m/s)"""
		i, o = inner.a, outer.a
		mu = self.mu
		dv1 = (mu/i)**.5*((2*o/(i+o))**.5-1)
		dv2 = (mu/i)**.5*(1-(2*i/(i+o))**.5)
		return dv1 + dv2

	def penumbra_at(self, distance: float) -> float:
		"""Penumbra radius at distance (m)"""
		planet, star = self, self.orbit.parent
		slope = (planet.radius+star.radius) / planet.orbit.a # line drawn from "top of star" to "bottom of planet"
		return slope*distance + planet.radius

	def roche(self, other) -> float:
		"""Roche limit of a body orbiting this one (m)"""
		m, rho = self.mass, other.density
		return (9*m/4/pi/rho)**(1/3)

	def umbra_at(self, distance: float) -> float:
		"""Umbra radius at distance (m)"""
		planet, star = self, self.orbit.parent
		slope = (planet.radius-star.radius) / planet.orbit.a # line drawn from "top of star" to "top of planet"
		return slope*distance + planet.radius


# ty https://www.python-course.eu/python3_inheritance.php
class Star(Body):
	@property
	def abs_mag(self) -> float:
		"""Absolute Magnitude (dimensionless)"""
		return -2.5 * log10(self.luminosity / L_0)

	@property
	def habitable_zone(self) -> (float, float):
		"""Inner and outer habitable zone (m)"""
		center = au*(self.luminosity/sun.luminosity)**.5
		inner = .95*center
		outer = 1.37*center
		return inner, outer

	@property
	def lifespan(self) -> float:
		"""Estimated lifespan (s)"""
		return 3e17*(self.mass/sun.mass)**-2.5162

	@property
	def luminosity(self) -> float:
		"""Luminosity (W)"""
		return self.properties['luminosity']
	
	@property
	def peakwavelength(self) -> float:
		"""Peak emission wavelength (m)"""
		return 2.8977729e-3/self.temperature

	@property
	def temperature(self) -> float:
		"""Temperature (K)"""
		return self.properties['temperature']

	@property
	def X(self) -> dict:
		"""Hydrogen composition (dimensionless)"""
		return self.composition['H']

	@property
	def Y(self) -> dict:
		"""Helium composition (dimensionless)"""
		return self.composition['He']

	@property
	def Z(self) -> dict:
		"""Metal composition (dimensionless)"""
		return 1 - self.X - self.Y

	# methods
	def app_mag(self, dist: float) -> float:
		"""Apparent Magnitude (dimensionless)"""
		return 5 * log10(dist / (10*pc)) + self.abs_mag

class System:
	"""Set of orbiting bodies"""
	def __init__(self, *bodies):
		"""Star system containing bodies.\nDoesn't need to be ordered."""
		self.bodies = set(bodies)

	@property
	def animation(self):
		"""Plot animated system with pyplot"""
		# see above plot for notes and sources
		n = 1000
		outerp = self.sorted_bodies[-1].orbit.p
		limit = sorted(list(self.bodies), key=lambda x: x.orbit.apo)[-1].orbit.apo

		fig = plt.figure(figsize=(7, 7))
		ax = Axes3D(fig)
		def update(i: int):
			plt.cla()
			ax.set_title('Orbit')
			ax.set_xlabel('x (m)')
			ax.set_ylabel('y (m)')
			ax.set_zlabel('z (m)')
			ax.set_xlim(-limit, limit)
			ax.set_ylim(-limit, limit)
			ax.set_zlim(-limit, limit)
			ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
			for body in self.bodies:
				cs = [body.orbit.cartesian((t+i)*outerp/n) for t in range(1)] # change to range(n) if orbits on
				xs, ys, zs, vxs, vys, vzs = zip(*cs)
				# ax.plot(xs, ys, zs, color='k', zorder=1)
				ax.scatter(xs[0], ys[0], zs[0], marker='o', s=15, zorder=3) # , color='b'

		xyanimation = FuncAnimation(fig, update, interval=50) # 20 fps
		plt.show()

	@property
	def any_body(self) -> Body:
		"""Returns a body (may or may not be the same each time)"""
		return list(self.bodies)[0]

	@property
	def plot(self):
		"""Plot system with pyplot"""
		# see above plot for notes and sources
		n = 1000

		fig = plt.figure(figsize=(7, 7))
		ax = Axes3D(fig)
		ax.set_title('Orbit')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		ax.set_zlabel('z (m)')
		ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
		for body in self.bodies:
			cs = [body.orbit.cartesian(t*body.orbit.p/n) for t in range(n)]
			xs, ys, zs, vxs, vys, vzs = zip(*cs)
			ax.plot(xs, ys, zs, color='k', zorder=1)
			ax.scatter(xs[0], ys[0], zs[0], marker='o', s=15, zorder=3)

		axisEqual3D(ax)
		plt.show()

	@property
	def plot2d(self):
		"""2D Plot system with pyplot"""
		# see above plot for notes and sources
		n = 1000

		fig = plt.figure(figsize=(7, 7))
		plt.title('Orbit')
		plt.xlabel('x (m)')
		plt.ylabel('y (m)')
		plt.scatter(0, 0, marker='*', color='y', s=50, zorder=2)
		for body in self.bodies:
			cs = [body.orbit.cartesian(t*body.orbit.p/n) for t in range(n)]
			xs, ys, zs, vxs, vys, vzs = zip(*cs)
			plt.plot(xs, ys, color='k', zorder=1)
			plt.scatter(xs[0], ys[0], marker='o', s=15, zorder=3)

		plt.show()
	
	@property
	def mass_pie(self):
		"""Mass pie chart"""
		system_masses = [i.mass for i in self.bodies]

		plt.subplot(1, 2, 1)
		plt.title('System mass')
		plt.pie(system_masses + [self.any_body.orbit.parent.mass])

		plt.subplot(1, 2, 2)
		plt.title('System mass (excl. primary)')
		plt.pie(system_masses)

		plt.show()

	@property
	def sorted_bodies(self) -> Body:
		"""List of bodies sorted by semimajor axis"""
		return sorted(list(self.bodies), key=lambda x: x.orbit.a)


# functions
def apsides2ecc(apo: float, peri: float) -> (float, float):
	return (apo+peri)/2, (apo-peri)/(apo+peri)


def plot_delta_between(body1: Body, body2: Body):
	"""Plot system with pyplot"""
	resolution = 100
	orbits = 8
	limit = max([body1, body2], key=lambda x: x.orbit.apo).orbit.apo*2
	outerp = max([body1, body2], key=lambda x: x.orbit.p).orbit.p

	fig = plt.figure(figsize=(7, 7))
	ax = Axes3D(fig)
	ax.set_title('Body Delta')
	ax.set_xlabel('dx (m)')
	ax.set_ylabel('dy (m)')
	ax.set_zlabel('dz (m)')
	ax.set_xlim(-limit, limit)
	ax.set_ylim(-limit, limit)
	ax.set_zlim(-limit, limit)
	ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
	b1s = [body1.orbit.cartesian(t*outerp/resolution) for t in range(orbits*resolution)]
	b2s = [body2.orbit.cartesian(t*outerp/resolution) for t in range(orbits*resolution)]
	cs = [[a-b for a, b in zip(b1, b2)] for b1, b2 in zip(b1s, b2s)]
	xs, ys, zs, vxs, vys, vzs = zip(*cs)
	ax.plot(xs, ys, zs, color='k', zorder=1)

	plt.show()


def plot_distance(body1: Body, body2: Body):
	"""Plot distance between two bodies over several orbits"""
	resolution = 1000
	orbits = 8
	outerp = max([body1, body2], key=lambda x: x.orbit.p).orbit.p

	fig = plt.figure(figsize=(7, 7))
	plt.title('Body Delta')
	plt.xlabel('time since epoch (s)')
	plt.ylabel('distance (m)')
	ts = [(t*outerp/resolution) for t in range(orbits*resolution)]
	xs = [body1.orbit.distance_to(body2.orbit, t) for t in ts]
	plt.plot(ts, xs, color='k')

	plt.show()


def plot_grav_acc(body1: Body, body2: Body):
	"""Plot gravitational acceleration from one body to another over several orbits"""
	resolution = 1000
	orbits = 8
	outerp = max([body1, body2], key=lambda x: x.orbit.p).orbit.p

	fig = plt.figure(figsize=(7, 7))
	plt.title('Body Delta')
	plt.xlabel('time since epoch (s)')
	plt.ylabel('acceleration (m/s^2)')
	plt.yscale('log')
	ts = [(t*outerp/resolution) for t in range(orbits*resolution)]
	xs = [body1.acc_towards(body2, t) for t in ts]
	plt.plot(ts, xs, color='k')

	plt.show()


def plot_grav_acc_vector(body1: Body, body2: Body):
	"""Plot gravitational acceleration vector from one body to another over several orbits"""
	resolution = 100
	orbits = 8
	outerp = max([body1, body2], key=lambda x: x.orbit.p).orbit.p

	fig = plt.figure(figsize=(7, 7))
	ax = Axes3D(fig)
	ax.set_title('Acceleration')
	ax.set_xlabel('x (m/s^2)')
	ax.set_ylabel('y (m/s^2)')
	ax.set_zlabel('z (m/s^2)')
	ts = [(t*outerp/resolution) for t in range(orbits*resolution)]
	xs, ys, zs = zip(*[body1.acc_vector_towards(body2, t) for t in ts])
	ax.plot(xs, ys, zs, color='k')
	axisEqual3D(ax)

	plt.show()


def stargen(m: float) -> Star:
	"""Generate star from mass"""
	m /= sun.mass
	# default exponents: .74,3,.505,-2.5
	# I find 0.96 a better approximation than 0.74, at least for smaller stars.
	# I find 0.54 a very slightly better approximation than 0.505, at least for smaller stars.
	# Luminosity and time values from https://www.academia.edu/4301816/On_Stellar_Lifetime_Based_on_Stellar_Mass
	# L
	if m > .45:
		lum = 1.148*m**3.4751
	else:
		lum = .2264*m**2.52
	return Star(**{
		'mass': m*sun.mass,
		'radius': sun.radius*m**0.96,
		'luminosity': sun.luminosity*lum,
		'temperature': 5772*m**.54,
	})


# bodies
sun = Star(**{
	'orbit': Orbit(**{
		'sma': 2.7e20,
	}),
	'rotation': Rotation(**{
		'period': 25.05*day,
		'tilt': .127,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 6050,
		'surface_pressure': .4,
	}),
	'mass': 1.9885e30,
	'radius': 6.957e8,
	'composition': {
		'H':  .7346,
		'He': .2483,
		'O':  .0077,
		'C':  .0029,
		'Fe': .0016,
		'Ne': .0012,
		'N':  .0009,
		'Si': .0007,
		'Mg': .0005,
		'S':  .0004,
	},
	# stellar properties
	'luminosity': 3.828e26,
	'temperature': 5778,
})

mercury = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 5.790905e10,
		'e': .20563,
		'i': .1223,
		'lan': .84354,
		'aop': .50831,
		'man': 3.05077,
	}),
	'rotation': Rotation(**{
		'period': 58.646 * day,
		'tilt': .00059, # to orbit
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 26000,
		'surface_pressure': 5e-10,
	}),
	'mass': 3.3011e23,
	'radius': 2.4397e6,
	'albedo': .142,
})

venus = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 1.08208e11,
		'e': .006772,
		'i': .0592466,
		'lan': 1.33832,
		'aop': .95791,
		'man': .87467,
	}),
	'rotation': Rotation(**{
		'period': 243.025 * day,
		'tilt': 3.0955,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 15900,
		'surface_pressure': 9.2e6,
		'composition': {
			'CO2': .965,
			'N2':  .035,
			'SO2': 1.5e-4,
			'Ar':  7e-5,
			'H2O': 2e-5,
			'CO':  1.7e-5,
			'He':  1.2e-5,
			'Ne':  7e-6,
			'HCl': 4e-7,
			'HF':  3e-9,
		},
	}),
	'mass': 4.8675e24,
	'radius': 6.0518e6,
	'albedo': .689,
})

earth = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 1.49598023e11,
		'e': .0167086,
		'i': 0, # by definition
		'lan': -.1965352,
		'aop': 1.9933027,
		'man': .1249,
	}),
	'rotation': Rotation(**{
		'period': 86164.100352,
		'tilt': .4090926295,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 8500,
		'surface_pressure': 101325,
		'composition': {
			'N2':  .78084,
			'O2':  .20946,
			'Ar':  .00934,
			'CO2': 4.08e-4,
			'Ne':  1.818e-5,
			'He':  5.24e-6,
			'CH4': 1.87e-6,
			'Kr':  1.14e-6,
		},
	}),
	'mass': 5.97237e24,
	'radius': 6.371e6,
	'albedo': .367,
})

ISS = Body(**{
	'orbit': Orbit(**{
		'parent': earth,
		'sma': 6.791e6,
		'e': 8.293e-4,
		'i': .9013,
		'lan': 5.643,
		'aop': .8070,
		'man': 0, # unknown
	}),
	'mass': 419725,
	'radius': (72.8**2 + 108.5**2 + 20**2)**.5 / 2,
})

_6Q0B44E = Body(**{
	'orbit': Orbit(**{
		# fixme epoch 2004
		'parent': earth,
		'sma': 7.84e8,
		'e': .254,
		'i': .1,
		'lan': 5.2,
		'aop': .6,
		'man': 5.6,
	}),
	'mass': 1e4, # assume like J002E3
	'radius': 5, # Its density has been estimated as around 15 kg/m3
	'albedo': .12,
})

moon = Body(**{
	'orbit': Orbit(**{
		'parent': earth,
		'sma': 3.84399e8,
		'e': .0549,
		'i': .0898,
		'lan': 0, # unknown
		'aop': 0, # unknown
		'man': 0, # unknown
	}),
	'rotation': Rotation(**{
		'period': 27.321661*day,
		'tilt': .02692,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 41860,
		'surface_pressure': 1e-7,
	}),
	'mass': 7.342e22,
	'radius': 1.7371e6,
})

mars = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 2.279392e11,
		'e': .0934,
		'i': .03229,
		'lan': .86495,
		'aop': 5.0004,
		'man': 0.33326,
	}),
	'rotation': Rotation(**{
		'period': 1.025957*day,
		'tilt': .4396, # to orbital plane
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 11100,
		'surface_pressure': 636,
		'composition': {
			'CO2': .949,
			'N2':  .026,
			'Ar':  .019,
			'O2':  .00174,
			'CO':  .000747,
			'H2O': .0003,
		},
	}),
	'mass': 6.4171e23,
	'radius': 3.3895e6,
	'albedo': .17,
})

ceres = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 2.7675*au,
		'e': .075823,
		'i': .18488,
		'lan': 1.40201,
		'aop': 1.26575,
		'man': 1.67533,
	}),
	'rotation': Rotation(**{
		'period': 9.074170*hour,
		'tilt': .07,
	}),
	'mass': 9.393e20,
	'radius': 473000,
	'albedo': .09,
})

jupiter = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 5.2044*au,
		'e': .0489,
		'i': .02774,
		'lan': 1.75343,
		'aop': 4.77988,
		'man': .34941,
	}),
	'rotation': Rotation(**{
		'period': 9.925*hour,
		'tilt': .0546, # to orbit
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 27000,
		'surface_pressure': 7e5,
	}),
	'mass': 1.8982e27,
	'radius': 6.9911e7,
	'albedo': .538,
})

# todo galileian moons

io = Body(**{
	'orbit': Orbit(**{
		'parent': jupiter,
		'sma': 421700,
		'e': .0041,
		'i': .05*deg, # equator
		'lan': 0, # unk
		'aop': 0,
		'man': 0,
	}),
	'rotation': Rotation(**{
		'period': 152853.5047,
	}),
	'atmosphere': Atmosphere(**{
		'surface_pressure': 7e5,
		'composition': {
			'SO2': .9,
		},
	}),
	'mass': 8.931938e22,
	'radius': 1.8216e6,
	'albedo': .63,
})

europa = Body(**{
	'orbit': Orbit(**{
		'parent': jupiter,
		'sma': 670900,
		'e': .009,
		'i': .470*deg, # equator
		'lan': 0, # unk
		'aop': 0,
		'man': 0,
	}),
	'rotation': Rotation(**{
		'period': 3.551181*day,
		'tilt': .1*deg,
	}),
	'atmosphere': Atmosphere(**{
		'surface_pressure': .1e-6,
	}),
	'mass': 4.799844e22,
	'radius': 1.5608e6,
	'albedo': .67,
})

ganymede = Body(**{
	'orbit': Orbit(**{
		'parent': jupiter,
		'sma': 1.0704e9,
		'e': .0013,
		'i': .2*deg, # equator
		'lan': 0, # unk
		'aop': 0,
		'man': 0,
	}),
	'rotation': Rotation(**{
		'period': 7.15455296*day,
		'tilt': .2*deg,
	}),
	'atmosphere': Atmosphere(**{
		'surface_pressure': .7e-6,
	}),
	'mass': 1.4819e23,
	'radius': 2.6341e6,
	'albedo': .43,
})

callisto = Body(**{
	'orbit': Orbit(**{
		'parent': jupiter,
		'sma': 1.8827e9,
		'e': .0074,
		'i': .192*deg, # to local Laplace planes
		'lan': 0, # unk
		'aop': 0,
		'man': 0,
	}),
	'rotation': Rotation(**{
		'period': 16.6890184*day,
		'tilt': 0, # known
	}),
	'atmosphere': Atmosphere(**{
		'surface_pressure': 7.5e-13,
		'composition': {
			'CO2':  .995,
			'O2':  .005,
		},
	}),
	'mass': 1.4819e23,
	'radius': 2.6341e6,
	'albedo': .43,
})

hektor = Body(**{ # 624 Hektor; largest trojan
	# fixme Epoch 23 March 2018
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 5.2571*au,
		'e': .0238,
		'i': .31706,
		'lan': 1.4020,
		'aop': 5.9828,
		'man': 2.3752,
	}),
	'rotation': Rotation(**{
		'period': 6.9205*hour,
	}),
	'mass': 9.95e18,
	'radius': 1e5,
	'albedo': .05,
})

saturn = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 9.5826*au,
		'e': .0565,
		'i': .04337,
		'lan': 1.98383,
		'aop': 5.92351,
		'man': 5.53304,
	}),
	'rotation': Rotation(**{
		'period': 10*hour + 33*minute + 38,
		'tilt': .4665, # to orbit
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 59500,
		'surface_pressure': 1.4e5,
	}),
	'mass': 5.6834e26,
	'radius': 5.8232e7,
	'albedo': .499,
})

halley = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 17.834*au,
		'e': .96714,
		'i': 2.8320,
		'lan': 1.020,
		'aop': 1.9431,
		'man': .6699,
	}),
	'rotation': Rotation(**{
		'period': 2.2*day,
	}),
	'mass': 2.2e14,
	'radius': 5.5e3,
	'albedo': .04,
})

uranus = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 19.2184*au,
		'e': .046381,
		'i': .0135,
		'lan': 1.2916,
		'aop': 1.6929494,
		'man': 2.48253189,
	}),
	'rotation': Rotation(**{
		'period': .71833*day,
		'tilt': 1.706, # to orbit
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 27700,
	}),
	'mass': 8.681e25,
	'radius': 2.5362e7,
	'albedo': .488,
})

neptune = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 30.11*au,
		'e': .009456,
		'i': .03085698,
		'lan': 2.30006,
		'aop': 4.82297,
		'man': 4.47202,
	}),
	'rotation': Rotation(**{
		'period': .6713*day,
		'tilt': .4943, # to orbit
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 19700,
	}),
	'mass': 1.02413e26,
	'radius': 2.4622e7,
	'albedo': .442,
})

pons_gambart = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 32.354*au,
		'e': .97509,
		'i': 136.39*deg,
		'lan': 0, # unk
		'aop': 0, # unk
		'man': 0, # unk
	}),
	'mass': 2e14, # SWAG
})

pluto = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 39.48*au,
		'e': .2488,
		'i': .2995,
		'lan': 1.92508,
		'aop': 1.98678,
		'man': .2536,
	}),
	'rotation': Rotation(**{
		'period': 6.38723*day,
		'tilt': 2.1386, # to orbit
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 60000,
		'surface_pressure': 1,
	}),
	'mass': 1.303e22,
	'radius': 1.1883e6,
	'albedo': .6,
})

ikeya_zhang = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 51.2136*au,
		'e': .990098,
		'i': .490785,
		'lan': 0, # unk
		'aop': 0, # unk
		'man': 0, # unk
	}),
	'mass': 2e14, # SWAG
})

eris = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 67.781*au,
		'e': .44068,
		'i': .768722,
		'lan': .627500,
		'aop': 2.63505,
		'man': 3.5633,
	}),
	'rotation': Rotation(**{
		'period': 25.9*hour,
	}),
	'mass': 1.66e22,
	'radius': 1.163e6,
	'albedo': .96,
})

sedna = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 506.8*au,
		'e': .85491,
		'i': .2081954,
		'lan': 2.52280,
		'aop': 5.4330,
		'man': 6.25112,
	}),
	'rotation': Rotation(**{
		'period': 10.3*hour,
	}),
	'mass': 9e20, # SWAG from Pluto
	'radius': 5e5,
	'albedo': .32,
})

planet_nine = Body(**{
	# http://www.findplanetnine.com/2019/02/version-2x.html
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 600*au,
		'e': .2,
		'i': .3,
		'lan': 1.6,
		'aop': 2.6,
		'man': 0, # could be literally anything in [0, 2pi)
	}),
	'mass': 3e25,
	'radius': 1.2e7, # SWAG
})

inner_solar_system = System(mercury, venus, earth, mars) # a <= mars
solar_system = System(mercury, venus, earth, mars, jupiter, saturn, uranus, neptune) # known planets
jupiter_system = System(io, europa, ganymede, callisto)
kuiper = System(neptune, pons_gambart, pluto, ikeya_zhang, eris, sedna, planet_nine) # a >= neptune
# todo rotational axis RA and DEC
# todo body1 body2 to orbit1 orbit2
# planet_nine.orbit.plot
