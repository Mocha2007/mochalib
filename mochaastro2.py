from math import cos, exp, inf, log10, pi

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


# classes
class Orbit:
	def __init__(self, **properties):
		self.properties = properties

	@property
	def a(self) -> float:
		"""Semimajor Axis (m)"""
		return self.properties['sma']

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
	def L_z(self) -> float:
		"""L_z Kozai constant (dimensionless)"""
		return (1-self.e**2)**.5*cos(self.i)

	@property
	def orbit_energy(self) -> float:
		"""Specific orbital energy (J)"""
		return -self.primary.mass*g/2/self.a

	@property
	def p(self) -> float:
		"""Period (seconds)"""
		mu = g*self.parent.mass
		return 2*pi*(self.a**3/mu)**.5

	@property
	def parent(self):
		"""Parent body"""
		return self.properties['parent'] # type: Body

	@property
	def peri(self) -> float:
		"""Periapsis (m)"""
		return (1-self.e)*self.a

	@property
	def v(self) -> float:
		"""Mean orbital velocity (m/s)"""
		mu = g*self.parent.mass
		return (self.a/mu)**-.5

	@property
	def v_apo(self) -> float:
		"""Orbital velocity at apoapsis (m/s)"""
		if e == 0:
			return self.v
		mu = g*self.parent.mass
		e = self.e
		return ((1-e)*mu/(1+e)/self.a)**.5

	@property
	def v_peri(self) -> float:
		"""Orbital velocity at periapsis (m/s)"""
		if e == 0:
			return self.v
		mu = g*self.parent.mass
		e = self.e
		return ((1+e)*mu/(1-e)/self.a)**.5

	# methods

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

	def v_at(self, r: float) -> float:
		"""Orbital velocity at radius (m/s)"""
		mu = g*self.parent.mass
		return (mu*(2/r-1/self.a))**.5


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

	# THIS IS WRONG
	# @property
	# def mass(self) -> float:
	# 	"""Atmospheric mass (kg)"""
	# 	# derived via integration
	# 	c = 6e9
	# 	return c * self.scale_height * self.surface_pressure

	@property
	def composition(self) -> dict:
		"""Composition (element: fraction)"""
		return self.properties['composition']

	@property
	def scale_height(self) -> float:
		"""Scale height (m)"""
		return self.properties['scale_height']

	@property
	def surface_pressure(self) -> float:
		"""Surface pressure (Pa)"""
		return self.properties['surface_pressure']

	# methods
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
		a, e, m, M = self.orbit.a, self.orbit.e, self.mass, self.orbit.primary.mass
		return a*(1-e)*(m/3/M)**(1/3)

	@property
	def soi(self) -> float:
		"""Sphere of influence (m)"""
		return self.orbit.a*(self.mass/self.orbit.parent.mass)**.4

	@property
	def temp(self) -> float:
		"""Planetary equilibrium temperature (K)"""
		a, R, sma, T = self.albedo, self.orbit.primary.radius, self.orbit.a, self.orbit.primary.temperature
		return T*(1-a)**.25*(R/2/sma)**.5

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

	# atmospheric properties
	@property
	def atmosphere(self) -> Atmosphere:
		return self.properties['atmosphere']

	# physical properties
	@property
	def area(self) -> float:
		"""Surface area (m^2)"""
		return 4*pi*self.radius**2

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
	def gravbinding(self) -> float:
		"""gravitational binding energy (J)"""
		return 3*g*self.mass**2/5/self.radius

	@property
	def mass(self) -> float:
		"""Mass (kg)"""
		return self.properties['mass']

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
		return g*self.mass/self.radius**2

	@property
	def schwarzschild(self) -> float:
		"""Schwarzschild radius (m)"""
		return 2*g*self.mass/c**2

	@property
	def v_e(self) -> float:
		"""Surface escape velocity (m/s)"""
		return (2*g*self.mass/self.radius)**.5

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
	
	# methods
	def bielliptic(self, inner: Orbit, mid: Orbit, outer: Orbit) -> float:
		"""Bielliptic transfer delta-v (m/s)"""
		i, m, o = inner.a, mid.a, outer.a
		mu = self.mass*g
		a1 = (i+m)/2
		a2 = (m+o)/2
		dv1 = (2*mu/i-mu/a1)**.5-(mu/i)**.5
		dv2 = (2*mu/m-mu/a2)**.5-(2*mu/m-mu/a1)**.5
		dv3 = (2*mu/o-mu/a2)**.5-(mu/o)**.5
		return dv1 + dv2 + dv3

	def hohmann(self, inner: Orbit, outer: Orbit) -> float:
		"""Hohmann transfer delta-v (m/s)"""
		i, o = inner.a, outer.a
		mu = self.mass*g
		dv1 = (mu/i)**.5*((2*o/(i+o))**.5-1)
		dv2 = (mu/i)**.5*(1-(2*i/(i+o))**.5)
		return dv1 + dv2

	def roche(self, other) -> float:
		"""Roche limit of a body orbiting this one (m)"""
		m, rho = self.mass, other.density
		return (9*m/4/pi/rho)**(1/3)


# ty https://www.python-course.eu/python3_inheritance.php
class Star(Body):
	@property
	def abs_mag(self) -> float:
		"""Absolute Magnitude (dimensionless)"""
		return -2.5 * log(self.luminosity / L_0)

	@property
	def habitable_zone(self) -> (float, float):
		"""Inner and outer habitable zone (m)"""
		center = au*self.luminosity**.5
		inner = .95*center
		outer = 1.37*center
		return inner, outer

	@property
	def lifespan(self) -> float:
		"""Estimated lifespam (s)"""
		return 3.97310184e17*self.mass**-2.5162

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

earth = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 2.7e20,
		'e': .0167086,
		'i': .1249,
	}),
	'rotation': Rotation(**{
		'period': 86164.100352,
		'tilt': .4090926295,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 8500,
		'surface_pressure': 101325,
	}),
	'mass': 5.97237e24,
	'radius': 6.371e6,
	'albedo': .367,
})

moon = Body(**{
	'orbit': Orbit(**{
		'parent': earth,
		'sma': 3.84399e8,
		'e': .0549,
		'i': .0898,
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

jupiter = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 5.2044*au,
		'e': .0489,
		'i': .02774,
	}),
	'rotation': Rotation(**{
		'period': 9.925*hour,
		'tilt': .0546, # to orbit
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 2.7e4,
		'surface_pressure': 7e5,
	}),
	'mass': 1.8982e27,
	'radius': 6.9911e7,
	'albedo': .538,
})

planet_nine = Body(**{
	# http://www.findplanetnine.com/2019/02/version-2x.html
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 600*au,
		'e': .2,
		'i': .3,
	}),
	'mass': 3e25,
	'radius': 1.2e7,
})
