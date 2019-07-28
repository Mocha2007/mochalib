from math import acos, atan, atan2, cos, erf, exp, inf, isfinite, log10, pi, sin, tan
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from mochaunits import Angle, Length, Mass, Time, pretty_dim

# constants
epoch = datetime(2000, 1, 1, 11, 58, 55, 816) # https://en.wikipedia.org/wiki/Epoch_(astronomy)#Julian_years_and_J2000

g = 6.674e-11 # m^3 / (kg*s^2); appx; standard gravitational constant
c = 299792458 # m/s; exact; speed of light
L_0 = 3.0128e28 # W; exact; zero point luminosity
L_sun = 3.828e26 # W; exact; nominal solar luminosity

lb = 0.45359237 # kg; exact; pound
minute = 60 # s; exact; minute
hour = 3600 # s; exact; hour
day = 86400 # s; exact; day
year = 31556952 # s; exact; gregorian year
jyear = 31536000 # s; exact; julian year
deg = pi/180 # rad; exact; degree
arcmin = deg/60 # rad; exact; arcminute
arcsec = arcmin/60 # rad; exact; arcsecond
atm = 101325 # Pa; exact; atmosphere

ly = c * jyear # m; exact; light-year
au = 149597870700 # m; exact; astronomical unit
pc = 648000/pi * au # m; exact; parsec
mi = 1609.344 # m; exact; mile

G_SC = L_sun / (4*pi*au**2) # W/m^2; exact*; solar constant;
# * - technically not b/c this is based on visual luminosity rather than bolometric


# functions
def axisEqual3D(ax: Axes3D):
	extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
	sz = extents[:, 1] - extents[:, 0]
	centers = np.mean(extents, axis=1)
	maxsize = max(abs(sz))
	r = maxsize/2
	for ctr, dim in zip(centers, 'xyz'):
		getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def linear_map(interval1: (float, float), interval2: (float, float)):
	"""Create a linear map from one interval to another"""
	return lambda x: (x - interval1[0]) / (interval1[1] - interval1[0]) * (interval2[1] - interval2[0]) + interval2[0]


def resonance_probability(mismatch: float, outer: int) -> float:
	"""Return probability a particular resonance is by chance rather than gravitational"""
	return 1-(1-abs(mismatch))**outer


# classes
class Orbit:
	def __init__(self, **properties):
		self.properties = properties

	@property
	def a(self) -> float:
		"""Semimajor Axis (m)"""
		return self.properties['sma']

	@property
	def aop(self) -> float:
		"""Argument of periapsis (radians)"""
		return self.properties['aop']

	@property
	def apo(self) -> float:
		"""Apoapsis (m)"""
		return (1+self.e)*self.a

	@property
	def copy(self):
		from copy import deepcopy
		return deepcopy(self)

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
	def v(self) -> float:
		"""Mean orbital velocity (m/s)"""
		return (self.a/self.parent.mu)**-.5

	@property
	def v_apo(self) -> float:
		"""Orbital velocity at apoapsis (m/s)"""
		if self.e == 0:
			return self.v
		e = self.e
		return ((1-e)*self.parent.mu/(1+e)/self.a)**.5

	@property
	def v_peri(self) -> float:
		"""Orbital velocity at periapsis (m/s)"""
		if self.e == 0:
			return self.v
		e = self.e
		return ((1+e)*self.parent.mu/(1-e)/self.a)**.5

	# double underscore methods
	def __gt__(self, other) -> bool:
		return other.apo < self.peri

	def __lt__(self, other) -> bool:
		return self.apo < other.peri

	def __str__(self) -> str:
		bits = [
			'<Orbit',
			'Parent: {parent}',
			'a:      {sma}',
			'e:      {e}',
			'i:      {i}',
			'Omega:  {lan}',
			'omega:  {aop}',
			'M:      {man}',
		]
		return '\n\t'.join(bits).format(**self.properties)+'\n>'

	def at_time(self, t: float):
		"""Get cartesian orbital parameters (m, m, m, m/s, m/s, m/s)"""
		new = self.copy
		new.properties['man'] += t/self.p * 2*pi
		new.properties['man'] %= 2*pi
		return new

	def cartesian(self, t: float = 0) -> (float, float, float, float, float, float):
		"""Get cartesian orbital parameters (m, m, m, m/s, m/s, m/s)"""
		# ~29μs avg.
		# https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
		# 2 GOOD eccentric anomaly
		E = self.eccentric_anomaly(t)
		# 3 true anomaly
		nu = self.true_anomaly(t)
		# 4 distance to central body
		a, e = self.a, self.e
		r_c = a*(1-e*cos(E))
		# 5 get pos and vel vectors o and o_
		mu = self.parent.mu
		o = tuple(r_c*i for i in (cos(nu), sin(nu), 0))
		o_ = tuple(((mu*a)**.5/r_c)*i for i in (-sin(E), (1-e**2)**.5*cos(E), 0))
		# transform o, o_ into inertial frame
		i = self.i
		omega, Omega = self.aop, self.lan
		co, C, s, S = cos(omega), cos(Omega), sin(omega), sin(Omega)

		def R(x: (float, float, float)) -> (float, float, float):
			return (
				x[0]*(co*C - s*cos(i)*S) - x[1]*(s*C + co*cos(i)*S),
				x[0]*(co*S + s*cos(i)*C) + x[1]*(co*cos(i)*C - s*S),
				x[0]*(s*sin(i)) + x[1]*(co*sin(i))
			)
		r, r_ = R(o), R(o_)
		# print([i/au for i in o], [i/au for i in r])
		return r + r_

	def close_approach(self, other, t: float = 0, n: float = 1, delta_t_tolerance: float = 1, after_only=True) -> float:
		"""Get close approach time between two orbits after epoch t, searching +/-n orbits of self. (s)"""
		# ~5ms total to compute, at least for earth -> mars
		delta_t = self.p * n
		if delta_t < delta_t_tolerance:
			return t
		d_0 = self.distance_to(other, t)
		d_aft = self.distance_to(other, t + delta_t)
		d_bef = self.distance_to(other, t - delta_t)
		# print(d_bef, d_0, d_aft)
		if d_bef > d_aft < d_0: # the best point must be between middle and positive end
			# print('aft')
			return self.close_approach(other, t+delta_t/2, n/2, delta_t_tolerance, after_only)
		if not after_only and d_bef < d_0: # the best point must be between middle and negative end
			# print('bef')
			return self.close_approach(other, t-delta_t/2, n/2, delta_t_tolerance)
		# the best point must be near middle
		# print('mid')
		return self.close_approach(other, t, n/2, delta_t_tolerance, after_only)

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
		# ~65μs avg.
		a, b = self.cartesian(t)[:3], other.cartesian(t)[:3]
		return sum((i-j)**2 for i, j in zip(a, b))**.5

	def eccentric_anomaly(self, t: float = 0) -> float:
		"""Eccentric anomaly (radians)"""
		# ~6μs avg.
		# get new anomaly
		tau = 2*pi
		tol = 1e-10
		e, p = self.e, self.p
		# dt = day * t
		M = (self.man + tau*t/p) % tau
		assert isfinite(M)
		# E = M + e*sin(E)
		E = M
		while 1: # ~2 digits per loop
			E_ = M + e*sin(E)
			if abs(E-E_) < tol:
				return E
			E = E_

	def get_resonance(self, other, sigma: int = 3) -> (int, int):
		"""Estimate resonance from periods, n-sigma certainty (outer, inner)"""
		# ~102μs avg.
		q = self.p / other.p
		if 1 < q:
			return other.get_resonance(self, sigma)
		outer = 0
		best = 0, 0, 1
		while 1:
			outer += 1
			inner = round(outer / q)
			d = outer/inner - q
			p = resonance_probability(d, outer)
			if p < best[2]:
				# print('\t {0}:{1}\t-> {2}'.format(outer, inner, p))
				best = outer, inner, p
			# certain?
			if best[2] < 1 - erf(sigma/2**.5):
				break
		return best[:2]

	def phase_angle(self, other) -> float:
		"""Phase angle for transfer orbits, fraction of other's orbit (rad)"""
		# https://forum.kerbalspaceprogram.com/index.php?/topic/62404-how-to-calculate-phase-angle-for-hohmann-transfer/#comment-944657
		d, h = other.a, self.a
		return pi/((d/h)**1.5)

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

	def relative_inclination(self, other) -> float:
		"""Relative inclination between two orbital planes (rad)"""
		t, p, T, P = self.i, self.lan, other.i, other.lan
		# vectors perpendicular to orbital planes
		v_self = np.array([sin(t)*cos(p), sin(t)*sin(p), cos(t)])
		v_other = np.array([sin(T)*cos(P), sin(T)*sin(P), cos(T)])
		return acos(np.dot(v_self, v_other))

	def resonant(self, ratio: float):
		"""Get a resonant orbit from this one"""
		p = {key: value for key, value in self.properties.items()}
		p['sma'] = self.a * ratio**(2/3)
		return Orbit(**p)

	def stretch_to(self, other):
		"""Stretch one orbit to another, doing the minimum to make them cross"""
		# ~156μs avg.
		# if already crossing
		if self.peri <= other.peri <= self.apo or self.peri <= other.apo <= self.apo:
			return self # nothing to do, already done!
		new = self.copy
		# if self is inferior to other
		if self.a < other.a:
			# increase apo to other.peri
			apo, peri = other.peri, self.peri
		# else must be superior
		else:
			# decrease peri to other.apo
			apo, peri = self.apo, other.apo
		a, e = apsides2ecc(apo, peri)
		new.properties['sma'] = a
		new.properties['e'] = e
		return new

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

	def transfer(self, other, t: float = 0,
		delta_t_tol: float = 1, delta_x_tol: float = 1e7, dv_tol: float = .1) -> (float, float, float, float):
		"""Compute optimal transfer burn (m/s, m/s, m/s, s)"""
		raise NotImplementedError('DO NOT USE THIS. It NEVER works, and takes ages to compute.')
		# initial guess needs to be "bring my apo up/peri down to the orbit
		initial_guess = self.stretch_to(other)
		System(*[Body(orbit=i) for i in (initial_guess, self, other)]).plot2d()
		time_at_guess = initial_guess.close_approach(other, t, 1, delta_t_tol)
		dv_of_guess = tuple(j-i for i, j in zip(self.cartesian(t), initial_guess.cartesian(t)))[-3:]
		dv_best = dv_of_guess + (t,) # includes time
		# compute quality of initial guess
		old_close_approach_dist = initial_guess.distance_to(other, time_at_guess)
		# order of deltas to attempt
		base_order = (
			(dv_tol, 0, 0),
			(-dv_tol, 0, 0),
			(0, dv_tol, 0),
			(0, -dv_tol, 0),
			(0, 0, dv_tol),
			(0, 0, -dv_tol),
		)
		delta_v_was_successful = True
		while delta_v_was_successful:
			if old_close_approach_dist < delta_x_tol: # success!
				# print('it finally works!')
				print('{0} < {1}'.format(*(Length(i, 'astro') for i in (old_close_approach_dist, delta_x_tol))))
				System(*[Body(orbit=i) for i in (burn_orbit, self, other)]).plot()
				return dv_best
			# try to change the time FIRST
			mul = 2**16
			while mul:
				# check if changing the time helps
				dt_mod = delta_t_tol*mul
				# start by testing if adding a minute dt improves close approach
				dt = dv_best[3]+dt_mod
				burn_orbit = self.at_time(dt)
				# now, to check if burn_orbit makes it closer...
				new_close_approach_time = burn_orbit.close_approach(other, dt, 1, delta_t_tol)
				new_close_approach_dist = burn_orbit.distance_to(other, new_close_approach_time)
				if new_close_approach_dist < old_close_approach_dist:
					print(Length(new_close_approach_dist, 'astro'), Length(old_close_approach_dist, 'astro'))
					# System(*[Body(orbit=i) for i in (burn_orbit, self, other)]).plot2d
					# good! continue along this path, then.
					print('good!', (0, 0, 0, dt_mod), '@', mul)
					dv_best = dv_best[:3] + (dt,)
					old_close_approach_dist = new_close_approach_dist
					continue
				# make mul neg if pos, make mul pos and halved if neg
				mul = -mul
				if 0 < mul:
					mul >>= 1
			delta_v_was_successful = False
			for modifiers in base_order:
				print('MODIFIERS', modifiers)
				mul = 2**13
				while 1 <= mul:
					dvx_mod, dvy_mod, dvz_mod = tuple(i*mul for i in modifiers)
					# start by testing if adding a minute dx improves close approach
					dvx, dvy, dvz, dt = dv_best[0]+dvx_mod, dv_best[1]+dvy_mod, dv_best[2]+dvz_mod, dv_best[3]
					old_cartesian = self.cartesian(dt)
					x, y, z, vx, vy, vz = old_cartesian
					new_cartesian = old_cartesian[:3] + (vx+dvx, vy+dvy, vz+dvz)
					burn_orbit = keplerian(self.parent, new_cartesian).at_time(-dt)
					# when t=0 for this orbit, the real time is dt, so we need to reverse it by dt seconds
					# print(self, burn_orbit)
					# now, to check if burn_orbit makes it closer...
					try:
						new_close_approach_time = burn_orbit.close_approach(other, dt, 1, delta_t_tol)
					except AssertionError:
						mul >>= 1
						continue
					new_close_approach_dist = burn_orbit.distance_to(other, new_close_approach_time)
					if new_close_approach_dist < old_close_approach_dist:
						print(Length(new_close_approach_dist, 'astro'), Length(old_close_approach_dist, 'astro'))
						# System(*[Body(orbit=i) for i in (burn_orbit, self, other)]).plot2d
						# good! continue along this path, then.
						print('good!', (dvx_mod, dvy_mod, dvz_mod, 0), '@', mul)
						dv_best = dvx, dvy, dvz, dt
						old_close_approach_dist = new_close_approach_dist
						delta_v_was_successful = True
						break
					# multiplier is too big!
					# print('old mul was', mul)
					mul >>= 1
				if delta_v_was_successful:
					break
			print('Transfer failed...', Length(old_close_approach_dist, 'astro'))
		# autopsy
		try:
			System(*[Body(orbit=i) for i in (burn_orbit, self, other)]).plot()
		except AssertionError:
			print(initial_guess, burn_orbit)
		errorstring = '\n'.join((
			'Arguments do not lead to a transfer orbit.',
			'Perhaps you set your tolerances too high/low?',
			'{0} < {1}'.format(*(Length(i, 'astro') for i in (delta_x_tol, old_close_approach_dist))),
			str(dv_best),
		))
		raise ValueError(errorstring)

	def true_anomaly(self, t: float = 0) -> float:
		"""True anomaly (rad)"""
		# ~8μs avg.
		E, e = self.eccentric_anomaly(t), self.e
		return 2 * atan2((1+e)**.5 * sin(E/2), (1-e)**.5 * cos(E/2))

	def v_at(self, r: float) -> float:
		"""Orbital velocity at radius (m/s)"""
		return (self.parent.mu*(2/r-1/self.a))**.5


class Rotation:
	def __init__(self, **properties):
		self.properties = properties

	@property
	def axis_vector(self) -> (float, float, float):
		"""Return unit vector of axis (dimensionless)"""
		theta, phi = self.dec, self.ra
		return sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)

	@property
	def dec(self) -> float:
		"""Declination of axis (rad)"""
		return self.properties['dec']

	@property
	def p(self) -> float:
		"""Period (s)"""
		return self.properties['period']

	@property
	def ra(self) -> float:
		"""Right ascension of axis (rad)"""
		return self.properties['ra']

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
		# initially based on trial and error
		# eventually I gave up and 90% of this is copied from a4x
		gh_p = self.greenhouse_pressure / atm
		atm_pressure = self.surface_pressure / earth.atmosphere.surface_pressure
		correction_factor = 1.319714531668124
		ghe_max = 3.2141846382913877
		return min(ghe_max, 1 + ((atm_pressure/10) + gh_p) * correction_factor)

	@property
	def greenhouse_pressure(self) -> float:
		"""Surface pressure (Pa)"""
		ghg = {
			'CH4',
			'CO2',
		}
		return sum(self.partial_pressure(i) for i in ghg if i in self.composition)

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
	def esi(self) -> float:
		# Radius, Density, Escape Velocity, Temperature
		"""Earth similarity index (dimensionless)"""
		r, rho, T = self.radius, self.density, self.temp
		r_e, rho_e = earth.radius, earth.density
		esi1 = 1-abs((r-r_e)/(r+r_e))
		esi2 = 1-abs((rho-rho_e)/(rho+rho_e))
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
	def star_dist(self): # -> Star
		"""Get the distance to the nearest star in the hierarchy"""
		p = self.orbit.parent
		return self.orbit.a if isinstance(p, Star) else p.star_dist

	@property
	def star_radius(self): # -> Star
		"""Get the radius of the nearest star in the hierarchy"""
		p = self.orbit.parent
		return p.radius if isinstance(p, Star) else p.star_radius

	@property
	def temp(self) -> float:
		"""Planetary equilibrium temperature (K)"""
		a, R, sma, T = self.albedo, self.star_radius, self.star_dist, self.star.temperature
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
	def atm_retention(self) -> float:
		"""Checks if v_e is high enough to retain compound; molmass in kg/mol"""
		# https://upload.wikimedia.org/wikipedia/commons/4/4a/Solar_system_escape_velocity_vs_surface_temperature.svg
		return 11 * (self.temp / self.v_e)**2

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
	def data(self) -> str:
		"""Data Table a la Space Engine"""
		# sadly the only alternative would be like, 5000 lines of try/except
		lines = (
			"'{} {}'.format(self.category.title(), self.properties['name'])",
			"'Class           {}'.format(self.setype)",
			"'Radius          {} ({} Earths)'.format(pretty_dim(Length(self.radius)), round(self.radius/earth.radius, 3))",
			"'Mass            {}'.format(str(Mass(self.mass, 'astro')))",
			"'ESI             {}'.format(round(self.esi, 3))",
			"'Absolute Mag.   {}'.format(round(self.app_mag_at(10*pc), 2))",
			"'Semimajor Axis  {}'.format(pretty_dim(Length(self.orbit.a, 'astro')))",
			"'Orbital Period  {}'.format(pretty_dim(Time(self.orbit.p, 'imperial')))",
			"'Rotation Period {}'.format(pretty_dim(Time(self.rotation.p, 'imperial')))",
			"'Solar Day       {}'.format(pretty_dim(Time(self.solar_day, 'imperial')))",
			"'Axial Tilt      {}'.format(Angle(self.rotation.tilt, 'deg'))",
			"'Gravity         {} g'.format(round(self.surface_gravity/earth.surface_gravity, 3))",
			# "'Atmosphere Composition {}'.format(', '.join(sorted(list(self.atmosphere.composition), key=lambda x: self.atmosphere.composition[x], reverse=True)[:5]))",
			"'Atmos. Pressure {} atm'.format(round(self.atmosphere.surface_pressure/earth.atmosphere.surface_pressure), 3)",
			"'Temperature     {} K'.format(round(self.atmosphere.greenhouse*self.temp, 2))",
			"'Greenhouse Eff. {} K'.format(round((self.atmosphere.greenhouse-1)*self.temp, 2))",
		)
		string = []
		for line in lines:
			try:
				string.append(eval(line))
			except KeyError:
				pass
		return '\n'.join(string)

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
	def mass(self) -> float:
		"""Mass (kg)"""
		return self.properties['mass']

	@property
	def metal_report(self) -> str:
		"""Information regarding important metals"""
		symbols = 'Fe Ni Cu Pt Au Ag U Co Al'.split(' ')
		ms, assume = self.metals
		string = 'COMPOSITION REPORT'
		Fe, Ni, Cu, Pt, Au, Ag, U, Co, Al = [Mass(ms[sym]*self.mass, 'astro') for sym in symbols]
		if assume:
			string += '\n(Assuming Earthlike composition)'
		if any([Fe, Ni]):
			string += '\nBase Metals\n\tFe: {}\n\tNi: {}'.format(Fe, Ni)
		if any([Pt, Au, Ag]):
			string += '\nPrecious\n\tAu: {}\n\tAg: {}\n\tPt: {}'.format(Au, Ag, Pt)
		if any([Cu, U]):
			string += '\nOther\n\tAl: {}\n\tCo: {}\n\tCu: {}\n\tU: {}'.format(Al, Co, Cu, U)
		return string

	@property
	def metals(self):
		"""Metal data for metal/mining reports"""
		symbols = 'Fe Ni Cu Pt Au Ag U Co Al'.split(' ')
		assume = False
		try:
			ms = {sym: (self.composition[sym] if sym in self.composition else None) for sym in symbols}
		except KeyError:
			ms = {sym: earth.composition[sym] for sym in symbols}
			assume = True
		return ms, assume

	@property
	def mining_report(self) -> str:
		"""Information regarding mining"""
		string = 'MINING REPORT\nMinable metal mass (<4km deep):'
		mass = self.density * -self.shell(-4000) # depest mines are 4km deep, some wiggle room
		production = {
			'Fe': 2.28e12,  # 2015: 2,280 million tons
			'Ni': 2.3e9,    # More than 2.3 million tonnes (t) of nickel per year are mined worldwide,
			'Au': 3.15e6,   # 3,150 t/yr
			'Ag': 3.8223e7, # 38,223 t/yr
			'Pt': 1.61e5,   # 2014: 161 t/yr
			'Cu': 1.97e10,  # 2017: 19.7 million t/yr
			'U':  6.0496e7, # worldwide production of uranium in 2015 amounted to 60,496 tonnes
			'Co': 1.1e8,    # 2017: 110,000 t/yr
			'Al': 5.88e10,  # 2016: 58.8 million t/yr
		}
		ms, assume = self.metals
		ms = {sym: Mass(ms[sym]*mass, 'astro') for sym in production}
		ms = {sym: (mass, Time(year * mass.value / production[sym], 'imperial')) for sym, mass in ms.items()}
		if assume:
			string += '\n(Assuming Earthlike composition)'
		string += '\n(Times assume earthlike extraction rates)'
		for sym, (mass, time) in sorted(list(ms.items()), key=lambda x: x[1][0], reverse=True):
			string += '\n\t{}: {} ({})'.format(sym, mass, time)
		return string

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
	def setype(self) -> float:
		"""Space Engine-like classifier"""
		temperature = self.temp
		try:
			temperature *= self.atmosphere.greenhouse
		except KeyError:
			pass
		if 800 < temperature:
			heat = 'Scorched'
		elif 400 < temperature:
			heat = 'Hot'
		elif 300 < temperature:
			heat = 'Warm'
		elif 250 < temperature:
			heat = 'Temperate'
		elif 200 < temperature:
			heat = 'Cool'
		elif 100 < temperature:
			heat = 'Cold'
		else:
			heat = 'Frozen'
		return heat + ' ' + self.category.title()

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
		return self.angular_diameter_at(dmax), self.angular_diameter_at(dmin)

	def angular_diameter_at(self, dist: float) -> float:
		"""Angular diameter at distance (rad)"""
		return 2*atan2(self.radius, dist)

	def app_mag(self, other: Orbit) -> (float, float):
		"""Apparent magnitude, min and max (dimensionless)"""
		dmin, dmax = self.orbit.distance(other)
		return self.app_mag_at(dmax), self.app_mag_at(dmin)

	def app_mag_at(self, dist: float) -> float:
		"""Apparent magnitude at distance (dimensionless)"""
		# https://astronomy.stackexchange.com/a/5983
		correction = 8 # solution given in SE gives high results
		a_p, r_p, d_s, v_sun = self.albedo, self.radius, self.star_dist, self.star.abs_mag
		v_planet = -2.5*log10(a_p * r_p**2 / (4*d_s**2)) - v_sun + correction
		return v_planet + 5*log10(dist / (10*pc))

	def atm_supports(self, molmass: float) -> bool:
		"""Checks if v_e is high enough to retain compound; molmass in kg/mol"""
		return molmass > self.atm_retention

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

	def force_between(self, other, t: float = 0) -> float:
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

	def lunar_eclipse(self):
		"""Draw maximum eclipsing radii"""
		satellite, planet = self, self.orbit.parent
		a = satellite.orbit.peri
		r = planet.radius
		moon_radius = satellite.angular_diameter_at(a-r)
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

	def net_grav_acc_vector(self, system, t: float) -> (float, float, float):
		"""Net gravitational acceleration vector (m/s^2)"""
		vectors = [self.acc_vector_towards(body, t) for body in system.bodies if body != self]
		return tuple(map(sum, zip(*vectors)))

	def penumbra_at(self, distance: float) -> float:
		"""Penumbra radius at distance (m)"""
		planet, star = self, self.orbit.parent
		slope = (planet.radius+star.radius) / planet.orbit.a # line drawn from "top of star" to "bottom of planet"
		return slope*distance + planet.radius

	def roche(self, other) -> float:
		"""Roche limit of a body orbiting this one (m)"""
		m, rho = self.mass, other.density
		return (9*m/4/pi/rho)**(1/3)

	def shell(self, other: float) -> float:
		"""Volume of a shell extending xxx meters above the surface (m^3)"""
		r = max(self.radius + other, 0)
		v = 4/3 * pi * r**3
		return v - self.volume

	def solar_eclipse(self):
		"""Draw maximum eclipsing radii"""
		satellite, planet, star = self, self.orbit.parent, self.orbit.parent.orbit.parent
		moon_radius = satellite.angular_diameter_at(satellite.orbit.peri - planet.radius)
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
	def X(self) -> float:
		"""Hydrogen composition (dimensionless)"""
		return self.composition['H']

	@property
	def Y(self) -> float:
		"""Helium composition (dimensionless)"""
		return self.composition['He']

	@property
	def Z(self) -> float:
		"""Metal composition (dimensionless)"""
		return 1 - self.X - self.Y

	# methods
	def app_mag(self, dist: float) -> float:
		"""Apparent Magnitude (dimensionless)"""
		return 5 * log10(dist / (10*pc)) + self.abs_mag

	def radiation_pressure_at(self, dist: float) -> float:
		"""Stellar radiation pressure at a distance"""
		wattage = self.luminosity / sun.luminosity
		return wattage * G_SC / (c*dist**2)

	def radiation_force_at(self, obj: Body, t: float = 0) -> float:
		"""Stellar radiation force on a planet at a time"""
		wattage = self.luminosity / sun.luminosity
		dist = sum(i**2 for i in obj.orbit.cartesian(t)[:3])**.5 / au
		area = obj.area / 2
		return wattage * G_SC / (c*dist**2) * area


class System:
	"""Set of orbiting bodies"""
	def __init__(self, *bodies):
		"""Star system containing bodies.\nDoesn't need to be ordered."""
		self.bodies = set(bodies)

	@property
	def any_body(self) -> Body:
		"""Returns a body (may or may not be the same each time)"""
		return list(self.bodies)[0]

	@property
	def sorted_bodies(self) -> list:
		"""List of bodies sorted by semimajor axis"""
		return sorted(list(self.bodies), key=lambda x: x.orbit.a)

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

	def plot2d(self):
		"""2D Plot system with pyplot"""
		# see above plot for notes and sources
		n = 1000

		plt.figure(figsize=(7, 7))
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

	def sim(self):
		"""Use pygame to produce a better simulation, albeit in 2D"""
		import pygame
		from time import sleep # , time

		orbit_res = 64
		dot_radius = 2
		black, blue, white = (0,)*3, (0, 0, 255), (255,)*3
		timerate = self.sorted_bodies[0].orbit.p/32
		max_a = self.sorted_bodies[-1].orbit.apo
		parent = self.any_body.orbit.parent

		size = 800, 800
		width, height = size

		pygame.init()
		screen = pygame.display.set_mode(size)
		refresh = pygame.display.flip
		title = str(self)
		pygame.display.set_caption(title)
		fontsize = 20
		font = pygame.font.SysFont('Courier New', fontsize)

		t = 0
		# precompute orbits
		orbits = {body: tuple((body.orbit.cartesian(t+i*body.orbit.p/orbit_res)[:2],
		body.orbit.cartesian(t+(i+1)*body.orbit.p/orbit_res)[:2]) for i in range(orbit_res)) for body in self.bodies}
		# frame
		while 1:
			# print('t =', t)
			t += timerate
			screen.fill(black)
			# show bodies
			# show star
			star_radius = round(parent.radius/max_a * width)
			try:
				pygame.draw.circle(screen, white, (width//2, height//2),
					star_radius if dot_radius < star_radius else dot_radius)
			except OverflowError:
				pass
			# show planets
			xmap = linear_map((-max_a, max_a), (0, width))
			ymap = linear_map((-max_a, max_a), (height, 0))
			# start_time = time()
			for body in self.bodies: # ~500 μs/body @ orbit_res = 64
				x, y, z, vx, vy, vz = body.orbit.cartesian(t)
				coords = int(round(xmap(x))), int(round(ymap(y)))
				# redraw orbit
				for start_pos, end_pos in orbits[body]:
					x, y = start_pos
					start_coords = int(round(xmap(x))), int(round(ymap(y)))
					x, y = end_pos
					end_coords = int(round(xmap(x))), int(round(ymap(y)))
					try:
						pygame.draw.line(screen, blue, start_coords, end_coords)
					except TypeError:
						pass
				try:
					body_radius = round(body.radius/max_a * width)
				except KeyError:
					body_radius = 0
				try:
					pygame.draw.circle(screen, white, coords,
						body_radius if dot_radius < body_radius else dot_radius)
				except OverflowError:
					pass
			# print((time() - start_time)/len(self.bodies))
			# print date
			textsurface = font.render(str(epoch+timedelta(seconds=t))+' (x{0})'.format(int(timerate)), True, white)
			screen.blit(textsurface, (0, 0))
			# print scale
			textsurface = font.render(str(Length(max_a, 'astro')), True, white)
			screen.blit(textsurface, (0, fontsize))
			refresh()
			# event handling
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.display.quit()
					pygame.quit()
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_KP_PLUS: # timerate up
						timerate *= 2
					elif event.key == pygame.K_KP_MINUS: # timerate down
						timerate /= 2
				elif event.type == pygame.MOUSEBUTTONDOWN:
					if event.button == 4: # zoom in
						max_a /= 2
					if event.button == 5: # zoom out
						max_a *= 2
			sleep(1/30)


# functions
def apsides2ecc(apo: float, peri: float) -> (float, float):
	return (apo+peri)/2, (apo-peri)/(apo+peri)


def keplerian(parent: Body, cartesian: (float, float, float, float, float, float)) -> Orbit:
	"""Get keplerian orbital parameters (a, e, i, O, o, m)"""
	# https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf
	r, r_ = np.array(cartesian[:3]), np.array(cartesian[3:])
	mu = parent.mu
	# 1a Calculate orbital momentum vector h
	h = np.cross(r, r_)
	# 1b Obtain the eccentricity vector e [1] from
	e = np.cross(r_, h) / mu - r / np.linalg.norm(r)
	# 1c Determine the vector n pointing towards the ascending node and the true anomaly nu with
	n = np.cross(np.transpose((0, 0, 1)), h)
	temp = acos(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r)))
	if 0 <= np.dot(r, r_):
		nu = temp
	else:
		nu = 2*pi - temp
	# 2 Calculate the orbit inclination i by using the orbital momentum vector h,
	# where h z is the third component of h:
	h_z = h[2]
	i = acos(h_z / np.linalg.norm(h))
	# 3 Determine the orbit eccentricity e [1],
	# which is simply the magnitude of the eccentricity vector e, and the eccentric anomaly E [1]:
	eccentricity = np.linalg.norm(e)
	E = 2 * atan(tan(nu/2) / ((1+eccentricity)/(1-eccentricity))**.5)
	# print(nu, eccentricity, '-> E =', E)
	# E = 2 * atan2(((1+eccentricity)/(1-eccentricity))**.5, tan(nu/2))
	# 4 Obtain the longitude of the ascending node Omega and the argument of periapsis omega:
	if np.linalg.norm(n):
		temp = acos(n[0] / np.linalg.norm(n))
		if 0 <= n[1]:
			Omega = temp
		else:
			Omega = 2*pi - temp
		temp = acos(np.dot(n, e) / (np.linalg.norm(n) * np.linalg.norm(e)))
		if 0 <= e[2]:
			omega = temp
		else:
			omega = 2*pi - temp
	else:
		Omega, omega = 0, 0
	# 5 Compute the mean anomaly M with help of Kepler’s Equation from the eccentric anomaly E
	# and the eccentricity e:
	M = E - eccentricity * sin(E)
	# print(E, eccentricity, '-> M =', M)
	# 6 Finally, the semi-major axis a is found from the expression
	a = 1 / (2/np.linalg.norm(r) - np.linalg.norm(r_)**2/mu)
	return Orbit(**{
		'parent': parent,
		'sma': a,
		'e': eccentricity,
		'i': i,
		'lan': Omega,
		'aop': omega,
		'man': M,
	})


def plot_delta_between(orbit1: Orbit, orbit2: Orbit):
	"""Plot system with pyplot"""
	resolution = 100
	orbits = 8
	limit = max([orbit1, orbit2], key=lambda x: x.apo).apo*2
	outerp = max([orbit1, orbit2], key=lambda x: x.p).p

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
	b1s = [orbit1.cartesian(t*outerp/resolution) for t in range(orbits*resolution)]
	b2s = [orbit2.cartesian(t*outerp/resolution) for t in range(orbits*resolution)]
	cs = [[a-b for a, b in zip(b1, b2)] for b1, b2 in zip(b1s, b2s)]
	xs, ys, zs, vxs, vys, vzs = zip(*cs)
	ax.plot(xs, ys, zs, color='k', zorder=1)

	plt.show()


def plot_distance(orbit1: Orbit, orbit2: Orbit):
	"""Plot distance between two bodies over several orbits"""
	resolution = 1000
	orbits = 8
	outerp = max([orbit1, orbit2], key=lambda x: x.p).p

	plt.figure(figsize=(7, 7))
	plt.title('Body Delta')
	plt.xlabel('time since epoch (s)')
	plt.ylabel('distance (m)')
	ts = [(t*outerp/resolution) for t in range(orbits*resolution)]
	xs = [orbit1.distance_to(orbit2, t) for t in ts]
	plt.plot(ts, xs, color='k')

	plt.show()


def distance_audio(orbit1: Orbit, orbit2: Orbit):
	"""Play wave of plot_distance
	Encoding   | Signed 32-bit PCM
	Byte order | little endian
	Channels   | 1 channel mono
	"""
	print("* recording")

	# begin plot_distance
	# if the product of these is 44100 it will last 1s
	resolution = 441
	orbits = 100 # this will be close to the output frequency
	outerp = max([orbit1, orbit2], key=lambda x: x.p).p
	ts = [(t*outerp/resolution) for t in range(orbits*resolution)]
	xs = [orbit1.distance_to(orbit2, t) for t in ts]
	# normalize xs to [-1, 1]
	xs_m, xs_M = np.amin(xs), np.amax(xs)
	xs = np.array([2 * (i-xs_m)/(xs_M-xs_m) - 1 for i in xs])
	# print(xs, np.amin(xs), np.amax(xs))
	frames = (0x7FFFFFFF * np.array(xs)).astype(np.int32)
	# print(frames, np.amin(frames), np.amax(frames))
	# end plot_distance

	print("* done recording")
	with open('audacity.txt', 'wb+') as file:
		file.write(frames.tobytes())


def value_parse(value) -> float:
	try:
		return eval(value) if isinstance(value, str) else value
	except (NameError, SyntaxError):
		return value


def convert_atmosphere(data: dict) -> Atmosphere:
	return Atmosphere(**{key: value_parse(value) for key, value in data.items()})


def convert_orbit(data: dict, current_universe: dict) -> Orbit:
	out = Orbit(**{key: value_parse(value) for key, value in data.items()})
	# string -> bod
	if isinstance(out.parent, str):
		out.properties['parent'] = value_parse(out.parent)
		# still string -> parent doesn't exist as a var maybe
		if isinstance(out.parent, str):
			out.properties['parent'] = current_universe[out.parent]
	return out


def convert_rotation(data: dict) -> Rotation:
	return Rotation(**{key: value_parse(value) for key, value in data.items()})


def convert_body(data: dict, current_universe: dict, datatype=Body) -> Body:
	body_data = {}
	for key, value in data.items():
		if key == 'atmosphere':
			body_data[key] = convert_atmosphere(value)
		elif key == 'orbit':
			body_data[key] = convert_orbit(value, current_universe)
		elif key == 'rotation':
			body_data[key] = convert_rotation(value)
		else:
			body_data[key] = value_parse(value)
	return datatype(**body_data)


def load_data(seed: dict) -> dict:
	import os
	from json import load

	loc = os.path.dirname(os.path.abspath(__file__)) + '\\mochaastro'
	universe_data = seed
	for file in [f for f in os.listdir(loc) if f.endswith('.json')]:
		print('Loading', file, '...')
		json_data = load(open(loc + '\\' + file, 'r'))
		for obj in json_data:
			universe_data[obj['name']] = convert_body(obj, universe_data, (Star if obj['class'] == 'star' else Body))

	return universe_data


def plot_grav_acc(body1: Body, body2: Body):
	"""Plot gravitational acceleration from one body to another over several orbits"""
	resolution = 1000
	orbits = 8
	outerp = max([body1, body2], key=lambda x: x.orbit.p).orbit.p

	plt.figure(figsize=(7, 7))
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


def search(name: str) -> Body:
	# try exact match
	if name in universe:
		return universe[name]
	# try case insensitive
	for key, val in universe.items():
		if name.lower() == key.lower():
			return val
	# try substring
	hits = set()
	for key, val in universe.items():
		if name.lower() in key.lower():
			hits.add(val)
	if len(hits) == 1:
		return list(hits)[0]
	raise KeyError(hits)


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


def test_functions():
	print(str(earth.orbit))
	print('~'*40)
	print(str(keplerian(sun, earth.orbit.cartesian(0))))


def universe_sim(parent: Body):
	# TODO
	# moon display
	# comet tails
	"""Use pygame to show the system of [parent] and all subsystems"""
	import pygame
	from pygame import gfxdraw
	from time import sleep, time
	from math import hypot
	from mochamath import dist
	from mochaunits import round_time

	orbit_res = 64
	dot_radius = 2
	black, blue, beige, white, grey = (0,)*3, (0, 0, 255), (255, 192, 128), (255,)*3, (128,)*3
	red = blue[::-1]
	fps = 30
	timerate = 1/fps
	paused = False
	vectors = False
	# target = parent # until user selects a new one
	t = 0
	mouse_sensitivity = 10 # pixels

	size = 1024, 640
	width, height = size
	center = width//2, height//2
	max_a = 20*parent.radius
	current_a = max_a
	selection = parent
	selection_coords = center
	current_coords = selection_coords
	v_exaggeration = max_a/1.4e4

	pygame.init()
	screen = pygame.display.set_mode(size, pygame.RESIZABLE)
	refresh = pygame.display.flip
	inverse_universe = {j: i for i, j in universe.items()}
	font_large = 20
	font_normal = 16
	font_small = 12

	# verify onscreen
	def is_onscreen(coords: (int, int), buffer: int=0) -> bool:
		x, y = coords
		return -buffer <= x <= width+buffer and -buffer <= y <= height+buffer

	def are_onscreen(points: tuple, buffer: int=0) -> bool:
		"""return true if any onscreen"""
		for point in points:
			if is_onscreen(point):
				return True
		return False

	def center_on_selection(coords: (int, int)) -> (int, int):
		return tuple(i-j+k for i, j, k in zip(coords, current_coords, center))

	def coord_remap(coords: (float, float), smooth: bool=True) -> (int, int):
		a = current_a if smooth else max_a
		b = height/width * a
		xmap = linear_map((-a, a), (0, width))
		ymap = linear_map((-b, b), (height, 0))
		x, y = coords
		return int(round(xmap(x))), int(round(ymap(y)))

	def is_hovering(body: Body, coords: (int, int)) -> bool:
		apparent_radius = body.radius * width/(2*current_a) if 'radius' in body.properties else dot_radius
		return dist(coords, pygame.mouse.get_pos()) < max(mouse_sensitivity, apparent_radius)

	# display body
	def point(at: (int, int), radius: float, color: (int, int, int)=white, fill: bool=True, highlight: bool=False):
		"""radius is in meters, NOT pixels!!!"""
		star_radius = round(radius * width/(2*current_a))
		r = star_radius if dot_radius < star_radius else dot_radius
		x, y = at
		try:
			pygame.gfxdraw.aacircle(screen, x, y, r, color)
			if fill:
				pygame.gfxdraw.filled_circle(screen, x, y, r, color)
			if highlight:
				pygame.gfxdraw.aacircle(screen, x, y, 2*r, red)
		except OverflowError:
			if is_onscreen(at):
				screen.fill(color)

	# display body
	def show_body(body: Body, coords: (int, int), name: str=''):
		"""Coords are the actual screen coords"""
		hovering = is_hovering(body, coords)
		r = body.radius if 'radius' in body.properties else 1
		point(coords, r, white, True, hovering)
		# show name
		apparent_radius = r * width/(2*current_a)
		name_coords = tuple(int(i+apparent_radius) for i in coords)
		text(name, name_coords, font_normal, grey)

	# display arrow
	def arrow(a: (int, int), b: (int, int), color: (int, int, int)=red):
		"""Coords are the actual screen coords"""
		tip_scale = 10
		# todo: the arrow "tip"
		displacement = tuple(j-i for i, j in zip(a, b))
		# -> point between the - and the >
		tip_base = tuple((tip_scale-3)/tip_scale * i for i in displacement)
		real_tip_base = tuple(i+j for i, j in zip(a, tip_base))
		# perpendicular vector 1/4 size
		perpendicular = tuple((-1)**j * i/tip_scale for j, i in enumerate(displacement[::-1]))
		# left tip
		left_tip = tuple(i+j+k for i, j, k in zip(a, tip_base, perpendicular))
		# right tip
		right_tip = tuple(i+j-k for i, j, k in zip(a, tip_base, perpendicular))
		# render
		pygame.draw.aaline(screen, color, a, real_tip_base)
		pygame.draw.aalines(screen, color, True, (left_tip, right_tip, b))

	# display text
	def text(string: str, at: (int, int), size: int=font_normal, color: (int, int, int)=white):
		if not is_onscreen(at):
			return None
		this_font = pygame.font.SysFont('Courier New', size)
		string = string.replace('\t', ' '*4)
		for i, line in enumerate(string.split('\n')):
			textsurface = this_font.render(line, True, color)
			x, y = at
			y += i*size
			screen.blit(textsurface, (x, y))

	# precompute orbits (time0, time1, ..., timeN)
	def precompute_orbit(obj: Body) -> tuple:
		return tuple(obj.orbit.cartesian(t+i*obj.orbit.p/orbit_res)[:2] for i in range(orbit_res))

	def zoom(r: float=0):
		nonlocal max_a
		nonlocal selection_coords
		max_a *= 2**r
		if selection != parent:
			try:
				selection_coords = coord_remap(selection.orbit.cartesian(t)[:2])
			except KeyError:
				pass

	# only get first tier, dc about lower tiers
	orbits = []
	for name, body in universe.items():
		if 'orbit' not in body.properties or 'parent' not in body.orbit.properties:
			continue
		# disable comets; sharp angles
		if 'class' in body.properties and body.properties['class'] == 'comet':
			continue
		# clean orbits of undrawable orbits
		try:
			body.orbit.cartesian()
		except KeyError:
			continue
		# good orbit then!
		if body.orbit.parent == parent:
			color = beige if 'class' in body.properties and body.properties['class'] != 'planet' else blue
			orbits.append((name, body, precompute_orbit(body), color))
	# main loop
	# frame
	while 1:
		start_time = time()
		t += timerate
		screen.fill(black)
		# recenter
		zoom()
		# show bodies
		# show star
		if is_onscreen((0, 0)):
			show_body(parent, center_on_selection(center), inverse_universe[parent])
		# show planets
		# for_start = time()
		for name, body, orbit, color in orbits: # ~1.1 ms/body @ orbit_res = 64
			# hide small orbits
			if not any(10 < abs(i-j//2) for i, j in zip(center_on_selection(coord_remap((body.orbit.a ,)*2)), (width, height))):
				continue
			# redraw orbit
			coords = center_on_selection(coord_remap(body.orbit.cartesian(t)[:2]))
			points = tuple(map(center_on_selection, map(coord_remap, orbit)))
			if not (are_onscreen(points) or is_onscreen(coords)):
				continue
			pygame.draw.lines(screen, color, True, points)
			# planet dot
			if not is_onscreen(coords):
				continue
			# tip of the arrow
			if vectors:
				# velocity
				mag = hypot(*body.orbit.cartesian(t)[3:5])
				vcoords = center_on_selection(coord_remap(tuple(
						v_exaggeration*i+j for i, j in
						zip(body.orbit.cartesian(t)[3:5], body.orbit.cartesian(t)[:2]))))
				arrow(coords, vcoords)
				# kinetic energy
				ke_str = ''
				if 'mass' in body.properties:
					ke = 1/2 * Mass(body.mass) * (Length(mag)/Time(1))**2
					ke_str = '\nKE = {}'.format(pretty_dim(ke, 0))
				# mag
				text('{}{}'.format(pretty_dim(Length(mag)/Time(1)), ke_str), vcoords, font_normal, red)
			show_body(body, coords, name)
			# change selection?
			if is_hovering(body, coords) and pygame.mouse.get_pressed()[0]:
				selection = body
				zoom()
		# print((time()-for_start)/len(orbits))
		# print date
		try:
			current_date = str(round_time(epoch+timedelta(seconds=t)))
		except OverflowError:
			current_date = '>10000' if 0 < t else '<0'
		information = current_date + ' (x{0}){1}'.format(int(fps*timerate), ' [PAUSED]' if paused else '') + '\n' + \
					'Width: '+pretty_dim(Length(2*current_a, 'astro'))
		text(information, (0, height-font_large*2), font_large)
		# print FPS
		text(str(round(1/(time()-start_time)))+' FPS', (width-font_normal*4, 0), font_normal, red)
		# print selection data
		text(selection.data, (0, 0), font_small, (200, 255, 200))
		# refresh
		refresh()
		# post-render operations
		# event handling
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_KP_PLUS: # zoom in
					zoom(-1)
				elif event.key == pygame.K_KP_MINUS: # zoom out
					zoom(1)
				elif event.key == pygame.K_PERIOD: # timerate up
					timerate *= 2
				elif event.key == pygame.K_COMMA: # timerate down
					timerate /= 2
				elif event.key == pygame.K_p: # pause
					timerate, paused = paused, timerate
				elif event.key == pygame.K_r: # reverse
					timerate = -timerate
				elif event.key == pygame.K_v: # vectors
					vectors = not vectors
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 1: # check for recenter on parent
					if is_hovering(parent, center_on_selection(center)):
						selection = parent
						selection_coords = center
				if event.button == 4: # zoom in
					zoom(-1)
				if event.button == 5: # zoom out
					zoom(1)
			elif event.type == pygame.VIDEORESIZE:
				width, height = event.size
				center = width//2, height//2
				selection_coords = center
				zoom()
				pygame.display.set_mode(event.size, pygame.RESIZABLE)
				# print(event.size) # debug
		# smooth zoom
		current_a = (max_a + current_a)/2
		current_coords = tuple((i + j)//2 for i, j in zip(selection_coords, current_coords))
		# refresh title
		title = '{}, {} System - {}'.format(inverse_universe[selection], inverse_universe[parent], current_date)
		pygame.display.set_caption(title)
		# sleep
		wait_time = 1/fps - (time() - start_time)
		if 0 < wait_time:
			sleep(wait_time)


def warnings():
	"""Attempt to find missing data"""
	for name, body in universe.items():
		# print('Checking {}...'.format(name))
		# check for missing albedo data
		if 'albedo' not in body.properties:
			if 6e19 < body.mass < 1.5e29 and body.orbit.a < 600*au:
				print('Albedo data missing from {}, should be easy to find'.format(name))
		# check for missing atm data
		elif 'atmosphere' not in body.properties and body.atm_retention < .131 and body.orbit.a < 67*au:
			print('Atmosphere data missing from {}, atmosphere predicted'.format(name))


# bodies - this file only contains the sun, moon, and planets. json files provide the rest.
# USE BOND ALBEDO PLEASE
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
		# https://web.archive.org/web/20151107043527/http://weft.astro.washington.edu/courses/astro557/LODDERS.pdf
		'Ni': 9.1e-5,
		'Cu': 1e-6,
		'Pt': 2.6e-9,
		'Ag': 9.4e-10,
		'Au': 3.7e-10,
		'U':  1.8e-11,
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
	'albedo': .088,
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
	'albedo': .76,
})

earth = Body(**{
	'orbit': Orbit(**{
		'parent': sun,
		'sma': 1.49598023e11,
		'e': .0167086,
		'i': 0, # by definition
		'lan': 0, # -.1965352,
		'aop': 0, # 1.9933027,
		'man': .1249,
	}),
	'rotation': Rotation(**{
		'period': 86164.100352,
		'tilt': .4090926295,
		'ra': 0,
		'dec': 90*deg,
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
	'composition': { # by mass https://en.wikipedia.org/wiki/Abundance_of_the_chemical_elements#Earth
		'Fe': .319,
		'O':  .297,
		'Si': .161,
		'Mg': .154,
		'Ni': 1.822e-2,
		'Ca': 1.71e-2,
		'Al': 1.59e-2,
		'S':  6.35e-3,
		'Cr': 4.7e-3,
		'Na': 1.8e-3,
		'Mn': 1.7e-3,
		'P':  1.21e-3,
		'Co': 8.8e-4,
		'Ti': 8.1e-4,
		'C':  7.3e-4,
		'H':  2.6e-4,
		'K':  1.6e-4,
		'V':  1.05e-4,
		'Cl': 7.6e-5,
		'Cu': 6e-5,
		'Zn': 4e-5,
		'N':  2.5e-5,
		'Sr': 1.3e-5,
		'Sc': 1.1e-5,
		'F':  1e-5,
		'Zr': 7.1e-6,
		'Ge': 7e-6,
		'Ba': 4.5e-6,
		'Ga': 3e-6,
		'Y':  2.9e-6,
		'Se': 2.7e-6,
		'Pt': 1.9e-6,
		'As': 1.7e-6,
		'Mo': 1.7e-6,
		'Ru': 1.3e-6,
		'Ce': 1.13e-6,
		'Li': 1.1e-6,
		'Pd': 1e-6,
		# skip a few... Au Ag U
		'Au': 1.6e-7,
		'Ag': 5e-8,
		'U':  2e-8,
	},
	'mass': 5.97237e24,
	'radius': 6.371e6,
	'albedo': .306,
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
	'albedo': .136,
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
	'albedo': .25,
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
	'albedo': .503,
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
	'albedo': .342,
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
	'albedo': .3,
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
	'albedo': .29,
})

# inner_solar_system = System(mercury, venus, earth, mars) # a <= mars
# solar_system = System(mercury, venus, earth, mars, jupiter, saturn, uranus, neptune) # known planets
# jupiter_system = System(io, europa, ganymede, callisto)
# kuiper = System(neptune, pons_gambart, pluto, ikeya_zhang, eris, sedna, planet_nine) # a >= neptune
# comets = System(earth, halley, pons_gambart, ikeya_zhang) # earth and comets
universe = load_data({
	'Sun': sun,
	'Mercury': mercury,
	'Venus': venus,
	'Earth': earth,
	'Moon': moon,
	'Mars': mars,
	'Jupiter': jupiter,
	'Saturn': saturn,
	'Uranus': uranus,
	'Neptune': neptune,
})
# todo rotational axis RA and DEC https://en.wikipedia.org/wiki/Axial_tilt#Solar_System_bodies
# planet_nine.orbit.plot
# distance_audio(earth.orbit, mars.orbit)
# solar_system.sim()
# burn = earth.orbit.transfer(mars.orbit)
universe_sim(sun)
