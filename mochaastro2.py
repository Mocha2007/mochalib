from math import acos, atan, atan2, cos, erf, exp, hypot, inf, isfinite, log, log10, pi, sin, sqrt, tan
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Patch
from datetime import datetime, timedelta
from mochaunits import Angle, Length, Mass, Time, pretty_dim # Angle is indeed used
from typing import Callable, Dict, Optional, Tuple
# pylint: disable=E1101,W0612
# module has no member; unused variable
# pylint bugs out with pygame, and I want var names for some unused vars,
# in case i need them in the future

# constants
J2000 = datetime(2000, 1, 1, 11, 58, 55, 816) # https://en.wikipedia.org/wiki/Epoch_(astronomy)#Julian_years_and_J2000
SQRT2 = sqrt(2)

GRAV = 6.674e-11 # m^3 / (kg*s^2); appx; standard gravitational constant
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
gas_constant = 8.31446261815324 # J/(Kmol); exact; ideal gas constant
N_A = 6.02214076e23 # dimensionless; exact; Avogadro constant


# simple functions
def apsides2ecc(apo: float, peri: float) -> Tuple[float, float]:
	return (apo+peri)/2, (apo-peri)/(apo+peri)


def axisEqual3D(ax: Axes) -> None:
	extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
	sz = extents[:, 1] - extents[:, 0]
	centers = np.mean(extents, axis=1)
	maxsize = max(abs(sz))
	r = maxsize/2
	for ctr, dim in zip(centers, 'xyz'):
		getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def blagg(n: int,
		A: float = 0.338163, B: float = 2.81213,
		a: float = 1.40708, b: float = 0.982812,
		base: float = 1.73809) -> float:
	"""https://en.wikipedia.org/wiki/Titius%E2%80%93Bode_law#Blagg_Formulation
	https://www.desmos.com/calculator/fpnxyw41r3"""
	f = lambda psi: cos(psi)/(3-cos(2*psi)) + 1/(6-4*cos(2*(psi-30*deg)))
	return A * base**n * (B + f(a + n*b))


def linear_map(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> Callable[[float], float]:
	"""Create a linear map from one interval to another"""
	return lambda x: (x - interval1[0]) / (interval1[1] - interval1[0]) * (interval2[1] - interval2[0]) + interval2[0]


def resonance_probability(mismatch: float, outer: int) -> float:
	"""Return probability a particular resonance is by chance rather than gravitational"""
	return 1-(1-abs(mismatch))**outer


def synodic(p1: float, p2: float) -> float:
	"""synodic period of two periods (s)"""
	return p1*p2/abs(p2-p1) if p2-p1 else inf


# classes
class Orbit:
	def __init__(self, **properties) -> None:
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
		# type: (Orbit) -> Orbit
		from copy import deepcopy
		return deepcopy(self)

	@property
	def e(self) -> float:
		"""Eccentricity (dimensionless)"""
		return self.properties['e']

	@property
	def epoch_offset(self) -> float:
		"""Based on the epoch in properties, return the time in seconds to subtract when computing position"""
		if 'epoch' not in self.properties:
			return 0
		return (datetime.strptime(self.properties['epoch'], '%d %B %Y') - J2000).total_seconds()

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
		return sqrt(1-self.e**2)*cos(self.i)

	@property
	def man(self) -> float:
		"""Mean Anomaly (radians)"""
		return self.properties['man']

	@property
	def mean_longitude(self) -> float:
		"""Mean Longitude (radians)"""
		return self.lan + self.aop + self.man
	
	@property
	def orbit_tree(self):
		# type: (Orbit) -> Tuple[Body, ...]
		if 'parent' not in self.properties:
			return tuple()
		o = [self.parent]
		if 'orbit' in self.parent.properties:
			o += self.parent.orbit.orbit_tree
		return tuple(o)

	@property
	def orbital_energy(self) -> float:
		"""Specific orbital energy (J)"""
		return -self.parent.mu/(2*self.a)

	@property
	def p(self) -> float:
		"""Period (seconds)"""
		return 2*pi*sqrt(self.a**3/self.parent.mu)

	@property
	def parent(self):
		 # type: (Orbit) -> Body
		"""Parent body"""
		return self.properties['parent']

	@property
	def peri(self) -> float:
		"""Periapsis (m)"""
		return (1-self.e)*self.a

	@property
	def peri_shift(self) -> float:
		"""Periapsis shift per revolution (rad)"""
		# https://en.wikipedia.org/wiki/Tests_of_general_relativity#Perihelion_precession_of_Mercury
		return 24*pi**3*self.a**2 / (self.p**2 * c**2 * (1-self.e**2))

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
		return sqrt((1-e)*self.parent.mu/(1+e)/self.a)

	@property
	def v_peri(self) -> float:
		"""Orbital velocity at periapsis (m/s)"""
		if self.e == 0:
			return self.v
		e = self.e
		return sqrt((1+e)*self.parent.mu/(1-e)/self.a)

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
		# type: (Orbit, float) -> Orbit
		"""Get cartesian orbital parameters (m, m, m, m/s, m/s, m/s)"""
		new = self.copy
		new.properties['man'] += t/self.p * 2*pi
		new.properties['man'] %= 2*pi
		return new

	def cartesian(self, t: float = 0) -> Tuple[float, float, float, float, float, float]:
		"""Get cartesian orbital parameters (m, m, m, m/s, m/s, m/s)"""
		# ~29μs avg.
        # account for abnormal epochs
		t -= self.epoch_offset
		# https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
		# 2 GOOD eccentric anomaly
		E = self.eccentric_anomaly(t)
		# 3 true anomaly
		nu = self.true_anomaly(t, E)
		# 4 distance to central body
		a, e = self.a, self.e
		r_c = a*(1-e*cos(E))
		# 5 get pos and vel vectors o and o_
		mu = self.parent.mu
		o = tuple(r_c*i for i in (cos(nu), sin(nu), 0))
		o_ = tuple((sqrt(mu*a)/r_c)*i for i in (-sin(E), sqrt(1-e**2)*cos(E), 0))
		# transform o, o_ into inertial frame
		i = self.i
		omega, Omega = self.aop, self.lan
		co, C, s, S = cos(omega), cos(Omega), sin(omega), sin(Omega)

		def R(x: Tuple[float, float, float]) -> Tuple[float, float, float]:
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

	def crosses(self, other) -> bool:
		"""do two orbits cross?"""
		return self.peri < other.peri < self.apo or other.peri < self.peri < other.apo

	def distance(self, other) -> Tuple[float, float]:
		# type: (Orbit, Orbit) -> Tuple[float, float]
		"""Min and max distances (m)"""
		# NEW FEATURE: it can now do planet => moon or any other relationship!
		ot1, ot2 = self.orbit_tree, other.orbit_tree
		assert ot1[-1] == ot2[-1]
		if ot1[0] == ot2[0]:
			ds = abs(self.peri - other.apo), abs(self.apo - other.peri), \
				abs(self.peri + other.apo), abs(self.apo + other.peri)
			return min(ds), max(ds)
		if 1 < len(ot2) and ot1[0] == ot2[1]: # planet, moon
			if self == ot2[0]: # a planet and its moon
				return other.peri, other.apo
			# a planet and the moon of another
			dmin, dmax = self.distance(other.parent.orbit)
			dmin -= other.apo
			dmax += other.apo
			return dmin, dmax
		if 1 < len(ot1) and ot1[1] == ot2[0]: # moon, planet
			return other.distance(self)
		if len(ot1) > 1 < len(ot2) and ot1[1] == ot2[1]: # moon, moon
			dmin, dmax = self.parent.orbit.distance(other.parent.orbit)
			dmin -= self.apo + other.apo
			dmax += self.apo + other.apo
			return dmin, dmax
		raise NotImplementedError('this type of orbital relationship is not supported')
		

	def distance_to(self, other, t: float) -> float:
		# type: (Orbit, Orbit, float) -> float
		"""Distance between orbits at time t (m)"""
		# ~65μs avg.
		a, b = self.cartesian(t)[:3], other.cartesian(t)[:3]
		return hypot(*(i-j for i, j in zip(a, b)))

	def eccentric_anomaly(self, t: float = 0) -> float:
		"""Eccentric anomaly (radians)"""
		# ~6μs avg.
		# get new anomaly
		tau = 2*pi
		tol = 1e-8
		e, p = self.e, self.p
		# dt = day * t
		M = (self.man + tau*t/p) % tau
		assert isfinite(M)
		# E = M + e*sin(E)
		E = M
		while 1: # ~2 digits per loop
			E_ = M + e*sin(E)
			if abs(E-E_) < tol:
				return E_
			E = E_

	def mean_anomaly_delta(self, t: float = 0) -> float:
		"""Change in mean anomaly over a period of time"""
		return ((t / self.p) % 1) * 2*pi

	def get_resonance(self, other, sigma: int = 3):
		# type: (Orbit, Orbit, int) -> Tuple[int, int]
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
			if best[2] < 1 - erf(sigma/SQRT2):
				break
		return best[:2]

	def phase_angle(self, other) -> float:
		"""Phase angle for transfer orbits, fraction of other's orbit (rad)"""
		# https://forum.kerbalspaceprogram.com/index.php?/topic/62404-how-to-calculate-phase-angle-for-hohmann-transfer/#comment-944657
		d, h = other.a, self.a
		return pi/((d/h)**1.5)

	def plot(self) -> None:
		"""Plot orbit with pyplot"""
		n = 1000
		ts = [i*self.p/n for i in range(n)]
		cs = [self.cartesian(t) for t in ts]
		xs, ys, zs, vxs, vys, vzs = zip(*cs)

		fig = plt.figure(figsize=(7, 7))
		ax = plt.axes(projection='3d')
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
		# type: (Orbit, float) -> Orbit
		"""Get a resonant orbit from this one"""
		p = {key: value for key, value in self.properties.items()}
		p['sma'] = self.a * ratio**(2/3)
		return Orbit(**p)

	def stretch_to(self, other):
		# type: (Orbit, Orbit) -> Orbit
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
		# type: (Orbit, Orbit) -> float
		"""Synodic period of two orbits (s)"""
		return synodic(self.p, other.p)

	def t_collision(self, other) -> float:
		# type: (Orbit, Body) -> float
		"""Collision timescale: other is a body object orbiting the same primary, out in (s)"""
		# Hamilton and Burns (Science 264, 550-553, 1994)
		# U_r / U is baaaasically 1, right? :^)
		assert self.crosses(other.orbit)
		line1 = pi * sqrt(sin(self.i)**2 + sin(other.orbit.i)**2)
		line2 = (other.orbit.a / other.radius)**2 * self.p
		timescale = line1*line2
		v_col = self.e * other.orbit.v
		if v_col < other.v_e:
			# from personal correspondence with D. Hamilton
			correction_factor = 1 + (other.v_e/v_col)**2
			return timescale / correction_factor
		return timescale

	def t_kozai(self, other) -> float:
		# type: (Orbit, Body) -> float
		"""Kozai oscillation timescale (s)"""
		# other is type Body
		e, M, m_2, p, p_2 = other.orbit.e, self.parent.mass, other.mass, self.p, other.orbit.p
		return M/m_2*p_2**2/p*(1-e**2)**1.5

	def tisserand(self, other) -> float:
		# type: (Orbit, Orbit) -> float
		"""Tisserand's parameter (dimensionless)"""
		a, a_P, e, i = self.a, other.a, self.e, self.relative_inclination(other)
		return a_P/a + 2*cos(i) * sqrt(a/a_P * (1-e**2))
	# """Compute optimal transfer burn (m/s, m/s, m/s, s)"""
	"""
	def transfer(self, other, t: float = 0,
		delta_t_tol: float = 1, delta_x_tol: float = 1e7, dv_tol: float = .1) -> Tuple[float, float, float, float]:

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
	"""

	def true_anomaly(self, t: float = 0, ecc_anom_cache = None) -> float:
		"""True anomaly (rad)"""
		# ~8μs avg.
		e = self.e
		E = self.eccentric_anomaly(t) if ecc_anom_cache is None else ecc_anom_cache
		return 2 * atan2(sqrt(1+e) * sin(E/2), sqrt(1-e) * cos(E/2))

	def v_at(self, r: float) -> float:
		"""Orbital velocity at radius (m/s)"""
		return sqrt(self.parent.mu*(2/r-1/self.a))


class Rotation:
	def __init__(self, **properties):
		self.properties = properties

	@property
	def axis_vector(self) -> Tuple[float, float, float]:
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
	def density(self) -> float:
		"""Density of atmosphere at surface, approximation, kg/m^3
		Does not account for composition or temperature, only pressure."""
		return 1.225 * self.surface_pressure / earth.atmosphere.surface_pressure

	@property
	def greenhouse(self) -> float:
		"""Increase in radiative forcing caused by greenhouse gases (W/m^2)"""
		ghg = {
			# https://en.wikipedia.org/wiki/IPCC_list_of_greenhouse_gases
			# in W/m^2 per mol fraction
			'CH4': 0.48 / (1801e-9 - 700e-9),
			'CO2': 1.82 / (391e-6 - 278e-6),
			'N2O': 0.17 / (324e-9 - 265e-9),
		}
		return sum(ghg[i] * self.partial_pressure(i)/atm for i in ghg if i in self.composition) * self.surface_pressure / atm

	@property
	def mesopause(self) -> float:
		"""Altitude of Mesopause, approximation, m
		See notes for Tropopause."""
		return self.altitude(earth.atmosphere.pressure(85000)) # avg.

	@property
	def scale_height(self) -> float:
		"""Scale height (m)"""
		return self.properties['scale_height']

	@property
	def stratopause(self) -> float:
		"""Altitude of Stratopause, approximation, m
		See notes for Tropopause."""
		return self.altitude(earth.atmosphere.pressure(52500)) # avg.

	@property
	def surface_pressure(self) -> float:
		"""Surface pressure (Pa)"""
		return self.properties['surface_pressure']

	@property
	def tropopause(self) -> float:
		"""Altitude of Tropopause, approximation, m
		Uses Earth's atmosphere as a model, so,
		for Earthlike planets, this is generally accurate.
		Otherwise, treat as a first-order approximation,
		likely within a factor of 2 from the true value.
		Might return negative numbers if atmosphere is too thin."""
		return self.altitude(earth.atmosphere.pressure(13000)) # avg.

	# methods
	def altitude(self, pressure: float) -> float:
		"""Altitude at which atm has pressure, in Pa"""
		return -self.scale_height*log(pressure/self.surface_pressure)

	def partial_pressure(self, molecule: str) -> float:
		"""Partial pressure of a molecule on the surface (Pa)"""
		return self.surface_pressure * self.composition[molecule]

	def pressure(self, altitude: float) -> float:
		"""Pressure at altitude (Pa)"""
		return self.surface_pressure * exp(-altitude / self.scale_height)

	def wind_pressure(self, v: float) -> float:
		"""Pressure of wind going v m/s, in Pa"""
		return self.density * v**2


class Body:
	def __init__(self, **properties):
		self.properties = properties

	# orbital properties
	@property
	def orbit(self) -> Orbit:
		return self.properties['orbit']

	@property
	def orbit_rot_synodic(self) -> float:
		"""Synodic period of moon orbit (self) and planetary rotation (s)"""
		return synodic(self.orbit.p, self.orbit.parent.rotation.p)

	@property
	def esi(self) -> float:
		# Radius, Density, Escape Velocity, Temperature
		"""Earth similarity index (dimensionless)"""
		r, rho, T = self.radius, self.density, self.temp
		r_e, rho_e = earth.radius, earth.density
		esi1 = 1-abs((r-r_e)/(r+r_e))
		esi2 = 1-abs((rho-rho_e)/(rho+rho_e))
		esi3 = 1-abs((self.v_e-earth.v_e)/(self.v_e+earth.v_e))
		esi4 = 1-abs((T-earth.temp)/(T+earth.temp))
		return esi1**(.57/4)*esi2**(1.07/4)*esi3**(.7/4)*esi4**(5.58/4)

	@property
	def hill(self) -> float:
		"""Hill Sphere (m)"""
		a, e, m, M = self.orbit.a, self.orbit.e, self.mass, self.orbit.parent.mass
		return a*(1-e)*(m/3/M)**(1/3)

	@property
	def max_eclipse_duration(self) -> float:
		"""Maximum theoretical eclipse duration (s)"""
		star_r = self.orbit.parent.star.radius
		planet_a = self.orbit.parent.orbit.apo
		planet_r = self.orbit.parent.radius
		moon_a = self.orbit.peri
		# moon_r = self.radius
		theta = atan2(star_r - planet_r, planet_a)
		shadow_r = planet_r-moon_a*tan(theta)
		orbit_fraction = shadow_r / (pi*self.orbit.a)
		return self.orbit.p * orbit_fraction

	@property
	def nadir_time(self) -> float:
		"""One-way speed of light lag between nadir points of moon (self) and planet (s)"""
		d = self.orbit.a - self.orbit.parent.radius - self.radius
		return d/c

	@property
	def soi(self) -> float:
		"""Sphere of influence (m)"""
		return self.orbit.a*(self.mass/self.orbit.parent.mass)**.4

	@property
	def star(self):
		# type: (Body) -> Star
		"""Get the nearest star in the hierarchy"""
		p = self.orbit.parent
		return p if isinstance(p, Star) else p.star

	@property
	def star_dist(self) -> float:
		"""Get the distance to the nearest star in the hierarchy"""
		p = self.orbit.parent
		return self.orbit.a if isinstance(p, Star) else p.star_dist

	@property
	def star_radius(self) -> float:
		"""Get the radius of the nearest star in the hierarchy"""
		p = self.orbit.parent
		return p.radius if isinstance(p, Star) else p.star_radius

	@property
	def temp(self) -> float:
		"""Planetary equilibrium temperature (K)"""
		a, R, sma, T = self.albedo, self.star_radius, self.star_dist, self.star.temperature
		return T * (1-a)**.25 * sqrt(R/2/sma)

	@property
	def temp_subsolar(self) -> float:
		"""Planetary temperature at subsolar point(K)"""
		# based on formula from Reichart
		return self.temp * SQRT2

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
		return inf if t == T else (t*T)/(T-t)

	@property
	def v_tan(self) -> float:
		"""Equatorial rotation velocity (m/s)"""
		return 2*pi*self.radius/self.rotation.p

	@property
	def rotational_energy(self) -> float:
		"""Rotational Energy (J)"""
		i = 2/5 * self.mass * self.radius**2
		omega = 2*pi / self.rotation.p
		return 1/2 * i * omega**2

	@property
	def atmosphere_mass(self) -> float:
		"""Mass of the atmosphere (kg)"""
		return self.atmosphere.surface_pressure * self.area / self.surface_gravity

	# atmospheric properties
	@property
	def atm_retention(self) -> float:
		"""Checks if v_e is high enough to retain compound; molmass in kg/mol"""
		# initial assessment from :
		# https://upload.wikimedia.org/wikipedia/commons/4/4a/Solar_system_escape_velocity_vs_surface_temperature.svg
		# revised based on formulas derived from:
		# Reichart, Dan. "Lesson 6 - Earth and the Moon". Astronomy 101: The Solar System, 1st ed.
		try:
			t = self.greenhouse_temp
		except KeyError:
			t = self.temp
		return 887.364 * t / (self.v_e)**2

	@property
	def atmosphere(self) -> Atmosphere:
		return self.properties['atmosphere']

	@property
	def greenhouse_temp(self) -> float:
		"""Planetary equilibrium temperature w/ greenhouse correction (K)"""
		# VENUS TARGET = 737
		# EARTH TARGET = 287.91
		# MARS TARGET = 213
		# TITAN TARGET = 94 (DO NOT USE)
		# https://www.desmos.com/calculator/p1dobf2cvm
		# todo: remove reliance on the C1 and C2 bits, they don't really make sense..
		# maybe try fitting the min() to arctan()???
		gh = self.atmosphere.greenhouse
		o = self.orbit
		while o.parent != self.star:
			o = o.parent.orbit
		insolation = self.star.radiation_pressure_at(o.a)
		C1 = 1.58725
		C2 = 0.0653011
		# print(self.temp, gh/insolation)
		return self.temp * C1 * (gh / insolation)**C2

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
	def categories(self) -> set:
		"""List all possible categories for body"""
		ice_giant_cutoff = 80*earth.mass # https://en.wikipedia.org/wiki/Super-Neptune
		categories = set()
		if isinstance(self, Star):
			return {'Star'}
		mass = 'mass' in self.properties and self.mass
		rounded = search('Miranda').mass <= mass # Miranda is the smallest solar system body which might be in HSE
		if 'Star' not in self.orbit.parent.categories:
			categories.add('Moon')
			if rounded:
				categories.add('Major Moon')
			else:
				categories.add('Minor Moon')
			return categories
		# substellar
		if 13*jupiter.mass < mass:
			return {'Brown Dwarf'}
		radius = 'radius' in self.properties and self.radius
		if mass and 1 <= self.planetary_discriminant:
			categories.add('Planet')
			# earth-relative
			if earth.mass < mass < ice_giant_cutoff:
				categories.add('Super-earth')
			elif mass < earth.mass:
				categories.add('Sub-earth')
				if search('Ceres').radius < radius < mercury.radius:
					categories.add('Mesoplanet')
			# USP
			if self.orbit.p < day:
				categories.add('Ultra-short period planet')
			# absolute
			if self.mass < 10*earth.mass or (mars.density <= self.density and self.mass < jupiter.mass): # to prevent high-mass superjupiters
				categories.add('Terrestrial Planet')
				if 'composition' in self.properties and set('CO') <= set(self.composition) and self.composition['O'] < self.composition['C']:
					categories.add('Carbon Planet')
				if 10*earth.mass < self.mass:
					categories.add('Mega-Earth')
				try:
					temperature = self.greenhouse_temp if 'atmosphere' in self.properties else self.temp
					if earth.greenhouse_temp < temperature < 300: # death valley avg. 298K
						# https://en.wikipedia.org/wiki/Desert_planet
						categories.add('Desert Planet')
					elif temperature < 260 and .9 < self.albedo:
						# https://en.wikipedia.org/wiki/Ice_planet
						categories.add('Ice Planet')
					elif 1000 < temperature:
						# https://en.wikipedia.org/wiki/Lava_planet
						categories.add('Lava Planet')
				except KeyError:
					pass
			else:
				categories.add('Giant Planet')
				if .1 < self.orbit.e:
					categories.add('Eccentric Jupiter')
				if 'atmosphere' in self.properties and 'composition' in self.atmosphere.properties and 'He' in self.atmosphere.composition and .5 < self.atmosphere.composition['He']:
					categories.add('Helium Planet')
				if mass < ice_giant_cutoff:
					categories.add('Ice Giant')
					if neptune.mass < self.mass:
						categories.add('Super-Neptune')
					if 1.7*earth.radius < self.radius < 3.9*earth.radius:
						categories.add('Gas Dwarf')
					if self.orbit.a < au:
						categories.add('Hot Neptune')
				else:
					categories.add('Gas Giant')
					if jupiter.mass < self.mass:
						categories.add('Super-Jupiter')
					if self.density < saturn.density:
						categories.add('Puffy Planet')
					if self.orbit.p < 10*day:
						categories.add('Hot Jupiter')
			return categories
		# subplanetary
		categories.add('Minor Planet')
		if 9e20 < mass: # smallest dwarf planet is Ceres
			categories.add('Dwarf Planet')
		# crossers
		# [print(n, b.categories) for n, b in universe.items() if any('Crosser' in c for c in b.categories)]
		for planet_name, body in solar_system.items():
			if planet_name in {'Moon', 'Sun'}:
				continue
			if self.orbit.peri < body.orbit.apo and body.orbit.peri < self.orbit.apo:
				categories.add('{} Crosser'.format(planet_name))
			if self.orbit.get_resonance(body.orbit, 2) == (1, 1): # even the worst matches for jupiter trojans are just over 2 sigma certainty
				categories.add('{} Trojan'.format(planet_name))
		# NEO
		if self.orbit.peri < 1.3*au:
			categories.add('NEO')
			if self.orbit.peri < earth.orbit.apo:
				if 70 < radius or 1e9 < mass:
					categories.add('PHO')
				if earth.orbit.a < self.orbit.a:
					categories.add('Apollo Asteroid')
			else:
				categories.add('Amor Asteroid')
			if self.orbit.a < earth.orbit.a:
				if earth.orbit.peri < self.orbit.apo:
					categories.add('Aten Asteroid')
				else:
					categories.add('Atira Asteroid')
		# orbit distance
		if self.orbit.a < jupiter.orbit.a: # asteroids which aren't necessarily NEOs
			if 2.06*au < self.orbit.a < 3.28*au:
				# [print(n, b.categories) for n, b in universe.items() if 'Asteroid Belt' in b.categories]
				categories.add('Asteroid Belt')
				if self.orbit.a < 2.5*au:
					categories.add('Inner Main Belt')
					if 2.26*au < self.orbit.a < 2.48*au and .035 < self.orbit.e < .162 and 5*deg < self.orbit.i < 8.3*deg:
						categories.add('Vesta Family')
					if 2.3*au < self.orbit.a and self.orbit.i < 18*deg:
						categories.add('Main Belt I Asteroid')
				elif self.orbit.a < 2.82*au:
					categories.add('Middle Main Belt')
					if self.orbit.i < 33*deg:
						if self.orbit.a < 2.76*au:
							categories.add('Main Belt IIa Asteroid')
						else:
							categories.add('Main Belt IIb Asteroid')
				else:
					categories.add('Outer Main Belt')
					if self.orbit.e < .35 and self.orbit.i < 30*deg:
						if self.orbit.a < 3.03*au:
							categories.add('Main Belt IIIa Asteroid')
						else:
							categories.add('Main Belt IIIb Asteroid')
				# Alinda
				if 2.45*au < self.orbit.a < 2.56*au:
					categories.add('Alinda Asteroid')
			# non-main group asteroids
			elif self.orbit.a < mercury.orbit.a:
				categories.add('Vulcanoid')
			elif .96*au < self.orbit.a < 1.04*au and self.orbit.i < 4.4*deg and self.orbit.e < .084:
				categories.add('Arjuna Asteroid')
			elif 1.78*au < self.orbit.a < 2*au and self.orbit.e < .18 and 16*deg < self.orbit.i < 34*deg:
				categories.add('Hungaria Group')
			# Jupiter resonance groups
			res_groups = {
				(1, 3): 'Alinda',
				(1, 2): 'Hecuba Gap',
				(4, 7): 'Cybele',
				(2, 3): 'Hilda',
				(3, 4): 'Thule',
			}
			resonance = self.orbit.get_resonance(jupiter.orbit)
			if resonance in res_groups:
				categories.add('Resonant Asteroid')
				categories.add('{} Asteroid'.format(res_groups[resonance]))
		elif self.orbit.a < neptune.orbit.a:
			# https://en.wikipedia.org/wiki/Centaur_(small_Solar_System_body)#Discrepant_criteria
			categories.add('Centaur')
		else:
			categories.add('TNO')
			if rounded:
				categories.add('Plutoid')
			# resonance
			resonance = self.orbit.get_resonance(neptune.orbit)
			if max(resonance) <= 11:
				categories.add('Resonant KBO')
				if resonance == (2, 3):
					categories.add('Plutino')
				elif resonance == (1, 2):
					categories.add('Twotino')
				elif resonance == (1, 1):
					categories.add('Neptune Trojan')
				else:
					categories.add(str(resonance)[1:-1].replace(', ', ':') + ' res')
			# KBO versus SDO
			if self.orbit.a < 50*au:
				categories.add('KBO')
				if self.orbit.e < .2 and 11 < max(resonance):
					# https://en.wikipedia.org/wiki/Classical_Kuiper_belt_object#DES_classification
					# slightly modified to allow Makemake to get in
					categories.add('Cubewano')
					if self.orbit.i < 5*deg and self.orbit.e < .1:
						categories.add('Cold Population Cubewano')
					else:
						categories.add('Hot Population Cubewano')
				# haumea family
				if 41.6*au < self.orbit.a < 44.2*au and .07 < self.orbit.e < .2 and 24.2*deg < self.orbit.i < 29.1*deg:
					categories.add('Haumea Family')
			else:
				categories.add('SDO')
				if 40*au < self.orbit.peri:
					categories.add('Detached Object')
				if 150*au < self.orbit.a:
					categories.add('ETNO')
					if 38*au < self.orbit.peri < 45*au and .85 < self.orbit.e:
						categories.add('ESDO')
					if 40*au < self.orbit.peri < 60*au:
						categories.add('EDDO')
					if 50*au < self.orbit.peri:
						categories.add('Sednoid')
		# Damocloid
		if self.orbit.peri < jupiter.orbit.a and 8*au < self.orbit.a and .75 < self.orbit.e:
			# https://en.wikipedia.org/wiki/Damocloid
			categories.add('Damocloid')
		return categories

	@property
	def category(self) -> str:
		"""Attempt to categorize the body [DEPRECATED, USE CATEGORIES INSTEAD]"""
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
			"'Density         {} kg/m³'.format(round(self.density))",
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
			"'Temperature     {} K'.format(round((self.atmosphere.greenhouse if 'atmosphere' in self.properties else 1)*self.temp, 2))",
			"'Greenhouse Eff. {} K'.format(round((self.atmosphere.greenhouse-1)*self.temp, 2))",
		)
		string = []
		for line in lines:
			try:
				string.append(eval(line))
			except (KeyError, OverflowError):
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
		return 3*GRAV*self.mass**2/(5*self.radius)

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
	def metals(self) -> Tuple[Dict[str, Optional[int]], bool]:
		"""Metal data for metal/mining reports"""
		symbols = 'Fe Ni Cu Pt Au Ag U Co Al'.split(' ')
		assume = False
		try:
			ms = {sym: (self.composition[sym] if sym in self.composition else None) for sym in symbols}
		except KeyError:
			if isinstance(self, Star):
				target = sun
			elif 10*earth.mass < self.mass:
				target = jupiter
				print('jupiter')
			else:
				target = earth
			ms = {sym: target.composition[sym] if sym in target.composition else 0 for sym in symbols}
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
		ms = {sym: Mass(ms[sym]*mass, 'astro') for sym in production if ms[sym]}
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
		return GRAV * self.mass

	@property
	def name(self) -> str:
		return self.properties['name'] if 'name' in self.properties else str(self)

	@property
	def planetary_discriminant(self) -> float:
		"""Margot's planetary discriminant (dimensionless)"""
		a, m, M = self.orbit.a/au, self.mass/earth.mass, self.orbit.parent.mass/sun.mass
		# https://en.wikipedia.org/wiki/Clearing_the_neighbourhood#Margot's_%CE%A0
		k = 807
		return k*m/(M**(5/2)*a**(9/8))

	@property
	def planetary_discriminant2(self) -> float:
		"""Stern-Levinston planetary discriminant (dimensionless)"""
		a, m = self.orbit.a, self.mass/self.orbit.parent.mass
		# https://en.wikipedia.org/wiki/Clearing_the_neighbourhood#Stern%E2%80%93Levison's_%CE%9B
		# https://www.boulder.swri.edu/~hal/PDF/planet_def.pdf
		k = 9.84e32
		return k * m**2 * a**-1.5

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
			temperature = self.greenhouse_temp
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
		return sqrt(2*self.mu/self.radius)

	@property
	def volume(self) -> float:
		"""Volume (m^3)"""
		return 4/3*pi*self.radius**3

	# satellite properties
	@property
	def synchronous(self) -> float:
		"""SMA of synchronous orbit (m)"""
		return (self.mu*self.rotation.p**2/4/pi**2)**(1/3)

	# double underscore methods
	def __gt__(self, other) -> bool:
		# type: (Body, Body) -> bool
		# WA uses radius, so despite my better judgement, so will I
		return other.radius < self.radius

	def __lt__(self, other) -> bool:
		# type: (Body, Body) -> bool
		return self.radius < other.radius

	# methods
	def acc_towards(self, other, t: float) -> float:
		# type: (Body, Body, float) -> float
		"""Acceleration of self towards other body at time t (m/s^2)"""
		return self.force_between(other, t) / self.mass

	def acc_vector_towards(self, other, t: float) -> Tuple[float, float, float]:
		# type: (Body, Body, float) -> Tuple[float, float, float]
		"""Acceleration vector of self towards other body at time t (m/s^2)"""
		a, b = self.orbit.cartesian(t)[:3], other.orbit.cartesian(t)[:3]
		dx, dy, dz = [i-j for i, j in zip(a, b)]
		scale_factor = hypot(dx, dy, dz)
		acc = self.acc_towards(other, t)
		ax, ay, az = [acc*i/scale_factor for i in (dx, dy, dz)] # scale_factor*i is in [0, 1]
		# print(self.acc_towards(other, t))
		# print(ax, ay, az)
		# print('plot_grav_acc_vector(sedna, planet_nine)')
		return ax, ay, az

	def angular_diameter(self, other: Orbit) -> Tuple[float, float]:
		"""Angular diameter, min and max (rad)"""
		dmin, dmax = self.orbit.distance(other)
		return self.angular_diameter_at(dmax), self.angular_diameter_at(dmin)

	def angular_diameter_at(self, dist: float) -> float:
		"""Angular diameter at distance (rad)"""
		return 2*atan2(self.radius, dist)

	def app_mag(self, other: Orbit) -> Tuple[float, float]:
		"""Apparent magnitude, min and max (dimensionless)"""
		dmin, dmax = self.orbit.distance(other)
		return self.app_mag_at(dmax), self.app_mag_at(dmin)

	def app_mag_at(self, dist: float) -> float:
		"""Apparent magnitude at distance (dimensionless)"""
		# https://astronomy.stackexchange.com/a/38377
		a_p, r_p, d_s, v_sun = self.albedo, self.radius, self.star_dist, self.star.abs_mag
		h_star = v_sun + 5*log10(au/(10*pc))
		d_0 = 2*au*10**(h_star/5)
		h = 5 * log10(d_0 / (2*r_p * sqrt(a_p)))
		return h + 5*log10(d_s * dist / au**2)

	def atm_supports(self, molmass: float) -> bool:
		"""Checks if v_e is high enough to retain compound; molmass in kg/mol"""
		return molmass > self.atm_retention

	def atmospheric_molecular_density(self, altitude: float) -> float:
		"""Molecular density at an altitude (m) in (mol/m^3)"""
		return self.atmosphere.pressure(altitude)/(gas_constant*self.greenhouse_temp)

	def bielliptic(self, inner: Orbit, mid: Orbit, outer: Orbit) -> float:
		"""Bielliptic transfer delta-v (m/s)"""
		i, m, o = inner.a, mid.a, outer.a
		mu = self.mu
		a1 = (i+m)/2
		a2 = (m+o)/2
		dv1 = sqrt(2*mu/i-mu/a1)-sqrt(mu/i)
		dv2 = sqrt(2*mu/m-mu/a2)-sqrt(2*mu/m-mu/a1)
		dv3 = sqrt(2*mu/o-mu/a2)-sqrt(mu/o)
		return dv1 + dv2 + dv3

	def force_between(self, other, t: float = 0) -> float:
		"""Force between two bodies at time t (N)"""
		r = self.orbit.distance_to(other.orbit, t)
		return self.mu * other.mass/r**2

	def hohmann(self, inner: Orbit, outer: Orbit) -> float:
		"""Hohmann transfer delta-v (m/s)"""
		i, o = inner.a, outer.a
		mu = self.mu
		dv1 = sqrt(mu/i)*(sqrt(2*o/(i+o))-1)
		dv2 = sqrt(mu/i)*(1-sqrt(2*i/(i+o)))
		return dv1 + dv2

	def lunar_eclipse(self) -> None:
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

	def net_grav_acc_vector(self, system, t: float) -> Tuple[float, float, float]:
		"""Net gravitational acceleration vector (m/s^2)"""
		vectors = [self.acc_vector_towards(body, t) for body in system.bodies if body != self]
		return tuple(map(sum, zip(*vectors)))

	def orbit_in(self, t: float) -> float:
		"""Returns SMA (m) of an orbit with period t (s)"""
		return (self.mu*(t/(2*pi))**2)**(1/3)

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
	def habitable_zone(self) -> Tuple[float, float]:
		"""Inner and outer habitable zone (m)"""
		center = au*sqrt(self.luminosity/sun.luminosity)
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
		"""Stellar radiation pressure at a distance (W/m^2)"""
		wattage = self.luminosity / sun.luminosity
		return G_SC * wattage / (dist/au)**2

	def radiation_force_at(self, obj: Body, t: float = 0) -> float:
		"""Stellar radiation force on a planet at a time (W)"""
		dist = hypot(*obj.orbit.cartesian(t)[:3]) / au
		area = obj.area / 2
		return self.radiation_pressure_at(dist) * area


class System:
	"""Set of orbiting bodies"""
	def __init__(self, parent: Body, *bodies: Body) -> None:
		"""Star system containing bodies.\nDoesn't need to be ordered."""
		self.parent = parent
		self.bodies = set(bodies)

	@property
	def sorted_bodies(self) -> list:
		"""List of bodies sorted by semimajor axis"""
		return sorted(list(self.bodies), key=lambda x: x.orbit.a)

	def plot(self) -> None:
		"""Plot system with pyplot"""
		# see above plot for notes and sources
		n = 1000

		fig = plt.figure(figsize=(7, 7))
		ax = plt.axes(projection='3d')
		ax.set_title('Orbit')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		ax.set_zlabel('z (m)')
		ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
		ax.text(0, 0, 0, self.parent.name, size=8, color='k', zorder=4)
		for body in self.bodies:
			cs = [body.orbit.cartesian(t*body.orbit.p/n) for t in range(n)]
			xs, ys, zs, vxs, vys, vzs = zip(*cs)
			ax.plot(xs, ys, zs, color='k', zorder=1)
			ax.scatter(xs[0], ys[0], zs[0], marker='o', s=15, zorder=3)
			ax.text(xs[0], ys[0], zs[0], body.name, size=8, color='k', zorder=4)

		axisEqual3D(ax)
		plt.show()

	def plot2d(self) -> None:
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

	def mass_pie(self) -> None:
		"""Mass pie chart"""
		system_masses = [i.mass for i in self.bodies]

		plt.subplot(1, 2, 1)
		plt.title('System mass')
		plt.pie(system_masses + [self.parent.mass])

		plt.subplot(1, 2, 2)
		plt.title('System mass (excl. primary)')
		plt.pie(system_masses)

		plt.show()

	def grav_sim(self, focus: Body = None, res = 10000, p=25, skip=100) -> None:
		"""Model newtonian gravity"""
		# from time import time
		# solar_system_object.grav_sim()
		parent = self.parent
		bodies = list(self.bodies) + [parent]
		steps = res*p
		timestep = (focus.orbit.p if focus else min(b.orbit.p for b in self.bodies)) / res

		body_xv = [body.orbit.cartesian() if body != parent else (0, 0, 0, 0, 0, 0) for body in bodies]
		body_x = [list(elem[:3]) for elem in body_xv]
		body_v = [list(elem[3:]) for elem in body_xv]
		body_a = [[0, 0, 0] for _ in bodies]
		body_mu_cache = [b.mu for b in bodies]
		xs = []
		# start = time()
		# compute positions
		for i in range(steps):
			xs.append([])
			for (bi, body) in enumerate(bodies):
				if (focus and body != focus):
					xs[i].append((0, 0, 0) if body == parent else body.orbit.cartesian(timestep * i)[:3])
					continue
				# copy xs to output list
				xs[i].append(body_x[bi][:])
				# new x, v
				for dim in range(3):
					body_x[bi][dim] += body_v[bi][dim] * timestep
					body_v[bi][dim] += body_a[bi][dim] * timestep
				# reset acc...
				body_a[bi] = [0, 0, 0]
				# new acc
				for (other_body_i, other_body_x) in enumerate(body_x):
					if other_body_i == bi:
						continue
					dx = tuple(other_body_x[dim] - body_x[bi][dim] for dim in range(3))
					r = hypot(*dx)
					a = body_mu_cache[other_body_i] / (r*r)
					for dim in range(3):
						body_a[bi][dim] += a * dx[dim] / r
		# print(time() - start)
		# draw
		fig = plt.figure(figsize=(7, 7))
		ax = plt.axes(projection='3d')
		ax.set_title('Orbit')
		ax.set_xlabel('x (m)')
		ax.set_ylabel('y (m)')
		ax.set_zlabel('z (m)')
		# ax.scatter(0, 0, 0, marker='*', color='y', s=50, zorder=2)
		for (p_i, planet_coords) in enumerate(list(zip(*xs))):
			x, y, z = zip(*planet_coords[::skip])
			ax.plot(x, y, z, color='k', zorder=p_i)
			ax.scatter(x[0], y[0], z[0], marker='o', s=15, zorder=len(bodies) + p_i)
			ax.scatter(x[-1], y[-1], z[-1], marker='o', s=15, zorder=2*len(bodies) + p_i)
			ax.text(*planet_coords[0], bodies[p_i].name + "^", size=8, color='k', zorder=3*len(bodies) + p_i)
			ax.text(*planet_coords[-1], bodies[p_i].name + "$", size=8, color='k', zorder=4*len(bodies) + p_i)

		axisEqual3D(ax)
		plt.show()


# functions
def accrete(star_mass: float = 2e30, particle_n: int = 25000) -> System:
	from random import randint, uniform
	# from time import time
	# constants
	# particle_n = 20000 # 500 -> 10 ms; 25000 -> 911 ms
	sweep_area = 1.35 # venus-earth
	# begin
	extra_mass = .0014 * star_mass
	particle_mass = extra_mass / particle_n
	a_range = mercury.orbit.a * star_mass/sun.mass, neptune.orbit.a * star_mass/sun.mass
	# list of [mass, sma]
	particles = [[particle_mass, uniform(*a_range)] for _ in range(particle_n)]
	# start = time()
	for _ in range(10*particle_n):
		i_c = randint(0, len(particles)-1)
		chosen_particle = particles[i_c]
		m_c, a_c = chosen_particle
		# find target
		for i, (m_o, a_o) in enumerate(particles):
			# merge
			if a_c/sweep_area < a_o < a_c*sweep_area and chosen_particle != (m_o, a_o):
				m_new = m_c + m_o
				a_new = (m_c*a_c + m_o*a_o)/m_new
				particles[i_c] = m_new, a_new
				del particles[i]
				break
		# early end
		if len(particles) <= 6:
			break
	# print(time() - start)
	# construct system
	star = stargen(star_mass)
	out = System(star, *(Body(**{'mass': mass, 'orbit': Orbit(**{
		'parent': star,
		'sma': a,
		'e': uniform(0, .1), 'i': uniform(0, 4*deg), 'lan': uniform(0, 2*pi), 'aop': uniform(0, 2*pi), 'man': uniform(0, 2*pi)
	})}) for mass, a in particles))
	return out # .plot2d()


def keplerian(parent: Body, cartesian: Tuple[float, float, float, float, float, float]) -> Orbit:
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
	E = 2 * atan(tan(nu/2) / sqrt((1+eccentricity)/(1-eccentricity)))
	# print(nu, eccentricity, '-> E =', E)
	# E = 2 * atan2(sqrt((1+eccentricity)/(1-eccentricity)), tan(nu/2))
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


def distance_audio(orbit1: Orbit, orbit2: Orbit):
	from mochaaudio import pcm_to_wav, play_file
	"""Play wave of plot_distance
	Encoding   | Signed 16-bit PCM
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
	frames = (0x7FFF * np.array(xs)).astype(np.int16)
	# print(frames, np.amin(frames), np.amax(frames))
	# end plot_distance

	print("* done recording")
	# with open('audacity.txt', 'wb+') as file:
	#	file.write(frames.tobytes())
	pcm_to_wav(frames.tobytes())
	# play audio
	play_file('output.wav')


def load_data(seed: dict) -> dict:
	import os
	from json import load

	def convert_body(data: dict, current_universe: dict, datatype=Body) -> Body:
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
		def value_parse(value) -> float:
			try:
				return eval(value) if isinstance(value, str) else value
			except (NameError, SyntaxError):
				return value

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
	def warnings(data) -> None:
		"""Attempt to find missing data"""
		for name, body in data.items():
			# print('Checking {}...'.format(name))
			# check for missing albedo data
			if 'albedo' not in body.properties:
				if 6e19 < body.mass < 1.5e29 and 'orbit' in body.properties and body.orbit.a < 600*au:
					print('Albedo data missing from {}, should be easy to find'.format(name))
			# check for missing atm data
			elif 'atmosphere' not in body.properties and \
					'mass' in body.properties and body.atm_retention < .131 and \
					body.orbit.a < 67*au:
				print('Atmosphere data missing from {}, atmosphere predicted'.format(name))

	loc = os.path.dirname(os.path.abspath(__file__)) + '\\mochaastro'
	universe_data = seed
	for file in [f for f in os.listdir(loc) if f.endswith('.json')]:
		print('Loading', file, '...')
		json_data = load(open(loc + '\\' + file, 'r'))
		for obj in json_data:
			universe_data[obj['name']] = convert_body(obj, universe_data, (Star if obj['class'] == 'star' else Body))
	warnings(universe_data)
	return universe_data


def plot_delta_between(orbit1: Orbit, orbit2: Orbit):
	"""Plot system with pyplot"""
	resolution = 100
	orbits = 8
	limit = max([orbit1, orbit2], key=lambda x: x.apo).apo*2
	outerp = max([orbit1, orbit2], key=lambda x: x.p).p

	fig = plt.figure(figsize=(7, 7))
	ax = plt.axes(projection='3d')
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


def plot_grav_acc(body1: Body, body2: Body) -> None:
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


def plot_grav_accs(body: Body, other_bodies: Tuple[Body]) -> None:
	"""Plot total gravitational acceleration from one body to several others over several orbits"""
	resolution = 1000
	orbits = 8
	outerp = max(body, *other_bodies, key=lambda x: x.orbit.p).orbit.p

	plt.figure(figsize=(7, 7))
	plt.title('Body Delta')
	plt.xlabel('time since epoch (s)')
	plt.ylabel('acceleration (m/s^2)')
	plt.yscale('log')
	ts = [(t*outerp/resolution) for t in range(orbits*resolution)]
	xs = [sum(body.acc_towards(body2, t) for body2 in other_bodies) for t in ts]
	plt.plot(ts, xs, color='k')

	plt.show()


def plot_grav_acc_vector(body1: Body, body2: Body) -> None:
	"""Plot gravitational acceleration vector from one body to another over several orbits"""
	resolution = 100
	orbits = 8
	outerp = max([body1, body2], key=lambda x: x.orbit.p).orbit.p

	fig = plt.figure(figsize=(7, 7))
	ax = plt.axes(projection='3d')
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


def test_functions() -> None:
	universe_sim(sun)
	solar_system_object.grav_sim()


def universe_sim(parent: Body, t: float=0, size: Tuple[int, int]=(1024, 640), selection: Body=None, smoothMotion=False) -> None:
	# TODO
	# moon display
	# comet tails
	"""Use pygame to show the system of [parent] and all subsystems"""
	import pygame
	from pygame import gfxdraw # DO NOT REMOVE THIS IT BREAKS SHIT
	from time import sleep, time
	from math import hypot
	from mochamath import dist
	from mochaunits import round_time

	orbit_res = 48
	dot_radius = 2
	black, blue, beige, white, grey = (0,)*3, (0, 0, 255), (255, 192, 128), (255,)*3, (128,)*3
	red = blue[::-1]
	fps = 30
	timerate = 1/fps
	paused = False
	vectors = False
	# target = parent # until user selects a new one
	mouse_sensitivity = 10 # pixels

	width, height = size
	center = width//2, height//2
	max_a = 20*parent.radius
	current_a = max_a
	if selection is None:
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
	def is_onscreen(coords: Tuple[int, int], buffer: int=0) -> bool:
		x, y = coords
		return -buffer <= x <= width+buffer and -buffer <= y <= height+buffer

	def are_onscreen(points: tuple, buffer: int=0) -> bool:
		"""return true if any onscreen"""
		for point in points:
			if is_onscreen(point):
				return True
		return False

	def center_on_selection(coords: Tuple[int, int]) -> Tuple[int, int]:
		return tuple(i-j+k for i, j, k in zip(coords, current_coords, center))

	def coord_remap(coords: Tuple[float, float]) -> Tuple[int, int]:
		a = current_a if smoothMotion else max_a
		b = height/width * a
		xmap = linear_map((-a, a), (0, width))
		ymap = linear_map((-b, b), (height, 0))
		x, y = coords
		return int(round(xmap(x))), int(round(ymap(y)))

	def is_hovering(body: Body, coords: Tuple[int, int]) -> bool:
		apparent_radius = body.radius * width/(2*current_a) if 'radius' in body.properties else dot_radius
		return dist(coords, pygame.mouse.get_pos()) < max(mouse_sensitivity, apparent_radius)

	# display body
	def point(at: Tuple[int, int], radius: float, color: Tuple[int, int, int]=white, fill: bool=True, highlight: bool=False) -> None:
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
	def show_body(body: Body, coords: Tuple[int, int], name: str='') -> None:
		"""Coords are the actual screen coords"""
		hovering = is_hovering(body, coords)
		r = body.radius if 'radius' in body.properties else 1
		point(coords, r, white, True, hovering)
		# show name
		apparent_radius = r * width/(2*current_a)
		name_coords = tuple(int(i+apparent_radius) for i in coords)
		text(name, name_coords, font_normal, grey)

	# display arrow
	def arrow(a: Tuple[int, int], b: Tuple[int, int], color: Tuple[int, int, int]=red) -> None:
		"""Coords are the actual screen coords"""
		tip_scale = 10
		# the arrow "tip"
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
		arrow_tip = left_tip, right_tip, b
		if any(is_onscreen(i) for i in arrow_tip):
			pygame.draw.aalines(screen, color, True, arrow_tip)

	# display text
	def text(string: str, at: Tuple[int, int], size: int=font_normal, color: Tuple[int, int, int]=white, shadow: bool=False) -> None:
		if not is_onscreen(at):
			return None
		this_font = pygame.font.SysFont('Courier New', size)
		string = string.replace('\t', ' '*4)
		# first, shadow
		if shadow:
			text(string, (at[0]+1, at[1]+1), size, black)
		# then, real text
		for i, line in enumerate(string.split('\n')):
			textsurface = this_font.render(line, True, color)
			x, y = at
			y += i*size
			screen.blit(textsurface, (x, y))

	# precompute orbits (time0, time1, ..., timeN)
	def precompute_orbit(obj: Body) -> Tuple[Tuple[int, int], ...]:
		total_res = int(orbit_res * (obj.orbit.e + 1)**3)
		return tuple(obj.orbit.cartesian(t+i*obj.orbit.p/total_res)[:2] for i in range(total_res))

	def zoom(r: float=0) -> None:
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
			if not any(20 < abs(i-j//2) for i, j in zip(coord_remap((body.orbit.a,)*2), (width, height))):
				continue
			# redraw orbit
			coords = center_on_selection(coord_remap(body.orbit.cartesian(t)[:2]))
			hovering = is_hovering(body, coords)
			points = tuple(map(center_on_selection, map(coord_remap, orbit)))
			if not (are_onscreen(points) or is_onscreen(coords)):
				continue
			pygame.draw.lines(screen, red if hovering else color, True, points)
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
			if hovering:
				if pygame.mouse.get_pressed()[0]:
					selection = body
					zoom()
				elif pygame.mouse.get_pressed()[2]:
					universe_sim(body, t, (width, height))
		# print((time()-for_start)/len(orbits))
		# print date
		try:
			current_date = str(round_time(J2000+timedelta(seconds=t)))
		except OverflowError:
			current_date = '>10000' if 0 < t else '<0'
		information = current_date + ' (x{0}){1}'.format(int(fps*timerate), ' [PAUSED]' if paused else '') + '\n' + \
					'Width: '+pretty_dim(Length(2*current_a, 'astro'))
		text(information, (0, height-font_large*2), font_large, white, True)
		# print FPS
		text(str(round(1/(time()-start_time)))+' FPS', (width-font_normal*4, 0), font_normal, red)
		# print selection data
		text(selection.data, (0, 0), font_small, (200, 255, 200), True)
		# refresh
		refresh()
		# post-render operations
		# event handling
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
				return print('Thank you for using mochaastro2.py!~')
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
				elif event.button == 3: # check for "go back!"
					if 'orbit' in parent.properties and 'parent' in parent.orbit.properties and \
						not any(is_hovering(body, center_on_selection(coord_remap(body.orbit.cartesian(t)[:2]))) for _, body, _, _ in orbits):
						universe_sim(parent.orbit.parent, t, (width, height), selection)
				elif event.button == 4: # zoom in
					zoom(-1)
				elif event.button == 5: # zoom out
					zoom(1)
			elif event.type == pygame.VIDEORESIZE:
				width, height = event.size
				center = width//2, height//2
				selection_coords = center
				zoom()
				pygame.display.set_mode(event.size, pygame.RESIZABLE)
				# print(event.size) # debug
		# smooth zoom
		current_a = (max_a + current_a)/2 if smoothMotion else max_a
		current_coords = tuple((i + j)//2 for i, j in zip(selection_coords, current_coords)) if smoothMotion else selection_coords
		# refresh title
		title = '{}, {} System - {}'.format(inverse_universe[selection], inverse_universe[parent], current_date)
		pygame.display.set_caption(title)
		# sleep
		wait_time = 1/fps - (time() - start_time)
		if 0 < wait_time:
			sleep(wait_time)


# bodies - this file only contains the sun, moon, and planets. json files provide the rest.
# USE BOND ALBEDO PLEASE
sun = Star(**{
	'name': 'Sun',
	'orbit': Orbit(**{
		'sma': 2.7e20,
	}),
	'rotation': Rotation(**{
		'period': 25.05*day,
		'tilt': .127,
		'ra': 286.13*deg,
		'dec': 63.87*deg,
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
	'name': 'Mercury',
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
		'ra': 281.01*deg,
		'dec': 61.42*deg,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 26000,
		'surface_pressure': 5e-10,
		'composition': { # appx.
			'H2': .75,
			'O2':  .25,
		},
	}),
	'mass': 3.3011e23,
	'radius': 2.4397e6,
	'albedo': .088,
})

venus = Body(**{
	'name': 'Venus',
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
		'ra': 272.76*deg,
		'dec': 67.16*deg,
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
	'name': 'Earth',
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
		'composition': { # https://en.wikipedia.org/wiki/Atmospheric_chemistry#Atmospheric_composition
			'N2':  .78084,
			'O2':  .20946,
			'H2O':  .01,
			'Ar':  .00934,
			'CO2': 4.2e-4,
			'Ne':  1.818e-5,
			'He':  5.24e-6,
			'CH4': 1.87e-6,
			'Kr':  1.14e-6,
			'H2':  5.5e-7,
			'N2O':  9e-8,
			'NO2':  2e-8,
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
		'Os': 9e-7,
		'Ir': 9e-7,
		'Nd': 8.4e-7,
		'Dy': 4.6e-7,
		'Nb': 4.4e-7,
		'La': 4.4e-7,
		'Rb': 4e-7,
		'Gd': 3.7e-7,
		'Br': 3e-7,
		'Te': 3e-7,
		'Er': 3e-7,
		'Yb': 3e-7,
		'Sm': 2.7e-7,
		'Sn': 2.5e-7,
		'Rh': 2.4e-7,
		'Pb': 2.3e-7,
		'B':  2e-7,
		'Hf': 1.9e-7,
		'Pr': 1.7e-7,
		'W':  1.7e-7,
		'Au': 1.6e-7,
		'Eu': 1e-7,
		'Ho': 1e-7,
		'Cd': 8e-8,
		'Re': 8e-8,
		'Tb': 7e-8,
		'Th': 6e-8,
		'Be': 5e-8,
		'Ag': 5e-8,
		'Sb': 5e-8,
		'I':  5e-8,
		'Tm': 5e-8,
		'Lu': 5e-8,
		'Cs': 4e-8,
		'Ta': 3e-8,
		'Hg': 2e-8,
		'U':  2e-8,
		'In': 1e-8,
		'Tl': 1e-8,
		'Bi': 1e-8,
	},
	'mass': 5.97237e24,
	'radius': 6.371e6,
	'albedo': .306,
})

moon = Body(**{
	'name': 'Moon',
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
		'ra': 270*deg,
		'dec': 66.54*deg,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 41860,
		'surface_pressure': 1e-7,
	}),
    'composition': { # https://www.permanent.com/l-apollo.htm
        'O':  .446,
        'Si': .21,
        'Al': .133,
        'Ca': .1068,
        'Fe': .0487,
        'Mg': .0455,
        'Na': 3.1e-3,
        'Ti': 3.1e-3,
        'Cr': 8.5e-4,
        'K':  8e-4,
        'Mn': 6.75e-4,
        'P':  5e-4,
        'C':  1e-4,
        'H':  5.6e-5,
        'Cl': 1.7e-5,
    },
	'mass': 7.342e22,
	'radius': 1.7371e6,
	'albedo': .136,
})

mars = Body(**{
	'name': 'Mars',
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
		'ra': 317.68*deg,
		'dec': 52.89*deg,
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
	'name': 'Jupiter',
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
		'ra': 268.05*deg,
		'dec': 64.49*deg,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 27000,
		'surface_pressure': 7e5,
		'composition': {
			'H2':    .89,
			'He':   .1,
			'CH4':  .003,
			'NH3':  .00026,
			'C2H6': .000006,
			'H2O':  .000004,
		},
	}),
	'mass': 1.8982e27,
	'radius': 6.9911e7,
	'albedo': .503,
})

saturn = Body(**{
	'name': 'Saturn',
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
		'ra': 40.60*deg,
		'dec': 83.54*deg,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 59500,
		'surface_pressure': 1.4e5,
		'composition': {
			'H2':    .963,
			'He':   .0325,
			'CH4':  .0045,
			'NH3':  .000125,
			'C2H6': .000007,
		},
	}),
	'mass': 5.6834e26,
	'radius': 5.8232e7,
	'albedo': .342,
})

uranus = Body(**{
	'name': 'Uranus',
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
		'ra': 257.43*deg,
		'dec': -15.10*deg,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 27700,
		'surface_pressure': 1e5, # arbitrary guess
		'composition': {
			'H2':    .83,
			'He':   .15,
			'CH4':  .023,
		},
	}),
	'mass': 8.681e25,
	'radius': 2.5362e7,
	'albedo': .3,
})

neptune = Body(**{
	'name': 'Neptune',
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
		'ra': 299.36*deg,
		'dec': 43.46*deg,
	}),
	'atmosphere': Atmosphere(**{
		'scale_height': 19700,
		'surface_pressure': 1e5, # arbitrary guess
		'composition': {
			'H2':    .8,
			'He':   .19,
			'CH4':  .015,
		},
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
solar_system = {
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
}
solar_system_object = System(sun, *(i for i in solar_system.values() if i is not sun and i is not moon))
universe = load_data(solar_system.copy())
# planet_nine.orbit.plot
# distance_audio(earth.orbit, mars.orbit)
# solar_system.sim()
# burn = earth.orbit.transfer(mars.orbit)
# universe_sim(sun)
# solar_system_object.grav_sim()