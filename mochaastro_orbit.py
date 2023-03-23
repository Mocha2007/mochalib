"""Defines the Orbit class"""
from datetime import datetime
from math import acos, atan2, cos, erf, hypot, isfinite, pi, sin, sqrt
from typing import Tuple
from mochaastro_common import apsides2ecc, axisEqual3D, c, J2000, \
	resonance_probability, SQRT2, synodic


class Orbit:
	"""Stores Keplerian orbital elements,
	and processes them through its methods."""
	def __init__(self, **properties) -> None:
		self.properties = properties

	@property
	def a(self) -> float:
		"""Semimajor Axis (m)"""
		return self.properties['sma']

	@property
	def anomalistic_period(self) -> float:
		"""The anomalistic orbital period (s)"""
		return synodic(self.p, 2*pi/self.apsidial_precession)

	@property
	def aop(self) -> float:
		"""Argument of periapsis (radians)"""
		return self.properties['aop']

	@property
	def apo(self) -> float:
		"""Apoapsis (m)"""
		return (1+self.e)*self.a

	@property
	def apsidial_precession(self) -> float:
		"""Apsidial precession rate predicted by general relativity (rad/s)"""
		# https://en.wikipedia.org/wiki/Apsidal_precession#General_relativity
		# note that the formula given is PER ORBIT and thus
		# needs to be divided by T to get it in seconds
		return 24*pi**3 * self.a**2 / (self.p**3 * c**2 * (1-self.e**2))

	@property
	def copy(self):
		# type: (Orbit) -> Orbit
		"""return a deep copy of this Orbit object"""
		from copy import deepcopy
		return deepcopy(self)

	@property
	def e(self) -> float:
		"""Eccentricity (dimensionless)"""
		return self.properties['e']

	@property
	def epoch_offset(self) -> float:
		"""Based on the epoch in properties,
		return the time in seconds to subtract when computing position"""
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
		"""Get ordered list of parent bodies"""
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

	def close_approach(self, other, t: float = 0, n: float = 1, \
		    delta_t_tolerance: float = 1, after_only=True) -> float:
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
		import matplotlib.pyplot as plt
		n = 1000
		ts = [i*self.p/n for i in range(n)]
		cs = [self.cartesian(t) for t in ts]
		xs, ys, zs, _, __, ___ = zip(*cs)

		plt.figure(figsize=(7, 7))
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
		import numpy as np
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
