"""Classes for celestial bodies for mochaastro.py"""
from math import atan2, cos, exp, hypot, inf, log, log10, pi, sin, sqrt, tan
from typing import Dict, Optional, Tuple
from mochaunits import Mass, Time
from mochaastro_common import atan, atm, au, bar, c, day, DEFAULT_DENSITIES, \
	deg, gas_constant, GRAV, G_SC, kB, L_0, MOCHAASTRO_DEBUG, MOLAR_MASS, \
	pc, photometry, planck, PHOTOMETRIC_FILTER, REDUCED_PLANCK, SQRT2, \
	search, stargen, STEFAN_BOLTZMANN, synodic, year
from mochaastro_orbit import Orbit

class Rotation:
	"""Stores information about the body's rotation and processes it."""
	def __init__(self, **properties):
		self.properties = properties

	@property
	def angular_velocity(self) -> float:
		"""Angular velocity (rad/s)"""
		return 2*pi/self.p

	@property
	def axis_vector(self) -> Tuple[float, float, float]:
		"""Return unit vector of axis (dimensionless)"""
		theta, phi = self.dec, self.ra
		return sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)

	@property
	def dec(self) -> float:
		"""Declination of axis (rad)"""
		return self.properties['dec'] if 'dec' in self.properties else 0

	@property
	def p(self) -> float:
		"""Period (s)"""
		return self.properties['period'] if 'period' in self.properties else inf

	@property
	def ra(self) -> float:
		"""Right ascension of axis (rad)"""
		return self.properties['ra'] if 'ra' in self.properties else 0

	@property
	def tilt(self) -> float:
		"""Axial tilt (rad)"""
		return self.properties['tilt'] if 'tilt' in self.properties else 0


class Atmosphere:
	"""Stores information about the body's atmosphere and processes it."""
	def __init__(self, **properties):
		self.properties = properties
		self.body = None # type: Body
		"""Parent body object"""

	@property
	def composition(self) -> dict:
		"""Composition (element: fraction)"""
		return self.properties['composition'] if 'composition' in self.properties else dict()

	@property
	def density(self) -> float:
		"""Density of atmosphere at surface, approximation, (kg/m^3)
		Does not account for composition or temperature, only pressure."""
		from mochaastro_data import earth
		return 1.225 * self.surface_pressure / earth.atmosphere.surface_pressure

	@property
	def greenhouse(self) -> float:
		"""Increase in radiative forcing caused by greenhouse gases (W/m^2)"""
		ghg = {
			# https://en.wikipedia.org/wiki/IPCC_list_of_greenhouse_gases
			# in W/m^2 per mol fraction
			'CH4': 0.48 / 1801e-9,
			'CO2': 1.82 / 391e-6,
			'H2O': 2.7 / 0.0025, # "which is responsible overall for about half of all atmospheric gas forcing."
			'N2O': 0.17 / 324e-9,
			# 'SO2': -1.3 / 15e-9, # https://en.wikipedia.org/wiki/File:Physical_Drivers_of_climate_change.svg
		}
		return sum(val * self.partial_pressure(key)/atm for (key, val) in ghg.items() \
	     	if key in self.composition) * self.surface_pressure / atm

	@property
	def mass(self) -> float:
		"""Mass of the atmosphere (kg)"""
		return self.surface_pressure * self.body.area / self.body.surface_gravity

	@property
	def mesopause(self) -> float:
		"""Altitude of Mesopause, approximation, m
		See notes for Tropopause."""
		from mochaastro_data import earth
		return self.altitude(earth.atmosphere.pressure(85000)) # avg.

	@property
	def molar_mass(self) -> float:
		"""Computed molar mass (kg/mol)"""
		return sum(MOLAR_MASS[chem]*amt for (chem, amt) in self.composition.items() if chem in MOLAR_MASS)

	@property
	def optical_depth(self) -> float:
		"""Optical depth (dimensionless?)"""
		# http://saspcsus.pbworks.com/w/file/fetch/64696386/planet%20temperatures%20with%20surface%20cooling%20parameterized.pdf
		return sum(self.partial_optical_depth(chem) for chem in self.composition)

	@property
	def predicted_scale_height(self) -> float:
		"""Computed scale height (m)"""
		return gas_constant * self.body.temperature / (self.molar_mass * self.body.surface_gravity)

	@property
	def scale_height(self) -> float:
		"""Scale height (m)"""
		return self.properties['scale_height'] if 'scale_height' in self.properties else self.predicted_scale_height

	@property
	def stratopause(self) -> float:
		"""Altitude of Stratopause, approximation, m
		See notes for Tropopause."""
		from mochaastro_data import earth
		return self.altitude(earth.atmosphere.pressure(52500)) # avg.

	@property
	def surface_pressure(self) -> float:
		"""Surface pressure (Pa)"""
		return self.properties['surface_pressure'] if 'surface_pressure' in self.properties else 0

	@property
	def thermopause(self) -> float:
		"""Altitude of Thermopause, approximation, m
		See notes for Tropopause."""
		from mochaastro_data import earth
		return self.altitude(earth.atmosphere.pressure(750000)) # varies b/w 500 and 1000 km

	@property
	def tropopause(self) -> float:
		"""Altitude of Tropopause, approximation, m
		Uses Earth's atmosphere as a model, so,
		for Earthlike planets, this is generally accurate.
		Otherwise, treat as a first-order approximation,
		likely within a factor of 2 from the true value.
		Might return negative numbers if atmosphere is too thin."""
		from mochaastro_data import earth
		return self.altitude(earth.atmosphere.pressure(13000)) # avg.

	# methods
	def altitude(self, pressure: float) -> float:
		"""Altitude at which atm has pressure, in Pa"""
		return -self.scale_height*log(pressure/self.surface_pressure)

	def molecular_density(self, altitude: float) -> float:
		"""Molecular density at an altitude (m) in (mol/m^3)"""
		return self.pressure(altitude)/(gas_constant*self.body.temperature)

	def partial_optical_depth(self, molecule: str) -> float:
		"""Optical depth (dimensionless?)"""
		# http://saspcsus.pbworks.com/w/file/fetch/64696386/planet%20temperatures%20with%20surface%20cooling%20parameterized.pdf
		gases = {
			'CO2': (0.025, 0.53),
			'H2O': (0.277, 0.3),
			# I tried est'ing with https://www.desmos.com/calculator/mjt8s2d0ja but failed so here are guesses
			'CH4': (5e-3, 0.6), # fitted to titan...
			'N2O': (1e-2, 0.6), # untested; wild guess
			'SO2': (-1e-2, 0.6), # untested; wild guess
		}
		return gases[molecule][0] * self.partial_pressure(molecule)**gases[molecule][1] if molecule in gases else 0

	def partial_pressure(self, molecule: str) -> float:
		"""Partial pressure of a molecule on the surface (Pa)"""
		return self.surface_pressure * self.composition[molecule]

	def pressure(self, altitude: float) -> float:
		"""Pressure at altitude (Pa)"""
		return self.surface_pressure * exp(-altitude / self.scale_height)

	def wind_pressure(self, v: float) -> float:
		"""Pressure of wind going v m/s, in Pa"""
		return self.density * v**2


class Hydrosphere:
	"""Stores information about the body's hydrosphere and processes it."""
	def __init__(self, **properties):
		self.properties = properties
		self.body = None # type: Body
		"""Parent body object"""

	@property
	def bottom_pressure(self) -> float:
		"""Pressure at bottom of hydrosphere (Pa)"""
		return self.pressure(self.max_depth)

	@property
	def composition(self) -> dict:
		"""Composition (element: fraction)"""
		return self.properties['composition'] if 'composition' in self.properties else dict()

	@property
	def coverage(self) -> dict:
		"""Fraction of the body's surface covered by the hydrosphere (dimensionless)"""
		return self.properties['coverage'] if 'coverage' in self.properties else 0

	@property
	def density(self) -> float:
		"""Density of hydrosphere at surface (kg/m^3)"""
		return self.properties['density'] if 'density' in self.properties else 0

	@property
	def max_depth(self) -> float:
		"""Maximum depth of hydrosphere (m)"""
		return self.properties['depth'] if 'depth' in self.properties else 0

	@property
	def surface_pressure(self) -> float:
		"""Surface pressure (Pa)"""
		return self.body.atmosphere.surface_pressure

	# methods
	def depth(self, pressure: float) -> float:
		"""Depth at which atm has pressure, in Pa"""
		return (pressure - self.surface_pressure) / (self.density * self.body.surface_gravity)

	def pressure(self, depth: float) -> float:
		"""Pressure at depth (Pa)"""
		return self.surface_pressure + self.density * self.body.surface_gravity * depth


class Body:
	"""Stores information about celestial bodies and processes it."""
	def __init__(self, **properties):
		self.properties = properties
		if 'atmosphere' in properties:
			self.atmosphere.body = self
		if 'hydrosphere' in properties:
			self.hydrosphere.body = self

	@property
	def temperature(self) -> float:
		"""The actual temperature, whether that be through just solar irradiation, greenhouse, or gravitational contraction... (K)"""
		t =  self.greenhouse_temp if 'atmosphere' in self.properties else self.temp if 'orbit' in self.properties else 0
		return max(stargen(self.mass).temperature, t)

	# orbital properties
	@property
	def orbit(self) -> Orbit:
		"""Returns orbit object."""
		return self.properties['orbit']

	@property
	def orbit_rot_synodic(self) -> float:
		"""Synodic period of moon orbit (self) and planetary rotation (s)"""
		return synodic(self.orbit.p, self.orbit.parent.rotation.p)

	@property
	def esi(self) -> float:
		"""Earth similarity index (dimensionless)"""
		from mochaastro_data import earth
		# Radius, Density, Escape Velocity, Temperature
		r, rho, T = self.radius, self.density if self.radius else DEFAULT_DENSITIES[self.classification], self.temperature
		T, v_e = self.temperature, self.v_e if self.radius else 0
		r_e, rho_e = earth.radius, earth.density
		esi1 = 1-abs((r-r_e)/(r+r_e))
		esi2 = 1-abs((rho-rho_e)/(rho+rho_e))
		esi3 = 1-abs((v_e-earth.v_e)/(v_e+earth.v_e))
		esi4 = 1-abs((T-earth.temperature)/(T+earth.temperature))
		return esi1**(.57/4)*esi2**(1.07/4)*esi3**(.7/4)*esi4**(5.58/4)

	@property
	def hill(self) -> float:
		"""Hill Sphere (m)"""
		a, e, m, M = self.orbit.a, self.orbit.e, self.mass, self.orbit.parent.mass
		return a*(1-e)*(m/3/M)**(1/3)

	@property
	def irradiance_total(self) -> float:
		"""Total solar irradiance at SMA. (W/m^2)
		https://en.wikipedia.org/wiki/Solar_irradiance"""
		o = self.orbit
		while o.parent != self.star:
			o = o.parent.orbit
		return self.star.radiation_pressure_at(o.a)

	@property
	def irradiance_global(self) -> float:
		"""Globally averaged solar irradiance at SMA. (W/m^2)
		https://en.wikipedia.org/wiki/Solar_irradiance"""
		return self.irradiance_total / 4

	@property
	def irradiance_surface(self) -> float:
		"""Globally averaged surface solar irradiance at SMA. (W/m^2)
		https://en.wikipedia.org/wiki/Solar_irradiance"""
		return (1 - self.albedo) * self.irradiance_global

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
		"""Get the distance to the nearest star in the hierarchy (m)"""
		p = self.orbit.parent
		return self.orbit.a if isinstance(p, Star) else p.star_dist

	@property
	def star_radius(self) -> float:
		"""Get the radius of the nearest star in the hierarchy (m)"""
		p = self.orbit.parent
		return p.radius if isinstance(p, Star) else p.star_radius

	@property
	def sudarsky_class(self) -> int:
		"""Sudarsky's gas giant classification (1-5, dimensionless)"""
		# https://en.wikipedia.org/wiki/Sudarsky's_gas_giant_classification
		t = max(self.temp, stargen(self.mass).temperature)
		if t < 150:
			return 1
		if t < 300:
			return 2
		if t < 850:
			return 3
		if t < 1400:
			return 4
		return 5

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
	def J2(self) -> float:
		"""J2 second dynamic form factor (dimensionless?)"""
		# https://en.wikipedia.org/wiki/Nodal_precession#Rate_of_precession
		return 2*self.oblateness/3 \
			- self.radius_equatorial**3 * self.rotation.angular_velocity**2 \
			/ (3 * self.mu)

	@property
	def oblateness_estimate(self) -> float:
		"""Returns estimated oblateness based on its density and rotation period.
		Theoretically, the oblateness should always be somewhat smaller due to
		mass concentration near the core. Decently accurate for terrestrial planets,
		but you may want to halve the result for gas/ice giants."""
		return 15*pi / (4 * GRAV * self.rotation.p**2 * self.density)

	@property
	def precession_period(self) -> float:
		"""Period of axial precession (s)"""
		return 2*pi / self.precession_rate

	@property
	def precession_rate(self) -> float:
		"""Rate of axial precession (rad/s)"""
		from mochaastro_data import universe
		# https://en.wikipedia.org/wiki/Axial_precession#Equations
		# http://www.columbia.edu/itc/ldeo/v1011x-1/jcm/Topic2/Topic2.html
		# (C-A) / A = H = oblateness
		# used to be / self.rotation.angular_velocity,
		# but that could cause a zero division error.
		# instead I use the inverse of it: self.p / (2*pi)
		SECOND_TERM = 3/(4*pi) * self.oblateness * cos(self.rotation.tilt) * self.rotation.p
		def dPsi(mu, a, e, i) -> float:
			return mu*(1-1.5*sin(i)**2)/(a**3 * (1-e**2)**1.5)
		STAR_RATE = dPsi(self.orbit.parent.mu, self.orbit.a, self.orbit.e, 0)
		MOONS = filter(lambda x: 'orbit' in x.properties and 'parent' in x.orbit.properties and \
			x.orbit.parent is self, universe.values())
		MOONS_RATE = sum(dPsi(moon.mu, moon.orbit.a, moon.orbit.e, moon.orbit.i) for moon in MOONS)
		# print(STAR_RATE*SECOND_TERM, MOONS_RATE*SECOND_TERM , '\n', list(MOONS))
		return (STAR_RATE + MOONS_RATE) * SECOND_TERM

	@property
	def rotation(self) -> Rotation:
		"""Returns the rotation object."""
		return self.properties['rotation']

	@property
	def solar_day(self) -> float:
		"""True solar day (s)"""
		t, T = self.rotation.p, self.orbit.p
		if pi/2 < self.rotation.tilt and 0 < t: # retrograde rotation
			t *= -1
		return inf if t == T else (t*T)/(T-t)

	@property
	def solar_year(self) -> float:
		"""Tropical year length in terms of local solar days"""
		return self.tropical_year / self.solar_day

	@property
	def tropical_year(self) -> float:
		"""Tropical year (s)"""
		return 1/(1/self.orbit.p + 1/self.precession_period)

	@property
	def v_r_orbit(self) -> float:
		"""Orbital velocity of a circular orbit around the body at a = R (m/s)"""
		return (self.radius/self.mu)**-.5

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

	# atmospheric properties
	@property
	def atmosphere(self) -> Atmosphere:
		"""Returns the atmosphere object."""
		return self.properties['atmosphere']

	@property
	def atm_retention(self) -> float:
		"""Checks if v_e is high enough to retain compound; molmass in kg/mol"""
		# initial assessment from :
		# https://upload.wikimedia.org/wikipedia/commons/4/4a/Solar_system_escape_velocity_vs_surface_temperature.svg
		# revised based on formulas derived from:
		# Reichart, Dan. "Lesson 6 - Earth and the Moon". Astronomy 101: The Solar System, 1st ed.
		return 887.364 * self.temperature / self.v_e**2

	@property
	def greenhouse_temp(self) -> float:
		"""Planetary equilibrium temperature w/ greenhouse correction (K)"""
		"""TARGETS:
				w/o		w/		this	e		%
		Venus	229 K	737 K	705 K	-32 K	-4.3%
		Earth	254 K	288 K	285 K	-3 K	-1.0%
		Mars	210 K	213 K	228 K	+15 K	+7.0%
		Titan	85 K	94 K	96 K	+2 K	+2.1%
		"""
		# http://saspcsus.pbworks.com/w/file/fetch/64696386/planet%20temperatures%20with%20surface%20cooling%20parameterized.pdf
		tau = self.atmosphere.optical_depth
		F = self.irradiance_surface
		# T_0 = self.temp * (1 + 0.75*tau)**0.25
		#tau_CO2 = self.atmosphere.partial_optical_depth('CO2')
		#tau_H2O = self.atmosphere.partial_optical_depth('H2O')
		F_CHEM = 0.75 * F * self.atmosphere.optical_depth
		tau_v = 0.354 + 0.0157*tau
		Fsi = F * exp(-tau_v)
		# https://www.desmos.com/calculator/mhnsmc5lrs
		# I did my own fit with my updated data...
		Fc = max(0, -27.6735 + 0.46872 * Fsi * tau)
		Fs = Fsi + F_CHEM - Fc
		emissivity = 0.95 # for venus and mars only; paper says 0.996 for earth but https://en.wikipedia.org/wiki/Emissivity#Emissivities_of_planet_Earth also says 0.95
		Ts = (Fs/(emissivity*STEFAN_BOLTZMANN))**0.25
		# Section 5
		# If water is liquid...
		"""
		if tau_H2O and water_phase(Ts, self.atmosphere.surface_pressure) == Phase.LIQUID:
			print('old tau_H2O =', tau_H2O)
			print('old Ts =', Ts)
			for _ in range(100):
				P_H2O = self.atmosphere.partial_pressure('H2O') * exp(0.0698 * (Ts - 288.15))
				tau_H2O = 0.277 * P_H2O ** 0.3
				F_H2O = 0.75 * F * tau_H2O
				Fs = Fsi + F_CO2 + F_H2O - Fc
				Ts_old = Ts
				Ts = (Fs/(emissivity*STEFAN_BOLTZMANN))**0.25
				if abs(Ts - Ts_old) < 0.001:
					print(_)
					break
			print('new tau_H2O =', tau_H2O)
			print('new Ts =', Ts)
		"""
		# Table 2
		if MOCHAASTRO_DEBUG:
			#print('tau_CO2', tau_CO2)
			#print('tau_H2O', tau_H2O)
			print('tau', tau)
			print()
			print('Fsi', Fsi)
			print('F_CHEM', F_CHEM)
			print('Fc', Fc)
			print()
			print('Fs', Fs)
			print('Ts', Ts)
		return Ts

	# hydrospheric properties
	@property
	def hydrosphere(self) -> Hydrosphere:
		"""Returns the hydrosphere object."""
		return self.properties['hydrosphere']

	# physical properties
	@property
	def albedo(self) -> float:
		"""Albedo (dimensionless)"""
		return self.properties['albedo'] if 'albedo' in self.properties else 0

	@property
	def area(self) -> float:
		"""Surface area (m^2)"""
		return 4*pi*self.radius**2

	@property
	def categories(self) -> set:
		"""List all possible categories for body"""
		from mochaastro_data import earth, jupiter, mars, mercury, neptune, saturn, solar_system, sun
		ice_giant_cutoff = 80*earth.mass # https://en.wikipedia.org/wiki/Super-Neptune
		categories = set()
		if isinstance(self, Star):
			return {'Star'}
		mass = 'mass' in self.properties and self.mass
		rounded = search('Miranda').mass <= mass
		HAS_ATMOSPHERE = 'atmosphere' in self.properties
		HAS_ATMOSPHERE_C = HAS_ATMOSPHERE and 'composition' in self.atmosphere.properties
		HAS_C = 'composition' in self.properties
		HAS_HYDROSPHERE = 'hydrosphere' in self.properties
		# Miranda is the smallest solar system body which might be in HSE
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
			if 'orbit' not in self.properties or 'parent' not in self.orbit.properties:
				categories.add('Rogue Planet')
			elif self.orbit.parent is not sun:
				categories.add('Exoplanet')
			else:
				categories.add(('Inferior' if self.orbit.a < earth.orbit.a else 'Superior') + ' Planet')
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
			# https://en.wikipedia.org/wiki/Hycean_planet
			if HAS_HYDROSPHERE and HAS_ATMOSPHERE_C and \
					self.hydrosphere.coverage > 0.5 < self.atmosphere.composition['H']:
				categories.add('Hycean Planet')
			# absolute
			if self.mass < 10*earth.mass or (mars.density <= self.density and self.mass < jupiter.mass): # to prevent high-mass superjupiters
				categories.add('Terrestrial Planet')
				if HAS_C and set('CO') <= set(self.composition) \
						and self.composition['O'] < self.composition['C']:
					categories.add('Carbon Planet')
				if 10*earth.mass < self.mass:
					categories.add('Mega-Earth')
				try:
					temperature = self.temperature
					if earth.temperature < temperature < 300: # death valley avg. 298K
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
				if HAS_ATMOSPHERE_C \
						and 'He' in self.atmosphere.composition \
						and 0.5 <= self.atmosphere.composition['He']:
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
					if 0.2 <= self.orbit.e:
						categories.add('Eccentric Jupiter')
			return categories
		# subplanetary
		categories.add('Minor Planet')
		if 9e20 < mass: # smallest dwarf planet is Ceres
			categories.add('Dwarf Planet')
		# solar system-centric categories...
		if self.orbit.parent is not sun:
			return categories # exit early
		# crossers
		# [print(n, b.categories) for n, b in universe.items()
		# if any('Crosser' in c for c in b.categories)]
		for planet_name, body in solar_system.items():
			if planet_name in {'Moon', 'Sun'}:
				continue
			if self.orbit.peri < body.orbit.apo and body.orbit.peri < self.orbit.apo:
				categories.add(f'{planet_name} Crosser')
			if self.orbit.get_resonance(body.orbit, 2) == (1, 1):
				# even the worst matches for jupiter trojans are just over 2 sigma certainty
				categories.add(f'{planet_name} Trojan')
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
					if 2.26*au < self.orbit.a < 2.48*au \
							and .035 < self.orbit.e < .162 \
							and 5*deg < self.orbit.i < 8.3*deg:
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
				categories.add(f'{res_groups[resonance]} Asteroid')
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
				if 41.6*au < self.orbit.a < 44.2*au \
						and .07 < self.orbit.e < .2 \
						and 24.2*deg < self.orbit.i < 29.1*deg:
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
	def circumference(self) -> float:
		"""Circumference (m)"""
		return 2*pi*self.radius

	@property
	def classification(self) -> str:
		"""Class name"""
		return self.properties['class']

	@property
	def composition(self) -> dict:
		"""Composition (element: fraction)"""
		return self.properties['composition'] if 'composition' in self.properties else dict()

	@property
	def data(self) -> str:
		"""Data Table a la Space Engine"""
		from mochaunits import Angle, Length, pretty_dim, Temperature # pylint: disable=unused-import
		from mochaastro_common import water_phase  # pylint: disable=unused-import
		from mochaastro_data import earth # pylint: disable=unused-import
		# used by eval
		# sadly the only alternative would be like, 5000 lines of try/except
		lines = (
			"'{} {}'.format(self.classification.title(), self.properties['name'])",
			"'Class              {}'.format(self.setype.title())",
			"'Radius             {} ({} Earth radii)'.format(pretty_dim(Length(self.radius)), round(self.radius/earth.radius, 3))",
			"'Mass               {}'.format(str(Mass(self.mass, 'astro', 'round=3')))",
			"'Density            {} kg/m³'.format(round(self.density))",
			"'ESI                {}'.format(round(self.esi, 3))",
			"'Composition        {}'.format(', '.join(map(lambda x: f'{x[0]} ({round(x[1]*100)}%)', sorted(list(self.composition.items()), key=lambda x: x[1], reverse=True)[:5])))",
			"'Absolute Mag.      {}'.format(round(self.app_mag(10*pc), 2))",
			"'Semimajor Axis     {}'.format(pretty_dim(Length(self.orbit.a, 'astro')))",
			"'Orbital Period     {} (sidereal)'.format(pretty_dim(Time(self.orbit.p, 'imperial')))",
			"'                   {} (tropical)'.format(pretty_dim(Time(self.tropical_year, 'imperial')))",
			"'                   {} (tropical sols)'.format(round(self.solar_year, 2))",
			"'Rotation Period    {} (sidereal)'.format(pretty_dim(Time(self.rotation.p, 'imperial')))",
			"'                   {} (synodic)'.format(pretty_dim(Time(self.solar_day, 'imperial')))",
			"'Axial Tilt         {}'.format(Angle(self.rotation.tilt, 'deg', 'round=1'))",
			"'Gravity            {} g'.format(round(self.surface_gravity/earth.surface_gravity, 3))",
			"'Atmos. Composition {}'.format(', '.join(map(lambda x: f'{x[0]} ({round(x[1]*100)}%)', sorted(list(self.atmosphere.composition.items()), key=lambda x: x[1], reverse=True)[:5])))",
			"'Atmos. Pressure    {} atm'.format(round(self.atmosphere.surface_pressure/earth.atmosphere.surface_pressure, 3))",
			"'Temperature        {}'.format(Temperature(self.temperature, 'celsius', 'round=2'))",
			"'Greenhouse Eff.    {}'.format(Temperature(self.greenhouse_temp - self.temp if 'atmosphere' in self.properties else 0, 'round=2'))",
			"'Water Phase        {}'.format(str(water_phase(self.temperature, self.atmosphere.surface_pressure if 'atmosphere' in self.properties else 0)))"
		)
		string = []
		for line in lines:
			try:
				string.append(eval(line)) # pylint: disable=eval-used
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
		return self.properties['mass'] if 'mass' in self.properties else 0

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
			string += f'\nBase Metals\n\tFe: {Fe}\n\tNi: {Ni}'
		if any([Pt, Au, Ag]):
			string += f'\nPrecious\n\tAu: {Au}\n\tAg: {Ag}\n\tPt: {Pt}'
		if any([Cu, U]):
			string += f'\nOther\n\tAl: {Al}\n\tCo: {Co}\n\tCu: {Cu}\n\tU: {U}'
		return string

	@property
	def metals(self) -> Tuple[Dict[str, Optional[int]], bool]:
		"""Metal data for metal/mining reports"""
		from mochaastro_data import earth, jupiter, sun
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
			string += f'\n\t{sym}: {mass} ({time})'
		return string

	@property
	def mu(self) -> float:
		"""Gravitational parameter (m^3/s^2)"""
		return GRAV * self.mass

	@property
	def name(self) -> str:
		"""Returns name of object, if available."""
		return self.properties['name'] if 'name' in self.properties else str(self)

	@property
	def oblateness(self) -> float:
		"""Oblateness (dimensionless)"""
		return self.properties['oblateness'] if 'oblateness' in self.properties else self.oblateness_estimate

	@property
	def planetary_discriminant(self) -> float:
		"""Margot's planetary discriminant (dimensionless)"""
		from mochaastro_data import earth, sun
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
		"""Mean Radius (m)"""
		return self.properties['radius'] if 'radius' in self.properties else 0

	@property
	def radius_equatorial(self) -> float:
		"""Equatorial Radius (m)"""
		return 2*self.radius / (2 - self.oblateness)

	@property
	def radius_polar(self) -> float:
		"""Polar Radius (m)"""
		return 2*(self.oblateness - 1)*self.radius / (self.oblateness - 2)

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
	def setype(self) -> str:
		"""Space Engine-like classifier"""
		from mochaastro_data import earth, jupiter
		GIANT = {'gas giant', 'ice giant'}
		# https://spaceengine.org/news/blog170924/
		me = self.mass / earth.mass
		mj = self.mass / jupiter.mass
		# Temperature
		temperature = self.temp
		try:
			temperature = self.greenhouse_temp
		except KeyError:
			pass
		if 1000 < temperature:
			heat = 'torrid'
		elif 500 < temperature:
			heat = 'hot'
		elif 330 < temperature:
			heat = 'warm'
		elif 250 < temperature:
			heat = 'temperate'
		elif 170 < temperature:
			heat = 'cool'
		elif 90 < temperature:
			heat = 'cold'
		else:
			heat = 'frigid'
		# volatiles
		HAS_ATMOSPHERE = 'atmosphere' in self.properties
		HAS_ATMOSPHERIC_COMPOSITION = HAS_ATMOSPHERE and 'composition' in self.atmosphere.properties
		p = self.atmosphere.surface_pressure if HAS_ATMOSPHERE and 'surface_pressure' in self.atmosphere.properties else 0
		hydrosphere_coverage = self.hydrosphere.coverage if 'hydrosphere' in self.properties else 0
		if p < 1e-9*bar:
			volatiles = 'airless'
		elif hydrosphere_coverage == 1:
			volatiles = 'superoceanic' if 100e3 < self.hydrosphere.max_depth else 'oceanic'
		elif 0.2 < hydrosphere_coverage:
			volatiles = 'marine'
		elif 0 < hydrosphere_coverage:
			volatiles = 'lacustrine'
		else: # corresponds to 'desertic' in the blogpost - I'm just guessing here...
			volatiles = 'arid' if HAS_ATMOSPHERIC_COMPOSITION and self.atmosphere.composition['H2O'] < 1e-4 else 'non-arid'
		# bulk
		INCLUDE_VOLATILES = True
		HAS_COMPOSITION = 'composition' in self.properties
		# I assume they mean https://en.wikipedia.org/wiki/Goldschmidt_classification ?
		siderophilic = 'Mn Fe Co Ni Mo Tc Ru Rh Pd W Re Os Ir Pt Au Sg Bh Hs Mt Ds Rg'.split(' ')
		siderophily = sum(self.composition[elem] for elem in siderophilic if elem in self.composition) \
			if HAS_COMPOSITION else 0
		carbon = self.composition['C'] if HAS_COMPOSITION and 'C' in self.composition else 0
		# I exclude O because that would cause Ice giants to be included
		lithophilic = 'Li Be B F Na Mg Al Si P Cl K Ca Sc Ti V Cr Br Rb Sr Y Zr Nb I Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta At Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rd Db Ts'.split(' ')
		lithophily = sum(self.composition[elem] for elem in lithophilic if elem in self.composition) \
			if HAS_COMPOSITION else 0
		nonmetals = 'H H2 He'.split(' ')
		nonmetallicity = sum(self.atmosphere.composition[elem] for elem in nonmetals if elem in self.atmosphere.composition) \
			if HAS_ATMOSPHERIC_COMPOSITION else 0
		rounded = 6.4e19 <= self.mass # Miranda
		if MOCHAASTRO_DEBUG:
			print(self.name, HAS_ATMOSPHERIC_COMPOSITION, HAS_COMPOSITION)
		if not rounded:
			bulk = 'asteroid'
		elif 'oceanic' in volatiles:
			bulk = 'aquaria'
		elif 0.5 < siderophily:
			bulk = 'ferria'
		elif 0.25 < carbon:
			bulk = 'carbonia'
		elif 0.25 < lithophily:
			bulk = 'terra'
		elif 0.25 < nonmetallicity:
			bulk = 'gas giant'
		else:
			bulk = 'ice giant'
		# insufficient data to safely predict, OVERRIDE!
		if not HAS_ATMOSPHERIC_COMPOSITION or (bulk not in GIANT and not HAS_COMPOSITION):
			bulk = 'gas giant' if 0.18 < mj else 'ice giant' if 7.3 < me else 'terra'
		if bulk in GIANT:
			INCLUDE_VOLATILES = False
		# prefix
		if bulk == 'ice giant':
			if me < 4:
				prefix = 'mini'
			elif me < 10:
				prefix = 'sub'
			elif me < 25:
				prefix = ''
			elif me < 62.5:
				prefix = 'super'
			else:
				prefix = 'mega'
		elif bulk == 'gas giant':
			if mj < 0.2:
				prefix = 'sub'
			elif mj < 2:
				prefix = ''
			elif mj < 10:
				prefix = 'super'
			else:
				prefix = 'mega'
		else:
			if me < 0.002:
				prefix = 'micro'
			elif me < 0.02:
				prefix = 'mini'
			elif me < 0.2:
				prefix = 'sub'
			elif me < 2:
				prefix = ''
			elif me < 10:
				prefix = 'super'
			else:
				prefix = 'mega'
		if ' ' in bulk:
			bulk = bulk.split(' ')
			bulk[1] = prefix + bulk[1]
			bulk = ' '.join(bulk)
			prefix = ''
		if INCLUDE_VOLATILES:
			return heat + ' ' + volatiles + ' ' + prefix + bulk
		return heat + ' ' + prefix + bulk

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

	def app_mag(self, dist: float) -> float:
		"""Apparent magnitude of this body at distance (dimensionless)"""
		# https://astronomy.stackexchange.com/a/38377
		a_p, r_p, d_s, v_sun = self.albedo, self.radius, self.star_dist, self.star.abs_mag
		h_star = v_sun + 5*log10(au/(10*pc))
		d_0 = 2*au*10**(h_star/5)
		h = 5 * log10(d_0 / (2*r_p * sqrt(a_p)))
		return h + 5*log10(d_s * dist / au**2)

	def app_mag_orbit(self, other: Orbit) -> Tuple[float, float]:
		"""Apparent magnitude of this body, min and max (dimensionless)"""
		dmin, dmax = self.orbit.distance(other)
		return self.app_mag(dmax), self.app_mag(dmin)

	def atm_supports(self, molmass: float) -> bool:
		"""Checks if v_e is high enough to retain compound; molmass in kg/mol"""
		return molmass > self.atm_retention

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
		"""Hohmann transfer delta-v (total) (m/s)"""
		return sum(self.hohmann_split(inner, outer))

	def hohmann_split(self, inner: Orbit, outer: Orbit) -> Tuple[float, float]:
		"""Hohmann transfer delta-v (by burn) (m/s)"""
		i, o = inner.a, outer.a
		mu = self.mu
		dv1 = sqrt(mu/i)*(sqrt(2*o/(i+o))-1)
		dv2 = sqrt(mu/o)*(1-sqrt(2*i/(i+o)))
		return dv1, dv2

	def lunar_eclipse(self) -> None:
		"""Draw maximum eclipsing radii"""
		import matplotlib.pyplot as plt
		from matplotlib.patches import Circle, Patch
		satellite, planet = self, self.orbit.parent
		a = satellite.orbit.peri
		r = planet.radius
		moon_radius = satellite.angular_diameter_at(a-r)
		umbra_radius = atan2(planet.umbra_at(a), a-r)
		penumbra_radius = atan2(planet.penumbra_at(a), a-r)

		_, ax = plt.subplots()
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
		slope = (planet.radius+star.radius) / planet.orbit.a
		# line drawn from "top of star" to "bottom of planet"
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
		import matplotlib.pyplot as plt
		from matplotlib.patches import Circle, Patch
		satellite, planet, star = self, self.orbit.parent, self.orbit.parent.orbit.parent
		moon_radius = satellite.angular_diameter_at(satellite.orbit.peri - planet.radius)
		star_radius = star.angular_diameter_at(planet.orbit.apo - planet.radius)

		_, ax = plt.subplots()
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
		slope = (planet.radius-star.radius) / planet.orbit.a
		# line drawn from "top of star" to "top of planet"
		return slope*distance + planet.radius


# ty https://www.python-course.eu/python3_inheritance.php
class Star(Body):
	"""Star object, which is like body but with a few extra parameters."""
	@property
	def abs_mag(self) -> float:
		"""Absolute Magnitude (dimensionless)"""
		return -2.5 * log10(self.luminosity / L_0)

	@property
	def color(self) -> Tuple[float, float, float]:
		"""Returns approximation of color in sRGB space (0-255 intensity)"""
		r = planck(PHOTOMETRIC_FILTER['RGB_R'], self.temperature)
		g = planck(PHOTOMETRIC_FILTER['RGB_G'], self.temperature)
		b = planck(PHOTOMETRIC_FILTER['RGB_B'], self.temperature)
		mag = max(r, g, b)
		r *= 255 / mag
		g *= 255 / mag
		b *= 255 / mag
		return tuple(map(round, (r, g, b)))

	@property
	def habitable_zone(self) -> Tuple[float, float]:
		"""Inner and outer habitable zone (m)"""
		from mochaastro_data import sun
		center = au * (self.temperature/sun.temperature)**2 * (self.radius / sun.radius) # sqrt(self.luminosity/sun.luminosity)
		inner = .95*center
		outer = 1.37*center
		return inner, outer

	@property
	def lifespan(self) -> float:
		"""Estimated lifespan (s)"""
		from mochaastro_data import sun
		# cf https://www.academia.edu/4301816/On_Stellar_Lifetime_Based_on_Stellar_Mass
		m, lum = self.mass / sun.mass, self.luminosity / sun.luminosity
		time = 8.839639544315635e17 * m / lum
		time *= 0.73 if m <= 0.45 else 0.4496 if m <= 2 else 0.511
		return time

	@property
	def luminosity(self) -> float:
		"""Luminosity (W)"""
		if 'luminosity' in self.properties:
			return self.properties['luminosity']
		if 'radius' in self.properties and 'temperature' in self.properties:
			return 4*pi*STEFAN_BOLTZMANN * self.temperature**4 * self.radius**2

	@property
	def metallicity(self) -> float:
		"""Metallicity, relative to the sun (dex)"""
		if 'metallicity' in self.properties:
			return self.properties['metallicity']
		from mochaastro_data import sun
		return log10(self.Z / sun.Z)

	@property
	def peakwavelength(self) -> float:
		"""Peak emission wavelength (m)"""
		return 2.8977729e-3/self.temperature

	@property
	def photosphere_pressure(self) -> float:
		"""Estimated pressure at bottom of photosphere (Pa)"""
		# okay here is my solution for pressure. I think of it this way.
		# It's probably proportional to surface gravity / surface area.
		# for the sun, that quotient is 4.5092805033996796e-17
		# but https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html says it's 125 mb (12500 Pa)
		# So I'll use that and hope for the best.
		return 12500/4.5092805033996796e-17 * self.surface_gravity / self.area

	@property
	def radius(self) -> float:
		"""Radius (m)"""
		# I am overriding Body.radius here because if that's unknown, but Lum and Temp are known, Radius can be computed directly from that instead
		if 'radius' in self.properties:
			return self.properties['radius'] # I could use super().radius instead
		if 'luminosity' in self.properties and 'temperature' in self.properties:
			return sqrt(self.luminosity / (4*pi*STEFAN_BOLTZMANN * self.temperature**4))

	@property
	def temperature(self) -> float:
		"""Temperature (K)"""
		if 'temperature' in self.properties:
			return self.properties['temperature']
		if 'radius' in self.properties and 'luminosity' in self.properties:
			return (self.luminosity / (4*pi*STEFAN_BOLTZMANN * self.radius**2))**0.25

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
		if 'metallicity' in self.properties:
			from mochaastro_data import sun
			return sun.Z * 10**self.metallicity
		return 1 - self.X - self.Y

	@property
	def BV(self) -> float:
		"""Estimated B-V difference from temperature (dimensionless)
		Maximum error = 0.0898 in the range [2380, 44900]"""
		return 1.25542 * atan(-4.22389e-4 * self.temperature + 1.53047) + 1.55418

	#@property
	#def BV2(self) -> float:
	#	"""Estimated B-V difference from temperature (dimensionless)
	#	Maximum error = 0.0898 in the range [2380, 44900]"""
	#	return photometry(self.temperature, 'B', 'V')

	@property
	def UB(self) -> float:
		"""Estimated U-B difference from temperature (dimensionless)
		This one in particular is rather bad, the R^2 was 0.899, compared to 0.987 and 0.99 for the other two.
		See notes for B-V"""
		# return 1.26 * self.BV - 0.516
		from math import cosh
		x = self.BV
		# took slope between endpoints, then added a regressed sech
		# to try to fix most of the error,
		# went from R^2 = 0.8988 to 0.9723
		return 1.393*x - 0.73 + 0.489/cosh(-5.69298*x + 0.244607)

	@property
	def VR(self) -> float:
		"""Estimated V-R difference from temperature (dimensionless)
		See notes for B-V"""
		return 0.803 * self.BV + 0.714

	@property
	def RI(self) -> float:
		"""Estimated R-I difference from temperature (dimensionless)
		See notes for B-V"""
		return 0.68 * self.BV - 0.0685

	@property
	def JH(self) -> float:
		"""Estimated J-H difference from temperature (dimensionless)
		Very poor quality.
		See notes for B-V"""
		# Methodology: http://exoplanets.org/plots
		# I plotted a histogram of (0.32 * BMV + 0.1135)/(J - H)
		# And just tweaked the slope to minimize the std
		# And tweaked the y-intercept to center the mean at 0
		# Cursed, but effective
		return 0.32 * self.BV + 0.1135

	@property
	def HK(self) -> float:
		"""Estimated H-K difference from temperature (dimensionless)
		Decent quality.
		See notes for B-V"""
		return 0.12 * self.BV + 0.00833

	# methods
	def app_mag(self, dist: float) -> float:
		"""Apparent Magnitude of this star at the given distance (dimensionless)"""
		return 5 * log10(dist / (10*pc)) + self.abs_mag

	def cool(self, t: float) -> float:
		"""Assuming this star is a stellar remnant (ie. a Neutron Star or White Dwarf),
		Finds the temperature (K) it cools down to after the given time (s).
		"""
		# http://hyperphysics.phy-astr.gsu.edu/hbase/thermo/cootime.html
		# N = self.photosphere_pressure*self.volume/(gas_constant*self.temperature)
		# ^
		# ideal gas law (my assumption pulled out of my ass...) https://chemistry.stackexchange.com/a/152172/94364
		# technically the volume and temp would vary with time,
		# but the number of mols would remain the same regardless so it doesn't matter
		# the only problem is pressure which I have NO idea...
		# I'm going to guesstimate the photosphere pressure based on a nasa doc, wish me luck
		# turns out I don't need that. White dwarves are like 50/50 carbon-12 and oxygen-16.
		# So I'll take the mean of the molar masses of those two (ie. 14 g/mol) and assume that's the molar mass of a white dwarf.
		# this turns out to be half of the previous value, which tells me I was on the right track though...
		# N = self.mass / 14000
		# print(N, 'mol')
		# A = self.area # I think?
		# emissivity = 1 # assume ideal
		# C = N*kB/(2*emissivity*STEFAN_BOLTZMANN*A)
		# print(C, N, A)
		# I expect this to be vaguely in the range of 8*10^27, but instead it's 3*10^-9!!!
		# I'll have to settle on a quick-and-dirty approximation then... sigh...
		# https://en.wikipedia.org/wiki/White_dwarf#Radiation_and_cooling
		# claims the coolest white dwarfs are just under 4000K.
		# Assuming they formed at the beginning of the universe, that means C should be...
		C = 2.1077e28 # https://www.desmos.com/calculator/wu1rudon6k
		return (t/C + 1/self.temperature**3)**(-1/3)

	def interplanetary_medium_electron_density(self, dist: float) -> float:
		"""Electron density at the given distance (m) from the sun. (Particles / m^3)
		Function designed with values of less than 20 Rsun in mind."""
		# formula taken from p. 154
		from mochaastro_data import sun
		l = self.luminosity / sun.luminosity
		r = dist / sun.radius
		return l * ((1.55*r**-6 + 2.99*r**-16) * 1e14 if r < 4 else 5e11 * r**-2)

	def photometry(self, filter_a: str = 'B', filter_b: str = 'V', correction: bool = True) -> float:
		"""Difference in magnitude of this star between the two specified photometric filters (dimensionless)"""
		return photometry(self.temperature, filter_a, filter_b, correction)

	def radiation_pressure_at(self, dist: float) -> float:
		"""Stellar radiation pressure at a distance (W/m^2)"""
		from mochaastro_data import sun
		wattage = self.luminosity / sun.luminosity
		return G_SC * wattage / (dist/au)**2

	def radiation_force_at(self, obj: Body, t: float = 0) -> float:
		"""Stellar radiation force on a planet at a time (W)"""
		dist = hypot(*obj.orbit.cartesian(t)[:3]) / au
		area = obj.area / 2
		return self.radiation_pressure_at(dist) * area


class BlackHole(Body):
	"""Black Hole object, which is like body but with a few extra parameters."""
	@property
	def dissipation_time(self) -> float:
		"""Dissipation time (s)"""
		# https://en.wikipedia.org/wiki/Hawking_radiation
		return 5120*pi*GRAV**2 * self.mass**3 / (REDUCED_PLANCK * c**4)

	@property
	def luminosity(self) -> float:
		"""Bekenstein-Hawking luminosity (W)"""
		# https://en.wikipedia.org/wiki/Hawking_radiation
		return REDUCED_PLANCK * c**6 / (15360*pi*GRAV**2*self.mass**2)

	@property
	def temperature(self) -> float:
		"""Hawking radiation temperature (K)"""
		# https://en.wikipedia.org/wiki/Hawking_radiation
		return REDUCED_PLANCK * c**3 / (8*pi*GRAV*self.mass*kB)
