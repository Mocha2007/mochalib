"""Constants and basic functions used throughout the mochaastro files"""
# pylint: disable=eval-used, no-member, unused-import, unused-variable
from datetime import datetime, timedelta
from enum import Enum
from math import acos, atan, cos, exp, hypot, inf, log, log10, pi, sin, sqrt, tan
from typing import Callable, Tuple
from mochaunits import Angle, Length, Mass, Time, pretty_dim # Angle is indeed used

# constants
MOCHAASTRO_DEBUG = False

# https://en.wikipedia.org/wiki/Epoch_(astronomy)#Julian_years_and_J2000
J2000 = datetime(2000, 1, 1, 11, 58, 55, 816)
SQRT2 = sqrt(2)

GRAV = 6.674e-11 # m^3 / (kg*s^2); appx; standard gravitational constant
c = 299792458
"""m/s; exact; speed of light"""
L_0 = 3.0128e28 # W; exact; zero point luminosity
L_sun = 3.828e26 # W; exact; nominal solar luminosity
STEFAN_BOLTZMANN = 5.670374419e-8 # W⋅m−2⋅K−4
"""W/(m^2 * K^4); appx; Stefan-Boltzmann constant (NOT Boltzmann: use kB for that)"""
C2K = 273.15 # K; exact; constant to add to convert from Celsius to Kelvin
PLANCK = 6.62607015e-34
"""J/Hz; exact; Planck constant, often written 'h'"""
REDUCED_PLANCK = PLANCK / (2*pi)
N_A = 6.02214076e23 # dimensionless; exact; Avogadro constant
WIEN = 2.897771955e-3 # m*K; appx; Wien's displacement constant
kB = 1.380649e-23
"""J/K; exact; Boltzmann constant (NOT Stefan-Boltzmann: use STEFAN_BOLTZMANN for that)"""
gas_constant = N_A*kB # J/(Kmol); exact; ideal gas constant

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
angstrom = 1e-10 # m; exact
bar = 1e5 # Pa; exact

ly = c * jyear # m; exact; light-year
au = 149597870700 # m; exact; astronomical unit
pc = 648000/pi * au # m; exact; parsec
mi = 1609.344 # m; exact; mile

G_SC = L_sun / (4*pi*au**2) # W/m^2; exact*; solar constant;
# * - technically not b/c this is based on visual luminosity rather than bolometric

# molar masses for common atmospheric compounds
MOLAR_MASS = {
	'H2': 1.008e-3*2,
	'H2O': 1.008e-3*2 + 15.999e-3,
	'He': 4.002602e-3,
	'CH4': 12.011e-3 + 1.008e-3*4,
	'CO2': 12.011e-3 + 15.999e-3*2,
	'N2': 14.007e-3*2,
	'O2': 15.999e-3*2,
	'SO2': 32.06e-3 + 15.999e-3*2,
	'Ar': 39.95e-3,
}

DEFAULT_DENSITIES = {
	'comet': 600,
}

PHOTOMETRIC_FILTER = {
	# https://en.wikipedia.org/wiki/UBV_photometric_system
	# Handbook of Space Astronomy and Astrophysics (2nd ed.). Cambridge University Press. p. 100
	# https://en.wikipedia.org/wiki/Photometric_system#Photometric_letters
	'U': 365e-9,
	'B': 445e-9,
	'G': 464e-9,
	'V': 551e-9,
	'R': 658e-9,
	'I': 806e-9,
	'Z': 900e-9,
	'Y': 1020e-9,
	'J': 1220e-9,
	'H': 1630e-9,
	'K': 2190e-9,
	'L': 3450e-9,
	'M': 4750e-9,
	'N': 10500e-9,
	'Q': 21000e-9,
	# https://physics.stackexchange.com/a/503176/277131
	'RGB_R': 612e-9,
	'RGB_G': 549e-9,
	'RGB_B': 464e-9,
}

PHOTOMETRY_CORRECTION_COEF = {
	'U-B': (7.96e-3, 0.268, -4, 8.74, 6.28, -17.2, 7.1),
	'B-V': (0.108, 1.24, 0.148, -0.129),
	'V-R': (0.126, 1.34),
	'R-I': (0.0697, 1.24, -1.92, 2.97),
	'J-H': (0.241, 0.492),
	'J-K': (0.306, 0.488),
}

# simple functions
def apsides2ecc(apo: float, peri: float) -> Tuple[float, float]:
	"""Converts an apoapsis and periapsis (any units)
	into a semimajor axis (same unit) and eccentricity (dimensionless)"""
	return (apo+peri)/2, (apo-peri)/(apo+peri)


def axisEqual3D(ax) -> None: # ax type: Axes
	"""Forces each axis to use the same scale"""
	import numpy as np
	extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
	sz = extents[:, 1] - extents[:, 0]
	centers = np.mean(extents, axis=1)
	maxsize = max(abs(sz))
	r = maxsize/2
	for ctr, dim in zip(centers, 'xyz'):
		getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)


def blagg(n: int,
		A: float = 0.338163, B: float = 2.81213,
		a: float = 1.40708, b: float = 0.982812,
		base: float = 1.73809) -> float:
	"""https://en.wikipedia.org/wiki/Titius%E2%80%93Bode_law#Blagg_Formulation
	https://www.desmos.com/calculator/fpnxyw41r3"""
	def f(psi: float) -> float:
		return cos(psi)/(3-cos(2*psi)) + 1/(6-4*cos(2*(psi-30*deg)))
	return A * base**n * (B + f(a + n*b))


def light_travel_time_OLD(z: float) -> float:
	"""Time it took for light to reach observer, given redshift z. Returns (s)"""
	# https://www.desmos.com/calculator/8r8h3ds9pf
	return 1e9*year * 8.97983*atan(1.27092*z)


def linear_map(interval1: Tuple[float, float], interval2: Tuple[float, float]) \
		-> Callable[[float], float]:
	"""Create a linear map from one interval to another"""
	return lambda x: (x - interval1[0]) / (interval1[1] - interval1[0]) \
		* (interval2[1] - interval2[0]) + interval2[0]


def photometry(temp: float, filter_a: str, filter_b: str, correction: bool = False) -> float:
	"""Difference in intensity of light between filters emitted at the given temperature (dimensionless)"""
	vega = 9602 # https://en.wikipedia.org/wiki/Vega
	l_a, l_b = PHOTOMETRIC_FILTER[filter_a], PHOTOMETRIC_FILTER[filter_b]
	# f_a, f_b = c/l_a, c/l_b
	va = planck(l_a, vega)
	vb = planck(l_b, vega)
	a = planck(l_a, temp)
	b = planck(l_b, temp)
	# https://astronomy.stackexchange.com/a/34062/48796
	C0 = -2.5*log10(a/b) + 2.5*log10(va/vb)
	# Corrective polynomial fits
	name = f"{filter_a}-{filter_b}"
	if correction and name in PHOTOMETRY_CORRECTION_COEF:
		C0 = sum(x*C0**i for (i, x) in enumerate(PHOTOMETRY_CORRECTION_COEF[name]))
	return C0

def photometryTest(temp: float = 5778) -> None:
	"""Used for testing accuracy of mochaastro_common.photometry"""
	#	U		B	V		R 	I	<- appx peaks
	t = 7900, 6550, 5250, 4400, 3600
	print('Name', 'λ', *t)
	for x in 'UBVRI':
		print(("{: >1} ({: >3} nm)" + " {: <20}"*5).format(x, int(PHOTOMETRIC_FILTER[x]*1e9), *(planck(PHOTOMETRIC_FILTER[x], tx)/1e4 for tx in t)))
	print('-'*70)
	print('T', 'U-B', 'B-V', 'V-R', 'R-I')
	classif = 'XOBAFGKM'
	for (i, x) in enumerate([temp, 42000, 30000, 9790, 7300, 5940, 5150, 3840]):
		print((classif[i] + " {: >5}" + " {: <6}"*4).format(x,
			round(photometry(x, 'U', 'B'), 3), round(photometry(x, 'B', 'V'), 3),
			round(photometry(x, 'V', 'R'), 3), round(photometry(x, 'R', 'I'), 3)))

# https://en.wikipedia.org/wiki/Planck's_law
def planck(wl: float, temp: float) -> float:
	"""Planck's law: spectral radiance emitted at a wavelength (m) given a temp (K)"""
	# https://phet.colorado.edu/sims/html/blackbody-spectrum/latest/blackbody-spectrum_all.html
	return 8*pi*PLANCK*c/wl**5 / (exp(PLANCK*c/(wl*kB*temp)) - 1)


def resonance_probability(mismatch: float, outer: int) -> float:
	"""Return probability a particular resonance is by chance rather than gravitational"""
	return 1-(1-abs(mismatch))**outer


def synodic(p1: float, p2: float) -> float:
	"""synodic period of two periods (s)"""
	return p1*p2/abs(p2-p1) if p2-p1 else inf


class Phase(Enum):
	"""Simple enum for chemical phases"""
	SOLID = 0
	LIQUID = 1
	GAS = 2
	SUPERCRITICAL_FLUID = 3
	PLASMA = 4
	def __str__(self) -> str:
		from re import search as regex
		return regex('[A-Z_]{2,}', self.__repr__()).group(0).replace('_', ' ').title()


def water_phase(t: float, p: float) -> Phase:
	"""Returns the predicted phase of water at the specified temperature and pressure"""
	if 3270 <= t: # https://en.wikipedia.org/wiki/Water_splitting#Thermal_decomposition_of_water
		return Phase.PLASMA # "It turned into a hydrogen-oxygen plasma"
	p = max(p, 1) # prevent domain error for low pressures
	CRIT_T = 647
	CRIT_P = 22.064e6
	TRIP_T = C2K # actually 273.16 but close enough...
	TRIP_P = 611.657
	def vapor_line(p_: float) -> float:
		"""Returns the temperature at which water boils at the given pressure"""
		y = log(p_)
		if p_ < atm:
			y0 = log(TRIP_P)
			y1 = log(atm) # 100C 1 atm
			return 100 * (y - y0)/(y1 - y0) + TRIP_T
		# else...
		y0 = log(atm) # 100C 1 atm
		y1 = log(CRIT_P)
		return (CRIT_T - (C2K + 100)) * (y - y0)/(y1 - y0) + C2K + 100
	WEIRD_ICE_P = 632.4e6
	def ice7_line(p_: float) -> float:
		"""Returns the temperature at which water melts from Ice VII (or related)"""
		y = log(p_)
		y0 = log(WEIRD_ICE_P)
		y1 = log(10.2e9)
		return (CRIT_T - C2K) * (y - y0)/(y1 - y0) + C2K
	def sublimation_line(p_: float) -> float:
		"""Returns the temperature at which water sublimates at the given pressure"""
		y = log(p_)
		y1 = log(TRIP_P)
		return (TRIP_T - 212.6) * y/y1 + 212.6
	if p < TRIP_P:
		return Phase.GAS if sublimation_line(p) < t else Phase.SOLID
	if t < C2K:
		return Phase.SOLID
	if WEIRD_ICE_P < p and ice7_line(p) < t:
		return Phase.SOLID
	return Phase.SUPERCRITICAL_FLUID if CRIT_P < p and CRIT_T < t else \
		Phase.GAS if vapor_line(p) < t else Phase.LIQUID


def wien(t: float) -> float:
	"""Wien's displacement law: given a temperature (K) return the peak emission wavelength (m)"""
	return WIEN / t


# advanced functions


def accrete(star_mass: float = 2e30, particle_n: int = 25000):
	"""Procedurally generate a star system."""
	from random import randint, uniform
	from mochaastro_body import Body, Rotation
	from mochaastro_data import mercury, neptune, sun
	from mochaastro_orbit import Orbit
	from mochaastro_system import System
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
	def body(mass: float, a: float) -> Body:
		def density_from_mass(mass: float) -> float: # todo better density
			if 2e26 < mass:
				return uniform(600, 1400)
			if 6e25 < mass:
				return uniform(1200, 1700)
			return uniform(3900, 5600)

		rotation =  Rotation(**{'period': 2.7573e7 * mass**-0.104259, 'tilt': uniform(0, 30)*deg})
		# this rotation formula is based on a regression of the mass for mars to neptune
		return Body(**{
			'mass': mass,
			'radius': (3*mass/(4*pi*density_from_mass(mass)))**(1/3),
			'oblateness': min(1, 4.4732e10 * rotation.p**-2.57479),
			# oblateness of 1 occurs when rotation ~ 3.8h
			'rotation' : rotation,
			'orbit': Orbit(**{
				'parent': star,
				'sma': a,
				'e': uniform(0, .1), 'i': uniform(0, 4*deg), 'lan': uniform(0, 2*pi),
				'aop': uniform(0, 2*pi), 'man': uniform(0, 2*pi)
			})})
	star = stargen(star_mass)
	out = System(star, *(body(mass, a) for mass, a in particles))
	return out # .plot2d()


def keplerian(parent, cartesian: Tuple[float, float, float, float, float, float]):
	"""Get keplerian orbital parameters (a, e, i, O, o, m)"""
	import numpy as np
	from mochaastro_orbit import Orbit
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


def distance_audio(orbit1, orbit2) -> None:
	"""Play wave of plot_distance
	Encoding   | Signed 16-bit PCM
	Byte order | little endian
	Channels   | 1 channel mono
	"""
	import numpy as np
	from mochaaudio import pcm_to_wav, play_file
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


def light_travel_time(z: float) -> float:
	"""Returns distance to object with observed redshift z. Distance in (m)"""
	# shamelessly ported from https://www.astro.ucla.edu/~wright/ACC.html
	# constants https://en.wikipedia.org/wiki/Lambda-CDM_model#Parameters
	H0 = 67.4 # 2018
	T0 = 2.72528
	mnue = 0.001
	mnumu = 0.009
	mnutau = 0.049
	Tyr = 977.8
	WM = 0.315 # 2018
	WV = 0.6847 # 2018 Omega-Lambda on the Wikipedia article; Omega-DE on the site
	w = -1.03 # 2018
	wp = 0
	n = 1000 # max iterations for integration
	#def DCMT() -> float:
	#	x = sqrt(abs(WK))*DCMR
	#	if x > 0.1: return (0.5*(exp(x)-exp(-x)) if WK > 0 else sin(x))/x*DCMR
	#	y = x**2
	#	if WK < 0: y = -y
	#	return (1 + y/6 + y*y/120)*DCMR
	def nurho(mnu: float) -> float:
		return (1+(mnurel/mnu)**1.842)**(1/1.842)
	#def VCM() -> float:
	#	x = sqrt(abs(WK))*DCMR
	#	if x > 0.1:	return (0.125*(exp(2*x)-exp(-2*x))-x/2 if WK > 0 else x/2 - sin(2*x)/4)/x**3*DCMR**3
	#	y = x**2
	#	if WK < 0: y = -y
	#	return (1 + y/5 + 2/105*y**2)*DCMR**3/3
	# main
	h = H0/100
	WR = 2.477E-5*(T0/2.72528)**4/h**2
	we = (mnue/93.64)*(T0/2.72528)**3
	wmu = (mnumu/93.90)*(T0/2.72528)**3
	wtau = (mnutau/93.90)*(T0/2.72528)**3
	mnurel = 6.13*(T0/2.72528)/11604.5
	Wnu = (we*nurho(mnue)+wmu*nurho(mnumu)+wtau*nurho(mnutau))/h**2
	WK = 1-WM-WR-WV
	WM -= Wnu
	az = 1/(1+z)
	# age = 0
	#for i in range(n):
	#	a = az*(i+0.5)/n
	#	rhoV = WV*a**(-3-3*w-6*wp) * exp(6*wp*(a-1))
	#	Wnu = (we*nurho(mnue*a)+wmu*nurho(mnumu*a)+wtau*nurho(mnutau*a))/h**2
	#	adot = sqrt(WK+(WM+Wnu)/a+WR/a**2+rhoV*a**2)
	#	age += 1/adot
	# zage = az*age/n
	# lpz = log10(1+z)
	# dzage = 0
	# if lpz > 15.107: dzage = 0.214
	# elif lpz > 14.081: dzage = 0.013 * (lpz - 14.081) +  0.201
	# elif lpz > 13.055: dzage = 0.013 * (lpz - 13.055) +  0.188
	# elif lpz > 12.382: dzage = 0.024 * (lpz - 12.382) +  0.171
	# elif lpz > 12.258: dzage = 0.461 * (lpz - 12.258) +  0.114
	# elif lpz > 11.851: dzage = 0.069 * (lpz - 11.851) +  0.086
	# elif lpz > 10.775: dzage = 0.035 * (lpz - 10.775) +  0.048
	# elif lpz > 10.000: dzage = 0.048
	# elif lpz >  9.500: dzage = 0.019 * (lpz -  9.500) +  0.039
	# elif lpz >  9.000: dzage = 0.020 * (lpz -  9.000) +  0.028
	# elif lpz >  8.500: dzage = 0.040 * (lpz -  8.500) +  0.008
	# elif lpz >  8.000: dzage = 0.014 * (lpz -  8.000) +  0.001
	# elif lpz >  7.500: dzage = 0.002 * (lpz -  7.500)
	# zage *= 10**dzage
	# zage_Gyr = (Tyr/H0)*zage
	DTT = 0
	# DCMR = 0
	for i in range(n):
		a = az+(1-az)*(i+0.5)/n
		rhoV = WV*a**(-3-3*w-6*wp)*exp(6*wp*(a-1))
		Wnu = (we*nurho(mnue*a)+wmu*nurho(mnumu*a)+wtau*nurho(mnutau*a))/(h*h)
		adot = sqrt(WK+(WM+Wnu)/a+WR/a**2+rhoV*a**2)
		DTT += 1/adot
		# DCMR += 1/(a*adot)
	DTT *= (1-az)/n
	# DCMR *= (1-az)/n
	# age = DTT+zage
	# age_Gyr = age*Tyr/H0
	DTT_Gyr = Tyr/H0*DTT
	# DCMR_Gyr = Tyr/H0*DCMR
	# DCMR_Mpc = c/H0*DCMR
	# DA = az*DCMT()
	# DA_Mpc = c/H0*DA
	# kpc_DA = DA_Mpc/206.264806
	# DA_Gyr = Tyr/H0*DA
	# DL = DA/az**2
	# DL_Mpc = c/H0*DL
	# DL_Gyr = Tyr/H0*DL
	# V_Gpc = 4*pi*(0.001*c/H0)**3 * VCM()
	return DTT_Gyr * 1e9 * year * c # convert from Gly to m


def plot_delta_between(orbit1, orbit2) -> None:
	"""Plot system with pyplot"""
	import matplotlib.pyplot as plt
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


def plot_distance(orbit1, orbit2) -> None:
	"""Plot distance between two bodies over several orbits"""
	import matplotlib.pyplot as plt
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


def plot_grav_acc(body1, body2) -> None:
	"""Plot gravitational acceleration from one body to another over several orbits"""
	import matplotlib.pyplot as plt
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


def plot_grav_accs(body, other_bodies) -> None:
	"""Plot total gravitational acceleration from one body to several others over several orbits"""
	import matplotlib.pyplot as plt
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


def plot_grav_acc_vector(body1, body2) -> None:
	"""Plot gravitational acceleration vector from one body to another over several orbits"""
	import matplotlib.pyplot as plt
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


def search(name: str):
	"""Try to find a body with the given name."""
	from mochaastro_data import universe
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


def stargen(m: float, variance: float = 0) :
	"""Generate star from mass (use variance to slightly fluctuate values from expected norm - I recommend something like 0.01)"""
	from mochaastro_body import Star
	from mochaastro_data import sun
	def e() -> float:
		from random import uniform
		return 1 + uniform(-variance, variance)
	m /= sun.mass
	# stellar radius
	# https://www.desmos.com/calculator/rpvpmewvyn
	# I manually tweaked the coefficients to get a good fit for M <= 11 sols
	def bell(mid: float) -> Callable[[float], float]:
		return lambda x: exp(-(log(x) - mid)**2)
	c_r = m and 1.2*bell(1)(m) - 1.13*bell(1.23)(m) + 0.9 # "and" is to prevent 0 from causing domain errors lol
	r = 1.012*m**c_r # 1.012 Rsun is the ref for a G2V star
	# stellar temp
	# https://www.desmos.com/calculator/mhhdatpcu0
	if m < 0.085:
		t = 5.23 * m
	elif m < 1:
		t = m ** (0.29 + 0.76 * sin(2.67*m + 0.1)*m)
	elif m < 2.1:
		t = m ** (0.58 + 0.14 * sin(3.3*m))
	else:
		t = 1.06 * m ** 0.57
	 # 5770 is the ref for a G2V star, but we've thus far been measuring in suns, so...
	t *= 5770 / 5778
	# error/variance...
	# 2023 Dec 29 - I realized you can manipulate two different formulas for the habitable zone
	# and get the relation T^4 R^2 = L
	# this relation seems to be an EXTREMELY close approximation, making me think it is
	# an exact relationship. Intuitively it makes sense, since Luminosity should be
	# proportional to area x flux. The full form is probably similar to the product of the RHS
	# of the sphere area formula and Stefan-Boltzmann law
	lum = 1.02 * t**4 * r**2
	return Star(**{
		'mass': m*sun.mass, # does not use variance because this is given
		'radius': r*sun.radius*e(),
		'luminosity': sun.luminosity*lum, # does not use variance since this is an exact relation
		'temperature': 5778*t*e(), # see note 14 lines above
	})


def test_functions() -> None:
	"""Debug."""
	from mochaastro_data import solar_system_object, sun
	universe_sim(sun)
	solar_system_object.grav_sim()


def universe_error_handler(func):
	"""Ignores errors when quitting pygame"""
	def wrapper(*args):
		from pygame import error # pylint: disable=no-name-in-module
		try:
			func(*args)
		except error:
			pass
	return wrapper


@universe_error_handler
def universe_sim(parent, t: float=0, size: Tuple[int, int]=(1024, 640), \
		selection=None, smoothMotion=False) -> None:
	# TODO
	# moon display
	# comet tails
	"""Use pygame to show the system of [parent] and all subsystems"""
	import pygame
	from pygame import gfxdraw # DO NOT REMOVE THIS IT BREAKS SHIT
	from time import sleep, time
	from mochamath import dist
	from mochaunits import round_time
	from mochaastro_body import Body
	from mochaastro_data import universe

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

	def are_onscreen(points: tuple) -> bool:
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
	def point(at: Tuple[int, int], radius: float, color: Tuple[int, int, int]=white, \
			fill: bool=True, highlight: bool=False) -> None:
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
	def text(string: str, at: Tuple[int, int], size: int=font_normal, \
			color: Tuple[int, int, int]=white, shadow: bool=False) -> None:
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
		if 'class' in body.properties and body.classification == 'comet':
			continue
		# clean orbits of undrawable orbits
		try:
			body.orbit.cartesian()
		except KeyError:
			continue
		# good orbit then!
		if body.orbit.parent == parent:
			color = beige if 'class' in body.properties and body.classification != 'planet' else blue
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
					ke_str = f'\nKE = {pretty_dim(ke, 0)}'
				# mag
				text(f'{pretty_dim(Length(mag)/Time(1))}{ke_str}', vcoords, font_normal, red)
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
		information = current_date + f' (x{int(fps*timerate)}){" [PAUSED]" if paused else ""}\n' + \
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
				print("Thank you for using mochaastro2.py!~")
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
		title = f'{inverse_universe[selection]}, {inverse_universe[parent]} System - {current_date}'
		pygame.display.set_caption(title)
		# sleep
		wait_time = 1/fps - (time() - start_time)
		if 0 < wait_time:
			sleep(wait_time)
