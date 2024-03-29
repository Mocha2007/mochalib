"""Astronomical data for mochaastro.py"""
from typing import Dict
from mochaastro_body import Atmosphere, Body, Hydrosphere, Rotation, Star
from mochaastro_common import atm, au, day, deg, hour, minute # pylint: disable=unused-import
from mochaastro_orbit import Orbit
from mochaastro_system import System


def load_data(seed: dict) -> dict:
	"""Load the data stored in the .json files."""
	import os
	from json import load

	def convert_body(data: dict, current_universe: dict, datatype=Body) -> Body:
		def convert_atmosphere(data: dict) -> Atmosphere:
			if isinstance(data, str):
				return value_parse(data)
			return Atmosphere(**{key: value_parse(value) for key, value in data.items()})
		def convert_hydrosphere(data: dict) -> Hydrosphere:
			if isinstance(data, str):
				return value_parse(data)
			return Hydrosphere(**{key: value_parse(value) for key, value in data.items()})
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
			if isinstance(data, str):
				return value_parse(data)
			return Rotation(**{key: value_parse(value) for key, value in data.items()})
		def value_parse(value) -> float:
			try:
				return eval(value) if isinstance(value, str) else value # pylint: disable=eval-used
			except (NameError, SyntaxError):
				return value

		body_data = {}
		for key, value in data.items():
			if key == 'atmosphere':
				body_data[key] = convert_atmosphere(value)
			elif key == 'hydrosphere':
				body_data[key] = convert_hydrosphere(value)
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
					print(f'Albedo data missing from {name}, should be easy to find')
			# check for missing atm data
			elif 'atmosphere' not in body.properties and \
					'mass' in body.properties and body.atm_retention < .131 and \
					body.orbit.parent == body.star and body.orbit.a < 67*au:
				print(f'Atmosphere data missing from {name}, atmosphere predicted')

	loc = os.path.dirname(os.path.abspath(__file__)) + '\\mochaastro'
	universe_data = seed
	for file in [f for f in os.listdir(loc) if f.endswith('.json')]:
		print('Loading', file, '...')
		json_data = load(open(loc + '\\' + file, 'r', encoding="utf8"))
		for obj in json_data:
			universe_data[obj['name']] = convert_body(obj, universe_data, \
				(Star if obj['class'] == 'star' else Body))
	warnings(universe_data)
	return universe_data


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
	'mass': 1.98847e30, # https://en.wikipedia.org/wiki/Solar_mass
	'radius': 6.9566e8, # https://en.wikipedia.org/wiki/Solar_radius
	'oblateness': 9e-6,
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
sun.atmosphere.properties['composition'] = sun.composition
# https://en.wikipedia.org/wiki/Stellar_corona#Physics_of_the_corona

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
	'composition': { # by mass https://en.wikipedia.org/wiki/Mercury_(planet)#Physical_characteristics
		'Fe': 0.7, # "70% metallic"
		# "30% Silicate, assuming Silicate ~ 30% Si 70% O"
		'O':  0.3 * 0.7,
		'Si': 0.3 * 0.3,
	},
	'mass': 3.3011e23,
	'radius': 2.4397e6,
	'oblateness': 9e-4,
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
		'period': 243.0226 * day,
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
	'oblateness': 1/29825.7222101,
	# upper bound; see https://ui.adsabs.harvard.edu/abs/1968AJS....73R.162A/abstract
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
			'Ar':  .00934,
			'H2O': .0025, # https://en.wikipedia.org/wiki/Atmosphere_of_Earth#Composition "Water vapor is about 0.25% by mass over full atmosphere"
			'CO2': 4.2e-4,
			'Ne':  1.818e-5,
			'He':  5.24e-6,
			'CH4': 1.87e-6,
			'Kr':  1.14e-6,
			'H2':  5.5e-7,
			'N2O': 9e-8,
			'Xe': 8.7e-8, # https://en.wikipedia.org/wiki/Xenon#Occurrence_and_production
			'NO2': 2e-8,
			'SO2': 1.5e-8, # https://en.wikipedia.org/wiki/Sulfur_dioxide#Occurrence
		},
	}),
	'hydrosphere': Hydrosphere(**{
		'coverage': 0.708,
		'density': 1025,
		'depth': 11034,
		'composition': { # https://en.wikipedia.org/wiki/Seawater#Chemical_composition
			'O':  .8584,
			'H':  .1082,
			'Cl': .0194,
			'Na': .0108,
			'Mg': .001292,
			'S':  .00091,
			'Ca': .0004,
			'K':  .0004,
			'Br': .000067,
			'C':  .000028,
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
	'oblateness': 1/298.257222101,
	'albedo': .306,
})
venus.properties['composition'] = earth.composition
# https://en.wikipedia.org/wiki/Venus#Internal_structure

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
	'oblateness': 1.2e-3,
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
			'H2': 15e-6, # https://en.wikipedia.org/wiki/Atmosphere_of_Mars#Other_trace_gases
			'Ne': 2.5e-6,
			'Kr': 0.3e-6,
			'Xe': 0.08e-6,
			'CH4': 10e-9, # https://en.wikipedia.org/wiki/Natural_methane_on_Mars
		},
	}),
	'composition': { # https://en.wikipedia.org/wiki/Composition_of_Mars#Elemental_composition
		'Fe': 0.639, # "Martian meteorite analysis suggests that the planet's mantle is about twice as rich in iron as the Earth's mantle."
		'O':  0.297 * (1 - 0.6517)/0.66322,
		'Si': 0.161 * (1 - 0.6517)/0.66322,
		'Mg': 0.154 * (1 - 0.6517)/0.66322,
		'Ni': 0.01822 * (1 - 0.6517)/0.66322,
		'Ca': 0.0171 * (1 - 0.6517)/0.66322,
		'Al': 0.0159 * (1 - 0.6517)/0.66322,
		'S':  0.0127, # "Second, its core is richer in sulphur."
		# 0.6517 vs 0.66322
	},
	'mass': 6.4171e23,
	'radius': 3.3895e6,
	'oblateness': 5.89e-3,
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
	'oblateness': 6.487e-2,
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
	'oblateness': 9.796e-2,
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
	'oblateness': 2.29e-2,
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
	'oblateness': 1.71e-2,
	'albedo': .29,
})

# asteroid compositions (most are guesses)
ASTEROID_COMPOSITION_C = { # Guess: 30% Ice crust, 50% Silicate mantle, 20% Iron core
	'O': 0.3*0.881 + 0.5*0.8,
	'H': 0.3*0.119,
	'Si': 0.5*0.2,
	'Fe': 0.2,
}
ASTEROID_COMPOSITION_M = { # https://starlust.org/what-are-asteroids-made-of/
	'Fe': 0.8,
	'Ni': 0.2,
}
ASTEROID_COMPOSITION_S = { # Guess: 25% Ice crust, 30% Silicate mantle, 45% Iron core
	'O': 0.25*0.881 + 0.3*0.8,
	'H': 0.25*0.119,
	'Si': 0.3*0.2,
	'Fe': 0.45,
}
ASTEROID_COMPOSITION_B = ASTEROID_COMPOSITION_C
ASTEROID_COMPOSITION_G = ASTEROID_COMPOSITION_C
ASTEROID_COMPOSITION_V = ASTEROID_COMPOSITION_S
CLASSICAL_KBO_COMPOSITION_HOT = { # Guess: 37.5% Ice crust, 37.5% Silicate mantle, 25% Iron core
	'O': 0.375*0.881 + 0.375*0.8,
	'H': 0.375*0.119,
	'Si': 0.375*0.2,
	'Fe': 0.25,
}
CLASSICAL_KBO_COMPOSITION_COLD = { # Guess: 40% Ice crust, 40% Silicate mantle, 20% Iron core
	'O': 0.4*0.881 + 0.4*0.8,
	'H': 0.4*0.119,
	'Si': 0.4*0.2,
	'Fe': 0.2,
}
SDO_COMPOSITION = CLASSICAL_KBO_COMPOSITION_HOT # wiki says they're similar in composition
PLUTINO_COMPOSITION = CLASSICAL_KBO_COMPOSITION_HOT # wild guess
HAUMEID_COMPOSITION = { # Densest type of KBO - Guess: 25% Ice crust, 40% Silicate mantle, 35% Iron core
	'O': 0.25*0.881 + 0.4*0.8,
	'H': 0.25*0.119,
	'Si': 0.4*0.2,
	'Fe': 0.35,
}
SEDNOID_COMPOSITION = CLASSICAL_KBO_COMPOSITION_HOT # wild guess
PROTOPLANET_ATM_COMPOSITION = { # Venus/Mars-like
	'CO2': 0.95,
	'N2': 0.03,
	'Ar': 0.02,
}

# inner_solar_system = System(mercury, venus, earth, mars) # a <= mars
# solar_system = System(mercury, venus, earth, mars, jupiter, saturn, uranus, neptune)
# jupiter_system = System(io, europa, ganymede, callisto)
# kuiper = System(neptune, pons_gambart, pluto, ikeya_zhang, eris, sedna, planet_nine)
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
solar_system_object = System(sun, *(i for i in solar_system.values() \
					if i is not sun and i is not moon))
universe = load_data(solar_system.copy()) # type: Dict[str, Body]
