from mochaastro2 import *
# from lyr import *

lyrsun = stargen(.3*sun.mass) # from orbital period and sma

zeus = Body(**{
	'orbit': Orbit(**{
		'parent': lyrsun,
		'sma': .55*au,
		'e': 3/11,
		'i': 0, # unk
		'lan': 0,
		'aop': 0,
		'man': 0,
	}),
	'mass': 600*earth.mass,
	'radius': 1e8,
	'albedo': .5, # SWAG
})

pegasia = Body(**{
	'orbit': Orbit(**{
		'parent': zeus,
		'sma': 5.7e8,
		'e': 0, # unk
		'i': 0,
		'lan': 0,
		'aop': 0,
		'man': 0,
	}),
	'rotation': Rotation(**{
		'period': 48*hour,
		'tilt': 3*deg,
	}),
	'atmosphere': Atmosphere(**{
		'surface_pressure': 1.6*earth.atmosphere.surface_pressure,
		'composition': {
			'N':   .68,
			'O':   .22,
			'Ar':  .08,
			'Ne':  .02,
			'CO2': 4e-4,
		},
	}),
	'mass': .5*earth.mass,
	'radius': 5.75e6,
	'albedo': .3,
})

tharn = Body(**{
	'orbit': Orbit(**{
		'parent': zeus,
		'sma': 1.18e9,
		'e': 0, # unk
		'i': 0,
		'lan': 0,
		'aop': 0,
		'man': 0,
	}),
	'rotation': Rotation(**{
		'period': 48*hour,
		'tilt': 16*deg,
	}),
	'atmosphere': Atmosphere(**{
		'surface_pressure': .24*earth.atmosphere.surface_pressure,
		'composition': {
			'N':   .65,
			'O':   .21,
			'Ar':  .09,
			'Ne':  .044,
			'CO2': .006,
		},
	}),
	'mass': .25*earth.mass,
	'radius': 4.15e6,
	'albedo': .3,
})

lyr = Body(**{
	'orbit': Orbit(**{
		'parent': lyrsun,
		'sma': 1.225*au,
		'e': 0.1,
		'i': 0, # unk
		'lan': 0,
		'aop': 0,
		'man': 0,
	}),
	'rotation': Rotation(**{
		'period': 12*hour,
		'tilt': 36*deg,
	}),
	'atmosphere': Atmosphere(**{
		'surface_pressure': 6*earth.atmosphere.surface_pressure,
		'composition': {
			'N':   .66,
			'Ne':  .14,
			'O':   .12,
			'Ar':  .06,
			'He':  .02,
			'CO2': 1.5e-4,
		},
	}),
	'mass': 7*earth.mass,
	'radius': 1.44e7,
	'albedo': .3, # slightly lower than earth's .367
})

oisin = Body(**{
	'orbit': Orbit(**{
		'parent': lyr,
		'sma': 5e8,
		'e': 0, # unk
		'i': 0, # "Interestingly, its inclination is nearly zero."
		'lan': 0, # unk
		'aop': 0,
		'man': 0,
	}),
	'rotation': Rotation(**{
		'period': 5.3*day,
		'tilt': 2*deg,
	}),
	'atmosphere': Atmosphere(**{
		'surface_pressure': 2500,
	}),
	'mass': .04*earth.mass,
	'radius': 2.5e6,
	'albedo': .8,
})

lyrsys = System(zeus, lyr)
zeussys = System(pegasia, tharn)
