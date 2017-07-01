# mochalib
Various mathematical and scientific functions I've needed

# Astronomical Library Documentation
## Fundamental Constants:

* c - Speed of Light. m/s
* g - Gravitational Constant. m^3/kgs^2
* sigma - Stefan-Boltzmann Constant. W/m^2K^4

## Length Constants:
* au - Astronomical Unit. m
* a_moon - Lunar Distance. m
* mi - Mile. m
* pc - Parsec. m
* r_e - Terran Radius. m
* r_j - Jovian Radius. m
* r_moon - Lunar Radius. m
* r_person - Mean Radius of the Average Human. m
* r_sun - Solar Radius. m
* ly - Light-Year. m

## Mass Constants

* lb - Pound. kg
* m_e - Terran Mass. kg
* m_j - Jovian Mass. kg
* m_moon - Lunar Mass. kg
* m_person - Average Human Mass. kg
* m_sun - Solar Mass. kg

## Time Constants

* day - Day. s
* hour - Hour. s
* jyear - Julian Year. s
* minute - Minute. s
* t_e - Earth's Rotation Period. s
* year - Gregorian Year. s

## Power Constants

* l_sun - Solar Luminosity. W

## Body Dictionaries:

### Each of these dictionaries contain four components:

* a - Semimajor Axis. m
* e - Eccentricity. [Dimensionless]
* m - Mass. kg
* r - Radius. m

### List:

* mars - Mars.

## Functions:

* absmag - Absolute magnitude of specified star. [Dimensionless],m -> [Dimensionless]
* acc - The derivative of *speed*. m,m,kg -> m/s^2
* apsides - Apsides of the specified orbit. Inverse of *apsides2ecc*. m,[Dimensionless] -> m,m
* apsides2ecc - Eccentricity of the specified orbit. Inverse of *apsides*. m,m -> m,[Dimensionless]
* bielliptic - Calculates Delta-v required to perform specified Bielliptic transfer manoeuvre. m,m,m,kg -> m/s
* biellipticinf - Calculates theoretical minimum Delta-v required to perform specified Bielliptic transfer manoeuvre. m,m,kg -> m/s
* bielliptictime - Calculates time required to perform specified Bielliptic transfer manoeuvre. m,m,m,kg -> s
* density - Density of the sphere. kg,m -> kg/m^3
* energy - Calculates rest energy of specified object. kg -> J
* gravbinding - Gravitational binding energy of specified body. m,kg -> J
* gravbinding2 - Gravitational binding energy of specified body, using density instead of mass. m,kg/m^3 -> J
* habitablezone - Approximates inner and outer bounds of the habitable zone of a hypothetical star with specified mass. kg-> m,m
* hill - Calculates hill sphere of specified body. cf. *soi*. kg,kg,m,[Dimensionless] -> m
* hohmann - Calculates Delta-v required to perform specified Hohmann transfer manoeuvre. m,m,kg -> m/s
* inclinationburn - Calculates Delta-v required to perform specified plane change manoeuvre. rad,[Dimensionless],rad,rad,rad,m -> m/s
* mass - Calculates relativistic mass of specified object. J -> kg
* mov - Mean orbital velocity of the specified orbit. kg,m -> m/s
* optimal - Calculates which of Hohmann or Bielliptic transfer manoeuvres is more efficient. m,m,kg->[String]
* p - Period of the specified orbit. kg,m -> s
* peakwavelength - Calculates peak wavelength of an object with temperature T. K -> m
* roche - Calculates roche limit of specified body. kg,kg/m^3 -> m
* schwarzschild - Calculates Schwarzschild radius of specified body. kg -> m
* soi - Sphere of influence of the specified body. cf. *hill*. kg,kg,m -> m
* speed - Supposedly calculates velocity of the body at a specified point in its orbit. Seems incorrect - use at own risk. m,m,kg -> m/s
* star - Approximates radius, luminosity, mean surface temperature, and lifespan of a hypothetical star of specified mass. kg-> m,W,K,s
* surfacegravity - Calculates surface gravity of specified body. kg,m->m/s^2
* synchronous - Semimajor axis of the synchronous orbit around a body. s,kg -> m
* synodic - Calculates synodic period of two specified bodies. s,s -> s
* temp - Calculates temperature of a body illuminated by a star. W,[Dimensionless],m -> K
* vescape - Calculates escape velocity of specified body. kg,m->m/s
* vextrema - Orbital velocity at periapsis and apoapsis of the specified orbit, respectively. m,[Dimensionless],kg -> m/s,m/s
* vtan - Calculates tangental velocity of specified rotating sphere. s,m->m/s

# Math Library Documentation
...

# Wavefunction Library Documentation
All functions require three components - *x* (Time in s), *freq* (Frequency in hz), and *amp* (Amplitude). They all return a y-value.
## Functions
* sawtooth - Sawtooth wave function. Uses modulo.
* sine - Sine wave function. Uses trigonometric functions.
* sine1 - Sine wave function. Uses complex powers to compute the wave instead of trigonometric functions.
* square - Square wave function. Uses flooring and modulo.
* square1 - Square wave function. Uses the imaginary part of ln(sin(x)).
* triangle - Triangle wave function.
