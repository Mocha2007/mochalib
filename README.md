# mochalib
Various mathematical and scientific functions I've needed

## MathFox
Adds functions/variables as classes to simplify, solve derivatives, etc.
sample usage:
x = Variable('x')
Arctan(x).derivative(x, 3) # calculates third derivative of arctan(x)

## 24 Game Library Documentation
### Functions
* allorder - Every number achievable with the four given integers.
* twentyfour - Every number achievable with the four given integers, used in order.
* operations - Performs every possible ordering of math operations on four integers, in that order.

## Astronomical Library Documentation
### Fundamental Constants

* c - Speed of Light. m/s
* g - Gravitational Constant. m^3/kgs^2

### Length Constants
* au - Astronomical Unit. m
* a_moon - Lunar Distance. m
* a_planck - Planck length. m
* mi - Mile. m
* pc - Parsec. m
* r_e - Terran Radius. m
* r_j - Jovian Radius. m
* r_moon - Lunar Radius. m
* r_person - Mean Radius of the Average Human. m
* r_sun - Solar Radius. m
* ly - Light-Year. m

### Mass Constants

* lb - Pound. kg
* m_e - Terran Mass. kg
* m_j - Jovian Mass. kg
* m_moon - Lunar Mass. kg
* m_person - Average Human Mass. kg
* m_planck - Planck Mass. kg
* m_sun - Solar Mass. kg

### Time Constants

* day - Day. s
* hour - Hour. s
* jyear - Julian Year. s
* minute - Minute. s
* t_e - Earth's Rotation Period. s
* t_planck - Planck Time. s
* year - Gregorian Year. s

### Power Constants

* l_planck - Planck Power. W
* l_sun - Solar Luminosity. W

### Body Dictionaries

#### Each of these dictionaries contain four components

* a - Semimajor Axis. m
* e - Eccentricity. \[Dimensionless]
* m - Mass. kg
* r - Radius. m

#### List

* earth - Earth.
* jupiter - Jupiter.
* mars - Mars.
* mercury - Mercury.
* neptune - Neptune.
* planetnine - Planet Nine.
* saturn - Saturn.
* uranus - Uranus.
* venus - Venus.

### Functions

* absmag - Absolute magnitude of specified star. \[Dimensionless],m -> \[Dimensionless]
* acc - The derivative of *speed*. m,m,kg -> m/s^2
* apsides - Apsides of the specified orbit. Inverse of *apsides2ecc*. m,\[Dimensionless] -> m,m
* apsides2ecc - Eccentricity of the specified orbit. Inverse of *apsides*. m,m -> m,\[Dimensionless]
* bielliptic - Calculates Delta-v required to perform specified Bielliptic transfer manoeuvre. m,m,m,kg -> m/s
* biellipticinf - Calculates theoretical minimum Delta-v required to perform specified Bielliptic transfer manoeuvre. m,m,kg -> m/s
* bielliptictime - Calculates time required to perform specified Bielliptic transfer manoeuvre. m,m,m,kg -> s
* density - Density of the sphere. kg,m -> kg/m^3
* drake - Drake Equation. Use -1 to specify default estimate. 1/y,\[Dimensionless],\[Dimensionless],\[Dimensionless],\[Dimensionless],\[Dimensionless],y -> \[Dimensionless]
* energy - Calculates rest energy of specified object. kg -> J
* esi - Calculates the Earth Similarity Index of specified object. m,kg,K -> \[Dimensionless]
* gravbinding - Gravitational binding energy of specified body. m,kg -> J
* gravbinding2 - Gravitational binding energy of specified body, using density instead of mass. m,kg/m^3 -> J
* habitablezone - Approximates inner and outer bounds of the habitable zone of a hypothetical star with specified mass. kg-> m,m
* hill - Calculates hill sphere of specified body. cf. *soi*. kg,kg,m,\[Dimensionless] -> m
* hohmann - Calculates Delta-v required to perform specified Hohmann transfer manoeuvre. m,m,kg -> m/s
* inclinationburn - Calculates Delta-v required to perform specified plane change manoeuvre. rad,\[Dimensionless],rad,rad,rad,m -> m/s
* itemp - Calculates SMA from Temp. K,m,\[Dimensionless],K -> m
* mass - Calculates relativistic mass of specified object. J -> kg
* mov - Mean orbital velocity of the specified orbit. kg,m -> m/s
* optimal - Calculates which of Hohmann or Bielliptic transfer manoeuvres is more efficient. m,m,kg->\[String]
* orbitenergy - Calculates the specific orbital energy of the orbit. kg,m -> m^2/s^2
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
* temp - Calculates planetary equilibrium temperature. K,m,,m\[Dimensionless] -> K
* vescape - Calculates escape velocity of specified body. kg,m->m/s
* vextrema - Orbital velocity at periapsis and apoapsis of the specified orbit, respectively. m,\[Dimensionless],kg -> m/s,m/s
* vtan - Calculates tangental velocity of specified rotating sphere. s,m->m/s

## Change Library Documentation
### Functions
* change - If there is a way to find n coins totalling $m, it will find it. Otherwise, returns False.

## Math Library Documentation
### Functions

* abundancy - Abundancy of n.
* area - Area of non-intersecting polygon defined by points.
* areacircle - Area of a circle.
* areacone - Surface area of a cone.
* areacylinder - Surface area of a cylinder.
* areasphere - Surface area of a sphere.
* areatetrahedron - Surface area of a tetrahedron.
* bernoulli - Bernoulli numbers.
* birthday - Solves the birthday problem for n people.
* bounding - Lower left and upper right points of the bounding box of all points in a set.
* cdf - Cumulative distribution function.
* circumference - Circumference of a circle.
* continuedfraction - Turns the series into decimal notation.
* dist - Distance between two points in n-dimensional space.
* divisors - Finds all divisors of n.
* divisorfunction - Finds number of divisors of n.
* doublefactorial - Finds n!!.
* dpoly - Finds nth derivative of a polynomial using the coefficients.
* fitts - Calculates Fitts' law.
* greedyegyptian - Returns the greedy egyptian fractional expansion of x.
* harmonic - Calculates harmonic numbers.
* heron - Calculates area of a triangle using its sidelengths.
* interiorangle - Calculates interior angle of a regular n-gon.
* interioranglesum - Calculates the sum of the interior angles for a regular n-gon.
* ipoly - Calculates the integral of a polynomial.
* lcm - Calculates the least common multiple of two integers.
* logistic - Logistic function.
* ncr - ncr.
* normal - Normal distribution curve.
* npr - npr.
* ord - Multiplicative order.
* quadratic - Solves for the zeroes of the quadratic with coefficients a, b, c.
* randomspherepoint - Returns an ACTUAL random point on a sphere, without polar clustering caused by naive methods.
* sgn - Sign function. Returns 0 for x=0
* snormal - Standard normal distribution curve.
* standardlogistic - Standard logistic function.
* sumofdivisors - Sum of divisors of n.
* totient - Euler's totient function.
* triangular - nth triangular number.
* volumecone - Volume of a cone.
* volumecylinder - Volume of a cylinder.
* volumensphere - Volume of an n-sphere.
* volumesphere - Volume of a sphere.
* volumetetrahedron - Volume of a tetrahedron.
* weierstrass - Weierstrass function.
* woodall - woodall function.
* zeta - Riemann zeta function.
* zipf - Zipf's law.

### Googological Functions

* ack - Ackermann Function.
* arrayed - Linear array notation.
* arrow - Conway's Arrow Notation.
* beaf - BEAF.
* chained - Chained arrow notation.
* graham - g_n.
* pentation - Tetration.
* tetration - Tetration.

### Random etc. tools

* averagetimestooccur - Average time for an event with probability p to occur.
* card - random playing card
* chanceofoccuring - more stat crap.
* cubicstats - Returns zeroes (soon!!!), extrema, and inflection points of a cubic.
* dice - ndn+bonus
* infiniroot - sqrt(n+sqrt(n+...
* maximizerect - Finds maximum area bounded by an m by n grid of rectangles with total perimeter P.
* mnm - random mnm based on published distributions by mars corp.
* pascal - Returns nth row of Pascal's triangle.
* pascalrow - Identical to *pascal*, deprecated

### Series

* iproduct - Infinite Summation.
* isum - Infinite Product.
* pproduct - Product.
* ssum - Summation.

### Sets

* isburningship - Is c in the burning ship set?
* ismandelbrot - Is c in the mandelbrot set?

## Primality Library Documentation
### Functions
* primality - Attempts to primecheck n.

## Runs or Straights Library Documentation
### Functions
* runs - Finds all runs in a set.
* sor - Returns greatest straight or run in a set.
* straights - Finds all straights in a set.

## Wavefunction Library Documentation
All functions require three components - *x* (Time in s), *freq* (Frequency in hz), and *amp* (Amplitude). They all return a y-value.
### Functions
* f2n - turns a frequency into a musical note
* n2f - turns a musical note into a frequency
* sawtooth - Sawtooth wave function. Uses modulo.
* sine - Sine wave function. Uses trigonometric functions.
* sine1 - Sine wave function. Uses complex powers to compute the wave instead of trigonometric functions.
* square - Square wave function. Uses flooring and modulo.
* square1 - Square wave function. Uses the imaginary part of ln(sin(x)).
* triangle - Triangle wave function.

## Chemistry Library Documentation

## Language Library Documentation

VERY UNSTABLE PLS DONT ACTUALLY USE THIS YET

## Matrix Library Documentation
### Matrix Variables
* lm - a dictionary of 5x5 matrices containing the alphabet
* long - a large sample markov matrix

### Matrix Functions
* adj - finds the adjugate of the matrix
* augmatrixsolve - solves the augmented matrix
* companion - finds the companion matrix of the given polynomial coefficients
* det - finds the determinant of the matrix
* disp - displays matrix in a more natural look
* disp2 - like disp, but displays block characters for 1s and blanks for other values
* echelon - converts the matrix to row echelon form
* eigenhelp - finds the rre of the matrix minus the eigenvalue times the identity matrix. The result is a solved system of equations.
* flipx - flips the matrix across the x-axis
* flipy - flips the matrix across the y-axis
* identity - finds the identity matrix with n rows
* infpower - finds the approximate infinite power of the matrix. may cause the program to hang if such a value does not exist.
* inverse - finds the inverse of the matrixmatrix
* matrixadd - adds two matrices
* matrixdiv - multiplies one matrix by the inverse of another
* matrixexp - raises a matrix to an integer power
* matrixmul - multiplies two matrices
* matrixsub - subtracts two matrices
* matrixscalar - multiplies a matrix by a scalar
* rmatrix - creates an mxn matrix with random values between 0 and 1
* rot - rotates the matrix a given number of times clockwise 90 degrees
* rotationmatrix - gives the rotation matrix for the given angle
* rre - converts the matrix to reduced row echelon form... I think.
* smallmatrixsqrt - attempts to find square roots of a 2x2 matrix
* trace - finds the trace of a square matrix
* transpose - finds the transposition of a matrix
* zero - finds the zero matrix with n rows