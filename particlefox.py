from math import ceil, pi

# numeric
# alpha = 137.035999084 # dimensionless
c = 299792458 # m/s
Da = 1.66053906660e-27 # kg; atomic mass unit AKA Dalton
# h = 6.62607015e-34 # J*s
# mu_0 = 4*pi*1.00000000082e-7 # H/m
N_A = 6.02214076e23 # kg/mol
q_e = 1.602176634e-19 # C
R_H = 1.097e7# 1/m; Rydberg constant
wavelength_violet = 380e-9
wavelength_red = 740e-9

# derived
eV = 1.602176634e-19 # J
# h_ = h/(2*pi) # J*s
keV = 1e3 * eV
MeV = 1e6 * eV # J
MeVc2 = MeV / c**2 # kg
m_e = 9.10938356e-31 # kg

# a_0 = h_ / (m_e * c * alpha) # m

# data
nobles = [2, 10, 18, 36, 54, 86, 172]
symbols = ['H', 'He', 
	'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
	'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 
	'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
	'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 
	'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
	'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]
# https://en.wikipedia.org/wiki/Electron_affinity_(data_page)
# in J/mol
affinity = [72769, -48000, 59632.6, -48000, 26989, 121776.3, -6800, 140976, 328164.9, -116000, 52867, -40000, 41762, 134068.4, 72037, 200410.1, 348575, -96000, 48383, 2370, 18000, 7289, 50911, 65210, -50000, 14785, 63898, 111650, 119235, -58000, 41000, 118935.2, 77650, 194958.7, 324537, -96000, 46884, 5023, 29600, 41806, 88516, 72100, 53000, 100960, 110270, 54240, 125862, -68000, 37043, 107298.4, 101059, 190161, 295153.1, -77000, 45505, 13954, 53000, 55000, 93000, 184870, 12450, 15630, 11200, 13220, 112400, 33960, 32610, 30100, 99000, -1930, 23040, 17180, 31000, 78760, 5827.3, 103990, 150940, 205041, 222747, -48000, 36400, 34418.3, 90924, 136000, 233000, -68000, 46890, 9648.5, 33770, 112720, 53030, 50940, 45850, -48330, 9930, 27170, -165240, -97310, -28600, 33960, 93910, -223220, -30040, 151000, 66600, 35300, 74900, 165900, 5403.18, 63870, 2030, 55000]


def electron_transition(z: int, n1: int, n2: int) -> float:
	return 1/(R_H * z**2 * (1/n1**2 - 1/n2**2))


def lorentz(v: float) -> float:
	"""Lorentz Factor (dimensionless)"""
	beta2 = v**2 / c**2
	return (1-beta2)**-.5


def shell_name(n: int) -> str:
	return chr(n+75)


def subshell_name(n: int) -> str:
	return 'spd'[n] if n < 3 else chr(n+99)


def subshell_size(n: int) -> int:
	return 4*n+2

def wavelength_to_rgb(wavelength: float) -> (float, float, float):
	# super simplistic cause i have no fucking idea what's goin on in http://www.fourmilab.ch/documents/specrend/
	wavelength *= 10 ** 9
	# infrared
	if 740 < wavelength:
		return 0, 0, 0
	# reds
	if 625 < wavelength:
		return 1, 0, 0
	# red (625) -> yellow (580)
	if 580 < wavelength:
		return 1, 1-(wavelength-580)/45, 0
	# yellow (580) -> green (530)
	if 530 < wavelength:
		return (wavelength-530)/50, 1, 0
	# green (530) -> cyan (505)
	if 505 < wavelength:
		return 0, 1, 1-(wavelength-505)/25
	# cyan (505) -> blue (470)
	if 470 < wavelength:
		return 0, (wavelength-470)/35, 1
	# blue (470) -> magenta (380)
	if 380 < wavelength:
		return 1-(wavelength-380)/90, 0, 1
	# ultraviolet
	return .0, .0, .0


def wiki(object) -> str:
	return 'https://en.wikipedia.org/wiki/' + object.name


class Particle:
	"""Elementary particle"""
	def __init__(self, name: str, properties: dict):
		self.name = name
		self.properties = properties

	@property
	def mass(self) -> float:
		return self.properties['mass']


class Composite:
	"""Composite particle"""
	def __init__(self, name: str, constituents: dict, **properties):
		self.name = name
		self.constituents = constituents # type: Dict[Particle, int]
		self.properties = properties

	@property
	def charge(self) -> float:
		return sum(i.charge for i in self.constituents)

	@property
	def mass(self) -> float:
		gamma = 103.10682287477916 # PROTONS lorentz(0.99995*c)
		gamma = 80.99701770276388 # NEUTRONS lorentz(0.99992*c)
		return self.properties['mass'] if 'mass' in self.properties else gamma*sum(i.mass for i in self.constituents)


class Isotope:
	def __init__(self, z: int=1, n: int=0, e: int=None, **properties):
		self.z = z
		self.n = n
		self.e = z if e is None else e
		self.properties = properties

	@property
	def a(self) -> int:
		"""Atomic Weight"""
		return self.z + self.n

	@property
	def block(self) -> str:
		return self.electron_configuration[-2]
	
	@property
	def electron_affinity(self) -> float:
		return affinity[self.z]

	@property
	def electron_configuration(self) -> str:
		"""Inaccurate above z=21(?)"""
		s = ''
		e = self.e
		shell = 0
		while 1:
			# todo https://en.wikipedia.org/wiki/Aufbau_principle
			for subshell in range(0, shell+1):
				s += str(shell+1)+subshell_name(subshell)
				size = subshell_size(subshell)
				if e < size:
					s += str(e)
					return s
				s += str(size)+' '
				e -= size
			shell += 1
	
	@property
	def electronegativity(self) -> float:
		"""Mulliken electronegativity"""
		return (self.ionization_energy(1) + self.electron_affinity/N_A)/2

	@property
	def electronegativity2(self) -> float:
		"""Mulliken pseudo-Pauling electronegativity"""
		return 0.374 * self.electronegativity/keV + 0.17

	@property
	def mass(self) -> float:
		z, n, e = self.z, self.n, self.e
		return z*proton.mass + n*neutron.mass + e*electron.mass

	@property
	def period(self) -> int:
		if self.z in nobles:
			return nobles.index(self.z) + 1
		return Isotope(self.z+1).period

	@property
	def mass_excess(self) -> float:
		return self.properties['mass'] - (self.z + self.n)*Da

	@property
	def nuclear_binding_energy(self) -> float:
		"""Nuclear binding energy (per nucleon), in MeV"""
		# https://en.wikipedia.org/wiki/Nuclear_binding_energy#Semiempirical_formula_for_nuclear_binding_energy
		a, b, c, d, e = 14, 13, .585, 19.3, 33
		A, Z, N = self.a, self.z, self.n
		pt = 1 if Z % 2 == 0 == N % 2 else (-1 if Z % 2 == 1 == N % 2 else 0)
		# print(pt, e/A**(7/4))
		return a - b/A**(4/3) - c*Z**2/A**(4/3) - d*(N-Z)**2/A**2 + pt*e/A**(7/4)

	@property
	def shells(self) -> int:
		"""Number of electron shells in an atom. Inaccurate above z=34"""
		return ceil((self.e-2)/4)

	@property
	def spectrum(self) -> set:
		"""SWAG of spectrum"""
		# http://hyperphysics.phy-astr.gsu.edu/hbase/hyde.html
		z = self.z
		n1 = 1
		# bring ns down to UV
		while wavelength_violet < electron_transition(z, n1, n1+1):
			n1 -= 1
		# now, increase, collect all good lambdas, meh
		wavelengths = set()
		while 1:
			n1 += 1
			# print('n1', n1)
			# if even n2=inf doesn't give a small enough wavelength, break
			if wavelength_red < electron_transition(z, n1, float('inf')):
				break
			n2 = n1
			while n2 - n1 < 8: # increase n2; n2 max set to 8 to prevent infinite loops
				n2 += 1
				# print('n2', n2)
				l = electron_transition(z, n1, n2)
				# print('\t->', int(l*10**9), 'nm')
				if l < wavelength_violet: # too far!!!
					break
				if electron_transition(z, n1, n2) < wavelength_red: # good lambda
					# wavelengths.add((l, n2 - n1))
					wavelengths.add(l)
					# print('\t   GOOD')
				# else you can keep going
			# if n2 == n1+1: # couldn't find any more
			#	break
		return wavelengths

	@property
	def symbol(self) -> str:
		return symbols[self.z-1]
	
	# decay modes

	@property
	def alpha_decay(self):
		z, n = self.z, self.n
		return Isotope(z-2, n-2)

	@property
	def beta_minus_decay(self):
		z, n = self.z, self.n
		return Isotope(z+1, n-1)

	@property
	def beta_plus_decay(self):
		z, n = self.z, self.n
		return Isotope(z-1, n+1)

	@property
	def electron_capture(self):
		z, n = self.z, self.n
		return Isotope(z-1, n)

	@property
	def neutron_emission(self):
		z, n = self.z, self.n
		return Isotope(z, n-1)

	@property
	def proton_emission(self):
		z, n = self.z, self.n
		return Isotope(z-1, n)

	@property
	def predict_decay(self):
		"""Very simplistic model"""
		z, n = self.z, self.n
		if 83 < z:
			return 'a', self.alpha_decay
		# extreme cases
		if 2.3 < n/z:
			return 'N', self.neutron_emission
		if n+4 < z:
			return 'EC', self.proton_emission
		
		# beta decay
		if z < 20:
			ratio = n/z
		else:
			ratio = (.61 * n + 5)/z
		if ratio < 1:
			return 'b+', self.beta_plus_decay
		return 'b-', self.beta_minus_decay	

	# double underscore methods
	
	def __eq__(self, other) -> bool:
		return all([self.z == other.z, self.n == other.n, self.e == other.e])
	
	def __hash__(self) -> int:
		return hash((self.z, self.n, self.e))
	
	def __str__(self) -> str:
		return '<Isotope {0}{1}>'.format(self.symbol, self.a)

	# methods
	def ionization_energy(self, n: int) -> float:
		# https://en.wikipedia.org/wiki/Ionization_energy#Electrostatic_explanation
		z = self.z
		return 13.6*eV*z**2/n**2
	
	def plot_spectrum(self):
		import matplotlib.pyplot as plt
		lambdas = self.spectrum
		plt.subplot(1, 1, 1)
		ilist = []
		zlist = []
		# todo plot roots
		for l in lambdas:
			x = round(l*10**9)
			plt.axvline(x=x, color=wavelength_to_rgb(l))
			# numeric label
			y = (l-wavelength_violet)/(wavelength_red-wavelength_violet)
			plt.text(x, y, '{0} nm'.format(int(x)), fontdict={'size': 8})
		plt.title('{} Spectral lines'.format(self.symbol))
		plt.xlabel('wavelength (nm)')
		plt.xlim(10**9 * wavelength_violet, 10**9 * wavelength_red)
		plt.ylim(0, 1)
		plt.show()

class Element:
	def __init__(self, isotopic_abundances: dict, **properties):
		self.isotopic_abundances = isotopic_abundances
		self.properties = properties
	
	@property
	def any_isotope(self) -> float:
		"""For when any isotope can be used for a calculation"""
		return list(self.isotopic_abundances)[0]
	
	@property
	def mass(self) -> float:
		s = sum(abundance for abundance in self.isotopic_abundances.values())
		return sum(None for isotope, abundance in self.isotopic_abundances.items())/s
	
	@property
	def period(self) -> str:
		return self.any_isotope.period
	
	@property
	def spectrum(self) -> str:
		return self.any_isotope.spectrum
	
	@property
	def symbol(self) -> str:
		return self.any_isotope.symbol

	# double underscore methods
	
	def __str__(self) -> str:
		return '<Element {0}>'.format(self.symbol)

	# methods
	
	def plot_spectrum(self):
		self.any_isotope.plot_spectrum()


class Molecule:
	def __init__(self, elements: list, bonds: set): # Set[(from_id: int, to_id: int, type: int)]
		self.elements = elements
		self.bonds = bonds

	@property
	def chemical_formula(self) -> str:
		unique = sorted(list(set(self.elements)), key=lambda x: x.symbol)
		return ''.join([element.symbol + (str(self.elements.count(element)) if 1 < self.elements.count(element) else '') for element in unique])

	@property
	def mass(self) -> float:
		return sum(element.mass for element in self.elements)

	@property
	def molar_mass(self) -> float:
		return self.mass * N_A

	# double underscore methods

	def __str__(self) -> str:
		return '<Molecule {0}>'.format(self.chemical_formula)


# Elementary Particles
down_quark = Particle('Down quark', {
	'mass': 4.7*MeVc2,
	'charge': -1/3*q_e,
})
electron = Particle('Electron', {
	'mass': m_e,
	'charge': -1*q_e,
})
up_quark = Particle('Up quark', {
	'mass': 2.2*MeVc2,
	'charge': 2/3*q_e,
})

# Composite Particles
neutron = Composite('Neutron', {
	up_quark: 1,
	down_quark: 2,
}, mass=1.674927471e-27)
proton = Composite('Proton', {
	up_quark: 2,
	down_quark: 1,
}, mass=1.67262192369e-27)

# Isotopes
protium = Isotope(mass=1.007825*Da)
deuterium = Isotope(1, 1, mass=2.013553212745*Da)
tritium = Isotope(1, 2, mass=3.0160492*Da)
He4 = Isotope(2, 2, mass=4.002602*Da)
C12 = Isotope(6, 6, mass=12*Da)
N14 = Isotope(7, 7, mass=14.00307400446*Da)
O16 = Isotope(8, 8, mass=15.99491461956*Da)
Fe56 = Isotope(26, 32, mass=55.9349363*Da)
Pb208 = Isotope(82, 126, mass=207.9766521*Da)
U238 = Isotope(92, 146, mass=238.05078826*Da)
Pu244 = Isotope(94, 150, mass=244.0642044*Da)
isotopes = {
	protium, deuterium, tritium,
	He4,
	C12,
	N14,
	O16,
	Fe56,
	Pb208,
	U238,
	Pu244,
}

# Elements
hydrogen = Element({protium: .99985, deuterium: .015, tritium: 0})
helium = Element({He4: .99999863})
carbon = Element({C12: .9893})
nitrogen = Element({N14: .99636})
oxygen = Element({O16: .9976})
iron = Element({Fe56: .9175})
lead = Element({Pb208: .524})
uranium = Element({U238: .992745})
plutonium = Element({Pu244: 8e7})
elements = { # sort by period
	hydrogen, helium,
	carbon, nitrogen, oxygen,
	iron,
	lead, uranium, plutonium,
}

# Molecules
water = Molecule(
	[hydrogen, oxygen, hydrogen],
	{(0, 1, 1), (2, 1, 1)}
)

# test
# helium.plot_spectrum()
