from math import ceil, pi

# numeric
# alpha = 137.035999084 # dimensionless
c = 299792458 # m/s
Da = 1.66053906660e-27 # kg; atomic mass unit AKA Dalton
# h = 6.62607015e-34 # J*s
# mu_0 = 4*pi*1.00000000082e-7 # H/m
N_A = 6.02214076e23 # kg/mol
q_e = 1.602176634e-19 # C

# derived
eV = 1.602176634e-19 # J
# h_ = h/(2*pi) # J*s
MeV = 1e6 * eV # J
MeVc2 = MeV / c**2 # kg
m_e = 9.10938356e-31 # kg

# a_0 = h_ / (m_e * c * alpha) # m


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


class Atom:
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
		# https://en.wikipedia.org/wiki/Electron_affinity_(data_page)
		affinity = [72769, 72807, -48000, 59632.6, 121776.3, 121775.5] # todo 
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
		return (self.ionization_energy(1) + self.electron_affinity)/2

	@property
	def mass(self) -> float:
		z, n, e = self.z, self.n, self.e
		return z*proton.mass + n*neutron.mass + e*electron.mass

	@property
	def shells(self) -> int:
		"""Number of electron shells in an atom. Inaccurate above z=34"""
		return ceil((self.e-2)/4)

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
	def symbol(self) -> str:
		symbols = [
			'H', 'He', 
			'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
		]
		return symbols[self.z-1]
	
	# decay modes

	@property
	def alpha_decay(self):
		z, n = self.z, self.n
		return Atom(z-2, n-2)

	@property
	def beta_decay(self):
		z, n = self.z, self.n
		return Atom(z+1, n-1)

	@property
	def beta_plus_decay(self):
		z, n = self.z, self.n
		return Atom(z-1, n+1)

	@property
	def electron_capture(self):
		z, n = self.z, self.n
		return Atom(z-1, n)

	# double underscore methods
	
	def __eq__(self, other) -> bool:
		return all([self.z == other.z, self.n == other.n, self.e == other.e])
	
	def __hash__(self) -> int:
		return hash((self.z, self.n, self.e))

	# methods
	def ionization_energy(self, n: int) -> float:
		# https://en.wikipedia.org/wiki/Ionization_energy#Electrostatic_explanation
		z = self.z
		return 13.6*eV*z**2/n**2

# todo: split Atom class into Isotope and Element classes, for a set of all isotopes


class Molecule:
	def __init__(self, atoms: list, bonds: set): # Set[(from_id: int, to_id: int, type: int)]
		self.atoms = atoms
		self.bonds = bonds

	@property
	def chemical_formula(self) -> str:
		unique = sorted(list(set(self.atoms)), key=lambda x: x.symbol)
		return ''.join([atom.symbol + (str(self.atoms.count(atom)) if 1 < self.atoms.count(atom) else '') for atom in unique])

	@property
	def mass(self) -> float:
		return sum(atom.mass for atom in self.atoms)

	@property
	def molar_mass(self) -> float:
		return self.mass * N_A


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

# Atoms
hydrogen = Atom(mass=1.007825*Da)
deuterium = Atom(1, 1, mass=2.013553212745*Da)
tritium = Atom(1, 2, mass=3.0160492*Da)
helium = Atom(2, 2, mass=4.002602*Da)
carbon = Atom(6, 6, mass=12*Da)
oxygen = Atom(8, 8, mass=15.99491461956*Da)
lead = Atom(82, 126, mass=207.9766521*Da)
uranium = Atom(92, 146, mass=238.05078826*Da)
isotopes = {
	hydrogen, deuterium, tritium,
	helium,
	carbon,
	oxygen,
	lead,
	uranium,
}

# Molecules
water = Molecule(
	[hydrogen, oxygen, hydrogen],
	{(0, 1, 1), (2, 1, 1)}
)