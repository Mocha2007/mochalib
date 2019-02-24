from random import random
from typing import Dict, Set, TypeVar
from json import load

inf = float('inf')
number = TypeVar('number', int, float, complex)


class Resource:
	def __init__(self, **kwargs):
		# name
		if 'name' in kwargs:
			self.name = kwargs['name'] # type: str
		else:
			self.name = str(random())[2:] # type: str
		# requirements (if any)
		if 'req' in kwargs:
			self.req = kwargs['req'] # type: Dict[str, number]
		else:
			self.req = {} # type: dict

	def price(self, in_economy) -> number: # in_economy type: Economy
		if self not in in_economy.demand:
			return 0
		if self not in in_economy.supply:
			return inf
		return in_economy.demand[self]/in_economy.supply[self]


class Recipe:
	def __init__(self, **kwargs):
		# name
		if 'name' in kwargs:
			self.name = kwargs['name'] # type: str
		else:
			self.name = str(random())[2:] # type: str
		# reagents (if any)
		if 'reagents' in kwargs:
			self.reagents = kwargs['reagents'] # type: Dict[Resource, number]
		else:
			self.reagents = {} # type: dict
		# products (if any)
		if 'products' in kwargs:
			self.products = kwargs['products'] # type: Dict[Resource, number]
		else:
			self.products = {} # type: dict

	def reagent_price(self, in_economy) -> number: # in_economy type: Economy
		return sum(i.price(in_economy) for i in self.reagents)

	def product_price(self, in_economy) -> number: # in_economy type: Economy
		return sum(i.price(in_economy) for i in self.products)


class PopType:
	def __init__(self, **kwargs):
		# name
		if 'name' in kwargs:
			self.name = kwargs['name'] # type: str
		else:
			self.name = str(random())[2:] # type: str
		# recipes (if any)
		if 'recipes' in kwargs:
			self.recipes = kwargs['recipes'] # type: Set[Recipe]
		else:
			self.recipes = set() # type: set

	def desirability(self, in_economy) -> number: # in_economy type: Economy
		return sum(i.product_price(in_economy) for i in self.recipes)


class Economy:
	def __init__(self, **kwargs):
		# name
		if 'name' in kwargs:
			self.name = kwargs['name'] # type: str
		else:
			self.name = str(random())[2:] # type: str
		# pops (if any)
		if 'pops' in kwargs:
			self.pops = kwargs['pops'] # type: Dict[PopType, number]
		else:
			self.pops = set() # type: set

		self.supply = {} # type: Dict[Resource, number]
		self.demand = {} # type: Dict[Resource, number]


resource_file = load(open('eco_resources.json', 'r'))
recipe_file = load(open('eco_recipes.json', 'r'))
pop_file = load(open('eco_pops.json', 'r'))
economy_file = load(open('eco_economies.json', 'r'))

resources = {Resource(**i) for i in resource_file}
recipes = {Recipe(**i) for i in recipe_file}
pops = {PopType(**i) for i in pop_file}
economies = {Economy(**i) for i in economy_file}

while 1:
	for economy in economies:
		for pop in economy.pops:
			# todo input
			# todo output
			pass
		# todo births
		# todo deaths
		# todo occupation changes
