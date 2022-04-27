from math import floor
from random import randint

def dice(n: int = 1, sides: int = 6) -> int:
	return sum(randint(1, sides+1) for _ in range(n))

def eu4_army_comp(units: int, cavalry_ratio: float = 0.5, use_artillery: bool = True):
	"""Determines best SAFE inf/cav/art ratio"""
	artillery = units // 2 # should be half or less
	units -= artillery
	cavalry = units * cavalry_ratio # should be strictly less than the ratio to prevent slight casualties from triggering neg modifier
	if cavalry % 1 == 0: # whole number
		cavalry = int(cavalry - 1)
	else:
		cavalry = floor(cavalry)
	infantry = units - cavalry
	return infantry, cavalry, artillery

def estimate_siege_time(siege_phase_time: int = 35, siege_status: int = 0, leader: int = 0, artillery: int = 0,
		blockade: int = 0, fort: int = 0, breach: int = 0, garrison: int = 1000,
		obsolete_fort_bonus: int = 0, attacker_max_fort: int = 2, trials: int = 1000) -> int:
	def simulate_siege() -> int:
		_siege_status = siege_status
		_breach = breach
		_garrison = garrison
		_siege_over = False
		ticks = 0
		def siege_tick() -> None:
			nonlocal _siege_status, _breach, _garrison, _siege_over
			unmodified_roll = dice(1, 14)
			roll = unmodified_roll + _siege_status + leader + artillery + blockade + fort + _breach
			# first, test if a breach occurs
			if _breach < 3 and 14 <= unmodified_roll + \
					(artillery + obsolete_fort_bonus) / 3 + attacker_max_fort / 10:
				_breach += 1

			match roll:
				case n if n < 5: # status quo / disease outbreak
					pass
				case n if n < 12: # supply shortage
					_siege_status += 1
					_garrison *= 0.99
				case n if n < 14: # food shortage
					_siege_status += 2
					_garrison *= 0.97
				case n if n < 15: # water shortage
					_siege_status += 3
					_garrison *= 0.95
				case n if n < 20: # defenders desert
					_siege_status += 2
					_garrison *= 0.9
				case _: # surrender
					_siege_over = True
		while not _siege_over and 100 <= _garrison:
			siege_tick()
			ticks += 1
		return ticks
	return sum(simulate_siege() * siege_phase_time for _ in range(trials)) / trials