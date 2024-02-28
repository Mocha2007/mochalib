from random import randint

def d6() -> int:
	return randint(1, 6)

def roll() -> int:
	"""Accounts for doubles (up to three times)"""
	s = 0
	for r in range(3):
		d0, d1 = d6(), d6()
		if d0 == d1 and r == 2: # speeding
			break
		s += d0 + d1
		if d0 != d1: # was this doubles?
			break
	return s

def stats_roll(n = 1000000) -> None:
	from numpy import histogram
	from matplotlib.pyplot import show, stairs
	# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
	data = [roll() for _ in range(n)]
	counts, bins = histogram(data, bins=range(36))
	stairs(counts, bins)
	show()

def dicey_descent() -> tuple[int, int, int, int]:
	"""Return the place of 4 players"""
	# https://wiisports.fandom.com/wiki/Dicey_Descent
	died_on = [0, 0, 0, 0]
	for roundn in range(1, 4):
		choices = [randint(0, 1) for _ in range(4)]
		lightning = randint(0, 1)
		for i, choice in enumerate(choices):
			if choice == lightning and not died_on[i]:
				died_on[i] = roundn
	place = [1 + sum(v0 < v1 for v1 in died_on) for v0 in died_on]
	return place


def risky_railroad() -> tuple[int, int, int, int]:
	"""Return the place of 4 players"""
	# https://wiisports.fandom.com/wiki/Risky_Railroad
	# eg. sum(len(set(risky_railroad())) == 4 for _ in range(1000000)) / 1000000
	died_on = [0, 0, 0, 0]
	for roundn in range(1, 11):
		choices = [randint(0, 2) for _ in range(4)]
		dead_end = randint(0, 2)
		for i, choice in enumerate(choices):
			if choice == dead_end and not died_on[i]:
				died_on[i] = roundn
	place = [1 + sum(v0 < v1 for v1 in died_on) for v0 in died_on]
	return place
