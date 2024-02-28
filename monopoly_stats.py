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
