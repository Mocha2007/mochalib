from random import randint

def d6() -> int:
	return randint(1, 6)

def roll() -> int:
	"""Accounts for doubles (up to three times)"""
	s = 0
	r = 0
	while (d0 := d6()) == (d1 := d6()) and (r := r + 1) < 3:
		s += d0 + d1
	s += d0 + d1
	return s

def stats_roll(n = 1000000) -> None:
	from numpy import histogram
	from matplotlib.pyplot import show, stairs
	# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
	data = [roll() for _ in range(n)]
	counts, bins = histogram(data, bins=range(37))
	stairs(counts, bins)
	show()
