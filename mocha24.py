def operations(n: int, lis: list) -> list:
	totals = []
	for i in lis:
		totals += [n+i, n-i, n*i]
		try:
			totals.append(n/i)
		except ZeroDivisionError:
			pass
	return totals


def twentyfour(a: int, b: int, c: int, d: int) -> list: # MUST be in the order a b c d
	return sorted(set(operations(a, operations(b, operations(c, [d])))))


def allorder(a: int, b: int, c: int, d: int):  # any order!
	t = twentyfour(a, b, c, d)
	t += twentyfour(a, b, d, c)
	t += twentyfour(a, c, b, d)
	t += twentyfour(a, c, d, b)
	t += twentyfour(a, d, b, c)
	t += twentyfour(a, d, c, b)
	t += twentyfour(b, a, c, d)
	t += twentyfour(b, a, d, c)
	t += twentyfour(b, c, a, d)
	t += twentyfour(b, c, d, a)
	t += twentyfour(b, d, a, c)
	t += twentyfour(b, d, c, a)
	t += twentyfour(c, a, b, d)
	t += twentyfour(c, a, d, b)
	t += twentyfour(c, b, a, d)
	t += twentyfour(c, b, d, a)
	t += twentyfour(c, d, a, b)
	t += twentyfour(c, d, b, a)
	t += twentyfour(d, a, b, c)
	t += twentyfour(d, a, c, b)
	t += twentyfour(d, b, a, c)
	t += twentyfour(d, b, c, a)
	t += twentyfour(d, c, a, b)
	t += twentyfour(d, c, b, a)
	return sorted(set(t))
