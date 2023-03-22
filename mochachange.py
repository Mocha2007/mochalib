"""Coin library for cheating at that one coin game lmao"""

def change(n: float, total: int):
	"""Figure out how to make the total with n coins"""
	coins = []
	while n > 0:
		n = round(n, 2)
		if n >= .5:
			coins.append(.5)
			n -= .5
		elif n >= .25:
			coins.append(.25)
			n -= .25
		elif n >= .1:
			coins.append(.1)
			n -= .1
		elif n >= .05:
			coins.append(.05)
			n -= .05
		else:
			coins.append(.01)
			n -= .01
	if len(coins) == total:
		return coins
	elif len(coins) > total:
		return False
	while 1:
		if total == len(coins):
			return coins
		elif total-len(coins) >= 4 and coins.count(.05) > 0:
			coins.remove(.05)
			coins += [.01]*5
		elif total-len(coins) >= 2 and coins.count(.25) > 0:
			coins.remove(.25)
			coins += [.1]*2+[.05]
		elif total-len(coins) >= 1 and coins.count(.5) > 0:
			coins.remove(.5)
			coins += [.25]*2
		elif total-len(coins) >= 1 and coins.count(.1) > 0:
			coins.remove(.1)
			coins += [.05]*2
		else:
			return False


while 1:
	try:
		a = float(input("$> "))
		b = int(input("n> "))
		# print(sorted(change(a,b)))
		halfdollar = 0
		quarter = 0
		dime = 0
		nickel = 0
		penny = 0
		for i in change(a, b):
			if i == .5:
				halfdollar += 1
			elif i == .25:
				quarter += 1
			elif i == .1:
				dime += 1
			elif i == .05:
				nickel += 1
			else:
				penny += 1
		if penny:
			print(".01", penny)
		if nickel:
			print(".05", nickel)
		if dime:
			print(".10", dime)
		if quarter:
			print(".25", quarter)
		if halfdollar:
			print(".50", halfdollar)
		if not change(a, b):
			print("Impossible")
	except:
		pass
