from collections import Counter


def straights(lis: list) -> list:
	tot = []
	count = Counter(lis)
	for i in set(lis):
		tot.append([i]*count[i])
	tot.sort()
	return tot


def runs(lis: list) -> list:
	tot = []
	for i in lis:
		run = [i]
		size = 1
		while i+size in lis:
			run.append(i+size)
			size += 1
		tot.append(run)
	tot.sort()
	return [tot[i] for i in range(len(tot)) if i == 0 or tot[i] != tot[i-1]]


def sor(lis: list) -> int:
	m = []
	for i in runs(lis):
		if len(m) < len(i):
			m = i
	for i in straights(lis):
		if len(m) < len(i):
			m = i
	return m
