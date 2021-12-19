from typing import Set

filename = 'phoible.csv'

langdata = {}

for line in open(filename, 'r', encoding='utf-8').read().split('\n')[1:]:
	# print(line) # debug
	lang, segment = line.split(',')
	if lang in langdata:
		langdata[lang].add(segment)
	else:
		langdata[lang] = {segment}

# now find languages with /p b t k g/ but no /d/

acceptable = {'p', 'b', 't', 'd', 'k', 'ɡ'}

# len(list(filter(lambda l: 'ɡ' in langdata[l] and 'b' not in langdata[l], langdata)))

def list6() -> None:
        for lang, data in langdata.items():
                print(data & acceptable)

def implies(x: str) -> Set[str]:
	s = None
	for data in langdata.values():
		if x in data:
			if s is None:
				s = data
			else:
				s &= data
	return s

def almost_implies(x: str, threshold: float = 0.95) -> Set[str]:
	xx = 0
	yy = {}
	for data in langdata.values():
		if x in data:
			xx += 1
			for y in data:
				if y in yy:
					yy[y] += 1
				else:
					yy[y] = 1
	# delete insufficient entries
	o = {}
	for i, y in yy.items():
		if i != x and threshold <= y/xx:
			o[i] = y/xx
	return o

def absence_implies(x: str) -> Set[str]:
	s = None
	for data in langdata.values():
		if x not in data:
			if s is None:
				s = data
			else:
				s &= data
	return s

def conditional_implication(x: str, y: str) -> float:
	"""Does x -> y ?"""
	xx = yy = 0
	for data in langdata.values():
		if x in data:
			xx += 1
			if y in data:
				yy += 1
	return yy/xx