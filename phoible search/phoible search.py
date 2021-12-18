filename = 'phoible.csv'

langdata = {}

for line in open(filename, 'r', encoding='utf-8').read().split('\n'):
	# print(line) # debug
	lang, segment = line.split(',')
	if lang in langdata:
		langdata[lang].add(segment)
	else:
		langdata[lang] = {segment}

# now find languages with /p b t k g/ but no /d/

acceptable = {'p', 'b', 't', 'k', 'g'}

print(list(filter(lambda l: acceptable <= langdata[l] and 'd' not in langdata[l], langdata)))
# len(list(filter(lambda l: 'ɡ' in langdata[l] and 'b' not in langdata[l], langdata)))