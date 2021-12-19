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

acceptable = {'p', 'b', 't', 'd', 'k', 'ɡ'}

# len(list(filter(lambda l: 'ɡ' in langdata[l] and 'b' not in langdata[l], langdata)))

def list6():
        for lang, data in langdata.items():
                print(data & acceptable)
