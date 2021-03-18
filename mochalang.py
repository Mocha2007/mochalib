from re import sub, compile
from random import choice

romanizations = [
	('[ΑАԱ]', 'A'),
	('[αаա]', 'a'),
	('[ΒБԲ]', 'B'),
	('[βбբ]', 'b'),
	('[ЧՃՉ]', 'Ch'),
	('[чճչ]', 'ch'),
	('[ΔДԴ]', 'D'),
	('[δдդ]', 'd'),
	('Ձ', 'Dz'),
	('ձ', 'dz'),
	('[ΕΗЭѢԵԷԸ]', 'E'),
	('[εηэѣեէը]', 'e'),
	('և', 'ev'),
	('[ФѲՖ]', 'F'),
	('[фѳֆ]', 'f'),
	('[ΓГԳ]', 'G'),
	('[γгգ]', 'g'),
	('Հ', 'H'),
	('հ', 'h'),
	('[ΙИІѴԻ]', 'I'),
	('[ιиіѵի]', 'i'),
	('[ЙՋ]', 'J'),
	('[йջ]', 'j'),
	('[ЯѦѨ]', 'Ja'),
	('[яѧѩ]', 'ja'),
	('Е', 'Je'),
	('е', 'je'),
	('Ё', 'Jo'),
	('ё', 'jo'),
	('[ЮѪѬ]', 'Ju'),
	('[юѫѭ]', 'ju'),
	('[ΚКԿ]', 'K'),
	('[κкկ]', 'k'),
	('[ΧХՔ]', 'Kh'),
	('[χхք]', 'kh'),
	('[ΛЛԼՂ]', 'L'),
	('[λлլղ]', 'l'),
	('[ΜМՄ]', 'M'),
	('[μмմ]', 'm'),
	('[ΝНՆ]', 'N'),
	('[νнն]', 'n'),
	('[ΟΩОѠՈՕ]', 'O'),
	('[οωоѡոօ]', 'o'),
	('[ΠПՊ]', 'P'),
	('[πпպ]', 'p'),
	('[ΦՓ]', 'Ph'),
	('[φփ]', 'ph'),
	('[ΨѰ]', 'Ps'),
	('[ψѱ]', 'ps'),
	('[ΡРՌՐ]', 'R'),
	('[ρрռր]', 'r'),
	('[ΣСՍ]', 'S'),
	('[σςсս]', 's'),
	('[ШЩՇ]', 'Sh'),
	('[шщշ]', 'sh'),
	('[ΤТՏ]', 'T'),
	('[τтտ]', 't'),
	('[ΘԹ]', 'Th'),
	('[θթ]', 'th'),
	('[ЦԾՑ]', 'Ts'),
	('[цծց]', 'ts'),
	('У', 'U'),
	('у', 'u'),
	('[ВՎՒ]', 'V'),
	('[вվւ]', 'v'),
	('[ΥЫՅ]', 'Y'),
	('[υыյ]', 'y'),
	('[ΞѮԽ]', 'X'),
	('[ξѯխ]', 'x'),
	('[ΖЗЅ]', 'Z'),
	('[ζзѕ]', 'z'),
	('[ЖԺ]', 'Zh'),
	('[жժ]', 'zh'),
	('[Ьь]', '\''),
	('[Ъъ]', ''),
]

morsekey = {
	'a': '.-',
	'b': '-...',
	'c': '-.-.',
	'd': '-..',
	'e': '.',
	'f': '..-.',
	'g': '--.',
	'h': '....',
	'i': '..',
	'j': '.---',
	'k': '-.-',
	'l': '.-..',
	'm': '--',
	'n': '-.',
	'o': '---',
	'p': '.--.',
	'q': '--.-',
	'r': '.-.',
	's': '...',
	't': '-',
	'u': '..-',
	'v': '...-',
	'w': '.--',
	'x': '-..-',
	'y': '-.--',
	'z': '--..',
	'1': '.----',
	'2': '..---',
	'3': '...--',
	'4': '....-',
	'5': '.....',
	'6': '-....',
	'7': '--...',
	'8': '---..',
	'9': '----.',
	'0': '-----',
	' ': ' ',
}


def lettersquare(n: int, mode: str) -> str:
	if mode == 'scrabble':
		source = 'e'*12+'ai'*9+'o'*8+'nrt'*6+'lsud'*4+'g'*3+'bcmpfhvwy'*2+'kjxqz'
	elif mode == 'freq':
		source = 'e'*172+'t'*122+'a'*110+'o'*101+'i'*94+'n'*91+'s'*86+'h'*82+'r'*81+'d'*57+'l'*54+'c'*38+'u'*37+'m'*33+'w'*32+'f'*30+'gy'*27+'p'*26+'b'*20+'v'*13+'k'*10+'jx'*2+'qz'
	else:
		source = 'abcdefghijklmnopqrstuvwxyz'
	o = '```'
	for _ in range(n):
		r = '\n'
		for _ in range(n):
			r += choice(source)
		o += r
	return o+'\n```'


def morse(string):
	string = list(sub(r'[^\da-z ]', '', string.lower()))
	string = list(map(lambda x: morsekey[x], string))
	return ' '.join(string)


def letterfreq(string):
	l = {}
	for c in string:
		if c not in l:
			l[c] = 1 / len(string)
		else:
			l[c] += 1 / len(string)
	return l


def sortdict(d):
	l = []
	for i in d:
		l.append((i, d[i]))
	return sorted(sorted(l), key=lambda x: x[1], reverse=True)  # first by freq, then by id


def scrabble(string):
	alphabet = 'abcdefghijklmnopqrstuvwxyz'
	scoring = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10]
	score = 0
	for char in string:
		score += scoring[alphabet.index(char)]
	return score


def unmojibake(string: str, fro: str, to: str) -> str:
	return bytes(string, fro).decode(to)


def romanize(string: str) -> str:
	for i in romanizations:
		string = sub(i[0], i[1], string)
	return string


def soundex(string: str) -> str:
	# Pre-rules
	string = string.title()
	# R1
	string = sub(r"[aeiouyhw'\n]", '', string)
	# R3
	s1 = ['b', 'f', 'p', 'v']
	s2 = ['c', 'g', 'j', 'k', 'q', 's', 'x', 'z']
	s3 = ['d', 't']
	s5 = ['m', 'n']
	# R3 - checking for dupe 1s
	for i in s1:
		for j in s1:
			string = sub(i + j, i, string)
	# R3 - checking for dupe 2s
	for i in s2:
		for j in s2:
			string = sub(i + j, i, string)
	# R3 - checking for dupe 3s
	for i in s3:
		for j in s3:
			string = sub(i + j, i, string)
	# R3 - checking for dupe 5s
	for i in s5:
		for j in s5:
			string = sub(i + j, i, string)
	# R2
	string = sub(r'[bfpv]', '1', string)
	string = sub(r'[cgjkqsxz]', '2', string)
	string = sub(r'[dt]', '3', string)
	string = sub(r'l', '4', string)
	string = sub(r'[mn]', '5', string)
	string = sub(r'r', '6', string)
	# R4
	string += '000'
	string = list(string)[0] + list(string)[1] + list(string)[2] + list(string)[3]
	return string


def uipa(word):
	word = sub('j', 'dʒ', word)
	word = sub('ôw', 'au', word)
	word = sub('öy', 'oi', word)
	word = sub('@r', 'ɚ', word)
	word = sub('@n', 'n̩', word)
	word = sub('@l', 'l̩', word)
	word = sub('ä', 'e', word)
	word = sub('â', 'æ', word)
	word = sub('ë', 'i', word)
	word = sub('ê', 'ɛ', word)
	word = sub('ï', 'ai', word)
	word = sub('î', 'ɪ', word)
	word = sub('ö', 'o', word)
	word = sub('ô', 'ɑ', word)
	word = sub('ü', 'ju', word)
	word = sub('û', 'ʌ', word)
	word = sub('ò', 'ɔ', word)
	word = sub('ù', 'ʊ', word)
	word = sub('@', 'ə', word)
	word = sub('y', 'j', word)
	word = sub('ñ', 'ŋ', word)
	word = sub('\+', 'θ', word)
	word = sub('\$', 'ʃ', word)
	word = sub('%', 'ʒ', word)
	word = sub('ç', 'tʃ', word)
	word = sub('r', 'ɹ', word)
	return word


def ipa(word):  # attempts to generate the IPA transcription of a word based on Zompist SC rules
	# http://www.zompist.com/spell.html
	vowel = ['a', 'e', 'i', 'o', 'u', 'ä', 'â', 'ë', 'ê', 'ï', 'î', 'ö', 'ô', 'ü', 'û', 'ò', 'ù', '@']
	consonant = ['p', 'b', 't', 'd', 'g', 'k', 'm', 'n', 'ñ', 'f', 'v', '\+', 'ð', 's', 'z', '\$', '%', 'ç', 'j', 'r',
				'l', 'h']  # underlined + and $ are edh and %
	# Rule 1
	replacement1 = [['ch', 'ç'], ['sh', '$'], ['ph', 'f'], ['th', '+'], ['qu', 'kw'], ['wr', 'r'], ['wh', 'w'],
					['xh', 'x'], ['rh', 'r']]
	word = sub('who', 'ho', word)
	for pair in replacement1:
		word = sub(pair[0], pair[1], word)
	# Rule 2
	for i in vowel:
		word = sub('ex' + i, 'egz' + i, word)
	word = sub('x', 'ks', word)
	# Rule 3
	word = sub("'", '', word)
	# Rule 4
	for i in vowel:
		word = sub('gh' + i, 'g' + i, word)
	# Rule 6
	word = sub('[ao]ught', 'òt', word)
	# Rule 7
	word = sub('ough', 'ö', word)
	# Rule 5
	word = sub('agh', 'ä', word)
	word = sub('egh', 'ë', word)
	word = sub('igh', 'ï', word)
	word = sub('ogh', 'ö', word)
	word = sub('ugh', 'ü', word)
	# Rule 8
	word = sub('gh', '', word)
	# Rule 9
	word = sub('^[gkm]n', 'n', word)
	word = sub('^pt', 't', word)
	word = sub('^ps', 's', word)
	word = sub('^tm', 'm', word)
	# Rule 10
	if compile('^[^aeiou]{0,3}y$').fullmatch(word) is not None:
		word = sub('y', 'ï', word)
	# Rule 11
	word = sub('ey', 'ë', word)
	word = sub('ay', 'ä', word)
	word = sub('oy', 'öy', word)
	# Rule 12
	if compile('[^aeiou]y').fullmatch(word) is not None:
		word = sub('y', 'i', word)
	# Rule 13
	if compile('stl[aeiouy]$').fullmatch(word) is not None:
		word = sub('stl', 'sl', word)
	# Rule 14
	for i in vowel:
		word = sub('[ct]i' + i, '$' + i, word)
	# Rule 15
	for i in vowel + ['r', 'l']:
		word = sub('tu' + i, 'çu' + i, word)
	# Rule 16
	word = sub('sio', '$o', word)
	word = sub('sur', '$ur', word)
	for i in vowel:
		word = sub('ks' + i, 'k$' + i, word)
	# Rule 17
	vowel17 = vowel
	del vowel17[0]  # a
	for i in vowel17:  # first vowel
		for j in vowel:  # second vowel
			word = sub(i + 's' + j, i + 'z' + j, word)
	# Rule 18
	rule18 = ['r', 's', 'm', 'd', 't']  # Don't forget final ll!
	for i in rule18:
		word = sub('al' + i, 'òl' + i, word)
	word = sub('all$', 'òl', word)
	# Rule 19
	# ! = temp
	word = sub('^alk', '!', word)
	word = sub('alk', 'òk', word)
	word = sub('!', 'alk', word)
	# Rule 20
	frontvowel = ['e', 'i', 'ä', 'â', 'ë', 'ê', 'ï', 'î']
	for i in frontvowel:
		word = sub('c' + i, 's' + i, word)
	word = sub('c', 'k', word)
	# Rule 21
	for i in frontvowel:
		word = sub('g' + i, 'j' + i, word)
	# Rule 22
	word = sub('$jea', '!', word)
	word = sub('jea', 'ja', word)
	word = sub('!', 'jea', word)
	word = sub('$jeo', '!', word)
	word = sub('jeo', 'jo', word)
	word = sub('!', 'jeo', word)
	# Rule 23
	word = sub('$gue*', 'g', word)
	word = sub('gue*', 'gw', word)
	# Rule 24
	if compile('[^aeiou]le$').fullmatch(word) is not None:
		word = sub('le$', '@l', word)
	if compile('[^aeiou]re$').fullmatch(word) is not None:
		word = sub('re$', '@r', word)
	# Rule 27
	word = sub('ind$', 'ïnd', word)
	word = sub('oss$', 'òs', word)
	word = sub('og$', 'òg', word)
	# Rule 28
	for c in consonant:
		word = sub('of' + c, 'òf' + c, word)
	# Rule 29
	rule29 = ['t', 'd', 'n', 's', '+']
	for c in rule29:
		word = sub('wa' + c, 'wô' + c, word)
	word = sub('wa$', 'wò$', word)
	word = sub('waç', 'wòç', word)
	# Rule 25
	for c in consonant:
		for v in vowel:
			word = sub('a' + c + v, 'ä' + c + v, word)
			word = sub('e' + c + v, 'ë' + c + v, word)
			word = sub('i' + c + v, 'ï' + c + v, word)
			word = sub('o' + c + v, 'ö' + c + v, word)
			word = sub('u' + c + v, 'ü' + c + v, word)
	# Rule 26 - broken for some reason, yet 25 isn't... wtf python
	for c in consonant:
		for v in consonant:
			word = sub('a' + c + v, 'â' + c + v, word)
			word = sub('e' + c + v, 'ê' + c + v, word)
			word = sub('i' + c + v, 'î' + c + v, word)
			word = sub('o' + c + v, 'ô' + c + v, word)
			word = sub('u' + c + v, 'û' + c + v, word)
	# Rule 30
	word = sub('ign$', 'ïn', word)
	for c in consonant:
		word = sub('ign' + c, 'ïn' + c, word)
	# Rule 31
	word = sub('eign', 'ein', word)
	# Rule 32
	word = sub('ous$', '@s', word)
	for c in consonant:
		word = sub('ous' + c, '@s' + c, word)
	# Rule 33
	if compile('[aeiou].*e$').fullmatch(word) is not None:
		word = sub('e$', '', word)
	# Rule 34
	for c in consonant:
		for v in vowel:
			word = sub('[aä]' + c + v + '$', 'â' + c + v, word)
			word = sub('[eë]' + c + v + '$', 'ê' + c + v, word)
			word = sub('[iï]' + c + v + '$', 'î' + c + v, word)
			word = sub('[oö]' + c + v + '$', 'ô' + c + v, word)
			word = sub('[uü]' + c + v + '$', 'û' + c + v, word)
	# Rule 35
	if compile('^[^aeiouäëïöüâêîôûòù@]*i[aeiouäëïöüâêîôûòù@]').fullmatch(word) is not None:
		word = list(word)
		spot = word.index('i')
		word[spot] = 'ï'
		word[spot + 1] = '@'
		word = ''.join(word)
	# Rule 36
	word = sub('ow$', 'ö', word)
	word = sub('ook', 'ùk', word)
	word = sub('sei', 'së', word)
	word = sub('ie$', 'ï', word)
	word = sub('ould$', 'ùd', word)
	# Rule 37
	word = sub('eau', 'ö', word)
	word = sub('ai', 'ä', word)
	word = sub('a[uw]', 'ò', word)
	word = sub('e[ea]', 'ë', word)
	word = sub('ei', 'ä', word)
	word = sub('eo', 'ë@', word)
	word = sub('e[uw]', 'ü', word)
	word = sub('ie', 'ë', word)
	for v in vowel:
		word = sub('i' + v, 'ë@', word)
	word = sub('o[ae]', 'ö', word)
	word = sub('oo', 'u', word)
	word = sub('o[uw]', 'ôw', word)
	word = sub('oi', 'öy', word)
	word = sub('ua', 'ü@', word)
	word = sub('u[ei]', 'u', word)
	# Rule 38
	word = sub('[aeiouäëïöüâêîôûòù@]l$', '@l', word)
	# Rule 39
	word = sub('[âêîôû]n$', '@n', word)
	# The past two may be problematic...
	# Rule 40
	word = sub('[ai]ble$', '@b@l', word)
	word = sub('lion$', 'ly@n', word)
	word = sub('nion$', 'ny@n', word)
	# Rule 41
	word = sub('m[bn]$', 'm', word)
	# Rule 42
	word = sub('a$', '@', word)
	word = sub('i$', 'ë', word)
	word = sub('o$', 'ö', word)
	# Rule 43
	word = sub('ôwr|ôr|òr', 'ör', word)
	# Rule 44
	word = sub('war$', 'wör', word)
	for c in consonant:
		word = sub('war' + c, 'wör' + c, word)
	word = sub('wor', 'w@r', word)
	# Rule 45
	word = sub('[êâ]rr', 'ärr', word)
	word = sub('êri', 'äri', word)
	# Rule 46
	word = sub('âr', 'ôr', word)
	# Rule 47
	word = sub('[êîû]r', '@r', word)
	# Rule 48
	rule48 = ['r', 'l', 'y', 'w']
	for c in rule48:
		word = sub('ng' + c, 'ñg' + c, word)
	# Rule 49
	word = sub('ng$', 'ñ', word)
	for c in consonant:
		word = sub('ng' + c, 'ñ' + c, word)
	# Rule 50
	word = sub('nk', 'ñk', word)
	word = sub('ng', 'ñg', word)
	# Rule 51
	word = sub('ôñ', 'òñ', word)
	word = sub('âñ', 'äñ', word)
	# Rule 52
	rule52 = ['b', 'd', 'g']
	for c in rule52:
		word = sub(c + 's', c + 'z', word)
	# Rule 53
	word = sub('sm$', 'zm', word)
	# Rule 54
	for c in consonant:
		word = sub(c + c, c, word)
	# Rule 55
	word = sub('tç', 'ç', word)
	word = sub('dj', 'j', word)
	# Rule 56
	word = sub('s[$]', '$', word)
	# Cleanup
	word = sub(r'\\', '', word)
	word = sub('i', 'î', word)
	return uipa(word)


def mochize(word):
	word = sub('ɔ', 'ɑ', word)
	word = sub('oi', 'ɔi', word)
	word = sub('nj', 'ɲɲ', word)
	word = sub('ə', '', word)  # Fix this rule, this is an overgeneralization
	word = sub('t', 'ʔ', word)
	# alveolar tap
	word = sub('hj', 'ç', word)
	return word


def frenchipa(word):  # https://en.wikipedia.org/wiki/French_orthography
	vowel = ['a', 'e', 'i', 'o', 'u', 'y', 'à', 'â', 'æ', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'œ', 'ù', 'û']  # add more
	consonant = ['b', 'c', 'ç', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'ʁ', 's', 't', 'v', 'w', 'x',
				'z', 'ŋ', 'ʒ', 'ʃ']
	# nasalizations and special combinations
	word = sub('ée[mn]$', 'éɛ̃', word)
	word = sub('ie[mn]$', 'iɛ̃', word)
	word = sub('ye[mn]$', 'yɛ̃', word)
	for c in consonant:
		word = sub('e[mn]' + c, 'ɛ̃' + c, word)
	word = sub('ein$', 'ɛ̃', word)
	for c in consonant:
		word = sub('ein' + c, 'ɛ̃' + c, word)
	word = sub('er$', 'é', word)
	word = sub('es$', '', word)
	word = sub('eun$|u[mn]$', 'œ̃', word)
	for c in consonant:
		word = sub('eun' + c + '|u[mn]' + c, 'œ̃' + c, word)
	word = sub('gea', 'ʒa', word)
	word = sub('geo', 'ʒo', word)
	word = sub('geu', 'ʒu', word)
	word = sub('[ae][mn]$|a[eo]n$', 'ɑ̃', word)
	for c in consonant:
		word = sub('[ae][mn]' + c + '|a[eo]n' + c, 'ɑ̃' + c, word)
	word = sub('ai[mn]$', 'ɛ̃', word)
	for c in consonant:
		word = sub('ai[mn]' + c, 'ɛ̃' + c, word)
	word = sub('cqu', 'k', word)
	somevowels = ['a', 'e', 'eu', 'œ', 'ou']  # ue is a special case and is dealt with separately
	for v in somevowels:
		word = sub(v + 'il$', v + 'j', word)
	word = sub('ueil$', v + 'œ', word)
	for v in somevowels:
		word = sub(v + 'ill', v + 'j', word)
	word = sub('ueill', v + 'œ', word)
	for c in consonant:
		word = sub('i[mn]' + c, 'ɛ̃' + c, word)
	word = sub('i[mn]$', 'ɛ̃', word)
	for c in consonant:
		word = sub('oin' + c, 'wɛ̃' + c, word)
	word = sub('oin$', 'wɛ̃', word)
	for c in consonant:
		word = sub('o[mn]' + c, 'ɔ̃' + c, word)
	word = sub('o[mn]$', 'ɔ̃', word)
	word = sub('qu', 'k', word)
	specialsituation = ['^', 's', 'x']
	for pre in specialsituation:
		for v in vowel:
			if pre != '^':
				word = sub(pre + 'ti' + v, pre + 'tj' + v, word)
			else:
				word = sub(pre + 'ti' + v, 'tj' + v, word)
	for v in vowel:
		word = sub('ti' + v, 'sj' + v, word)  # or si, but much less commonly
	for c in consonant:
		word = sub('y[mn]' + c, 'ɛ̃' + c, word)
	word = sub('y[mn]$', 'ɛ̃', word)
	# unsorted
	word = sub('[bcdfgpt]s', '', word)
	# 1:1 rules
	word = sub('bb', 'b', word)
	word = sub('dd', 'd', word)
	word = sub('ff', 'f', word)
	word = sub('j', 'ʒ', word)
	word = sub('ll', 'l', word)
	word = sub('mm', 'm', word)
	word = sub('nn', 'n', word)
	word = sub('ng', 'ŋ', word)
	word = sub('r+', 'ʁ', word)
	# z rules
	word = sub('z$', '', word)
	# s rules
	for v1 in vowel:
		for v2 in vowel:
			word = sub(v1 + 's' + v2, v1 + 'z' + v2, word)
	word = sub('s$', '', word)
	word = sub('sce', 'se', word)
	word = sub('sci', 'si', word)
	word = sub('scy', 'sy', word)
	word = sub('sc', 'sk', word)
	word = sub('ss', 's', word)
	# x rules
	word = sub('^x', 'ks', word)
	word = sub('xce', 'kse', word)
	word = sub('xci', 'ksi', word)
	word = sub('xcy', 'ksy', word)
	word = sub('xc', 'ksk', word)
	vlcons = ['f', 'k', 'p', 's', 't']  # voiceless consonant list
	for c in vlcons:
		word = sub(c + 'x', c + 'ks', word)
	for c in vlcons:
		word = sub('x' + c, 'ks' + c, word)
	word = sub('x$', '', word)
	word = sub('x', 'gz', word)
	# c rules
	word = sub('cce', 'kse', word)
	word = sub('cci', 'ksi', word)
	word = sub('ccy', 'ksy', word)
	word = sub('cc', 'k', word)
	word = sub('[cs]h', 'ʃ', word)  # added sh for loans
	word = sub('nct', 'n', word)
	word = sub('ct', 'kt', word)
	word = sub('ce', 'se', word)
	word = sub('ci', 'si', word)
	word = sub('cy', 'sy', word)
	word = sub('c', 'k', word)
	# g rules
	word = sub('gn', 'ɲ', word)
	word = sub('ge', 'ʒe', word)
	word = sub('gi', 'ʒi', word)
	word = sub('gy', 'ʒy', word)
	word = sub('gge', 'gʒe', word)
	word = sub('ggi', 'gʒi', word)
	word = sub('ggy', 'gʒy', word)
	word = sub('gg', 'g', word)
	word = sub('g$', '', word)
	# continued from special combos up top
	word = sub('gue', 'ge', word)
	word = sub('gui', 'gi', word)
	word = sub('guy', 'gy', word)
	for c in consonant:
		word = sub(c + 'ill', c + 'ij', word)
	# p rules
	word = sub('p$', '', word)
	word = sub('ph', 'f', word)
	# q rules
	if compile('q[^u]').fullmatch(word) is not None:
		word = sub('q', 'k', word)
	# t rules
	word = sub('t+$', '', word)
	word = sub('t+', 't', word)
	word = sub('th', 't', word)
	# final consonant rules
	word = sub('ç', 's', word)
	word = sub('h', '', word)
	# vowels
	word = sub('â', 'ɑ', word)
	word = sub('aî|aie|ay$|è|ê|ei|eî', 'ɛ', word)
	word = sub('aye', 'ɛi', word)  # DANGER DANGER makes an i
	word = sub('ay', 'ɛj', word)
	# e rules
	for c1 in consonant:
		for c2 in consonant:
			word = sub('e' + c1 + c2, 'ɛ' + c1 + c2, word)
	# TODO: in monosyllabic words before a silent consonant
	word = sub('e$', '', word)
	word = sub('e', 'ə', word)
	# continued...
	word = sub('[eœ]u$|eû', 'ø', word)
	word = sub('[eœ]u|œ', 'œ', word)
	for v in vowel:
		word = sub('i' + v, 'j' + v, word)
	if compile('o$|oz').fullmatch(word) is None:
		word = sub('o', 'ɔ', word)
	for v in vowel:
		word = sub('^y' + v, 'j' + v, word)
	word = sub('y', 'i', word)
	for v in vowel:
		word = sub('u' + v, 'ɥ' + v, word)
	word = sub('e*au|ô|ow', 'o', word)  # i needed to move this
	word = sub('u|û|ue$', 'y', word)
	word = sub('ue', 'ɥɛ', word)
	word = sub('à', 'a', word)
	# producing monophthongs with ascii ipa
	word = sub('à', 'a', word)
	word = sub('æ|ai|é', 'e', word)
	word = sub('î|ie$', 'i', word)
	# producing diphthongs with ascii ipa
	word = sub('aï', 'ai', word)
	word = sub('ao[uû]', 'au', word)
	word = sub('eoi|oê|oie*', 'wa', word)
	word = sub('oe|oë', 'ɔe', word)
	word = sub('oï', 'oi', word)
	word = sub('oue$', 'u', word)
	for v in vowel:
		word = sub('o[uù]' + v, 'w' + v, word)
	word = sub('o[uùû]', 'u', word)
	word = sub('oy', 'waj', word)
	# last shit
	word = sub('^ï', 'j', word)
	for v1 in vowel:
		for v2 in vowel:
			word = sub(v1 + 'ï' + v2, v1 + 'j' + v2, word)
	# getting rid of symbols
	word = sub('-', '', word)
	return word


def spanishipa(
		word):  # I assume Spanish dialect. Replace thetas with s if you have seseo or s with theta if you have ceceo
	vowel = ['a', 'e', 'i', 'o', 'u', 'y']
	# x rules
	consonant = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']
	for v1 in vowel:
		for v2 in vowel:
			word = sub(v1 + 'x' + v2, v1 + 'ks' + v2, word)
	word = sub('^x', 'ks', word)
	for c in consonant:
		word = sub('x' + c, 'ks' + c, word)
	# b/v rules
	word = sub('v', 'b', word)
	word = sub('(?<!m)b', 'β', word)
	word = sub('^β', 'b', word)
	# c rules
	word = sub('ce', 'θe', word)
	word = sub('ci', 'θi', word)
	word = sub('ch|t[xz]', 'tʃ', word)
	word = sub('c', 'k', word)
	# d rules
	word = sub('(?<![ln])d', 'ð', word)
	word = sub('^ð', 'd', word)
	# g rules
	word = sub('ge', 'xe', word)
	word = sub('gi', 'xi', word)
	word = sub('(?<!n)g', 'ɣ', word)
	word = sub('$gua', 'gwa', word)
	word = sub('ngua', 'ŋgwa', word)
	word = sub('ŋguo', 'ŋgwo', word)
	word = sub('gua', 'ɣwa', word)
	word = sub('guo', 'ɣwo', word)
	word = sub('$gue', 'ge', word)
	word = sub('$gui', 'gi', word)
	word = sub('ngue', 'ŋge', word)
	word = sub('ngui', 'ŋgi', word)
	word = sub('gue', 'ɣe', word)
	word = sub('gui', 'ɣi', word)
	word = sub('ɣue', 'ɣe', word)  # need these two due to an odd bug
	word = sub('ɣui', 'ɣi', word)  #
	word = sub('$güe', 'gwe', word)
	word = sub('$güi', 'gwi', word)
	word = sub('ngüe', 'ŋgwe', word)
	word = sub('ngüi', 'ŋgwi', word)
	word = sub('güe', 'ɣwe', word)
	word = sub('güi', 'ɣwi', word)
	# j rules
	word = sub('j', 'x', word)
	# i/u before vowel
	for v in vowel:
		word = sub('i' + v, 'j' + v, word)
		word = sub('u' + v, 'w' + v, word)
	# h rules
	word = sub('sh', 'ʃ', word)
	for v in vowel:
		word = sub('hi' + v, 'j' + v, word)
		word = sub('hu' + v, 'w' + v, word)
	word = sub('h', '', word)
	# ll rules
	word = sub('ll', 'ʎ', word)
	# m/n rules
	word = sub('m$', 'n', word)
	mntrigger = ['m', 'b', 'f', 'β']
	yntrigger = ['j', 'ʎ', 'y']
	gntrigger = ['x', 'ɣ', 'k', 'g']
	for labial in mntrigger:
		word = sub('n' + labial, 'm' + labial, word)
	for palatal in yntrigger:
		word = sub('n' + palatal, 'ɲ' + palatal, word)
	for velar in gntrigger:
		word = sub('n' + velar, 'ŋ' + velar, word)
	word = sub('ñ', 'ɲ', word)
	# q rules
	word = sub('qu', 'k', word)
	# r rules
	word = sub('r', 'ɾ', word)
	word = sub('ɾɾ', 'r', word)
	word = sub('$ɾ', 'r', word)
	word = sub('lɾ', 'r', word)
	word = sub('nɾ', 'r', word)
	word = sub('sɾ', 'r', word)
	# tl rules
	word = sub('tl', 'tɬ', word)
	# y rules
	for v in vowel:
		word = sub('y' + v, 'j' + v, word)
	for v in vowel:
		word = sub(v + 'y', v + 'j', word)
	# z rules
	word = sub('z', 'θ', word)
	# s rules - MAKE THESE LAST
	voiced = ['b', 'β', 'd', 'ð', 'g', 'ɣ', 'j', 'l', 'ʎ', 'm', 'n', 'r', 'ɾ', 'w']
	for c in voiced:
		word = sub('s' + c, 'z' + c, word)
	# y->i
	word = sub('[yíý]', 'i', word)
	word = sub('á', 'a', word)
	word = sub('é', 'e', word)
	word = sub('ó', 'o', word)
	word = sub('ú', 'u', word)
	return word


def germanipa(word):
	vowel = ['a', 'ä', 'e', 'i', 'o', 'ö', 'u', 'ü', 'y']
	# these rules need to be done early
	word = sub('tsch|t*zsch', 'tʃ', word)
	word = sub('sch', 'ʃ', word)
	# b rules - as close as i can get sadly
	word = sub('b$', 'p', word)
	# d rules
	word = sub('dsch', 'dʒ', word)
	word = sub('dt', 't', word)
	word = sub('d$', 't', word)
	# c rules
	word = sub('c(?=[äei])', 'ts', word)
	word = sub('chs', 'ks', word)
	word = sub('ck', 'k', word)
	word = sub('(?<=[aou])ch', 'x', word)
	word = sub('ch', 'ç', word)
	word = sub('c', 'k', word)
	# g rules
	word = sub('ig$', 'iç', word)
	word = sub('g$', 'k', word)
	# n rules
	word = sub('ng', 'ŋ', word)
	word = sub('nk', 'ŋk', word)
	# p rules
	word = sub('ph', 'f', word)
	# qu rules
	word = sub('qu', 'kv', word)
	# r rules
	for v in vowel:
		word = sub('r' + v, 'ʁ' + v, word)
	word = sub('r', 'ɐ', word)
	# s rules
	for v in vowel:
		word = sub('s' + v, 'z' + v, word)
	word = sub('$s(?=[pt])', 'ʃ', word)
	word = sub('schen$', 'sçen', word)
	word = sub('ss|ß', 's', word)
	# t rules
	word = sub('th', 't', word)
	word = sub('ti(?=on|är|al|ell)', 'tsɪ̯', word)
	word = sub('t*z', 'ts', word)
	# v rules
	word = sub('v', 'f', word)
	# w rules
	word = sub('w', 'v', word)
	# x rules
	word = sub('x', 'ks', word)
	# vowel rules - mostly hopeful guesses, the article wasn't really helpful
	# irregularers
	word = sub('eɐ$', 'ɐ', word)
	# regularers
	word = sub('[ae][iy]', 'aɪ', word)
	word = sub('[äe]u', 'ɔʏ̯', word)
	word = sub('au', 'aʊ', word)
	word = sub('(?<!ae)i(?![eh])', 'ɪ', word)
	word = sub('(?<!ae)ie*h*', 'iː', word)
	word = sub('ü(?!h)|y', 'ʏ', word)
	word = sub('üh', 'yː', word)
	word = sub('öh', 'øː', word)
	word = sub('ö', 'ø', word)
	word = sub('äh', 'ɛː', word)
	word = sub('ä', 'ɛ', word)
	word = sub('e[he]', 'eː', word)
	word = sub('e(?![iuyː])', 'ə', word)
	word = sub('(?<![aäe])u(?!h)', 'ʊ', word)
	word = sub('uh', 'uː', word)
	word = sub('o[oh]', 'oː', word)
	word = sub('o', 'ɔ', word)
	word = sub('a[ah]', 'aː', word)
	word = sub('a(?![iuyʊɪ])', 'a', word)
	return word


def hungarianipa(word):
	# foreign letters treated as something else
	word = sub('qu', 'kv', word)
	word = sub('w', 'v', word)
	word = sub('x', 'ks', word)
	word = sub('y', 'i', word)
	# historic spellings
	word = sub('ch', 'cs', word)
	word = sub('[ct]z', 'c', word)
	word = sub('gh', 'g', word)
	word = sub('th', 't', word)
	word = sub('a[aá]', 'á', word)
	word = sub('eé', 'é', word)
	word = sub('e[öw]', 'ö', word)
	word = sub('oó', 'ó', word)
	word = sub('ay', 'ai', word)
	word = sub('dy', 'di', word)
	word = sub('ey', 'ei', word)
	# consonants which won't screw up later ones
	# allophones of /j/
	word = sub('(?<=[cfhkpqstx])j', 'ç', word)
	word = sub('(?<=[bdgjlmnrvwyz])j', 'ʝ', word)
	# gy
	word = sub('gy', 'ɟ', word)
	# allophones of /h/
	word = sub('(?<![aáeéiíoóöőuúüű])h(?![aáeéiíoóöőuúüű])', 'ɦ', word)
	word = sub('h$', '', word)
	# allophones of /n/
	word = sub('n(?:[kg])', 'ŋ', word)
	# ny
	word = sub('ny', 'ɲ', word)
	# s
	word = sub('cs', 'tʃ', word)
	word = sub('sz', '!', word)
	word = sub('s', 'ʃ', word)
	word = sub('!', 's', word)
	# zs
	word = sub('zs', 'ʒ', word)
	# consonantws which may screw up later ones
	word = sub('c', 'ts', word)
	word = sub('ly', 'j', word)
	word = sub('ty', 'c', word)
	# vowels which won't screw up later vowels
	word = sub('a', 'ɒ', word)
	word = sub('e', 'ɛ', word)
	word = sub('ö', 'ø', word)
	word = sub('ő', 'øː', word)
	# vowels which'd screw up later ones
	word = sub('á', 'aː', word)
	word = sub('ë', 'e', word)
	word = sub('é', 'eː', word)
	word = sub('í', 'iː', word)
	word = sub('ó', 'oː', word)
	word = sub('ú', 'uː', word)
	word = sub('ü', 'y', word)
	word = sub('ű', 'yː', word)
	return word


def italianipa(word):
	# foreign letters
	word = sub('y', 'i', word)
	# merge like vowels
	word = sub('à', 'a', word)
	word = sub('é', 'e', word)
	word = sub('[ìíî]', 'i', word)
	word = sub('ó', 'o', word)
	word = sub('[ùú]', 'u', word)
	word = sub('è', 'ɛ', word)
	word = sub('ò', 'ɔ', word)
	# c/g rules
	word = sub('sc(?=[eiɛ])', 'ʃ', word)
	word = sub('sci(?=[aoɔu])', 'ʃ', word)
	word = sub('c(?=[eiɛ])', 'tʃ', word)
	word = sub('g(?=[eiɛ])', 'dʒ', word)
	word = sub('ci(?=[aoɔu])', 'tʃ', word)
	word = sub('gi(?=[aoɔu])', 'dʒ', word)
	word = sub('ch', 'k', word)
	word = sub('gh', 'g', word)
	word = sub('c', 'k', word)
	# h
	word = sub('h', '', word)
	# i/u rules
	word = sub('i(?=[aeɛɔou])', 'j', word)
	word = sub('u(?=[aeioɛɔ])', 'w', word)
	# s/z rules
	word = sub('z(?=.[fkpqstxʃ])', '!', word)
	word = sub('z(?=i[aeouɛɔ])', '!', word)
	word = sub('(?<=l)z', '!', word)
	word = sub('ezza$', 'et!a', word)
	word = sub('ozza$', 'ot!a', word)
	word = sub('uzzo$', 'ut!o', word)
	word = sub('azzare$', 'at!are', word)
	word = sub('anza$', 'an!a', word)
	word = sub('enza$', 'en!a', word)
	word = sub('onzolo$', 'on!olo', word)
	word = sub('s(?=bdgjmnrvw)', 'z', word)
	word = sub('s(?=bdgjmnrvw)', 'z', word)
	word = sub('(?<=aeioujwɛɔ)z(?=aeioujwɛɔ)', 'zz', word)  # doubled b/w vowels
	word = sub('zz', 'dd%', word)  # otherwise, voiced
	word = sub('z', 'd%', word)  # otherwise, voiced
	word = sub('!', 'ts', word)
	word = sub('%', 'dz', word)
	# foreign letters
	word = sub('w', 'v', word)
	word = sub('x', 'ks', word)
	return word


def polishipa(word):
	# b rules
	word = sub('b(?=cćfhkpsśtx)', 'p', word)
	# f rules
	word = sub('f(?=bdgqrvwzźż)', 'v', word)
	# g rules
	word = sub('g(?=cćfhkpsśtx)', 'k', word)
	# (c)h rules
	word = sub('c?h(?=bdgqrvwzźż)', 'ɣ', word)
	word = sub('c?h', 'x', word)
	# k rules
	word = sub('k(?=bdgqrvwzźż)', 'g', word)
	# p rules
	word = sub('p(?=bdgqrvwzźż)', 'b', word)
	# t rules
	word = sub('t(?=bdgqrvwzźż)', 'd', word)
	# w rules
	word = sub('w(?=cćfhkpsśtx)', 'f', word)
	word = sub('w', 'v', word)
	# c rules
	word = sub('cz(?=bdgqrvwzźż)', 'd͡ʐ', word)
	word = sub('cz', 't͡ʂ', word)
	word = sub('c(?=bdgqrvwzźż)', 'd͡z', word)
	word = sub('c', 't͡s', word)
	word = sub('ć(?=bdgqrvwzźż)', 'd͡ʑ', word)
	word = sub('ć', 't͡ɕ', word)
	# d rules
	word = sub('dz(?=cćfhkpsśtx)', 'ts', word)
	word = sub('dz', 'd͡z', word)
	word = sub('dź(?=cćfhkpsśtx)', 't͡ɕ', word)
	word = sub('dź', 'd͡ʑ', word)
	word = sub('dz(?=cćfhkpsśtx)', 't͡ʂ', word)
	word = sub('dz', 'd͡ʐ', word)
	word = sub('d(?=cćfhkpsśtx)', 't', word)
	# s rules
	word = sub('sz(?=bdgqrvwzźż)', 'ʐ', word)
	word = sub('sz', 'ʂ', word)
	word = sub('s(?=bdgqrvwzźż)', 'z', word)
	word = sub('ś(?=bdgqrvwzźż)', 'ʑ', word)
	word = sub('ś', 'ɕ', word)
	# z rules
	word = sub('z(?=cćfhkpsśtx)', 's', word)
	word = sub('ź(?=bdgqrvwzźż)', 'ɕ', word)
	word = sub('ź', 'ʑ', word)
	word = sub('ż(?=bdgqrvwzźż)|rz(?=bdgqrvwzźż)', 'ʂ', word)
	word = sub('ż|rz', 'ʐ', word)
	# 1:1 rules
	word = sub('ł', 'w', word)
	word = sub('ń', 'ɲ', word)
	# Palatalization
	word = sub('ci(?!aąeęioóuy)', 't͡ɕi', word)
	word = sub('ci(?=aąeęioóuy)', 't͡ɕ', word)
	word = sub('dzi(?!aąeęioóuy)', 'd͡ʑi', word)
	word = sub('dzi(?=aąeęioóuy)', 'd͡ʑ', word)
	word = sub('si(?!aąeęioóuy)', 'ɕi', word)
	word = sub('si(?=aąeęioóuy)', 'ɕ', word)
	word = sub('ni(?!aąeęioóuy)', 'ɲi', word)
	word = sub('ni(?=aąeęioóuy)', 'ɲ', word)
	# final vowel shit
	word = sub('(?<=aąeęioóuy)u', 'w', word)
	word = sub('ó', 'u', word)
	word = sub('ą', 'ɔ̃', word)
	word = sub('e', 'ɛ', word)
	word = sub('ę', 'ɛ̃', word)
	word = sub('o', 'ɔ', word)
	word = sub('ó', 'u', word)
	word = sub('y', 'ɨ', word)
	return word
