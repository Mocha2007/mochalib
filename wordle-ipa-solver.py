"""
A solver for http://manishearth.github.io/ipadle
by Mocha2007 (Luna)
"""

from random import choice

dictionary_file = "ignore/cmudict-0.7b" # http://www.speech.cs.cmu.edu/cgi-bin/cmudict

# http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols
replacements = [
	('0', ''),
	('1', ''),
	('2', ''),
	('AA', 'ɑ'),
	('AE', 'æ'),
	('AH', 'ʌ'),
	('AO', 'ɔ'),
	('AW', 'aʊ'),
	('AY', 'aɪ'),
	('CH', 'tʃ'),
	('DH', 'ð'),
	('EH', 'ɛ'),
	('ER', 'ɚ'),
	('EY', 'eɪ'),
	('HH', 'h'),
	('IH', 'ɪ'),
	('IY', 'i'),
	('JH', 'dʒ'),
	('NG', 'ŋ'),
	('OW', 'oʊ'),
	('OY', 'ɔɪ'),
	('R', 'ɹ'),
	('SH', 'ʃ'),
	('TH', 'θ'),
	('UH', 'ʊ'),
	('UW', 'u'),
	('Y', 'j'),
	('ZH', 'ʒ'),
	(' ', ''),
]

color_aliases = {
	'gray': 'grey',
	'gry': 'grey',
	'gy': 'grey',
	'grn': 'green',
	'gn': 'green',
	'yel': 'yellow',
	'yw': 'yellow',
	'y': 'yellow',
}

quit_keywords = {'quit', 'exit'}

def test_word(test: str, solution: str):
	return tuple('green' if x == y else 'yellow' if x in solution else 'grey' for x, y in zip(test, solution))

# preload dict

with open(dictionary_file) as file:
	raw_data = file.read()

print('Dictionary file loaded, processing...')

# process their weird syntax

dictionary = dict()

for line in raw_data.splitlines():
	try:
		[key, val] = line.split('  ')
	except ValueError:
		continue
	# ok now do some simple text replacements
	for replacement in replacements:
		val = val.replace(*replacement)
	# make lowercase
	val = val.lower()
	# push
	dictionary[key] = val

print('Dictionary processed!', len(dictionary), 'entries.')

# now theoretically the dictionary should be fully processed.

# MAIN LOOP

while 1:
	word_length = int(input("Please enter word length: "))
	# prefilter dict
	filtered = [val for val in dictionary.values() if len(val) == word_length]
	# ok now the sub-loop
	while 1:
		print(len(filtered), "valid words.")
		print("Random valid words:", *(choice(filtered) for _ in range(5)))
		test = input("Please enter your next attempt (eg. 'dʒinz'):")
		# process exit request
		if test in quit_keywords:
			break
		result = input("Please enter the result for this word (eg. 'green grey yellow grey grey' or 'gn gy yw gy gy')")
		result = result.split(' ')
		# replace color name aliases
		for i, w in enumerate(result):
			if w in color_aliases:
				result[i] = color_aliases[w]
		result = ' '.join(result)
		print("the program understood this as <", result, ">")
		# now filter dict by valid words
		filtered = [word for word in filtered if ' '.join(test_word(test, word)) == result]
	# retry?
	if input("Would you like to solve another puzzle? (Type anything to quit, or press enter to continue...)"):
		break

print('Bye-bye!~ :3')
