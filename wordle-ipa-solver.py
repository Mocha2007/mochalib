"""
A solver for http://manishearth.github.io/ipadle
by Mocha2007 (Luna)
"""
from random import choice
from time import time

dictionary_file = "ignore/cmudict-0.7b" # http://www.speech.cs.cmu.edu/cgi-bin/cmudict

# http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols
replacements = [
	('AH0', 'ə'), # this one seems to mark schwa? see "about"
	# stress markers
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
	'gray': 0,
	'grey': 0,
	'gry': 0,
	'gy': 0,
	'green': 2,
	'grn': 2,
	'gn': 2,
	'yellow': 1,
	'yel': 1,
	'yw': 1,
	'y': 1,
}

quit_keywords = {'quit', 'exit'}

def test_word(test: str, solution: str):
	if (test, solution) not in test_word_cache:
		test_word_cache[(test, solution)] = sum(x*3**i for i, x in enumerate(2 if a == b else 1 if a in solution else 0 for a, b in zip(test, solution)))
	return test_word_cache[(test, solution)]
# uncached: OFC for 2-letter words takes 32.878 s
# cached: takes 6.420 s instead! :D
test_word_cache = dict()

def optimize_first_choice(valid_length_range = range(3, 12)):
	# we want to MINIMIZE the average of how many results are possible after filtering
	word_score = dict()
	print("Preprocessing dictionary...")
	resized_dicts = {i: {word for word in dictionary.values() if len(word) == i} for i in valid_length_range}
	words_to_test = {word for word in dictionary.values() if len(word) in valid_length_range}
	print("Pretesting", len(words_to_test), "words...")
	test_results = {a: {b: test_word(a, b) for b in words_to_test if len(a) == len(b)} for a in words_to_test}
	print("Searching sample space...")
	for test in words_to_test:
		# print("Testing <", test, "> (", i+1, "/", len(words_to_test) ,")...")
		word_length = len(test)
		score = 0
		solution_space = resized_dicts[word_length]
		for solution in solution_space:
			# if j % (len(solution_space) // 10) == 0:
			# 	print(round(j/len(solution_space) * 100), "% of", len(solution_space) ,"tests done...")
			# print('... against', solution, len(resized_dicts[word_length]), 'words of this size')
			result = test_results[test][solution]
			for word in solution_space:
				if test_results[test][word] == result:
					score += 1
		score /= len(solution_space)
		print('Word <', test, '> has score', score, '(lower is better)')
		word_score[test] = score
	# now get the min for each length!
	return {i: min(((w, s) for w, s in word_score.items() if len(w) == i), key=lambda x: x[1]) for i in valid_length_range}

def optimize_first_choice2(valid_length_range = range(3, 12), skip = 22):
	# we want to MINIMIZE the average of how many results are possible after filtering
	word_score = dict()
	print("Preprocessing dictionary...")
	resized_dicts = {i: [word for word in dictionary.values() if len(word) == i][::skip] for i in valid_length_range}
	words_to_test = [word for word in dictionary.values() if len(word) in valid_length_range][::skip]
	print("Pretesting", len(words_to_test), "words...")
	test_results = {a: {b: test_word(a, b) for b in words_to_test if len(a) == len(b)} for a in words_to_test}
	print("Searching sample space...")
	for test in words_to_test:
		# print("Testing <", test, "> (", i+1, "/", len(words_to_test) ,")...")
		word_length = len(test)
		score = 0
		solution_space = resized_dicts[word_length]
		for solution in solution_space:
			# if j % (len(solution_space) // 10) == 0:
			# 	print(round(j/len(solution_space) * 100), "% of", len(solution_space) ,"tests done...")
			# print('... against', solution, len(resized_dicts[word_length]), 'words of this size')
			result = test_results[test][solution]
			for word in solution_space:
				if test_results[test][word] == result:
					score += 1
		score /= len(solution_space)
		print('Word <', test, '> has score', score, '(lower is better)')
		word_score[test] = score
	# now get the min for each length!
	return {i: min(((w, s) for w, s in word_score.items() if len(w) == i), key=lambda x: x[1]) for i in valid_length_range}

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

def main():
	while 1:
		word_length = int(input("Please enter word length: "))
		# prefilter dict
		filtered = [val for val in dictionary.values() if len(val) == word_length]
		# ok now the sub-loop
		while 1:
			print(len(filtered), "valid words.")
			if len(filtered) <= 10:
				print("All valid words:", *filtered)
			else:
				print("5 random valid words:", *(choice(filtered) for _ in range(5)))
			test = input("Please enter your next attempt (eg. 'klɪnt'):")
			# process exit request
			if test in quit_keywords:
				break
			result = input("Please enter the result for this word (eg. 'green grey yellow grey grey' or 'gn gy yw gy gy')")
			result = result.split(' ')
			# replace color name aliases
			for i, w in enumerate(result):
				if w in color_aliases:
					result[i] = color_aliases[w]
			result = sum(x*3**i for i, x in enumerate(result))
			# print("the program understood this as <", result, ">")
			# now filter dict by valid words
			filtered = [word for word in filtered if test_word(test, word) == result]
		# retry?
		if input("Would you like to solve another puzzle? (Type anything to quit, or press enter to continue...)"):
			break
	print('Bye-bye!~ :3')

main()
# print(optimize_first_choice(range(3, 4)))
# print(optimize_first_choice2(range(5, 6)))