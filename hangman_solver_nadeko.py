from json import load
from re import match, search
from typing import Set

alphabet = 'abcdefghijklmnopqrstuvwxyz'

hangman_data = load(open('hangman.json', 'r'))
all_words = [[i['Word'] for i in word] for word in [category for category in hangman_data.values()]]
all_words = {j.lower() for i in all_words for j in i} # type: Set[str]


def solve_all(pattern: str, ignore_set: str = '') -> Set[str]:
	re_alphabet = ''.join([letter for letter in alphabet if letter not in set(pattern)])
	re_pattern = pattern.replace('_', '['+re_alphabet+']')
	possible_words = set()
	for word in all_words:
		if ignore_set and search('['+ignore_set+']', word):
			continue
		if len(word) == len(pattern) and match(re_pattern, word):
			possible_words.add(word)
	print(possible_words)
	return possible_words


def recommend_letter(word_set: Set[str], ignore_set: str = '') -> str:
	"""Given a set of possible answers, return a guess to whittle down as many answers as possible."""
	letter_histogram = {letter: abs(len(word_set)//2-sum(letter in word for word in word_set))
						for letter in alphabet if letter not in ignore_set}
	return sorted(letter_histogram.items(), key=lambda x: x[1])[0][0]


while 1:
	status = input('>>> ')
	try:
		if ';' in status:
			status, ignore = status.split(';')
		else:
			ignore = ''
		solution_set = solve_all(status, ignore)
		print(list(solution_set)[0] if len(solution_set) == 1 else recommend_letter(solution_set, ignore))
	except Exception as e:
		print(e)
