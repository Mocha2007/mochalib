from re import sub
from random import choice


def clean_text(text: str) -> str:
	return sub("[^'a-z]+", ' ', text.lower())


def corpus_from(text: str) -> set:
	return set(text)


def freq(text: str) -> dict:
	"""Fastest of three methods tested:
	king james bible - 683 ms
	"""
	output = {}
	for word in text.split():
		if word not in output:
			output[word] = 1
		else:
			output[word] += 1
	return output


def make_markov(text: str) -> dict:
	"""
	king james bible - 1053 ms
	"""
	output = {}
	words = text.split()
	for i, word in enumerate(words):
		if i == len(words) - 1:
			break
		next_word = words[i+1]
		if word not in output:
			output[word] = {}
		if next_word in output[word]:
			output[word][next_word] += 1
		else:
			output[word][next_word] = 1
	return output


def markov_from(filename: str) -> dict:
	return make_markov(clean_text(open(filename, 'r').read()))


def random_word(freq: dict) -> str:
	wordlist = []
	for word, count in freq.items():
		wordlist += [word]*count
	return choice(wordlist)


def speak(markov_chain: dict, length: int=16) -> str:
	words = [choice(list(markov_chain))]
	for _ in range(length):
		words.append(random_word(markov_chain[words[-1]]))
	return ' '.join(words)


def test():
	# from time import time
	# start_time = time()
	kjb = markov_from('kjb.txt')
	# print(kjb['sin'])
	print(speak(kjb))
	# print(time() - start_time)
