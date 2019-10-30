from re import sub
from typing import Dict, Set

corpus_filename = 'books/star wars.txt'
corpus_file = open(corpus_filename, 'r').read()


def clean(text: str) -> str:
	text = clean_whitespace(text)
	text = clean_contractions(text)
	text = text.lower()
	text = clean_punctuation(text)
	return text


def clean_whitespace(text: str) -> str:
	"""Multispace -> singlespace"""
	return sub('[ \n\t]+', ' ', text)


def clean_contractions(text: str) -> str:
	"""Turn contractions into separate words"""
	return text.replace('\'', ' \'')


def clean_punctuation(text: str) -> str:
	"""chars which aren't alphabetic, space, period, or \' get removed. ?! -> ."""
	text = sub('[.!?]+', '.', text)
	return sub('[^a-z.\' ]', '', text)


def get_sentences(text: str) -> Set[str]:
	return set(text.split('.'))


def main(text: str) -> Dict[str, Dict[str, int]]:
	mapping = {}
	text = clean(text)
	for sentence in get_sentences(text):
		words = set(sentence.split(' '))
		if '' in words:
			words.remove('')
		for word_1 in words:
			if word_1 in mapping:
				for word_2 in words:
					if word_1 == word_2:
						continue
					if word_2 in mapping[word_1]:
						mapping[word_1][word_2] += 1
					else:
						mapping[word_1][word_2] = 1
			else:
				mapping[word_1] = {word_2: 1 for word_2 in words if word_1 != word_2}
	return mapping


def pretty_associations(raw_corp: str, target: str, limit: int = 20):
	"""-1 for no limit"""
	clean_corpus = clean(raw_corp)
	word_dict = main(raw_corp)[target] # type: Dict[str, int]
	counts = {word: word_dict[word] / clean_corpus.count(word)**.7 for word in word_dict} # optimal exp somewhere between .5 and .8?
	sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
	for i, (word, r) in enumerate(sorted_words):
		if 0 <= limit and limit < i:
			break
		print(word, '\t', r)


def counts_to_wilson(word_dict_count: int, clean_corpus_count: int) -> float:
	up = word_dict_count
	n = clean_corpus_count
	down = n - up
	z = 1.96
	p1 = (up+z**2/2)/(n+z**2)
	p2 = z/(n+z**2)*(up*down/n+z**2/4)**.5
	return p1 - p2


def pretty_associations_wilson(raw_corp: str, target: str, limit: int = 20):
	"""-1 for no limit"""
	clean_corpus = clean(raw_corp)
	word_dict = main(raw_corp)[target] # type: Dict[str, int]
	counts = {word: counts_to_wilson(word_dict[word], clean_corpus.count(word)) for word in word_dict}
	sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
	for i, (word, r) in enumerate(sorted_words):
		if 0 <= limit and limit < i:
			break
		print(word, '\t', r)


# output = main(corpus_file)
# pretty_associations(corpus_file, 'luke')
# print('\n')
pretty_associations_wilson(corpus_file, 'luke')
