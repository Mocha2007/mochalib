from re import sub
from random import choice


def clean_text(filename: str) -> str:
	text = open(filename, 'r', encoding='utf8').read()
	return sub("[^'a-z\.]+", ' ', text.lower().replace('.', ' . '))


def corpus_from(text: str, minimum_attestation: int=0) -> set:
	if minimum_attestation == 0:
		return set(text.split())
	print(len(text))
	return {word for word, count in freq(text).items() if 3 <= count}


def export_citations(text: str, minimum_attestation: int=3):
	text = text.replace('. ', '')
	template = '<dt>{}</dt><dd><ul>'+'<li>{}</li>'*minimum_attestation+'</ul></dd>'
	corpus = sorted(list(corpus_from(text, minimum_attestation)))
	document = ['<dl>']
	for word in corpus:
		word_ = ' {} '.format(word)
		occurences = []
		for i in range(minimum_attestation):
			if i == 0:
				index = text.find(word_)
			else:
				index = text.find(word_, occurences[-1]+1)
			occurences.append(index)
		args = [word]
		for index in occurences:
			args.append('...{}...'.format(text[index-20:index+len(word)+20].replace(word_, ' <b>{}</b> '.format(word))))
		document.append(template.format(*args))
	with open('books/{}.html'.format(abs(hash(text))), 'w+') as file:
		file.write('\n'.join(document))


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
	return make_markov(clean_text(filename))


def pretty_freq(text: str):
	text = text.replace('. ', '')
	length = len(text.split())
	lemmata = len(corpus_from(text, 3))
	lengths = word_lengths(text)
	header = 'TOTAL\t{} = {} min reading\n\t{} unique words attested at least three times, of length {}-{}\n'.format(length, int(.22*length/60), lemmata, min(lengths), max(lengths))
	top_ten_words = sorted(freq(text).items(), key=lambda x: x[1], reverse=True)[:25]
	return header+'\n'.join('{}\t{}\t{}%'.format(words, count, round(100*count/length, 3)) for words, count in top_ten_words)


def random_word(freq: dict) -> str:
	wordlist = []
	for word, count in freq.items():
		wordlist += [word]*count
	return choice(wordlist)


def speak(markov_chain: dict, max_len: int=50) -> str:
	words = [choice(list(markov_chain['.']))]
	for _ in range(max_len):
		words.append(random_word(markov_chain[words[-1]]))
		if words[-1] == '.':
			break
	return ' '.join(words).replace(' .', '')


def word_lengths(text: str) -> set:
	return {len(word) for word in corpus_from(text)}


def test(filename: str='star wars'):
	filename = 'books/' + filename + '.txt'
	# from time import time
	# start_time = time()
	kjb = markov_from(filename)
	# print(kjb['sin'])
	print(speak(kjb))
	kjb_text = clean_text(filename)
	print(pretty_freq(kjb_text))
	export_citations(kjb_text)
	# print(time() - start_time)
