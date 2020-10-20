from re import sub
from random import choice


alphabet = ''.join(chr(i) for i in range(97, 123))
numerals = ''.join(str(i) for i in range(10))


def char_count(text: str) -> dict:
	return {char: text.count(char) for char in set(text)}


def clean_text(text_name: str) -> str:
	return sub("[^'a-z.]+", ' ', load(text_name).lower().replace('.', ' . '))


def corpus_from(text: str, minimum_attestation: int = 0) -> set:
	if minimum_attestation == 0:
		return set(text.split())
	print(len(text))
	return {word for word, count in freq(text).items() if 3 <= count}


def export_citations(text: str, minimum_attestation: int = 3):
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


def letter_count(text: str) -> dict:
	text = text.lower()
	return {char: text.count(char) for char in set(text) if char in alphabet}


def load(text_name: str) -> str:
	return open('books/{}.txt'.format(text_name), 'r', encoding='utf8').read()


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


def pretty_letter_count(text: str) -> str:
	count = sorted(letter_count(text).items(), key=lambda x: x[1], reverse=True)
	string = ['\t%']
	total = sum(i[1] for i in count)
	maximum = max(i[1] for i in count)
	for char, hz in count:
		string.append('{}\t{}%\t{}'.format(char, round(100*hz/total, 3), '█'*round(64*hz/maximum)))
	return '\n'.join(string)


def pretty_freq(text: str, n: int = 25) -> str:
	text = text.replace('. ', '')
	length = len(text.split())
	lemmata = len(corpus_from(text, 3))
	lengths = word_lengths(text)
	header = 'TOTAL\t{} = {} min reading\n\t{} unique words attested at least three times, of length {}-{}\n'.format(
		length, int(.22*length/60), lemmata, min(lengths), max(lengths))
	top_words = sorted(freq(text).items(), key=lambda x: x[1], reverse=True)[:n]
	return header+'\n'.join('{}\t{}\t{}%'.format(words, count, round(100*count/length, 3)) for words, count in top_words)


def random_word(frequency: dict) -> str:
	wordlist = []
	for word, count in frequency.items():
		wordlist += [word]*count
	return choice(wordlist)


def speak(markov_chain: dict, max_len: int = 50) -> str:
	words = [choice(list(markov_chain['.']))]
	for _ in range(max_len):
		words.append(random_word(markov_chain[words[-1]]))
		if words[-1] == '.':
			break
	return ' '.join(words).replace(' .', '')


def word_lengths(text: str) -> set:
	return {len(word) for word in corpus_from(text)}


def test(filename: str = 'star wars'):
	# from time import time
	# start_time = time()
	kjb = markov_from(filename)
	# print(kjb['sin'])
	print(speak(kjb))
	kjb_text = clean_text(filename)
	print(pretty_freq(kjb_text, 100))
	export_citations(kjb_text)
	# print(time() - start_time)
