from math import e, log, pi
from random import choice, random
from typing import Dict, Set, Tuple


def log_normal(x: float, sigma: float, mu: float) -> float:
	return e**-((log(x)-mu)**2/(2*sigma**2)) / (x*sigma*(2*pi)**.5)


class Phoneme:
	def __init__(self, name: str = '', **features: Dict[str, bool]):
		self.name = name
		self.features = features

	def __repr__(self) -> str:
		return self.name


class Morpheme:
	def __init__(self, phonemes: Tuple[Phoneme] = tuple()):
		self.phonemes = phonemes

	def __repr__(self) -> str:
		return ''.join([str(i) for i in self.phonemes])

	def __getitem__(self, item: int) -> Phoneme:
		return self.phonemes[item]

	def __len__(self) -> int:
		return len(self.phonemes)

	def append(self, phoneme: Phoneme):
		self.phonemes = list(self.phonemes)
		self.phonemes.append(phoneme)
		self.phonemes = tuple(self.phonemes)


class Environment:
	def __init__(self, before: Set[Phoneme], after: Set[Phoneme]):
		self.before = before
		self.after = after

	def obeys(self, before: Phoneme, after: Phoneme) -> bool:
		b = before is None or before in self.before
		a = after is None or after in self.after
		return b and a


class Phonotactics:
	def __init__(self, syllable_structure: Tuple[Tuple[Tuple[Set[Phoneme], bool]]], # bool is for "is this mandatory?"
				constraints: Tuple[Tuple[Set[Phoneme], Environment, bool]], syllable_count: (int, int) = (1, 4),
				distribution: (float, float) = (0.48, 0.95)):
		# distribution defaults to English-like; use (0.38, 1.13) for a French-like distribution
		self.syllable_strcuture = syllable_structure
		self.constraints = constraints
		assert 0 < syllable_count[0] < syllable_count[1]
		self.syllable_count = syllable_count
		self.distribution = distribution

	def generate_morpheme(self) -> Morpheme:
		morph = Morpheme()
		for _ in range(self.generate_syllable_count()):
			syllable_template = choice(self.syllable_strcuture)
			for phoneme_set, b in syllable_template:
				if b or random() < .5:
					morph.append(choice(tuple(phoneme_set)))
		return morph if self.obeys(morph) else self.generate_morpheme()

	def generate_syllable_count(self) -> int:
		"""Randomly generate syllable count"""
		items = [[i]*int(100*log_normal(i, *self.distribution)) for i in range(*self.syllable_count)]
		items = [i for j in items for i in j]
		return choice(items)

	def obeys(self, morph: Morpheme, check_syllables: bool = False) -> bool:
		for i, phoneme in enumerate(morph):
			for phoneme_set, environment, b in self.constraints:
				if phoneme not in phoneme_set:
					continue
				before = morph[i-1] if i-1 else None
				after = morph[i+1] if i+1 < len(morph) else None
				if not (environment.obeys(before, after) == b):
					return False
		if check_syllables:
			# check for syllable structure here - may or may not be broken
			for structure in self.syllable_strcuture:
				looking_at = 0
				works = True
				for phones, mandatory in structure:
					if looking_at in phones or not mandatory:
						looking_at += 1
					else:
						works = False
						break
				if works:
					return self.obeys(Morpheme(morph.phonemes[looking_at:]), check_syllables=True)
		return True


phoneme_a = Phoneme('a', **{'open': True, 'front': True, 'rounded': False, 'vowel': True})
phoneme_b = Phoneme('b', **{'voiced': True, 'bilabial': True, 'plosive': True, 'consonant': True})
syllable_a = ({phoneme_b}, False), ({phoneme_a}, True) # type: Tuple[Tuple[Set[Phoneme], bool]]
phonotactics_a = Phonotactics((syllable_a,), tuple())
new_morpheme = phonotactics_a.generate_morpheme()
print(new_morpheme)
