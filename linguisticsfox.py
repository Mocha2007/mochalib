from typing import Dict, Set, Tuple
from random import choice, random


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

	def __getitem__(self, item) -> Phoneme:
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
				constraints: Tuple[Tuple[Set[Phoneme], Environment, bool]], syllable_count: (int, int) = (1, 4)):
		self.syllable_strcuture = syllable_structure
		self.constraints = constraints
		self.syllable_count = syllable_count

	def generate_morpheme(self) -> Morpheme:
		morph = Morpheme()
		for _ in range(*self.syllable_count):
			syllable_template = choice(self.syllable_strcuture)
			for phoneme_set, b in syllable_template:
				if b or .5 < random():
					morph.append(choice(tuple(phoneme_set)))
		return morph if self.obeys(morph) else self.generate_morpheme()

	def obeys(self, morph: Morpheme) -> bool:
		for i, phoneme in enumerate(morph):
			for phoneme_set, environment, b in self.constraints:
				if phoneme not in phoneme_set:
					continue
				before = morph[i-1] if i-1 else None
				after = morph[i+1] if i+1 < len(morph) else None
				if not (environment.obeys(before, after) == b):
					return False
		return True


phoneme_a = Phoneme('a', **{'open': True, 'front': True, 'rounded': False, 'vowel': True})
phoneme_b = Phoneme('b', **{'voiced': True, 'bilabial': True, 'plosive': True, 'consonant': True})
syllable_a = ({phoneme_b}, False), ({phoneme_a}, True) # type: Tuple[Tuple[Set[Phoneme], bool]]
phonotactics_a = Phonotactics((syllable_a,), tuple())
new_morpheme = phonotactics_a.generate_morpheme()
print(new_morpheme)
