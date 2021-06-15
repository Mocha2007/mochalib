from random import randint
from typing import List, Tuple

trials = 10000

def d6(n: int=1) -> List[int]:
	return [randint(1, 6) for _ in range(n)]

def scoring(dice: List[int]) -> Tuple[int, int]:
	# does NOT account for no scoring dice;
	# that is handled in the main program
	score = 0
	# three of a kind
	# four of a kind = 2*three of a kind score
	# five of a kind = 2*four of a kind score
	# etc...
	potentially_usable = 0
	for unique in set(dice):
		count = dice.count(unique)
		if 2 < count:
			score += 2**(count-3) * 100 * \
				(unique if unique != 1 else 10*unique)
			potentially_usable += count
	# run, three pair
	if len(set(dice)) == 3 and all(dice.count(i) == 2 for i in dice):
		score += 1500
		potentially_usable = 6
	# ones are 100 each unless there are 3+
	if dice.count(1) < 3:
		score += 100*dice.count(1)
		potentially_usable += dice.count(1)
	# fives are 50 each unless there are 3+
	if dice.count(5) < 3:
		score += 50*dice.count(5)
		potentially_usable += dice.count(5)
	return score, potentially_usable if potentially_usable < 6 else 6

def scoring_plus(dice: List[int]) -> int:
	held_dice = 6-len(dice)
	excuse_six_unhelds = lambda score: 500 if not (score or held_dice) else score
	raw_score, used_dice = scoring(dice)
	score = excuse_six_unhelds(raw_score)
	if used_dice + held_dice == 6: # free roll
		score += scoring_plus(d6(6))
	return score

def mean_next_roll(unheld_dice: int=6) -> float:
	return sum(scoring_plus(d6(unheld_dice)) for _ in range(trials))/trials

def zilch_chance_next_roll(unheld_dice: int=6) -> float:
	if unheld_dice == 6:
		return 0
	return sum(0 == scoring(d6(unheld_dice)) for _ in range(trials))/trials

def expected(current_score: int=0, unheld_dice: int=6) -> float:
	mnr = mean_next_roll(unheld_dice)
	zc = zilch_chance_next_roll(unheld_dice)
	return (current_score + mnr)*(1-zc)

def should_i_roll(current_score: int=0, unheld_dice: int=6) -> bool:
	return current_score < expected(current_score, unheld_dice)