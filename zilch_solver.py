from random import randint
from typing import List

trials = 10000

def d6(n: int=1) -> List[int]:
	return [randint(1, 6) for _ in range(n)]

def scoring(dice: List[int]) -> int:
	# does NOT account for no scoring dice;
	# that is handled in the main program
	score = 0
	# three of a kind
	# four of a kind = 2*three of a kind score
	# five of a kind = 2*four of a kind score
	# etc...
	for unique in set(dice):
		count = dice.count(unique)
		if 2 < count:
			score += 2**(count-3) * 100 * \
				(unique if unique != 1 else 10*unique)
	# run, three pair
	if len(set(dice)) == 3 and all(dice.count(i) == 2 for i in dice):
		score += 1500
	# ones are 100 each unless there are 3+
	if dice.count(1) < 3:
		score += 100*dice.count(1)
	# fives are 50 each unless there are 3+
	if dice.count(5) < 3:
		score += 50*dice.count(5)
	return score

def mean_next_roll(unheld_dice: int=6) -> float:
	held_dice = 6-unheld_dice
	excuse_six_unhelds = lambda score: score if score else 500 if not held_dice else 0
	average = sum(excuse_six_unhelds(scoring(d6(unheld_dice))) for _ in range(trials))/trials
	return average

def zilch_chance_next_roll(unheld_dice: int=6) -> float:
	if unheld_dice == 6:
		return 0
	return sum(0<scoring(d6(unheld_dice)) for _ in range(trials))/trials

def should_i_roll(current_score: int=0, unheld_dice: int=6):
	mnr = mean_next_roll(unheld_dice)
	zc = zilch_chance_next_roll(unheld_dice)
	return current_score < (current_score + mnr)*(1-zc)