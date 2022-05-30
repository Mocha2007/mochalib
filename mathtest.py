"""
Math testing for the BOI
"""
from random import randint

def expect_yn(prompt: str = "") -> bool:
	oo = {
		'n': False,
		'no': False,
		'y': True,
		'yes': True,
	}
	while (o := input(prompt).lower()) not in oo:
		print("Invalid response.")
	return oo[o]

def random_factorable_quadratic(cmax: int = 9) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]:
	a1, a2, b1, b2 = [randint(-cmax, cmax) for _ in range(4)]
	return (a1, b1), (a2, b2), (a1*a2, a1*b2+a2*b1, b1*b2)

def test_factoring(a_must_be_positive: bool = True) -> None:
	"""Factoring tester"""
	while 1:
		l1, l2, q = random_factorable_quadratic()
		if l1[0] < 0 and l2[0] < 0:
			l1 = tuple(-x for x in l1)
			l2 = tuple(-x for x in l2)
		if a_must_be_positive and q[0] < 0:
			if l1[0] < 0:
				l1 = tuple(-x for x in l1)
			else:
				l2 = tuple(-x for x in l2)
			q = tuple(-x for x in q)
		print(f"{q[0]}x² + {q[1]} x + {q[0]}")
		input("Solve, and then press enter to see the answer:")
		print(f"({l1[0]}x + {l1[1]})({l2[0]}x + {l2[1]})")
		print(f"[Alternative] ({-l1[0]}x + {-l1[1]})({-l2[0]}x + {-l2[1]})")
		print(f"[Alternative] ({l2[0]}x + {l2[1]})({l1[0]}x + {l1[1]})")
		print(f"[Alternative] ({-l2[0]}x + {-l2[1]})({-l1[0]}x + {-l1[1]})")
		if expect_yn("Would you like to go back to the menu?\n\t"):
			return


tests = {
	"factoring": test_factoring,
	"quit": quit,
}

def main() -> None:
	"""Menu for selecting problems"""
	while 1:
		print("Welcome to Mocha's Mathtest Thingy for the boi UwU\nOptions:")
		for i, _ in tests.items():
			print(f"\t{i}")
		while (option := input("Select choice:\n\t").lower()) not in tests:
			print(f"{option} is not a valid choice; please try again:")
		tests[option]()

main()