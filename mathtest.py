"""
Math testing for the BOI
"""
from random import randint

settings = {
	"a_must_be_positive": True,
	"a_must_be_1": False,
}

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

def random_factorable_quadratic(cmax: int = 9,
		a_must_be_positive: bool = settings["a_must_be_positive"],
		a_must_be_1: bool = settings["a_must_be_1"]) \
		-> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]:
	a1, a2, b1, b2 = [randint(-cmax, cmax) for _ in range(4)]
	if a1*a2 == 0: # bad value
		return random_factorable_quadratic(cmax, a_must_be_positive)
	# normalize
	if a_must_be_1:
		a1 = a2 = 1
	elif a_must_be_positive and a1 < 0 and a2 < 0:
		a1, b1 = -a1, -b1
		a2, b2 = -a2, -b2
	elif a_must_be_positive and a1*a2 < 0:
		if a1 < 0:
			a1, b1 = -a1, -b1
		else:
			a2, b2 = -a2, -b2
	return (a1, b1), (a2, b2), (a1*a2, a1*b2+a2*b1, b1*b2)

def test_factoring() -> None:
	"""Factoring tester"""
	while 1:
		l1, l2, q = random_factorable_quadratic()
		print(f"{q[0]}x² + {q[1]} x + {q[0]}")
		input("Solve, and then press enter to see the answer:")
		print(f"({l1[0]}x + {l1[1]})({l2[0]}x + {l2[1]})")
		print(f"[Alternative] ({-l1[0]}x + {-l1[1]})({-l2[0]}x + {-l2[1]})")
		print(f"[Alternative] ({l2[0]}x + {l2[1]})({l1[0]}x + {l1[1]})")
		print(f"[Alternative] ({-l2[0]}x + {-l2[1]})({-l1[0]}x + {-l1[1]})")
		if expect_yn("Would you like to go back to the menu?\n\t"):
			return

def settings_menu() -> None:
	while 1:
		print("~~~~ Settings ~~~~")
		# list settings
		for i, j in settings.items():
			print(f" {i} = {j}")
		# try get key
		while (option := input("Select option:\n\t").lower()) not in settings:
			print(f"{option} is not a valid option; please try again:")
		t = type(settings[option])
		if t == bool: # python's bool thing doesn't work right with strings
			t = lambda x: x.lower() in {"true", "t"}
		# try get value
		while 1:
			try:
				value = t(input("New Value: "))
				break
			except:
				print("Invalid setting, please try again.")
		settings[option] = value
		print(f"{option} set to {value}")
		if expect_yn("Would you like to go back to the menu?\n\t"):
			return


tests = {
	"factoring": test_factoring,
	"settings": settings_menu,
	# quit
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