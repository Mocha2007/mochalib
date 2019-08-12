from textwrap import wrap

def string_checksum(string: str, b: int=2) -> int:
	"""outputs an 8b-bit integer (default: 16-bit)
	b	sigma
	1	2.885    8-bit
	2	4.324   16-bit
	3	5.419   24-bit
	4	6.337   32-bit
	8	9.155   64-bit
	16	13.108 128-bit
	"""
	string += '\0'*(len(string) % b) # so the string length is divisible by three
	checksum = 0
	for substring in wrap(string, b):
		checksum ^= string_to_int(substring)
	return checksum


def string_to_int(string: str) -> int:
	return sum(ord(char)*256**i for i, char in enumerate(string))
