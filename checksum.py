from textwrap import wrap

def string_checksum(string: str, b: int=4) -> int:
	"""outputs an 8b-bit integer (default: 32-bit)"""
	string += '\0'*(len(string) % b) # so the string length is divisible by three
	checksum = 0
	for substring in wrap(string, b):
		checksum ^= string_to_int(substring)
	return checksum


def string_to_int(string: str) -> int:
	return sum(ord(char)*256**i for i, char in enumerate(string))
