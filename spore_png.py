# tool to decode Spore PNG files
# as originally documented
# http://www.rouli.net/2008/08/spores-png-format-illustrated.html
# https://web.archive.org/web/20111107050031/http://forums.somethingawful.com/showthread.php?threadid=2886188
# currently WIP and non-functional
from PIL import Image


def bits_to_bytes(bits: list[int]) -> bytes:
	return bytes(bytearray(bit_array_to_byte_array(bits)))


def bit_array_to_byte_array(bits: list[int]) -> list[int]:
	# https://stackoverflow.com/a/20545957/2579798
	return [sum([byte[b] << b for b in range(0,8)])
			for byte in zip(*(iter(bits),) * 8)
		]


def int_to_bytes(i: int) -> bytes:
	# https://stackoverflow.com/a/30375198/2579798
	return i.to_bytes(2, byteorder='big')


def decode(filename: str):
	img = Image.open(filename)
	bits = []
	i = 0
	for x in range(128):
		for y in range(128):
			r, g, b, a = img.getpixel((x, y))
			bit = a & 1
			bits.append(bit)
			i += 1
	data = bits_to_bytes(bits)
	# now, we compute the key, and then xor the data with the key to get the decoded model data
	# ... except that algorithm has been lost to time so just return the data for now
	return data


def test() -> None:
	print(decode(r'spore_png/CrownedLaggie.png'))
