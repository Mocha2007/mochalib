import pygame
import os
from random import choice
from time import sleep, time
from typing import Set

black = 0, 0, 0
blue = 64, 128, 255
green = 0, 255, 0

font_size = 16
fps = 30
origin = 0, 0
window_size = 570, 560

special = {
	7: ' \\a', # BEL
	8: ' \\b', # BS
	9: ' \\t', # HT
	10: ' \\n', # LF
	11: ' \\v', # VT
	12: ' \\f', # FF
	13: ' \\r', # CR
}


def get_filenames_from_dir(location: str, extension: str = None) -> Set[str]:
	filenames = set()
	for file in os.listdir(location):
		if not extension or file.endswith('.' + extension):
			filenames.add(location+'\\'+file)
	return filenames


def get_hex(filename: str, start: int = 0, size: int = 256) -> bytes:
	file = open(filename, 'rb')
	file.seek(start)
	return file.read(size)


def hex_representation(integer: int, min_digits: int = 2) -> str:
	h = hex(integer)[2:]
	padding = max(0, min_digits - len(h))
	return '0'*padding + h


def pretty_bytes(bytestring: bytes) -> str:
	string = ''
	for i, byte in enumerate(bytestring):
		if i % 0x10 == 0:
			if i:
				string += '\n'
		else:
			string += '  '
		string += hex_representation(byte)
	return string


def pretty_chars(bytestring: bytes) -> str:
	string = ''
	for i, byte in enumerate(bytestring):
		if i % 0x10 == 0:
			if i:
				string += '\n'
		else:
			string += ' '
		string += ('  ' + chr(byte)) if 0x20 <= byte <= 0x7E else (special[byte] if byte in special else '   ') # '\\' + hex_representation(byte)
	return string


def random_filename(directory: str, extension: str = None) -> str:
	return choice(list(get_filenames_from_dir(directory, extension)))


def show_hex(filename: str):
	pygame.init()
	screen = pygame.display.set_mode(window_size)
	
	filesize = '\t({} b)'.format(os.path.getsize(filename))

	def text(string: str, coords: (int, int) = (0, 0), color: (int, int, int) = green):
		string = string.replace('\t', ' '*4)
		myfont = pygame.font.SysFont('Consolas', font_size)
		for i, line in enumerate(string.split('\n')):
			textsurface = myfont.render(line, True, color)
			x, y = coords
			y += i*font_size
			screen.blit(textsurface, (x, y))

	start = 0
	while 1:
		start_time = time()
		# render
		screen.fill(black)
		data = get_hex(filename, start)
		hex_text = pretty_bytes(data)
		char_text = pretty_chars(data)
		screen_text = '\n\n'.join([filename+filesize, hex_text])
		text(screen_text)
		text(char_text, (0, 300), blue)
		pygame.display.flip()
		# events
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 4:
					start -= 16
					if start < 0:
						start = 0
				elif event.button == 5:
					start += 16
		# pacing
		wait_time = start_time + 1/fps - time()
		if 0 < wait_time:
			sleep(wait_time)


def split_every(iterable, n: int) -> list:
	return [iterable[i:i+n] for i in range(0, len(iterable), n)]


current_dir = os.getenv('UserProfile') + '\\Desktop'
current_filename = random_filename(current_dir, 'txt')
# current_filename = input('> ')
show_hex(current_filename)
