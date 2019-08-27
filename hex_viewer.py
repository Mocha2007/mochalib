import os
import pygame
import sys
from random import choice
from time import sleep, time
from typing import List, Set

black = 0, 0, 0
blue = 64, 128, 255
green = 0, 255, 0
red = 255, 0, 0

font_size = 16
fps = 30
origin = 0, 0
window_size = 640, 560

special = {
	7: ' \\a', # BEL
	8: ' \\b', # BS
	9: ' \\t', # HT
	10: ' \\n', # LF
	11: ' \\v', # VT
	12: ' \\f', # FF
	13: ' \\r', # CR
}


def bytes_to_int(bytes: List[int]) -> int:
	s = 0
	for i in bytes[::-1]:
		s *= 0x100
		s += i
	return s


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


def plot_bytes(filename: str, bytes_per_coord: int = 2):
	import matplotlib.pyplot as plt

	bytestring = get_hex(filename)

	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_facecolor('#000000')
	
	plt.title('Bytespace')
	plt.xlabel('x ({} b)'.format(bytes_per_coord))
	plt.ylabel('y ({} b)'.format(bytes_per_coord))

	for i in range(0, len(bytestring), bytes_per_coord*2): # todo implement chunking
		if len(bytestring) <= i + bytes_per_coord:
			break
		x = bytestring[i:i+bytes_per_coord]
		y = bytestring[i+bytes_per_coord:i+2*bytes_per_coord]
		x, y = bytes_to_int(x), bytes_to_int(y)
		plt.scatter(x, y, marker='o', s=15, zorder=3, c='#00ff00')

	plt.show()


def pretty_bytes(bytestring: bytes) -> str:
	string = ''
	for i, byte in enumerate(bytestring):
		if i % 0x10 == 0:
			if i:
				string += '\n'
			string += ' '
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

	def scroll(lines: int):
		nonlocal start
		start += 16 * lines
		if start < 0:
			start = 0

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
		text(char_text, (0, 304), blue)
		# indices
		# top x
		text('  ' + '   '.join(hex_representation(i, 1) for i in range(0x10)), (0, 16), red)
		# bottom x
		text('  ' + '   '.join(hex_representation(i, 1) for i in range(0x10)), (0, 288), red)
		# top y (right)
		text('\n'.join(hex_representation(i) for i in range(start, start+256, 0x10)), (590, 32), red)
		# bottom y (right)
		text('\n'.join(hex_representation(i) for i in range(start, start+256, 0x10)), (590, 304), red)
		# end indices
		pygame.display.flip()
		# events
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 4: # up
					scroll(-1)
				elif event.button == 5: # down
					scroll(1)
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP: # up
					scroll(-1)
				elif event.key == pygame.K_DOWN: # down
					scroll(1)
				elif event.key == pygame.K_PAGEUP: # up x 16
					scroll(-16)
				elif event.key == pygame.K_PAGEDOWN: # down x 16
					scroll(16)
		# keyhold
		pressed = pygame.key.get_pressed()
		if pressed[pygame.K_w]:
			scroll(-1)
		elif pressed[pygame.K_s]:
			scroll(1)
		# pacing
		wait_time = start_time + 1/fps - time()
		if 0 < wait_time:
			sleep(wait_time)


def split_every(iterable, n: int) -> list:
	return [iterable[i:i+n] for i in range(0, len(iterable), n)]


if 1 < len(sys.argv):
	current_filename = sys.argv[1]
else:
	current_filename = input('> ')
	if not current_filename:
		current_dir = os.getenv('UserProfile') + '\\Desktop'
		current_filename = random_filename(current_dir, 'txt')

show_hex(current_filename)
# plot_bytes(current_filename)
