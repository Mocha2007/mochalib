import os
import pygame
import sys
from math import ceil
from random import choice
from struct import unpack
from time import sleep, time
from typing import List, Set

black = 0, 0, 0
blue = 64, 128, 255
green = 0, 255, 0
red = 255, 0, 0

font_size = 16
fps = 30
origin = 0, 0
window_size = 950, 560

special = {
	7: ' \\a', # BEL
	8: ' \\b', # BS
	9: ' \\t', # HT
	10: ' \\n', # LF
	11: ' \\v', # VT
	12: ' \\f', # FF
	13: ' \\r', # CR
}


def bytes_to_int(bytestring: List[int]) -> int:
	s = 0
	for i in bytestring[::-1]:
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


def histogram(filename: str):
	import matplotlib.pyplot as plt

	bytestring = get_hex(filename)

	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_facecolor('#000000')

	plt.title('Bytes')
	plt.xlabel('Byte')
	plt.ylabel('Count')
	plt.hist([i for i in bytestring], 0x100)
	plt.show()


def plot_bytes(filename: str):
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	bytestring = open(filename, 'rb').read(0x2000000) # read at most 32 Mb

	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_facecolor('#000000')
	ax.axis('equal')

	plt.title('Byte Transitions')
	plt.xlabel('i-th byte')
	plt.ylabel('i+1-th byte')
	plt.xlim((0, 255))
	plt.ylim((0, 255))

	xs, ys = list(bytestring)[:-1], list(bytestring)[1:]

	plt.hist2d(xs, ys, bins=(256, 256), norm=mpl.colors.LogNorm())
	plt.show()


def plot_bytes2(filename: str):
	bytestring = open(filename, 'rb').read(0x200000) # read at most 2 Mb

	dim = ceil((len(bytestring)/3)**.5)
	pygame.init()
	screen = pygame.display.set_mode((dim,)*2)

	for i, r in list(enumerate(bytestring))[::3]:
		if len(bytestring) <= i + 2:
			break
		g, b = bytestring[i+1:i+3]
		coords = (i//3) % dim, (i//3) // dim
		screen.set_at(coords, (r, g, b))

	pygame.display.flip()

	while 1:
		# events
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
		sleep(1/30)


def plot_bytes3(filename: str, key: str='type'): # todo prerender the image first, then just move the image lol
	def scroll(lines: int):
		nonlocal start
		# scroll
		height = size[1]//2
		start += height * lines * 4
		if start < 0:
			start = 0
		# rerender
		file = open(filename, 'rb')
		file.seek(start)
		bytestring = file.read(0x40000) # read at most 256 kb
		screen.fill(black)
		for i, byte in list(enumerate(bytestring)):
			if key == 'type':
				color = bytecolor[byte]
			elif key == 'val':
				color = (byte,)*3
			# top half (256 kb)
			coords = i // height, i % height
			screen.set_at(coords, color)
			# bottom half (1 kb)
			if coords[0] < 4:
				rect = i, height, 1, height
				pygame.draw.rect(screen, color, rect)
		pygame.display.flip()

	start = 0
	size = 1024, 512
	pygame.init()
	screen = pygame.display.set_mode(size)
	# 				operators			symbols				digits
	# 				symbols				uppercase			symbols
	# 				lowercase			symbols				DEL
	bytecolor = [(255, 0, 255)]*0x21 + [(255, 0, 0)]*15 + [(255, 255, 0)]*10 + \
				[(255, 0, 0)]*7 + [(0, 255, 0)]*26 + [(255, 0, 0)]*6 + \
				[(0, 255, 255)]*26 + [(255, 0, 0)]*4 + [(255, 0, 255)] + [(128, 0, 255)]*0x80

	scroll(0)

	while 1:
		# events
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()
		# keyhold
		pressed = pygame.key.get_pressed()
		if pressed[pygame.K_a]:
			scroll(-1)
		elif pressed[pygame.K_d]:
			scroll(1)
		sleep(1/30)


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
		string += ('  ' + chr(byte)) if 0x20 <= byte <= 0x7E else (special[byte] if byte in special else '   ')
		# '\\' + hex_representation(byte)
	return string


def random_filename(directory: str, extension: str = None) -> str:
	return choice(list(get_filenames_from_dir(directory, extension)))


def show_hex(filename: str):
	pygame.init()
	screen = pygame.display.set_mode(window_size)
	cursor = [0, 0]
	filesize = '\t({} b)'.format(os.path.getsize(filename))

	def cmove(dx: int, dy: int = 0):
		nonlocal cursor
		cursor[0] += dx
		cursor[1] += dy
		cursor[0] %= 16
		cursor[1] %= 15 # buffer needs as many as 8 bytes

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
		# display cursor byte address
		cursor_rel = cursor[0] + 16*cursor[1]
		c_addr = hex(start + cursor_rel)[2:]
		c_rgb = tuple(data[cursor_rel+i] for i in range(3))
		typing = (
			('b', 1),
			('B', 1),
			('h', 2),
			('H', 2),
			('i', 4),
			('I', 4),
			('q', 8),
			('Q', 8),
			('f', 4),
			('d', 8),
		)
		join_data = tuple(str(unpack(t, data[cursor_rel:cursor_rel+size]))[1:-2] for t, size in typing)
		join_data = (c_addr,) + join_data + (c_rgb,)
		lines = [
			'\nAddress\t{}\n',
			'sbyte  \t{}',
			'byte   \t{}',
			'short  \t{}',
			'ushort \t{}',
			'int    \t{}',
			'uint   \t{}',
			'long   \t{}',
			'ulong  \t{}',
			'float  \t{}',
			'double \t{}',
			'RGB    \t{}',
		]
		cursor_tooltip = '\n'.join(lines).format(*join_data)
		text(cursor_tooltip, (640, 0))
		# swatch
		rect = 720, 16*(3+len(typing)), 16, 16
		pygame.draw.rect(screen, c_rgb, rect)
		# highlight
		if int(time()*2) % 2:
			# draw rect at byte
			byte_x, byte_y = (cursor[0])*18*2+10, (cursor[1]+2)*16
			rect = [byte_x, byte_y, 18, 14]
			pygame.draw.rect(screen, green, rect)
			# draw rect at char
			rect[1] += 16*17
			pygame.draw.rect(screen, blue, rect)
			# black byte
			text(hex_representation(data[cursor_rel]), (byte_x, byte_y), black)
			# todo black char
			text(pretty_chars(bytes([data[cursor_rel]])), (byte_x-10, byte_y+16*17), black)
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
				elif event.key == pygame.K_END: # end
					start = ceil(len(open(filename, 'rb').read())/16)*16 - 256
				elif event.key == pygame.K_HOME: # beginning
					start = 0
				elif event.key == pygame.K_PAGEUP: # up x 16
					scroll(-16)
				elif event.key == pygame.K_PAGEDOWN: # down x 16
					scroll(16)
				elif event.key == pygame.K_KP1: # cursor
					cmove(-1, 1)
				elif event.key == pygame.K_KP2: # cursor
					cmove(0, 1)
				elif event.key == pygame.K_KP3: # cursor
					cmove(1, 1)
				elif event.key == pygame.K_KP4 or event.key == pygame.K_LEFT: # cursor
					cmove(-1)
				elif event.key == pygame.K_KP6 or event.key == pygame.K_RIGHT: # cursor
					cmove(1)
				elif event.key == pygame.K_KP7: # cursor
					cmove(-1, -1)
				elif event.key == pygame.K_KP8: # cursor
					cmove(0, -1)
				elif event.key == pygame.K_KP9: # cursor
					cmove(1, -1)
		# keyhold
		pressed = pygame.key.get_pressed()
		shifting = pressed[pygame.K_LSHIFT] | pressed[pygame.K_RSHIFT]
		if pressed[pygame.K_w]:
			scroll(-16 if shifting else -1)
		elif pressed[pygame.K_s]:
			scroll(16 if shifting else 1)
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

# plot_bytes(current_filename)
show_hex(current_filename)
