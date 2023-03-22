import os
from math import log10
from typing import Tuple

def get_size(start_path = '.'):
	"""Get size of directory"""
	total_size = 0
	for dirpath, dirnames, filenames in os.walk(start_path):
		for f in filenames:
			fp = os.path.join(dirpath, f)
			try:
				total_size += os.path.getsize(fp)
			except FileNotFoundError:
				pass
	return total_size

def get_subdirectories():
	return next(os.walk('.'))[1]

def get_sizes_of_subdirectories():
	subs = get_subdirectories()
	return sorted([(name, get_size(name)) for name in subs], key=lambda x: x[1])

def prettify_size(entry: Tuple[str, int]) -> str:
	name, size = entry
	prefixes = '', 'k', 'M', 'G', 'T'
	level = int(log10(size)/3) if size else 0
	return str(size//10**(3*level)) + f' {prefixes[level]}b\t' + name

while 1:
	input('\n'.join(map(prettify_size, get_sizes_of_subdirectories())))
