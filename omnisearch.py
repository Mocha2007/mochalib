from os import listdir
from os.path import dirname, isfile, join, realpath
from re import findall

dir_path = dirname(realpath(__file__))
files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

def search_file(filename: str, regexp: str):
	with open(filename, 'r', errors='ignore') as file:
		matches = findall(regexp, file.read())
	if matches:
		print('{0} -> {1}'.format(filename, len(matches)))

def search_all_files(regexp: str):
	for file in files:
		search_file(file, regexp)

while 1:
	search_all_files(input('> '))
