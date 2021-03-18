from time import time
from typing import List

mem = [0]*256


def tonum(x: str) -> int:
	try:
		return int(x)
	except ValueError:
		pass
	if 1 == len(x):
		return ord(x)
	else:
		return len(x)


def run(prog: str, inp: List[str]) -> str:
	inp = list(map(tonum, inp))
	o = []
	inpn = 0
	start = time()
	pointer = 0
	i = 0
	while 1:
		try:
			command = prog[i]
		except IndexError:
			break

		if command == '>':
			pointer += 1
		elif command == '<':
			pointer -= 1
		elif command == '+':
			mem[pointer] += 1
		elif command == '-':
			mem[pointer] -= 1
		elif command == ',':
			try:
				mem[pointer] = inp[inpn]
			except IndexError:
				return 'Error: End of input'
			inpn += 1
		elif command == '.':
			o += [mem[pointer]]
		elif command == '[':# jump to the next matching ]
			if mem[pointer] == 0:
				balance = 1
				while balance:
					i += 1
					try:
						if prog[i] == '[':
							balance += 1
						elif prog[i] == ']':
							balance -= 1
					except IndexError:
						return 'Error: End of input'
				i -= 1
		elif command == ']': # jump to the next matching [
			if mem[pointer]:
				balance = -1
				while balance:
					i -= 1
					if prog[i] == '[':
						balance += 1
					elif prog[i] == ']':
						balance -= 1

		pointer = pointer % 256
		mem[pointer] = mem[pointer] % 256
		i += 1
		if 5 < time()-start:
			return 'Took too long (5s) Stopped at char '+str(i)
	return str(o)+'\n'+''.join(map(chr, o))
