from math import ceil, log
from random import choice, random
from time import time

alphabet = 'abcdefghijklmnopqrstuvwxyz'
inf = float('inf')


def mode(x):
	"""Return the mode of an iterable"""
	return max(set(x), key=x.count)


def alphabet_generator(x: str) -> str:
	assert len(x) == 1
	if x in alphabet.upper():
		return alphabet[:alphabet.index(x.lower())+1].upper()
	return alphabet[:alphabet.index(x)+1]


def random_program(n: int) -> str:
	valid_chars = list(type_specific_behavior.keys()) + list(functions.keys()) + list("#':;\\_`")
	return ''.join(choice(valid_chars) for _ in range(n))


def error(**kwargs):
	pointer = ' '*kwargs['i'] + '^'
	kwargs['p'] = pointer
	kwargs['b'] = stack_history[-1]
	print('{x}\n{p}\nchar {i} {c}: {e}\ns: {b} -> {s}\nf: {f}'.format(**kwargs))


type_specific_behavior = {
	'$': {
		len: 1,
		# (int,): lambda x: stack[-x],
		(list,): sorted,
		(str,): sorted,
	},
	'"': {
		len: 1,
		(int,): chr,
		(list,): lambda x: ''.join([chr(i) for i in x]),
		(str,): lambda x: [ord(i) for i in x],
	},
	'(': {
		len: 1,
		(float,): lambda x: x-1,
		(int,): lambda x: x-1,
		(list,): lambda x: [i-1 for i in x],
		(str,): lambda x: ''.join([chr(ord(i)-1) for i in x]),
	},
	')': {
		len: 1,
		(float,): lambda x: x+1,
		(int,): lambda x: x+1,
		(list,): lambda x: [i+1 for i in x],
		(str,): lambda x: ''.join([chr(ord(i)+1) for i in x]),
	},
	'-': {
		len: 2,
		(float, float): lambda a, b: a-b,
		(float, int): lambda a, b: a-b,
		(int, int): lambda a, b: a-b,
		(int, float): lambda a, b: a-b,
		(list, float): lambda a, b: list(filter(lambda x: x != b, a)),
		(list, int): lambda a, b: list(filter(lambda x: x != b, a)),
		(list, list): lambda a, b: list(filter(lambda x: x not in b, a)),
		(list, str): lambda a, b: list(filter(lambda x: x != b, a)),
		(str, int): lambda a, b: a[:-b],
		(str, str): lambda a, b: ''.join(list(filter(lambda x: x not in b, list(a)))),
	},
	'/': {
		len: 2,
		(float, float): lambda a, b: a/b,
		(float, int): lambda a, b: a/b,
		(int, int): lambda a, b: a/b,
		(int, float): lambda a, b: a/b,
		(list, float): lambda a, b: a[:len(a)//b],
		(list, int): lambda a, b: a[:len(a)//b],
		(list, list): lambda a, b: a.count(b),
		(list, str): lambda a, b: a.count(b),
		(str, float): lambda a, b: a[:len(a)//b],
		(str, int): lambda a, b: a[:len(a)//b],
		(str, str): lambda a, b: a.count(b),
	},
	'M': {
		len: 1,
		(list,): lambda x: sum(x)/len(x),
	},
	'[': {
		len: 1,
		(float,): int,
		(int,): int,
		(list,): min,
		(str,): lambda x: x.lower(),
	},
	']': {
		len: 1,
		(float,): ceil,
		(int,): int,
		(list,): max,
		(str,): lambda x: x.upper(),
	},
	'm': {
		len: 1,
		(float,): lambda x: mode(str(x)),
		(int,): lambda x: mode(str(x)),
		(list,): mode,
		(str,): mode,
	},
	'n': {
		len: 1,
		(int,): lambda x: -x,
		(str,): alphabet_generator,
	},
	'|': {
		len: 1,
		(float,): abs,
		(int,): abs,
		(list,): len,
		(str,): len,
	},
	'~': {
		len: 1,
		(float,): lambda x: -x,
		(int,): lambda x: -x,
		(list,): lambda x: x[::-1],
		(str,): lambda x: x[::-1],
	},
}


# special functions
def type_specific_function(char: str, stack):
	args = tuple(stack.pop() for _ in range(type_specific_behavior[char][len]))
	return type_specific_behavior[char][tuple(type(arg) for arg in args)](*args)


# functions
functions = {
	'!': lambda stack: int(not stack.pop()),
	# " (USED) chr/ord
	# # (USED) comments
	# $ (USED) int -> stack item at index; list/string -> sort
	'%': lambda stack: stack.pop() % stack.pop(),
	'&': lambda stack: stack.pop() & stack.pop(),
	# ' (USED) string creation
	# ( (USED) decrement
	# ) (USED) increment
	'*': lambda stack: stack.pop() * stack.pop(),
	'+': lambda stack: stack.pop() + stack.pop(),
	',': lambda *_: 0,
	# - (USED) numeric/set subtraction
	'.': lambda stack: stack[-1],
	# / (USED) numeric division, special behavior with lists/strings
	'0': lambda stack: stack.pop()*10,
	'1': lambda stack: stack.pop()*10+1,
	'2': lambda stack: stack.pop()*10+2,
	'3': lambda stack: stack.pop()*10+3,
	'4': lambda stack: stack.pop()*10+4,
	'5': lambda stack: stack.pop()*10+5,
	'6': lambda stack: stack.pop()*10+6,
	'7': lambda stack: stack.pop()*10+7,
	'8': lambda stack: stack.pop()*10+8,
	'9': lambda stack: stack.pop()*10+9,
	# : (USED) assignment
	# ; (USED) pop
	'<': lambda stack: int(stack.pop() < stack.pop()),
	'=': lambda stack: int(stack.pop() == stack.pop()),
	'>': lambda stack: int(stack.pop() > stack.pop()),
	'?': lambda stack: (lambda *x: x[1] if x[0] else x[2])(stack.pop(), stack.pop(), stack.pop()),
	'@': lambda stack: stack.pop(0),
	# M (USED) mean
	# [ (USED) floor/lowercase/min
	# \ (USED) swap top two
	# ] (USED) ceiling/uppercase/max
	'^': lambda stack: stack.pop()**stack.pop(),
	# _ (USED) collapse stack into array
	# ` (USED) function call
	'a': lambda *_: [],
	'l': lambda stack: log(stack.pop()),
	# m (USED) mode
	# n (USED) 5 n -> [1, 2, 3, 4, 5]; 'c' n -> 'abc'
	# 'p': lambda: [stack.pop()] + stack.pop(),
	'r': lambda *_: random(),
	't': lambda *_: time(),
	'z': lambda stack: list(zip(stack.pop())),
	# {
	# | (USED) abs/len
	# }
	# ~ (USED) negation/reversal
}


# execute line
def run(program: str, **kwargs):
	if 'declared_functions' in kwargs:
		declared_functions = kwargs['declared_functions']
	else:
		declared_functions = {}
	if 'external_stack' in kwargs:
		external_stack = kwargs['external_stack']
	else:
		external_stack = Stack(0)
	stack = Stack(external_stack=external_stack)
	is_comment = False
	is_string = False
	is_escape = False
	for i, char in enumerate(program):
		stack_history.append(list(stack))
		if is_string:
			# don't add strings to ints n shit
			if not stack or type(stack[-1]) != str:
				stack.append('')
			# escape chars
			if is_escape:
				is_escape = False
				stack[-1] += char
			elif char == '\\':
				is_escape = True
			elif char == '\'':
				is_string = False
			else:
				# otherwise, concat
				stack[-1] += char
			continue
		try:
			# SPECIAL FUNCTIONS THAT MUST BE RUN BEFORE
			# comments
			if char == '#':
				is_comment = not is_comment
			# strings
			elif char == '\'':
				is_string = True
			# function assignment
			elif char == ':':
				a, b = stack.pop(), stack.pop()
				declared_functions[a] = str(b)
			# stack pop
			elif char == ';':
				stack.pop()
			# top two in stack swap
			elif char == '\\':
				a, b = stack.pop(), stack.pop()
				stack.append(b)
				stack.append(a)
			# collapse stack into array
			elif char == '_':
				stack.list = [stack.list]
			# function execution
			elif char == '`':
				stack.append(run(declared_functions[stack.pop()], declared_functions=declared_functions, external_stack=stack))
			# really special
			elif char in type_specific_behavior:
				stack.append(type_specific_function(char, stack))
			# nop
			if char not in functions:
				continue
			# main code
			stack.append(functions[char](stack))
		except Exception as this_exception:
			error(c=char, e=this_exception, f=declared_functions, i=i, s=stack, x=program)
			break
	return stack


class Stack:
	count = 0

	def __init__(self, n=0, **kwargs):
		self.list = [0]*n
		if 'external_stack' in kwargs:
			self.external_stack = kwargs['external_stack']

	def __getitem__(self, item):
		return self.list[item]

	def __iter__(self):
		for i in self.list:
			yield i

	def __len__(self) -> int:
		return len(self.list)

	def __repr__(self) -> str:
		return str(self.list).replace(',', '')

	def __setitem__(self, key, value):
		self.__dict__[key] = value

	def append(self, x):
		self.push(x)

	def pop(self, *args):
		try:
			return self.list.pop(*args)
		except IndexError:
			try:
				return self.external_stack.pop(*args)
			except (AttributeError, IndexError):
				return 0

	def push(self, x):
		self.list.append(x)


# main
while 1:
	# stack = Stack() # [0] <> Stack()
	stack_history = []
	code = input('>>> ')
	if not code:
		code = random_program(20)
	print(code)
	output = run(code)
	print(output)
