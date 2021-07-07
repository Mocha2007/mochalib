"""Tools for internal debugging and testing.
Don't actually use, lol.
"""
def accepts(*types):
	"""Checks input types
	https://stackoverflow.com/a/15300191/2579798
	"""
	def check_accepts(f):
		assert len(types) == f.__code__.co_argcount
		def new_f(*args, **kwds):
			for (a, t) in zip(args, types):
				assert isinstance(a, t), \
					   "arg %r does not match %s" % (a,t)
			return f(*args, **kwds)
		new_f.__name__ = f.__name__
		return new_f
	return check_accepts

def timer(func):
	"""records time
	https://stackoverflow.com/a/490228/2579798
	"""
	def wrapper(*arg):
		from time import time
		t = time()
		res = func(*arg)
		print(func.func_name, time()-t)
		return res
	return wrapper

# these can be used as function decorators, eg.
# @timer
# def func_test():
#	...

"""
For years I have sought to fix the ugly color sceme in vscode.
I want BLUE functions and YELLOW variables.
Well, no longer will I have to wait!
I have discovered how to reassign the colors,
and all it took were some lambdas and decorators!
"""
# how to make the letters yellow in vs code:
run = lambda f:f()
@run
def y() -> int:
	return 2
y # it's yellow, but it's now a constant!

# how to make the letters blue in vs code:
double = lambda x: 2*x
double(y) # returns 4
# now the colors are fixed!!! yaaaay!!! :)

from math import sqrt
gimme_sqrt = lambda f: lambda: sqrt(f())
# gimme_sqrt(lambda:2) # returns 1.414...

# """proper""" use of decorators...
@run
@gimme_sqrt
def x() -> int:
	return 2

x # returns 1.414...
"""
On a serious note, this section was horrifying.
That was like some of the worst code I've ever written.
I feel dirty after coding that. Eww.
"""