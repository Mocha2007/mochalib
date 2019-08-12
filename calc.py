from time import sleep
import tkinter as tk
# ty https://www.python-course.eu/tkinter_buttons.php <3

digits = '0123456789'
keys = [
	['~', 'R', 'S', '^', 'L'],
	['7', '8', '9', '/'],
	['4', '5', '6', '*'],
	['1', '2', '3', '-'],
	['0', '\\', '%', '+'],
]
buttons = keys
key_coords = {}
for i, row in enumerate(keys):
	for j, k in enumerate(row):
		key_coords[k] = i, j

# set up vars

stack = [0]
history = []

# functions

def error(name: str='Error'):
	print(name)
	screen.config(text=name, bg='red')
	root.update()
	sleep(1)
	numpad('clear')


def numpad(n: str):
	global history
	global stack
	print(n)
	history.append(n)
	if n in digits: # 48-57
		n = int(n)
		stack[-1] *= 10
		stack[-1] += n if 0 <= n else -n
	# speshul
	elif n == 'clear':
		stack = [0]
		history = []
	elif n == '↵':
		stack.append(0)
	# other than special
	elif n == '%': # 37
		stack[-1] /= 100
	elif n == '*': # 42
		if 1 < len(stack):
			stack.append(stack.pop() * stack.pop())
		else:
			stack[-1] = 0
	elif n == '+': # 43
		if 1 < len(stack):
			stack.append(stack.pop() + stack.pop())
	elif n == '-': # 45
		if 1 < len(stack):
			stack.append(stack.pop(-2) - stack.pop())
		else:
			stack[-1] *= -1
	elif n == '/': # 47
		if 1 < len(stack):
			stack.append(stack.pop() / stack.pop())
		else:
			if stack[-1]:
				stack[-1] = 0
			else:
				error('ZeroDivisionError')
	elif n == '\\': # 92
		if 1 < len(stack):
			stack.append(stack.pop(-2))
	elif n == '^': # 94
		if 1 < len(stack):
			if stack[-2:] == [0, 0]:
				error('ZeroDivisionError')
			else:
				stack.append(stack.pop(-2) ** stack.pop())
		else:
			if stack[-1]:
				stack[-1] = 0
			else:
				error('ZeroDivisionError')
	elif n == '~': # 126
		stack[-1] *= -1
	screen_update()

def screen_update():
	screen.config(text='\n'.join(str(i) for i in stack), bg='white')
	history_screen.config(text=' '.join(history))

# make the gui 
 
root = tk.Tk()
root.title("MoCalc")
root.resizable(False, False)
# tk.Font(family="Consolas", size=12)
history_screen = tk.Label(root, anchor='e', width=35, height=1)
history_screen.grid(row=0, columnspan=len(keys[0])+1)
screen = tk.Label(root, anchor='e', width=35, height=5)
screen.grid(row=1, columnspan=len(keys[0])+1)
for i, row in enumerate(keys):
	for j, k in enumerate(row):
		buttons[i][j] = tk.Button(root, text=k, height=2, width=6, command=(lambda k: lambda: numpad(k))(k))
		buttons[i][j].grid(row=i+2, column=j)
del i, row, j, k
tk.Button(root, text='CLEAR', height=5, width=6, command=lambda: numpad('clear')).grid(row=3, column=4, rowspan=2)
tk.Button(root, text='ENTER', height=5, width=6, command=lambda: numpad('↵')).grid(row=5, column=4, rowspan=2)
screen_update()
root.mainloop()
