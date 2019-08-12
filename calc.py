from time import sleep
import tkinter as tk
# ty https://www.python-course.eu/tkinter_buttons.php <3

digits = '0123456789'
keys = [
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

# functions

def error(name: str='Error'):
	n = 5
	for i in range(n):
		label.config(text=name, bg='red' if i % 2 else 'white')
		sleep(1/n)
	stack = [0]
	screen_update()


def numpad(n: str):
	global stack
	print(n)
	if n in digits: # 48-57
		n = int(n)
		stack[-1] *= 10
		stack[-1] += n if 0 <= n else -n
	# speshul
	elif n == 'clear':
		stack = [0]
	elif n == 'enter':
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
	screen_update()

def screen_update():
	label.config(text='\n'.join(str(i) for i in stack), bg='white')

# make the gui 
 
root = tk.Tk()
root.title("MoCalc")
# tk.Font(family="Consolas", size=12)
label = tk.Label(root, anchor='e', width=35, height=5)
label.grid(row=0, columnspan=len(keys[0])+1)
for i, row in enumerate(keys):
	for j, k in enumerate(row):
		buttons[i][j] = tk.Button(root, text=k, height=2, width=6, command=(lambda k: lambda: numpad(k))(k))
		buttons[i][j].grid(row=i+1, column=j)
del i, row, j, k
tk.Button(root, text='CLEAR', height=5, width=6, command=lambda: numpad('clear')).grid(row=1, column=4, rowspan=2)
tk.Button(root, text='ENTER', height=5, width=6, command=lambda: numpad('enter')).grid(row=3, column=4, rowspan=2)
screen_update()
root.mainloop()
