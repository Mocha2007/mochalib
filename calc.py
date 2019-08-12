from copy import copy
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

def numpad(n: str):
	print(1, type(n), n)
	if n in digits:
		print(2)
		n = int(n)
		stack[-1] *= 10
		stack[-1] += n if 0 <= n else -n
	screen_update()

def screen_update():
	label.config(text=str(stack[-1]))

# make the gui 
 
root = tk.Tk()
root.title("MoCalc")
# tk.Font(family="Consolas", size=12)
label = tk.Label(root, anchor='e', width=30)
label.grid(row=0, columnspan=len(keys[0]))
for i, row in enumerate(keys):
	for j, k in enumerate(row):
		buttons[i][j] = tk.Button(root, text=k, height=2, width=6, command=(lambda k: lambda: numpad(k))(k))
		buttons[i][j].grid(row=i+1, column=j)
del i, row, j, k
screen_update()
root.mainloop()
