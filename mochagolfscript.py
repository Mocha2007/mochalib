def pre(prog: str) -> str:
	return prog


def main(prog: str) -> str:
	var = {'n': '\n'}
	comment = False
	string = False
	esc = False
	stack = []
	cnum = ''
	for i in range(len(prog)):
		command = prog[i]
		errorcode = 'error @ char '+str(i)+': '+command+'\n\tstack ('+str(len(stack))+'): '+str(stack)+'\n\tcode '
		# TODO " and or xor print p puts rand do while until if abs zip base
		try:
			isdefined = prog[i-1] == ':'
		except IndexError:
			isdefined = False
		if not comment and not isdefined:
			if (string and command not in '\\\'') or command in '0123456789':
				cnum += command
			elif command == ' ':
				try:
					stack.append(int(cnum))
				except ValueError:
					pass
				cnum = ''
			elif command == '\'' and string:
				stack.append(cnum)
				cnum = ''
				string = False
			elif command == '\\' and string:
				esc = True
			elif esc:
				if command in '\\\'':
					cnum += command
					esc = False
				else:
					return errorcode+'a'
			elif command == '#':
				comment = True
			elif command == '\'':
				string = True
			else:
				if cnum != '':
					try:
						stack.append(int(cnum))
						cnum = ''
					except ValueError:
						return errorcode+'b'
				if command in var:
					stack.append(var[command])
				elif len(stack): # for commands requiring at least ONE var
					temp = stack.pop()
					if command == '~':
						if isinstance(temp, int):
							stack.append(~temp)
						elif isinstance(temp, str):
							stack.append(run(temp))
						elif isinstance(temp, list):
							for j in temp:
								stack.append(j)
						else:
							return errorcode+'c'
					elif command == '`':
						stack.append(str(temp))
					elif command == '!':
						stack.append(int(not temp))
					elif command == '$':
						if isinstance(temp, int):
							stack.append(stack[len(stack)-1-temp])
						else:
							stack.append(sorted(temp))
					elif command == ':':
						stack.append(temp)
						var[prog[i+1]] = temp
					elif command == ';':
						pass
					elif command == ',':
						if isinstance(temp, int):
							stack.append(list(range(temp)))
						else:
							stack.append(len(temp))
					elif command == '.':
						stack.append(temp)
						stack.append(temp)
					elif command == '(':
						if isinstance(temp, int):
							stack.append(temp-1)
						else:
							try:
								stack.append(temp[0])
								stack.append(temp[1:])
							except IndexError:
								return errorcode+'d'
					elif command == ')':
						if isinstance(temp, int):
							stack.append(temp-1)
						else:
							try:
								stack.append(temp[:-1])
								stack.append(temp[-1])
							except IndexError:
								return errorcode+'e'
					elif len(stack): # for commands requiring at least TWO vars
						temp2 = stack.pop()
						if command == '+':
							if type(temp) == int == type(temp2) or type(temp) == list == type(temp2): # int int or arr arr
								stack.append(temp+temp2)
							elif isinstance(temp, str) or isinstance(temp2, str): # either string
								stack.append(str(temp2)+str(temp))
							elif isinstance(temp, int): # int arr
								stack.append(temp2 + [temp])
							elif isinstance(temp2, int): # arr int
								stack.append([temp2]+temp)
							else:
								return errorcode+'f'
						elif command == '-':
							try:
								stack.append(temp2-temp)
							except TypeError:
								return errorcode+'g'
						elif command == '*':
							if isinstance(temp, int):
								stack.append(temp2*temp)
							elif isinstance(temp2, int):
								stack.append(temp*temp2)
							elif isinstance(temp2, list) and isinstance(temp, str):
								stack.append(temp.join(temp2))
							elif isinstance(temp2, str) and isinstance(temp, str):
								stack.append(temp.join(temp2))
							elif isinstance(temp2, list) and isinstance(temp, str):
								stack.append(temp.join(''.join(temp2)))
							else: # str arr | arr arr
								if isinstance(temp2, str):
									temp2 = list(temp2)
								q = str(temp2).replace(',', str(temp)[1:-1])[1:-1].split(',')
								stack.append(q)
						elif command == '/': # UNLIKE OFFICIAL DOCS, ONLY DIVIDES
							stack.append(int(temp2/temp))
						elif command == '%': # UNLIKE OFFICIAL DOCS, ONLY MODULUS
							stack.append(temp2 % temp)
						elif command == '|':
							try:
								stack.append(temp2 | temp)
							except TypeError:
								return errorcode+'h'
						elif command == '&':
							try:
								stack.append(temp2 & temp)
							except TypeError:
								return errorcode+'i'
						elif command == '^':
							try:
								stack.append(temp2 ^ temp)
							except TypeError:
								return errorcode+'j'
						elif command == '\\':
							stack.append(temp)
							stack.append(temp2)
						elif command == '<':
							try:
								stack.append(temp2 < temp)
							except TypeError:
								return errorcode+'k'
						elif command == '>':
							try:
								stack.append(temp2 > temp)
							except TypeError:
								return errorcode+'l'
						elif command == '=':
							try:
								stack.append(temp2 == temp)
							except TypeError:
								return errorcode+'m'
						elif command == '?':
							if type(temp) == int == type(temp2):
								stack.append(temp2**temp)
							else:
								try:
									stack.append(temp.index(temp2))
								except (AttributeError, TypeError, ValueError):
									stack.append(-1)
						elif len(stack): # for commands requiring at least THREE vars
							temp3 = stack.pop()
							if command == '@':
								stack.append(temp2)
								stack.append(temp)
								stack.append(temp3)
							else:
								return errorcode+'n'
						else:
							return errorcode+'o'
					else:
						return errorcode+'p'
				else:
					return errorcode+'q'
		elif command == '\n':
			comment = False
	if cnum != '':
		try:
			stack.append(int(cnum))
		except ValueError:
			pass
	return str(stack)


def run(prog: str) -> str:
	return main(pre(prog))
