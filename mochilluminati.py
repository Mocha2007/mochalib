def anagram(a,b):#https://stackoverflow.com/questions/8286554/using-python-find-anagrams-for-a-list-of-words#8286639
    return sorted(a)==sorted(b)

def illuminati(name):
	name=name.lower()
	isymbols=['eye','triangle']
	fsymbols=['apron','ashlar','charity','compass','faith','forget-me-not','g','gavel','hope','jabulon','jahbulon','level','lodge','pencil','plumb','skull','square','sun','trowel']
	names=['illuminati','mason']
	people=[
		('adam','weishaupt','was the founder of the Bavarian Illuminati.'),
		('adolph','knigge','was a recruiter for the Illuminati.')
		]
	#see if already llumin
	if name in isymbols:
		print(name,'is an illuminati symbol.')
		return 'Illuminati Confirmed.'
	if name in fsymbols:
		print(name,'is a masonic symbol.')
		return 'Illuminati Confirmed.'
	if name in names:
		print(name,'is an illuminati name.')
		return 'Illuminati Confirmed.'
	#find names first
	for person in people:
		#is a forename
		if person[0]==name or person[1]==name:
			print(person[0],person[1],person[2])
			return 'Illuminati Confirmed.'
		#check if anagram
		if anagram(name,person[0]):
			print(name,'is an anagram of',person[0]+'.')
			return illuminati(person[0])
		#check if anagram
		if anagram(name,person[1]):
			print(name,'is an anagram of',person[1]+'.')
			return illuminati(person[1])
		#contains a forename
		if person[0] in name or person[1] in name:
			print(name,'contains the name',person[0]+'.')
			return illuminati(person[0])
		#same first char forename
		if person[0][0]==name[0]:
			print(name,'has the same first letter as',person[0],person[1]+'.')
			return illuminati(person[0])
		#same first char surname
		if person[1][0]==name[0]:
			print(name,'has the same first letter as',person[1]+',',person[0]+'.')
			return illuminati(person[1])
	#check if 3 or 4
	for item in ['3','4','three','four']:
		if item in name:
			print(name,'contains the word',item+'.')
			if item in ['3','three']:
				print('Triangles have',item,'sides.')
				return illuminati('triangle')
			else:
				print('Squares have',item,'sides.')
				return illuminati('square')
	#then bs
	for item in isymbols+fsymbols+names:
		#substring
		if item in name:
			print(name,'contains the word',item+'.')
			return illuminati(item)
		#check if anagram
		if anagram(name,item):
			print(name,'is an anagram of',item+'.')
			return illuminati(item)
		#same length
		if len(name)==len(item):
			print(name,'has',len(name),'letters,',item,'also has',len(name),'letters.')
			return illuminati(item)
		#same first char
		if name[0]==item[0]:
			print(name,'has the same first letter as',item+'.')
			return illuminati(item)
	return 'Inconclusive'
while 1:
	print(illuminati(input('Enter name:\n> ')))