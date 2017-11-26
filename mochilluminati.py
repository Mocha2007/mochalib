def illuminati(name):
	name=name.lower()
	symbols=['compass','eye','square','triangle']
	names=['illuminati','mason']
	people=[
		('adam','weishaupt','was the founder of the Bavarian Illuminati.'),
		('adolph','knigge','was a recruiter for the Illuminati.')
		]
	#see if already llumin
	if name in symbols:
		print(name,'is an illuminati symbol.')
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
		#contains a forename
		if person[0] in name or person[1] in name:
			print(name,'contains the name',person[0]+'.')
			return illuminati(person[0])
	#then bs
	for item in symbols+names:
		#substring
		if item in name:
			print(name,'contains the word',item+'.')
			return illuminati(item)
		#same length
		if len(name)==len(item):
			print(name,'has',len(name),'letters,\n',item,'also has',len(name),'letters.')
			return illuminati(item)
	return 'Inconclusive'
while 1:
	print(illuminati(input('Enter name:\n> ')))