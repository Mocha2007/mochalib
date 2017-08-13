from random import choice
recipesfile=open("recipes.cfg", "r")
elementsfile=open("elements.cfg", "r")
ignorefile=open("ignore.cfg", "r")
aliasfile=open("alias.cfg", "r")
#make elements list
elements=[]
for line in elementsfile:
	finalname=''
	for char in line:
		if char!='\n':
			finalname+=char
	elements+=[finalname]
#make recipes list
recipes=[]
for line in recipesfile:
	firstelement=''
	for char in line:
		if char!='+':
			firstelement+=char
		else:
			break
	secondelement=''
	passedplus=False
	for char in line:
		if passedplus==True and char!='=':
			secondelement+=char
		elif passedplus==False and char=='+':
			passedplus=True
		elif passedplus==False:
			pass
		else:
			break
	thirdelement=''
	passedequals=False
	for char in line:
		if passedequals==True and char!='=' and char!='\n':
			thirdelement+=char
		elif passedequals==False and char=='=':
			passedequals=True
		elif passedequals==False:
			pass
		else:
			break
	#now that we know what each element is, the magic can happen!
	recipes+=[[firstelement,secondelement,thirdelement]]
#need to make combolist
combos=[]
for i in recipes:
	combos+=[[i[0],i[1]]]
#ignore list
ignore=[]
for line in ignorefile:
	finalname=''
	for char in line:
		if char!='\n':
			finalname+=char
	ignore+=[finalname]
#alias list
alias=[]
for line in aliasfile:
	chem=''
	for char in line:
		if char!='=':
			chem+=char
		else:
			break
	aka=''
	passedequals=False
	for char in line:
		if passedequals==True and char!='\n':
			aka+=char
		elif passedequals==False and char=='=':
			passedequals=True
		elif passedequals==False:
			pass
		else:
			break
	alias+=[[chem,aka]]
	
def verify(string):
	if string not in elements:
		for entry in alias:
			if entry[1]==string:
				option=input('Mochem has detected that '+string+' is an unindexed compound.\nHowever, it is an alias for '+entry[0]+'.\nWould you like to use this compound instead? y/n\na> ')
				if option=='y':
					return entry[0]
	return string
cations=['hydrogen','sodium','potassium','magnesium','calcium','ammonium']
anions=['chloride','oxide','sulfide','nitride','phosphide','nitrite','nitrate','sulfite','sulfate','hydroxide','phosphate','carbonate','chlorate','acetate','bicarbonate']
def allpairs():
	lst=[]
	for cation in cations:
		for anion in anions:
			lst+=[cation+' '+anion]
	return lst
while 1:
	command=input('c> ')
	if command in ['help','list']:
		print('alias X\nclear ~ clr ~ cls\ncombo X\nexit\nhelp ~ list\nhow X\nignore X\nindex\nmake X\nnames X\nsearch X\nstats\nunnamed\nuse X')
	elif command[0:5]=='alias':
		chem=verify(command[6:])
		aka=input('aka> ')
		if aka=='EXIT':
			pass
		elif [chem,aka] in alias:
			print('The attempted alias already exists!')
		elif chem not in elements:	
			print('This chemical is unindexed!')
		else:
			alias+=[[chem,aka]]
			open("alias.cfg", "a").write('\n'+chem+'='+aka)
	elif command in ['clear','clr','cls']:
		print('\n'*25)
	elif command[0:5]=='combo':
		e1=verify(command[6:])
		e2=verify(input('e2> '))
		if [e1,e2] in combos or [e2,e1] in combos:
			for recipe in recipes:
				if [recipe[0],recipe[1]]==[e1,e2] or [recipe[0],recipe[1]]==[e2,e1]:
					print(recipe[0],'+',recipe[1],'->',recipe[2])
		else:print('Unknown combination!')
	elif command=='exit':
		break
	elif command[0:3]=='how':
		e=verify(command[4:])
		for recipe in recipes:
			if recipe[2]==e:
				print(recipe[0],'+',recipe[1],'->',recipe[2])
	elif command[0:6]=='ignore':
		e=verify(command[7:])
		ignore+=[e]
		open("ignore.cfg", "a").write("\n"+e)
	elif command=='index':
		lst=[]
		for element in elements:
			if element not in ignore:
				lst+=[element]
		lst.sort()
		print(lst)
	elif command[0:4]=='make':
		out=verify(command[5:])
		e1=verify(input('e1> '))
		e2=verify(input('e2> '))
		#error detect thingy
		if e1 not in elements:
			print(e1,'unknown!')
		elif e2 not in elements:
			print(e2,'unknown!')
		elif True: #[e1,e2] not in combos and [e2,e1] not in combos
			recipes+=[[e1,e2,out]]
			combos+=[[e1,e2]]
			open("recipes.cfg", "a").write("\n"+e1+"+"+e2+"="+out)
			if out not in elements:
				elements+=[out]
				open("elements.cfg", "a").write("\n"+out)
		else:
			print('That is already an existing combination!')
	elif command[0:5]=='names':
		search=command[6:]
		for entry in alias:
			if entry[0]==search:
				print(entry[1])
			elif entry[1]==search:
				realname=entry[0]
				print(realname)
				for entry in alias:
					if entry[0]==realname and entry[1]!=search:
						print(entry[1])
	elif command[0:6]=='search':
		search=command[7:]
		for element in elements:
			if search in element:
				aka=''
				for entry in alias:
					if entry[0]==element:
						aka=entry[1]
						break
				if aka!='':
					print(element,'(',aka,')')
				else:
					print(element)
		for element in alias:
			if search in element[1]:
				print(element[0],'=',element[1])
	elif command=='stats':
		#figuring out how many cation-anion pairs have already been made
		lst=allpairs()
		number=0
		for compound in alias:
			if compound[1] in lst:
				number+=1
		print('',len(elements),'molecules\n',len(recipes),'recipes\n',len(alias),'aliases\n',100*number//len(lst),'% cation-anion pairs indexed')
	elif command=='unnamed':
		retour=elements[:]#has to have this, otherwise it won't actually copy the list
		for entry in alias:
			if entry[0] in retour:
				retour.remove(entry[0])
		for entry in ignore:
			if entry in retour:
				retour.remove(entry)
		print(retour)
	elif command[0:3]=='use':
		e=verify(command[4:])
		for recipe in recipes:
			if recipe[0]==e or recipe[1]==e:
				print(recipe[0],'+',recipe[1],'->',recipe[2])
	elif command=='':
		foundone=0
		while foundone==0:
			foundone=1
			cation=choice(cations)#+
			anion=choice(anions)#-
			#checks if this is already an alias
			for entry in alias:
				if cation+' '+anion==entry[1]:
					foundone=0
					break
			if foundone==1:
				print('Random suggestion: make',cation,anion)