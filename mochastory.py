from random import choice as c
from random import random as r
from time import sleep as w

introductions=	[
				['Suddenly,','appeared.'],
				['A novelty arrived:','.'],
				['Futher exploration yielded','.'],
				['Uninvited,','entered.'],
				['Abruptly,','came to the area.']
				]
iactions=	[
			['Suddenly,','became dizzy.'],
			['Hungrily,','ate.'],
			['Thirstily,','drank.'],
			['Drunkenly,','fell.']
			]
tactions=	[
			['Hurriedly,','turned on','.'],
			['Never the one to be phased,','continued to','with haste.'],
			['Forgivingly,','apologized to','.'],
			['Sensually,','mounted','.']
			]
speech=	["Well, that's that.","Will you kindly shut up?"]
farewells=	[
			['And that was the last anyone ever heard of','.'],
			['And just as quickly as it arrived,','left.'],
			['Abruptly','exited the area.'],
			['Suddenly,','was lost forever.'],
			['Sadly,','died. Ripperoni in pizza.'],
			['Boldly,','entered oblivion, never to be heard from again.'],
			['Suspiciously,','perished in an unfortunate accident involving a runaway carriage.']
			]
actors=	[
	['abacus','abaci','an','the'],
	['applesauce','applesauce','an','the'],
	['building','buildings','a','the'],
	['cat','cats','a','the'],
	['coyote','coyotes','a','the'],
	['dog','dogs','a','the'],
	['fox','foxes','a','the'],
	['hat','hats','a','the'],
	['lion','lions','a','the'],
	['loaf','loaves','a','the'],
	['tiger','tigers','a','the'],
	['wolf','wolves','a','the']
	]
adj=	['calm','tense','uneventful','unsoiled']
spoke=	['commented','noted','remarked','said','stated']

currentactors=[]

while 1:
	w(1)
	if len(currentactors)==0 or r()<1/len(currentactors):#no actors
		#choosing an actor
		entrance=c(actors)
		currentactors+=[entrance]
		#introducing them
		script=c(introductions)
		print(script[0],entrance[2],entrance[0],script[1])
	elif len(currentactors)>1 and r()<.4:
		#choosing actors - preventing self-actions is seemingly impossible
		actora=c(currentactors)
		actorb=c(currentactors)
		#action
		script=c(tactions)
		if actora!=actorb:
			print(script[0],actora[3],actora[0],script[1],actorb[3],actorb[0],script[2])
		else:
			print(script[0],actora[3],actora[0],script[1],actorb[3],'other',actorb[0],script[2])
	elif len(currentactors)>0 and r()<.4:
		#choosing actor
		actor=c(currentactors)
		#action
		script=c(iactions)
		print(script[0],actor[3],actor[0],script[1])
	elif len(currentactors)>0 and r()<.4:
		#choosing actor
		actor=c(currentactors)
		#action
		script=c(speech)
		print('"'+script+'",',c(spoke),actor[3],actor[0]+'.')
	elif len(currentactors)>0 and r()<.5:
		#choosing actor
		actor=c(currentactors)
		#action
		script=c(farewells)
		print(script[0],actor[3],actor[0],script[1])
		currentactors.remove(actor)
	else:
		print('The moment was',c(adj)+'.')
