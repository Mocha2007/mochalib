def change(n,total):
	coins=[]
	while n>0:
		n=round(n,2)
		if n>=.5:
			coins+=[.5]
			n-=.5
		elif n>=.25:
			coins+=[.25]
			n-=.25
		elif n>=.1:
			coins+=[.1]
			n-=.1
		elif n>=.05:
			coins+=[.05]
			n-=.05
		else:
			coins+=[.01]
			n-=.01
	if len(coins)==total:return coins
	elif len(coins)>total:return False
	while 1:
		if total==len(coins):return coins
		elif total-len(coins)>=4 and coins.count(.05)>0:
			coins.remove(.05)
			coins+=[.01]*5
		elif total-len(coins)>=2 and coins.count(.25)>0:
			coins.remove(.25)
			coins+=[.1]*2+[.05]
		elif total-len(coins)>=1 and coins.count(.5)>0:
			coins.remove(.5)
			coins+=[.25]*2
		elif total-len(coins)>=1 and coins.count(.1)>0:
			coins.remove(.1)
			coins+=[.05]*2
		else:return False
while 1:
	try:
		a=float(input("$> "))
		b=int(input("n> "))
		#print(sorted(change(a,b)))
		halfdollar=0
		quarter=0
		dime=0
		nickel=0
		penny=0
		for i in change(a,b):
			if i==.5:halfdollar+=1
			elif i==.25:quarter+=1
			elif i==.1:dime+=1
			elif i==.05:nickel+=1
			else:penny+=1
		if penny>0:
			print(".01",penny)
		if nickel>0:
			print(".05",nickel)
		if dime>0:
			print(".10",dime)
		if quarter>0:
			print(".25",quarter)
		if halfdollar>0:
			print(".50",halfdollar)
		if change(a,b)==False:
			print("Impossible")
	except:
		pass