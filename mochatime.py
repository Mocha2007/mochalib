def t2d(n):#doesn't acct for negs yet
	mos=(31,59,90,120,151,181,212,243,273,304,334,365)
	mnames=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
	dy=365.25
	hd=24
	mh=60
	sm=60
	remainder=n
	years=remainder//(dy*hd*mh*sm)
	remainder=remainder%(dy*hd*mh*sm)
	days=remainder//(hd*mh*sm)
	remainder=remainder%(hd*mh*sm)
	hours=remainder//(mh*sm)
	remainder=remainder%(mh*sm)
	minutes=remainder//sm
	seconds=remainder%sm
	monthmonth=0
	for i in range(len(mos)):
		print(days,mos[i])
		if days<=mos[i]:
			monthmonth=i
			break
	days=days-mos[monthmonth-1]
	return int(1970+years),mnames[monthmonth],int(days),int(hours),int(minutes),seconds

def t2oneia(n):
	epoch=1497151176#SUN 2017 JUN 11 03:19:36 UTC
	year=6477407.605917404
	day=104148
	remainder=n-epoch
	years=remainder//year
	remainder=remainder%year
	days=remainder//day
	remainder=remainder%day
	first=remainder//(day/10)
	remainder=remainder%(day/10)
	second=remainder//(day/100)
	third=remainder%(day/100)/(day/1000)
	return int(years),int(days),int(first),int(second),third

def oneiapretty(qqq):
	return 'Year '+str(qqq[0])+', Day '+str(qqq[1])+', '+str(qqq[2])+':'+str(qqq[3])+':'+str(int(qqq[4]))

def ostats(qqq):
	stats={'e':False,4:0,'s':False}
	#eclipse season
	dates=((33,40),(55,61))
	if qqq[1] in range(dates[0][0],dates[0][1]+1) or qqq[1] in range(dates[1][0],dates[1][1]+1):stats['e']=True
	#month
	if qqq[1]<16:stats[4]='Spring'
	elif qqq[1]<32:stats[4]='Summer'
	elif qqq[1]<47:stats[4]='Fall'
	else:stats[4]='Winter'
	#showers
	showers=(11,12,18,24,36,42)
	showernames=('Vernal Nwarbid','Vernal Oikhurtid','Millenniid','Aestive Oikhurtid','Autumnal Nwarbid','Na\'unid',)
	if qqq[1] in showers:stats['s']=showernames[showers.index(qqq[1])]
	#zodiac
	zday=qqq[1]+10%62
	zodiac=('Drum','Bucket','Mice','Trowel','Ruler','Hawks','Candle','Road','Inkwell','Treaty','Die','Void','Void')#sans 'The'
	stats['z']='The '+zodiac[zday//5]
	return stats
