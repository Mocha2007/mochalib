from collections import Counter
from itertools import groupby
def straights(list):
	tot=[]
	count=Counter(list)
	for i in set(list):
		tot+=[[i]*count[i]]
	tot.sort()
	return tot
			
def runs(list):
	tot=[]
	for i in list:
		run=[i]
		size=1
		while 1:
			if i+size in list:
				run+=[i+size]
				size+=1
			else:break
		tot+=[run]
	tot.sort()
	return [tot[i] for i in range(len(tot)) if i == 0 or tot[i] != tot[i-1]]

def sor(list):
	max=[]
	for i in runs(list):
		if len(i)>len(max):
			max=i
	for i in straights(list):
		if len(i)>len(max):
			max=i
	return max