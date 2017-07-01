from math import floor,log
from time import time
def primality(n):
	patience=1
	try:n=int(n)
	except:return False
	if n<2:return False
	if n==2 or n==3:return True
	#see if the number is of form 6k+/-1
	if n%6!=1 and n%6!=5:
		if n%2==0:return False#,2
		return False#,3
	#Lucas-Lehmer test
	if 2**floor(log(n,2))-1==n:
		p=floor(log(n,2))
		s=4
		i=0
		#write the sequence, stopping at s_(p-2)
		while i<p-2:
			s=s**2-2
			i+=1
		if s%n==0:return True
		return False
	#factor as much as possible given 1 second
	try:
		starttime=time()
		for i in range(1,int(n**.5//2)+1):
			if n%(i*2+1)==0:return False#,i*2+1
			if time()-starttime>patience:break
		if time()-starttime<patience:return True
	except OverflowError:
		for i in range(1,10**6):
			if n%(i*2+1)==0:return False#,i*2+1
	#can't disprove primality
	return "Probable Prime"