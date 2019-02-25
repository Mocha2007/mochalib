# mocha's math library
from math import acos, atan, ceil, cos, e, erf, factorial, gcd, pi, log, sin
from random import choice, randint, random
# math.factorial()
# shoelace formula
# area((0,0),(1,0),(1,1),(0,1))
# area((3,4),(5,6),(9,5),(12,8),(5,11))


def area(*points) -> float:
	points.append(points[0])
	return sum((points[i][0]*points[i+1][1] - points[i][1]*points[i+1][0]) for i in range(len(points)-1))/2


def areacircle(r: float) -> float:
	return pi*r**2


def areacone(r: float, h: float) -> float:
	return pi*r*(r+(r**2+h**2)**.5)


def areacylinder(r: float, h: float) -> float:
	return 2*pi*r*(r+h)


def areasphere(r: float) -> float:
	return 4*pi*r**2


def areatetrahedron(l: float) -> float:
	return l*3**.5


def bounding(*points): # eg point: (1,2,3)
	return tuple(map(min, zip(*points))), tuple(map(max, zip(*points)))


def cdf(x: float) -> float: # cumulative distribution function
	return (1+erf(x/2**.5))/2


def circumference(r: float) -> float:
	return 2*pi*r


def continuedfraction(*series):
	series = series[::-1]
	s = 0
	for i, si in enumerate(series):
		s = si+1/s if i else si
	return s


def dpoly(coefficients: list, n: int) -> list:
	"""derivative of a polynomial: input is a LIST or TUPLE and the nth derivative you want"""
	if coefficients == [0]:
		return [0]
	newcfs = []
	for i in range(len(coefficients)):
		newcfs += [coefficients[i]*(len(coefficients)-i-1)]
	del newcfs[len(newcfs)-1]
	if not newcfs:
		return [0]
	if n == 1:
		return newcfs
	return dpoly(newcfs, n-1)


def dist(a: list, b: list) -> float:
	"""works in any dimension! a and b are lists"""
	s = 0
	for i, ai in enumerate(a):
		s += (ai-b[i])**2
	return s**.5


def divisors(n: int) -> set:
	if n in (0, 1):
		return {n}
	allproducts = []
	for i in range(1,int(n**.5)+1):
		if n % i == 0:
			allproducts.append(i)
	for i in list(allproducts):
		allproducts.append(n//i)
	return set(allproducts)


def divisorfunction(n: int, x: int) -> int:
	divisorlist = divisors(n)
	s = 0
	for i in divisorlist:
		s += i**x
	return s


def doublefactorial(n: int) -> int:
	if n in (1, 2):
		return n
	return n*doublefactorial(n-2)

def fitts(a,b,d,w):
	return a+b*log(2*d/w,2)
	
def greedyegyptian(real):
	series=[]
	try:
		while 1:
			series+=[ceil(1/real)]
			real-=1/ceil(1/real)
	except ZeroDivisionError:
		return series
	
def harmonic(n,m):
	s=0
	for k in range(1,n+1):
		s+=1/k**m
	return s
	
def heron(a,b,c):
	s=(a+b+c)/2
	return (s*(s-a)*(s-b)*(s-c))**.5
#interior angle of a regular n-gon in degrees
def interiorangle(x):
	return (x-2)*180/x
#interior angle sum of a regular n-gon in degrees
def interioranglesum(x):
	return (x-2)*180

def ipoly(coefficients,n):
	newcfs=[]
	for i in range(len(coefficients)):
		if coefficients[i]!="c":newcfs+=[coefficients[i]/(len(coefficients)-i)]
		else:newcfs+=["c"]
	newcfs+=["c"]
	if n==1:return newcfs
	return ipoly(newcfs,n-1)

def lcm(a,b):
	return int(a*b/gcd(a,b))
	
def logistic(x,x0,L,k):
	return L/(1+e**(-k*(x-x0)))
	
def npr(a,b):
	return factorial(a)/factorial(a-b)
	
def bernoulli(m,n):
	bigsum=0
	for k in range(m+1):
		littlesum=0
		for v in range(k+1):
			littlesum+=(-1)**v*ncr(k,v)*(n+v)**m/(k+1)
		bigsum+=littlesum
	return bigsum

def birthday(n):
	return npr(365,n)/365**n
	
def ncr(a,b):
	return npr(a,b)/factorial(b)

def ord(a,n):#multiplicative order
	if gcd(a,n)!=1:raise ValueError("Cannot find the multiplicative order of two integers whose greatest common denominator is not 1!")
	k=1
	while n**k%a!=1:
		k+=1
	return k

def pascal(n):
	line=[1]
	for k in range(n):
		line+=[line[k]*(n-k)//(k+1)]
	return line
	
def quadratic(a,b,c):
	return (-b+(b**2-4*a*c)**.5)/2/a,(-b-(b**2-4*a*c)**.5)/2/a
	
def sgn(x):
	if x==0:return 0
	return x//abs(x)
	
def snormal(x):
	return e**(-x**2/2)/(2*pi)**.5

def normal(x,mu,sigma):
	return snormal((x-mu)/sigma)/sigma

def randomspherepoint():
	return 2*pi*random(), acos(2*random()-1)
	
def standardlogistic(x):
	return 1/(1+e**-x)
	
def sumofdivisors(n):
	return sum(divisors(n))

def abundancy(n):
	return sumofdivisors(n)/2/n

def totient(n):
	s=0
	for i in range(n):
		if gcd(n,i)==1:s+=1
	return s
	
def triangular(n: int) -> int:
	return (n**2+n)//2

def volumecone(r,h):
	return 1/3*pi*r**2*h

def volumecylinder(r,h):
	return pi*r**2*h

def volumensphere(r,n):
	if n==1:return 2*r
	if n==2:return pi*r**2
	return 2*pi/n*volumensphere(r,n-2)

def volumesphere(r):
	return 4/3*pi*r**3

def volumetetrahedron(l):
	return l/6/2**.5


# weierstrass(0,5/6,7)
def weierstrass(x: float, a: float, b: int) -> float:
	if not 0 < a < 1:
		raise ValueError("A must be between 0 and 1 exclusive!")
	if b % 2 == 0 or b < 0:
		raise ValueError("B must be a positive odd integer!")
	if not a*b > 1+3*pi/2:
		raise ValueError("A and B must be such that ab>1+3*pi/2")
	s = 0
	n = -1
	oldsum = -1
	while oldsum != s:
		oldsum = s
		n += 1
		s += a**n*cos(b**n*pi*x)
	return s


def woodall(n):
	return n*2**n-1
	
def zeta(s):
	#one
	if s==1:raise ValueError("Zeta is undefined at s=1")
	#non-positive integers
	if s%1==0 and s<1:
		#negative even integers and zero
		if s%2==0:return 0
		#negative odd integers
		return (-1)**-s*bernoulli(1-s,0)/(1-s)#i think bernoulli is 0 by default if no number is specified, but unsure.
	#non-integer reals under one
	if s<1:raise ValueError("Sorry, this algorithm can't compute non-integer zeta numbers below 1")
	#reals above one - even positive integers can be calculated using bernoulli numbers, however, i am unsure of the efficacy of that method
	summation=0
	i=1
	old=-1
	while old!=summation:
		old=summation
		summation+=i**-s
		i+=1
	return summation
	
def zipf(k,s,n):
	return 1/(k**s*harmonic(n,s))
	
#random tools

def averagetimestooccur(chance):#eg an event with a 6% chance of occuring has a 50% chance after n number of times
	return -log(2)/log(1-chance)

def chanceofoccuring(chance,times):#eg an event with a 6% chance of occuring has had the opportunity to happen 15 times
	return 1-(1-chance)**times
	
def card():
	faces=["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
	deck=[]
	for i in faces:
		deck+=[(i+"♦")]
		deck+=[(i+"♥")]
		deck+=[(i+"♣")]
		deck+=[(i+"♠")]
	return choice(deck)

def dice(n,sides,bonus):
	s=0
	for i in range(n):
		s+=randint(1,sides)
	return s+bonus

def mnm():
	bag=["Blue"]*24+["Brown"]*13+["Green"]*16+["Orange"]*20+["Red"]*13+["Yellow"]*14
	return choice(bag)
	
#return the nth row of pascal's triangle. Taken from http://stackoverflow.com/a/40067541
def pascalrow(n):
     n-=1
     line=[1]
     for k in range(max(n,0)):             
         line.append(line[k]*(n-k)//(k+1))             
     return line
	 
#sum. function must use x.
def ssum(min,max,function):
	total=0
	for x in range(min,max+1):
		total+=eval(function)
	return total
#product. function must use x.
def pproduct(min,max,function):
	total=1
	for x in range(min,max+1):
		total=total*eval(function)
	return total
	
#infinite sum
def isum(min,function):
	previous=-1
	s=0
	x=min
	while previous!=s and abs(s)<10**10:
		previous=s
		s+=eval(function)
		x+=1
		#print(x,s)
	if s<10**10:
		return s
	elif s>10**10:
		return "+inf"
	return "-inf"
#infinite product
def iproduct(min,function):
	previous=-1
	s=1
	x=min
	while previous!=s and abs(s)<10**10:
		previous=s
		s=s*eval(function)
		x+=1
		#print(x,s)
	if s<10**10:
		return s
	elif s>10**10:
		return "+inf"
	return "-inf"
	
#is in set?

def isburningship(c):
	z=0
	for i in range(1000):
		z=(abs(z.real)+1j*abs(z.imag))**2+c
		if abs(z)>2:return False
	return True

def ismandelbrot(c):
	z=0
	for i in range(1000):
		z=z**2+c
		if abs(z)>2:return False
	return True
	
#for really big numbers

def tetration(x,y):
	o=1
	for i in range(y):o=x**o
	return o

def pentation(x,y):
	o=1
	for i in range(y):o=tetration(x,o)
	return o

def arrow(a,b,power):
	if power==1:return a**b
	if b==1: return a
	return arrow(a,arrow(a,b-1,power),power-1)

def graham(n):
	gn=arrow(3,3,4)
	for n in range(1,n):
		gn=arrow(3,3,gn)
	return gn
	#graham's number is n=64

#DONT CHANGE TO *chain
def chained(chain):#chain is a LIST, function returns an INTEGER
	#http://googology.wikia.com/wiki/Chained_arrow_notation
	#rules
	#checking for any 1s
	for i in range(len(chain)-1):
		if chain[i]==1:
			for j in range(i,len(chain)-1):
				del chain[j]
	if chain==[]:return 1#best thing i can come up with
	if len(chain)==1:return chain[0]
	if len(chain)==2:return chain[0]**chain[1]
	if len(chain)==3:return arrow(chain[0],chain[1],chain[2])
	#end rules
	newchain1=chain
	newchain1[len(chain)-2]-=1
	newchain1=chained(newchain1)
	newchain2=chain
	newchain2[len(chain)-1]-=1
	newchain2[len(chain)-2]=newchain1
	return chained(newchain2)
	
def arrayed(array):#has to be a list, sadly, not *array. BROKEN. Don't know why. Fuck this shit. I spent hours trying to make this work and it only breaks. Nice.
	if len(array)==1:return array[0]
	if len(array)==2:return array[0]**array[1]
	if array[len(array)-1]==1:
		del array[len(array)-1]
		return arrayed(array)
	if array[1]==1:return array[0]
	if array[2]==1:
		for i in range(len(array)):
			if array[i]==1:
				newarray=array
				newarray[1]-=1
				array[i]=arrayed(newarray)
				#decrement the next value in the array
				array[i+1]-=1
				return arrayed(array)#maybe break instead of return? i forgot how this code even works
			i=array[0]
	#rule five
	newarray=array
	newarray[1]-=1
	array[1]=arrayed(newarray)
	array[2]-=1
	return arrayed(array)
	
def beaf(array):
	#rule 0
	if len(array)==0:return 1
	if len(array)==1:return array[0]
	#rule 1
	if array[1]==1:return array[0]
	#rule 2
	if len(array)==2:return array[0]**array[1]
	#rule 3.1
	for i in range(2,len(array)):
		if array[i]!=1:
			array[i]-=1
			#rule 3.2
			if i!=2:
				newarray=array
				newarray[0]-=1
				array[i-1]=beaf(newarray)
			#rule 3.3
			for j in range(i+1,len(array)):
				array[j]=array[0]
	return beaf(array)
	
def ack(m,n):
	if m==0:return n+1
	if m>0 and n==0: return ack(m-1,1)
	return ack(m-1,ack(m,n-1))
	
# NEW FUNCTIONS

def cubicstats(a,b,c,d):
	print("Zeroes","*shrug*")
	print("Extrema",quadratic(3*a,2*b,c))
	print("Inflection Point",-b/3/a)
	
#sqrt(n+sqrt(n+sqrt(n+...
#equivalent to positive solution to x^2-x-n=0
def infiniroot(n):
	return (1+(1+4*n)**.5)/2
	
def maximizerect(P,m,n):
	return P**2/(4*(m+1)*(n+1))
#coord systems
def cyl2rect(r,theta,z):
	'''Converts from cylindrical coordinates to rectangular coordinates'''
	return r*cos(theta),r*sin(theta),z
def rect2cyl(x,y,z):
	'''Converts from rectangular coordinates to cylindrical coordinates'''
	return (x**2+y**2)**.5,atan(y/x),z
def rect2sphere(x,y,z):
	'''Converts from rectangular coordinates to spherical coordinates'''
	return (x**2+y**2+z**2)**.5,atan(y/x),acos(z/(x**2+y**2+z**2)**.5)
def sphere2rect(rho,theta,phi):
	'''Converts from spherical coordinates to rectangular coordinates'''
	return rho*sin(phi)*cos(theta),rho*sin(phi)*sin(theta),rho*cos(phi)

# 21 Dec 2018


def prime_factors(n: int) -> set:
	for i in range(2, int(n**.5)+1):
		if n % i == 0:
			return factors(n//i).add(i)
	return {n}


def factors(n: int) -> dict:
	for i in range(2, int(n**.5)+1):
		if n % i == 0:
			dictionary = factors(n//i)
			if i in dictionary:
				dictionary[i] += 1
			else:
				dictionary[i] = 1
			return dictionary
	return {n: 1}


def product(*n) -> float:
	if len(n) == 0:
		return 0
	if len(n) == 1:
		return n[0]
	return product(*n[:-1]) * n[-1]


def is_prime(n: int) -> bool:
	for i in range(2, int(n**.5)+1):
		if n % i == 0:
			return False
	return True
