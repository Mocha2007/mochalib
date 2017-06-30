#Advanced-ish math
#sign of a number
def factorial(x):
	if x==0 or x==1:
		return 1
	p=1
	for i in range(2,x+1):
		p=p*i
	return p
def sign(x):
	if x==0:return 0
	return x//abs(x)
#zeroes of a quadratic function
def quadratic(a,b,c):
	return [(-b+(b**2-4*a*c)**.5)/(2*a),(-b-(b**2-4*a*c)**.5)/(2*a)]
#triangular number
def triangular(x):
	return (x**2+x)/2
#interior angle of a regular n-gon in degrees
def interiorangle(x):
	return (x-2)*180/x
#interior angle sum of a regular n-gon in degrees
def interioranglesum(x):
	return (x-2)*180
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