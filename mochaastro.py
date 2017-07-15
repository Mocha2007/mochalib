from math import cos,inf,log,pi,sin
g=6.67408e-11#standard gravitational constant
sigma=5.670373e-8#Stefan-Boltzmann Constant
c=299792458#m/s

pc=3.0857e16
ly=9.4607e15
au=149597870700
a_moon=3.84399e8
a_planck=2.176470e-8

m_sun=1.98855e30
m_j=1.8986e27
m_e=5.97237e24
m_moon=7.342e22
m_person=69

r_sun=6.957e8
r_j=6.9911e7
r_e=6.371e6
r_moon=1737100
r_person=.2544#assuming a perfect sphere
r_planck=1.616229e-35

t_e=86164.100352
t_planck=5.39116e-44

l_sun=3.828e26#Watts
l_planck=3.62831e52#Watts

lb=0.45359237
mi=1609.344
minute=60
hour=3600
day=86400
year=31556952
jyear=31536000

mercury={'a':0.387098*au,'e':0.205630,'r':2439700,'m':3.3011e23}
venus={'a':0.723332*au,'e':0.006772,'r':6051800,'m':4.8675e24}
earth={'a':au,'e':0.0167086,'r':r_e,'m':m_e}
mars={'a':1.523679*au,'e':0.0934,'r':3389500,'m':6.4171e23}
jupiter={'a':5.2026*au,'e':0.048498,'r':r_j,'m':m_j}
saturn={'a':9.5549*au,'e':0.05555,'r':58232000,'m':5.6836e26}
uranus={'a':19.2184*au,'e':0.046381,'r':25362000,'m':8.681e25}
neptune={'a':30.110387*au,'e':0.009456,'r':24622000,'m':1.0243e26}
planetnine={'a':700*au,'e':0.6,'r':20000000,'m':6e25}
#hohmann(362600000,405400000,5.97237e24)
def hohmann(inner,outer,mass):
	mu=mass*g
	dv1=(mu/inner)**.5*((2*outer/(inner+outer))**.5-1)
	dv2=(mu/inner)**.5*(1-(2*inner/(inner+outer))**.5)
	return dv1+dv2
#bielliptic(362600000,405400000,4054000000,5.97237e24)
def bielliptic(inner,outer,mid,mass):
	mu=mass*g
	a1=(inner+mid)/2
	a2=(outer+mid)/2
	dv1=(2*mu/inner-mu/a1)**.5-(mu/inner)**.5
	dv2=(2*mu/mid-mu/a2)**.5-(2*mu/mid-mu/a1)**.5
	dv3=(2*mu/outer-mu/a2)**.5-(mu/outer)**.5
	return dv1+dv2+dv3
#biellipticinf(362600000,405400000,5.97237e24)
def biellipticinf(inner,outer,mass):
	mu=mass*g
	return (mu/inner)**.5*(2**.5-1)*(1+(inner/outer)**.5)
#bielliptictime(362600000,405400000,4054000000,5.97237e24)
def bielliptictime(inner,outer,mid,mass):
	mu=mass*g
	a1=(inner+mid)/2
	a2=(outer+mid)/2
	t1=pi*(a1**3/mu)**.5
	t2=pi*(a2**3/mu)**.5
	return t1+t2
	
def optimal(inner,outer,mass):
	if biellipticinf(inner,outer,mass)>hohmann(inner,outer,mass):return "Bi-Elliptic"
	return "Hohmann"
	
def inclinationburn(deltai,ecc,aop,trueanomaly,meanmotion,sma):
	return 2*sin(deltai/2)*(1-ecc**2)**.5*cos(aop+trueanomaly)*meanmotion*sma/(1+ecc*cos(trueanomaly))
	
def p(mass,sma):#period
	mu=g*mass
	return 2*pi*(sma**3/mu)**.5
#mov(1.98855e30,149598023000)
def mov(mass,sma):#mean orbital velocity
	mu=g*mass
	return (sma/mu)**-.5
	
def apsides(a,e):
	return (1-e)*a,(1+e)*a
	
def vextrema(a,e,mass):
	if e==0:return mov(mass,a)
	mu=g*mass
	v_peri=((1+e)*mu/(1-e)/a)**.5
	v_apo=((1-e)*mu/(1+e)/a)**.5
	return v_peri,v_apo
	
def apsides2ecc(apo,peri):
	return (apo+peri)/2,(apo-peri)/(apo+peri)
	
def density(m,r):
	return m/(4/3*pi*r**3)
	
def synchronous(t,m):
	mu=g*m
	return (mu*t**2/4/pi**2)**(1/3)
	
def soi(m,M,a):
	return a*(m/M)**.4
	
def speed(a,r,mass):
	mu=g*mass
	return (mu*(2/r-1/a))**.5
#based on dv/dr(speed)
def acc(a,r,mass):
	mu=g*mass
	return mu/(2*a**2*(mu*(2/r-1/a))**.5)
	
def surfacegravity(m,r):
	return g*m/r**2
	
def vescape(m,r):
	return (2*g*m/r)**.5
	
def vtan(t,r):#Equatorial rotation velocity
	return 2*pi*r/t
	
def synodic(p1,p2):#synodic period
	try:
		return p1*p2/(p2-p1)
	except:
		return inf
	
def star(mass):
	m=mass/m_sun
	return r_sun*m**.74/2,l_sun*m**3,5772*m**.505,12e9*365.2425*60*60*m**-2.5#Radius (m),Luminosity (solar lumin.),Temp (K),Lifespan (s)
	
def habitablezone(mass):
	m=mass/m_sun
	return au*.95*(star(mass)[1]/l_sun)**.5,au*1.37*(star(mass)[1]/l_sun)**.5
	
def temp(luminosity,albedo,a):
	return (luminosity*(1-albedo)/(16*pi*sigma*a**2))**.25
	
def absmag(appmag,distance):
	return appmag-5*(log(distance/pc)-1)
	
def peakwavelength(T):
	return 2.8977729e-3/T
	
def roche(M,density):
	return (9*M/4/pi/density)**(1/3)
	
def hill(m,M,a,e):
	return a*(1-e)*(m/3/M)**(1/3)
	
#NEW!
	
def gravbinding(m,r):#gravitational binding energy (in joules)
	return 3*g*m**2/5/r	

def gravbinding2(r,rho):#gravitational binding energy (in joules), using density for mass
	return g*pi**2*16*rho**2*r**5/15
	
def schwarzschild(m):#schwarzschild radius
	return 2*g*m/c**2
	
def energy(m):#rest energy
	return m*c**2

def mass(e):#relativistic mass
	return e/c**2

def orbitenergy(m,a):#Specific orbital energy
	return -m*g/2/a

#source for the formula: http://phl.upr.edu/projects/earth-similarity-index-esi
#esi(mars['r'],mars['m'],210)
def esi(r,m,T):#Radius,Density,Escape Velocity,Temperature
	esi1=1-abs((r-r_e)/(r+r_e))
	esi2=1-abs((density(m,r)-density(m_e,r_e))/(density(m,r)+density(m_e,r_e)))
	esi3=1-abs((vescape(m,r)-vescape(m_e,r_e))/(vescape(m,r)+vescape(m_e,r_e)))
	esi4=1-abs((T-255)/(T+255))
	return esi1**(.57/4)*esi2**(1.07/4)*esi3**(.7/4)*esi4**(5.58/4)
