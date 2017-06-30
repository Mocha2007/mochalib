from math import pi,sin
import cmath

#using mod
def square(x,freq,amp):
	return amp*(2*(x/freq//1%2)-1)
#using logsine
def square1(x,freq,amp):
	return amp*(2/pi*cmath.log(sin(2*x*pi/frequency)).imag-1)
#using abs and mod
def triangle(x,freq,amp):
	return amp*(abs(4*((x/freq)%1)-2)-1)
def sine(x,freq,amp):
	return amp*sin(2*pi*x/freq)
#using powers of i
def sine1(x,freq,amp):
	return (amp*1j**(4*x/freq)).imag
def sawtooth(x,freq,amp):
	return amp*(2*((x/freq)%1)-1)
