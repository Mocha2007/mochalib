from json import load
from io import StringIO
import urllib.request
from re import compile,sub,findall,search

#https://openweathermap.org/current

key = open("../openweathermap.txt", "r").read()

linkpattern = r'\[\[[^\]]+?\]\]'
limit = 499

user_agent = 'MochaMW/1.0 (https://github.com/Mocha2007/mochalib)'
headers={'User-Agent':user_agent}

def k2c(t):
	return t-273.16

def k2f(t):
	return 1.8*t-459.4

def m2mi(v):
	return v/1609.344*60*60

def dir2w(deg):
	if deg<22.5:return 'N'
	if deg<22.5+45:return 'NE'
	if deg<22.5+45*2:return 'E'
	if deg<22.5+45*3:return 'SE'
	if deg<22.5+45*4:return 'S'
	if deg<22.5+45*5:return 'SW'
	if deg<22.5+45*6:return 'W'
	if deg<22.5+45*7:return 'NW'
	return 'N'

def l(loc): #load
	url='http://api.openweathermap.org/data/2.5/weather?q='+loc+'&appid='+key
	request=urllib.request.Request(url,None,headers)
	webpage=urllib.request.urlopen(request).read().decode("utf-8")
	return load(StringIO(webpage))

def cleanup(j):
	state = j['weather'][0]['main']
	temp = round(k2c(j['main']['temp']),2)
	tempf = round(k2f(j['main']['temp']),2)
	cloudiness = ' ('+str(j['clouds']['all'])+'% cloudy)'
	dir = j['wind']['deg']
	dirw = dir2w(dir)
	wind = str(j['wind']['speed'])+' m/s ('+str(round(m2mi(j['wind']['speed']),1))+' mi/h) '+str(dir)+'\xb0 ('+dirw+')'
	return '```\n'+str(temp)+'\xb0C ('+str(tempf)+'\xb0F)\n'+state+cloudiness+'\nWind '+wind+'\n```'

def main(loc):
	# eg main('London')
	return cleanup(l(loc))