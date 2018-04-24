from json import load
from io import StringIO
import urllib.request,telnetlib
from re import compile,sub,findall,search,MULTILINE

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

huracan = telnetlib.Telnet(host='rainmaker.wunderground.com')
def hurricane(x):
	try:
		x = int(x)
		if not 0<x<6:x=1
		x = str(x)
	except:x='1'
	huracan.read_until(b':',timeout=1) # PRESS RETURN TO CONTINUE
	huracan.write(b'\n')
	huracan.read_until(b'-- ',timeout=1) # PRESS RETURN FOR MENU
	huracan.write(b'\n')
	huracan.read_until(b'n:',timeout=1) # selection:
	huracan.write(b'8\n')
	huracan.read_until(b':',timeout=1) # selection:
	huracan.write(x.encode('ascii')+b'\n')

	y = huracan.read_until(b'$$',timeout=1).decode('ascii')[:-2]
	if x in '45':
		y = sub(r'^[\w\W]+Rmks\/\n+','',y)
		y = sub(r'\n+[^\n]+exit$','',y)
	return '```\n'+y+'\n```'

th = telnetlib.Telnet(host='telehack.com')
def phoon():
	th.read_until(b'\n.',timeout=1)
	th.write(b'phoon\r\n')

	x = th.read_until(b'\n.',timeout=1)
	return '```\n'+x.decode('ascii')[7:-1]+'\n```'

def quake(): #load
	url='https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.csv'
	request=urllib.request.Request(url,None,headers)
	csv=urllib.request.urlopen(request).read().decode('ascii')
	csv=sub(compile(r'(,[^,]*){8}$',MULTILINE),'',csv)
	csv=sub(r',(?! )','\t',csv).replace('"','')
	csv=sub(r'\tmagType\tnst\tgap\tdmin\trms\tnet\tid\tupdated|\t[Mm][bdeiLlsw].+(?=\t\d+km)','\t',csv) # del magType -> updated
	return '```\n'+csv[:1992]+'\n```'