import urllib.request,mochamw,telnetlib
from re import compile,sub,findall,search,M
from random import randint
from json import load
from io import StringIO

linkpattern = r'\[\[[^\]]+?\]\]'
limit = 499

user_agent = 'MochaWeb/1.0 (https://github.com/Mocha2007/mochalib)'
headers={'User-Agent':user_agent,}

def l(url): #load
	request=urllib.request.Request(url,None,headers)
	webpage=urllib.request.urlopen(request).read().decode("utf-8")
	return webpage

def udcleanup(string):
	string = search(compile(r'(?<=class="meaning">)[^\0]+?(?=<\/div>)'),string).group(0) # get first def
	string = string.replace('<br/>','\n') # newline
	string = string.replace('&quot;','"') # quote
	string = sub(compile(r'<.+?>'),'',string)#.group(0) # remove links
	return string

def wtcleanup(string):
	string = sub(r'----[\w\W]+','',string) # crap in end
	string = sub('^[^#].+$','',string,flags=M) # delete ANY line not beginning with a hash
	string = string.replace('#','\n#') # ol
	string = sub('^[^#]+','',string) # crap in beginning
	string = sub('#[:*].+','',string) # crap in middle
	string = sub(r'\n{2,}','\n',string) # multiple returns
	string = sub(r'Category:[\w:]+','',string) # category removal
	string = sub(r'^[#*][^\w\n]+\n','',string) # empty defs
	n = 0
	for c in string:# start numbering the definitions
		if c=='#':
			n+=1
			string = string.replace(c,'**'+str(n)+'.**',1)
	return string[:2000]

def wikicleanup(string):
	string = sub(r'\s?\(.*?\)','',string) # text in parens
	return ''.join(findall('[^.]+.',string)[:3]) # first three sentences

def ud(word):
	# eg ud('test')
	return udcleanup(l('https://www.urbandictionary.com/define.php?term='+word))

def dfprop(word):
	art = mochamw.read2('dwarffortresswiki.org','DF2014:'+word)
	temps = findall(compile(r'{{ct\|\d+}}'),art)
	for m in temps: # template ct
		art = art.replace(m,sub(r'\D','',m)+'\xb0U')
	props = search(compile(r'(?<=properties=\n)[\w\W]+?(?=}}{{av}})'),art).group(0)
	return mochamw.cleanup(props)

def xkcdcleanup(string):
	string = string.replace('&#39;','\'').replace('&quot;','"')
	title = search(r'(?<=ctitle">)[^<]+?(?=<)',string).group(0)
	img = 'https:'+search(r'(?<=src=")[^"]+?(?=" t)',string).group(0)
	alt = search(r'(?<=title=")[^"]+?(?=" a)',string).group(0)
	return '**'+title+'**\n*'+alt+'*\n'+img

def xkcd(arg):
	try:arg = 'https://xkcd.com/'+str(int(arg))
	except:arg = 'https://c.xkcd.com/random/comic/'
	return xkcdcleanup(l(arg))

def numbersapi(n):
	x = l('http://numbersapi.com/'+n)
	if 'numbersapi' in x:return n+' is a gay-ass number.'
	return x

htn = telnetlib.Telnet(host='horizons.jpl.nasa.gov',port=6775)
def horizons(name):
	htn.read_until(b'Horizons> ',timeout=1)
	htn.write(name.encode('ascii')+b'\n')
	x = htn.read_until(b'Horizons> ',timeout=1)
	if b'<cr>=yes' in x:
		htn.write(b'\n')
		x = htn.read_until(b'Select ...',timeout=1)
	return sub(r'^[^*]*\*+|\*+[^*]*$','',x.decode('ascii'))

fixerioapikey = open('../fixer.io.txt','r').read()
j = False
def currency(a,b):
	global j
	if not j:j = load(StringIO(l('http://data.fixer.io/api/latest?access_key='+fixerioapikey)))

	if a != 'EUR':a = j['rates'][a]
	else:a = 1

	if b != 'EUR':b = j['rates'][b]
	else:b = 1
	
	return b/a