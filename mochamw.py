from json import load
from io import StringIO
import urllib.request
from re import compile,sub,findall,search

linkpattern = r'\[\[[^\]]+?\]\]'
limit = 499

user_agent = 'MochaMW/1.0 (https://github.com/Mocha2007/mochalib)'
headers={'User-Agent':user_agent,}

def l(site,page): #load
	url='http://'+site+'/api.php?action=query&titles='+page+'&prop=revisions&rvprop=content&format=json&formatversion=2'
	request=urllib.request.Request(url,None,headers)
	webpage=urllib.request.urlopen(request).read().decode("utf-8")
	return load(StringIO(webpage))

def read(site,page):
	return l(site,page)['query']['pages'][0]['revisions'][0]['content']

def read2(site,page):
	t = l(site,page)['query']['pages']
	for i in t:
		v = i
		break
	return t[v]['revisions'][0]['*']

def deltemplates(string):
	sublimit = 3 # should be enough
	for i in range(sublimit):
		string = sub(compile(r'{{[^{}]+?}}'),'',string)
	return sub(compile(r'{{[\w\W]+?}}'),'',string)

def link2text(string):
	if search(r'\[\[File:[^\]]+?]]',string):return '' # file
	if search(r'\[\[[^\]|]+?]]',string):return string [2:-2] # no alt text
	try:return search(r'\|[^\]|]+?]]',string).group(0)[1:-2] # alt text
	except:return ''

def cleanup(string):
	string = sub(compile(r'<!--[^>]+?-->'),'',string) # remove comments
	string = string.replace('{&nbsp;}',' ')
	string = deltemplates(string)
	string = string.replace("'''",'**')
	string = string.replace("''",'*')
	string = sub(r'\n+','\n',string)
	string = sub(r'<ref>[^>]*<\/ref>','',string)#refs
	for i in findall(compile(linkpattern),string):
		alt = link2text(i)
		string = string.replace(i,alt)
	return string

def main(site,page):
	# eg main('dwarffortresswiki.org','DF2014:Pig')
	if site == 'en.wiktionary.org/w':return cleanup(read(site,page))#[:1997]+'...'
	return cleanup(read(site,page))[:limit]+'...'

def main2(site,page):
	return cleanup(read2(site,page))[:limit]+'...'