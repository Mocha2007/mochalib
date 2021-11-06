from os import replace
from re import findall, match, sub
from typing import Dict, Set
from urllib.error import HTTPError
from mochaweb import l as load

def fixlink(url: str, domain: str) -> str:
	# trim hash
	url = sub(r'#.+', '', url)
	# already has http+dom?
	if url[:4] == 'http':
		return url
	# add domain
	url = url if domain in url else \
		domain + ('' if domain[-1] == '/' else  '/') + url
	# add http
	return 'http://' + url

def getURLs(s: str) -> Set[str]:
	# need the stupid g[0] nonsense because python things groups and matches are the same thing
	return set(g[0] for g in findall(r'((?<=href=([\'"])).+?(?=\2))', s))

def linksTo(url: str, domain: str) -> Set[str]:
	return set(filter(lambda u: not match(r'https?://', u) or domain in u,
		set(fixlink(uwu, domain) for uwu in getURLs(load(url).lower()))))

def sitemap(start: str, domain: str, max_depth: int = 3, ignore: Set[str] = set()) -> Dict[str, Set[str]]:
	print(f'loading (depth {max_depth}) {start} ...')
	try:
		urlmap = {start: linksTo(start, domain)}
	except (HTTPError, UnicodeEncodeError) as e: # 404 or invalid encoding
		print(f'caught {e} @ {start}')
		return {start: set()}
	if max_depth == 0:
		return urlmap
	max_depth -= 1
	for url in urlmap[start]:
		if url in urlmap or url in ignore:
			continue
		for u, uu in sitemap(url, domain, max_depth, set(urlmap)|ignore).items():
			if u in urlmap:
				continue
			urlmap[u] = uu
	return urlmap

def test():
	global verd, vs
	verd = 'http://www.zompist.com/virtuver.htm'
	vs = load(verd).lower()
	# return getURLs(vs)
	return sitemap(verd, 'zompist.com')
