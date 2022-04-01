from base64 import encode
import urllib.request
from os import mkdir
from os.path import isdir, isfile
from re import findall, match
from typing import List

user_agent = 'MochaWeb/1.0 (https://github.com/Mocha2007/mochalib)'
headers = {'User-Agent': user_agent}
scraper_dir = "scraper"

def get_filename_from_url(url: str) -> str:
	destination = match(r"(?<=\/)[^\/]+$", url)
	return f"{scraper_dir}/{destination}"

def get_files(source: str) -> List[str]:
	return findall(r"(?<=src=([\"']))[^\"']+?(?=\1)", source)

def get_source(url: str) -> str: # load from mochaweb.py
	request = urllib.request.Request(url, None, headers)
	webpage = urllib.request.urlopen(request).read().decode("utf-8", errors='ignore')
	return webpage

def get_urls(source: str) -> List[str]:
	return findall(r"(?<=href=([\"']))[^\"']+?(?=\1)", source)

def normalize_url(url: str) -> str:
	if match(f"^https?:\/\/", url):
		return # already good
	return f"https://{url}"

def save_url(url: str, src: str) -> bool:
	destination = get_filename_from_url(url)
	if isfile(destination): # don't overwrite existing files
		return False
	if not isdir(scraper_dir): # create directory if it doesn't already exist
		mkdir(scraper_dir)
	with open(destination, "w") as file:
		print(f"saving {destination}...")
		file.write(src, encode='utf-8', errors='ignore')
	return True

def scrape_domain(domain: str, url: str = None) -> None:
	"""Main method
	sample usage: scrape_domain("zompist.com")
	"""
	print(f"scraping {domain} @ {url}...")
	# normalize URL
	if url is None:
		url = domain
	url = normalize_url(url)
	# continue
	source = get_source(url)
	if not save_url(url, source):
		return # ignore duplicates
	# get files
	for src in get_files(source):
		urllib.request.urlretrieve(src, get_filename_from_url(src))
	# get links
	for iurl in get_urls(source):
		if "http" not in iurl:
			scrape_domain(domain, f"{domain}/{iurl}")
		elif domain in iurl:
			scrape_domain(domain, iurl)
	print(f"finished scraping outlets from {url}")