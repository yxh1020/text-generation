#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web scraper
@author: yuanhu
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup

url_base  = "https://papers.nips.cc"
urls_index = [(1987, "https://papers.nips.cc/book/neural-information-processing-systems-1987")]
for i in range(1, 32):
    year = i+1987
    urls_index.append((year, "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-%d-%d" % (i, year)))

# get urls for paper pages
urls_papers = []
for year, url_page in urls_index:
    response = requests.get(url_page)

    soup = BeautifulSoup(response.text, "html.parser")
        
    divs = soup.findAll('div', {'class', 'main-container'})
    for div in divs:
        uls = div.find_all('ul', attrs={'class': None})
        for ul in uls:
            for li in ul.find_all('li'):
                a = li.find('a')
                urls_papers.append((year, url_base + a['href']))

# get paper information
nips = []
for year, url_paper in urls_papers:
    response_paper = requests.get(url_paper)
    soup_paper = BeautifulSoup(response_paper.text, "html.parser")
    title = soup_paper.find('h2', attrs={'class': 'subtitle'}).contents[0]
    try: 
        abstract = soup_paper.find('p', attrs={'class': 'abstract'}).contents[0]
    except:
        print("No abstract!")
        abstract = None
    
    nips.append((year, title, abstract))

# save paper information in csv file
pd.DataFrame(nips, columns=["Year", "Title", "Abstract"]).to_csv("nips.csv", index=False)
