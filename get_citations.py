import urllib, urllib.request
import feedparser
import time
import pandas as pd
import requests
from collections import defaultdict


ihep_search_arxiv = "https://inspirehep.net/api/arxiv/"
ihep_search_article = "https://inspirehep.net/api/literature?sort=mostcited&size=150&page=1&q=refersto%3Arecid%3A"

year = [str(x+1) for x in range(2009,2022)]



def recollect_data():

    df = pd.read_pickle(r"d:/genti/Desktop/datasets/arxiv dataset/my_data.pkl")

    arxiv_id = df["id"].map(lambda x: str(x)[:-2]).to_list()
    
    return arxiv_id


def count_year(year, input_list):
    
    year_count = {}
    for y in year:
        if input_list[0] == 'NaN':
            year_count[y] = 0
        else:
            year_count[y] = input_list.count(y)

    return year_count


    

def get_number_cite():


    citation_count = []

    for id in arxiv_id:

        inspirehep_url_arxiv = f"{ihep_search_arxiv}{id}"

        citation_count.append(requests.get(inspirehep_url_arxiv).json()["metadata"]["citation_count"])

    return citation_count


def get_cnumber():


    citation_url = []

    for id in arxiv_id:

        inspirehep_url_arxiv = f"{ihep_search_arxiv}{id}"

        control_number = requests.get(inspirehep_url_arxiv).json()["metadata"]["control_number"]
        
        citation_url.append(f"{ihep_search_article}{control_number}")


    return citation_url


def get_citations():

    citation_url = get_cnumber()

    citation_per_year = pd.DataFrame(columns=year)

    citation_date = defaultdict(list)

    for i, url in enumerate(citation_url):

        data_article = requests.get(url).json()

        if len(data_article["hits"]["hits"]) == 0:

            citation_date[i].append('NaN')

        else : 

            for j, _ in enumerate(data_article["hits"]["hits"]):

                citation_date[i].append(data_article["hits"]["hits"][j]["created"][:4])


    for p, article in enumerate(citation_date):
        citation_per_year = citation_per_year.append(count_year(year,citation_date[p]), True)


    citation_per_year.insert(0,"arxiv_id",arxiv_id,True)
    

    return citation_per_year


arxiv_id = recollect_data()

print(get_citations())

