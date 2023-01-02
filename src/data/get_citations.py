import urllib, urllib.request
import feedparser
import time
import pandas as pd
import requests
from collections import defaultdict
import sys


ihep_search_arxiv = "https://inspirehep.net/api/arxiv/" #link to get the metadata for an arxiv id
ihep_search_article = "https://inspirehep.net/api/literature?sort=mostcited&size=500&page=1&q=refersto%3Arecid%3A" #link to get the citing papers and their metadata

year = [str(x+1) for x in range(2009,2022)]

def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def recollect_data(df):
    #collecting the arxiv ids that were harvested with get_data.py
    arxiv_id = df["id"].map(lambda x: x.partition("v")[0]).to_list()#for instance 1001.0034v22 becomes 1001.0034

    print('There are initially %i arXiV articles' % len(arxiv_id))

    for i,id in enumerate(arxiv_id):
        inspirehep_url_arxiv = f"{ihep_search_arxiv}{id}"
        try:
            data_arxiv = requests.get(inspirehep_url_arxiv).json()['metadata']
        except KeyError:
            arxiv_id.pop(i)
            print("The arXiV id %s with index %i gives an error. It has been removed." % (id,i))

        progress_bar(i,len(arxiv_id),'fetching arxiv ids')


    print("There are finally %i valid arXiV articles" % len(arxiv_id))


    return arxiv_id


def count_year(year, input_list):
    #Count the occurrences of each year in the citing articles.
    year_count = {}
    for y in year:
        if input_list[0] == 'NaN':
            year_count[y] = 0
        else:
            year_count[y] = input_list.count(y)

    return year_count


def get_number_cite():
    #get the total number citations for each arxiv id. Can be modified to look for only single count citations.
    citation_count = pd.DataFrame(columns=['arxiv_id','Number of total citations'])

    for i, id in enumerate(arxiv_id):

        inspirehep_url_arxiv = f"{ihep_search_arxiv}{id}"
        cc = {'arxiv_id':id,'Number of total citations': requests.get(inspirehep_url_arxiv).json()["metadata"]["citation_count"]}
        citation_count = citation_count.append(cc,True)
        progress_bar(i,len(arxiv_id),'fetching total citations')

    citation_count.to_pickle("data/arxiv_id_total_citation_th.pkl")

    return citation_count


def get_cnumber():
    #get the control number in order to search the article metadata from each arxiv id
    citation_url = []

    for id, article_id in enumerate(arxiv_id):

        inspirehep_url_arxiv = f"{ihep_search_arxiv}{article_id}"
        control_number = requests.get(inspirehep_url_arxiv).json()["metadata"]["control_number"]
        citation_url.append(f"{ihep_search_article}{control_number}")
        progress_bar(id,len(arxiv_id),'fetching urls')
    
    return citation_url


def get_citations():

    citation_url = get_cnumber()
    max_results = len(citation_url)
    citation_per_year = pd.DataFrame(columns=year)
    citation_date = defaultdict(list)

    for i, url in enumerate(citation_url):

        data_article = requests.get(url).json()
        if len(data_article["hits"]["hits"]) == 0:
            citation_date[i].append('NaN')
        else : 
            for j, _ in enumerate(data_article["hits"]["hits"]):
                citation_date[i].append(data_article["hits"]["hits"][j]["created"][:4]) #getting the year of publication of a citing article

        progress_bar(i,max_results,'fetching citations')

    for p, _ in enumerate(citation_date):
        citation_per_year = citation_per_year.append(count_year(year,citation_date[p]), True)

    citation_per_year.insert(0,"arxiv_id",arxiv_id,True)
    
    return citation_per_year



if __name__ == "__main__":

    df = pd.read_pickle(r"data/full_data_th_2010_2015_10k.pkl")

    arxiv_id = recollect_data(df)

    df_citations = get_citations()

    df_citations.to_pickle("data/arxiv_id_citation_year_th.pkl")

