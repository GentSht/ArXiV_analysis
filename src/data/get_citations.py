import pandas as pd
import requests
from collections import defaultdict
import sys
import time
from get_data import *


ihep_search_arxiv = "https://inspirehep.net/api/arxiv/" #link to get the metadata for an arxiv id
ihep_search_article = "https://inspirehep.net/api/literature?sort=mostcited&size=500&page=1&q=refersto%3Arecid%3A" #link to get the citing papers and their metadata

year = [str(x+1) for x in range(int(start_year)-1,2022)]

def progress_bar(count, total, status=''):
    #just a progress bar to know where we are at
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def recollect_data(df):
    #collecting the arxiv ids that were harvested with get_data.py
    arxiv_ID = df["arxiv_id"].to_list()

    arxiv_id = []
    citation_url = []

    print("Harvesting now in the InspireHep database")
    print('There are initially %i arXiV articles' % len(arxiv_ID))
    #to my knowledge, it is not possible to get a link to the citations with the inspireHep API with only the arxiv id
    for i,id in enumerate(arxiv_ID):
        inspirehep_url_arxiv = f"{ihep_search_arxiv}{id}"
        try:
            control_number = requests.get(inspirehep_url_arxiv).json()["metadata"]["control_number"]#important number to search for the citations
            citation_url.append(f"{ihep_search_article}{control_number}")
            arxiv_id.append(id)
        except KeyError:
            print("The arXiV id %s with index %i gives an error. It has been removed." % (id,i))#some arxiv ids can't be found in inspireHep
        
        progress_bar(i,len(arxiv_ID),'fetching arxiv ids')

    print("There are finally %i valid arXiV articles" % len(arxiv_id))

    valid_links = {'arxiv_id':arxiv_id,'citation_link':citation_url}
    link_df = pd.DataFrame(valid_links)
    link_df.to_pickle(f'data/arxiv_id_citation_links_{start_year}_{end_year}.pkl')
    print(f'File data/arxiv_id_citation_links_{start_year}_{end_year}.pkl has been created')   

    return arxiv_id, citation_url

def count_year(year, input_list):
    #Count the occurrences of each year in the citing articles.
    year_count = {}
    for y in year:
        if input_list[0] == 'NaN':
            year_count[y] = 0
        else:
            year_count[y] = input_list.count(y)

    return year_count

def get_citations(arxiv_id, citation_url):

    max_results = len(citation_url)
    citation_per_year = pd.DataFrame(columns=year)
    citation_date = defaultdict(list)
    citation_tot = []

    for i, url in enumerate(citation_url):

        data_article = requests.get(url).json()
        citation_tot.append(data_article["hits"]["total"])
        if len(data_article["hits"]["hits"]) == 0:
            citation_date[i].append('NaN')#some articles have zero citations, otherwise no index in dictionnary->probably a smarter way to do this...
        else : 
            for j, _ in enumerate(data_article["hits"]["hits"]):
                citation_date[i].append(data_article["hits"]["hits"][j]["created"][:4]) #getting the year of publication of a citing article

        
        progress_bar(i,max_results,'fetching citations')

    for p, _ in enumerate(citation_date):
        citation_per_year = citation_per_year.append(count_year(year,citation_date[p]), True)

    citation_per_year.insert(0,"arxiv_id",arxiv_id,True)
    citation_per_year.insert(1,"Total",citation_tot,True)
    
    return citation_per_year

def partition(arxiv_id, citation_url):
    #just a function to partition the dataset into smaller sets to not overload the API
    rd = (round(len(arxiv_id)/10**3))*10**3
    rest = len(arxiv_id)

    for i in range(0,rd,1000):
        a = i
        b = i+10**3
        df_citations = get_citations(arxiv_id[a:b], citation_url[a:b])
        print(df_citations.head())
        print("The size of the dataframe is : ", df_citations.shape)
        file_path = f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}_{a}_{b}.pkl"
        df_citations.to_pickle(file_path)
        print(f"File {file_path} has been created")

    df_citations = get_citations(arxiv_id[rd:rest], citation_url[rd:rest])
    print(df_citations.head())
    print("The size of the dataframe is : ", df_citations.shape)
    file_path = f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}_{rd}_{rest}.pkl"
    df_citations.to_pickle(file_path)
    print(f"File {file_path} has been created")

def merge():
    #merge all the dataframes created in the partition process
    rd = (round(len(arxiv_id)/10**3))*10**3
    rest = len(arxiv_id)
    df = pd.read_pickle(f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}_0_1000.pkl")
    for i in range(1000,rd,1000):
        j = i+10**3
        df_p = pd.read_pickle(f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}_{i}_{j}.pkl")
        df = pd.concat([df,df_p],ignore_index=True)
    df_rest = pd.read_pickle(f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}_{rd}_{rest}.pkl")
    df = pd.concat([df,df_rest],ignore_index=True)
    
    df.to_pickle(f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}.pkl")
    print("File data/arxiv_id_citation_year_th.pkl has been created. See structure below:")
    print(df.head())
    print(df.shape)


if __name__ == "__main__":

    df = pd.read_pickle(f"data/full_data_th_{start_year}_{end_year}.pkl")
    arxiv_id, citation_url = recollect_data(df)
    
    if len(arxiv_id)>1000: #we expect at least 1000 articles to be scraped
        partition(arxiv_id,citation_url)
        merge()
    else:
        print("Not enough data. Please change the parameters")



