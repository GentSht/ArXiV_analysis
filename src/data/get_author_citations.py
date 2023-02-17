import requests
import pandas as pd
from progress.bar import IncrementalBar
from collections import defaultdict
import time


ihep_search_arxiv = "https://inspirehep.net/api/arxiv/"
ihep_litterature_search = "https://inspirehep.net/api/literature/"
ihep_author_articles = "https://inspirehep.net/api/literature?sort=mostrecent&size=500&q=a%20"
start_year = 2010
end_year = 2015

def get_cnumber():
    
    df = pd.read_pickle('data/arxiv_id_total_citation_year_th_2010_2015.pkl')
    arxiv_ID = df["arxiv_id"].to_list()
    bar = IncrementalBar('Getting the control numbers', suffix='%(percent)d%%', max=len(arxiv_ID))
    control_number = []
    
    for i,id in enumerate(arxiv_ID):
        inspirehep_url_arxiv = f"{ihep_search_arxiv}{id}"
        control_number.append(requests.get(inspirehep_url_arxiv).json()["metadata"]["control_number"])
        bar.next()

    bar.finish()

    id_cnumber = {'arxiv_id':arxiv_ID,'control_number':control_number}
    cnumber_df = pd.DataFrame(id_cnumber)
    print(cnumber_df.head())
    cnumber_df.to_pickle(f'data/arxiv_id_control_number_{start_year}_{end_year}.pkl')
    print(f'File data/arxiv_id_control_number_{start_year}_{end_year}.pkl has been created')

    return

def get_BAI():

    df = pd.read_pickle(f'data/arxiv_id_control_number_{start_year}_{end_year}.pkl')
    control_number = df["control_number"].to_list()
    arxiv_id = df["arxiv_id"].to_list()
    bar = IncrementalBar('Getting the author ids', suffix='%(percent)d%%', max=len(arxiv_id))

    author_dict = defaultdict(list)
    author_list = []

    for i, cnum in enumerate(control_number):
        link = f"{ihep_litterature_search}{cnum}"
        data_author = requests.get(link).json()["metadata"]["authors"]
        for _ , author in enumerate(data_author):
            try:
                for _ , el in enumerate(author["ids"]):
                    if not el["schema"] == 'INSPIRE BAI': continue
                    author_dict[i].append(el["value"])
            except KeyError:
                pass
        bar.next()
    bar.finish()

    for j,_ in enumerate(author_dict):
        author_list.append(author_dict[j])

    author_df = pd.DataFrame({'arxiv_id':arxiv_id,'authors':author_list})
    print(author_df)

    author_df.to_pickle(f'data/arxiv_id_author_ids_{start_year}_{end_year}.pkl')
    print(f'File data/arxiv_id_author_ids_{start_year}_{end_year}.pkl has been created')
    
    return

def get_citations_authors(BAI, c_number, year):

    url = ihep_author_articles+BAI
    data_articles = requests.get(url).json()["hits"]

    total = len(data_articles["hits"])
    count_dist = 0

    for i in range(total):
        if data_articles["hits"][i]["id"] != c_number and data_articles["hits"][i]["created"] <= year:
            count_dist += 1

    return count_dist

def prRed(skk): print("\033[91m {}\033[00m" .format(skk))


#WARNING : SOME IDENTIFIERS ARE NOT WELL DEFINED. RETRIEVING ALL THE AUTHORS CITATIONS CAN TAKE UP TO 10 HOURS. BETTER TO SPLIT THE QUERY IN 1000 SMALL ONES.

def create_table_author_citation(split):
    
    year = df_year["created"].to_list()
    authors = df_author["authors"].to_list()
    arxiv_id = df_author["arxiv_id"].to_list()
    c_number = df_control["control_number"].to_list()
    author_citations = [] 
    
    for i, article in enumerate(authors[:split]):
        dict_author = {}
        for _,auth in enumerate(article):
            try:
                citation = get_citations_authors(auth,c_number[i],year[i])
                dict_author[auth]=citation
            except Exception as F:
                dict_author[auth]='NaN'
                prRed(f"{auth} link didn't work")
                
        author_citations.append(dict_author)
    
    print(author_citations)

    df_final = pd.DataFrame({"arxiv_id":arxiv_id[:10],"created":year[:10],"author_citations":author_citations}) 
    print(df_final)      
    df_final.to_pickle(f"data/arxiv_id_author_citations_{start_year}_{end_year}.pkl") 
    return

if __name__ == "__main__":
    df_author = pd.read_pickle(f"data/arxiv_id_author_ids_{start_year}_{end_year}.pkl")
    df_year = pd.read_pickle(f"data/full_data_th_{start_year}_{end_year}.pkl")
    df_control = pd.read_pickle(f"data/arxiv_id_control_number_{start_year}_{end_year}.pkl")

    create_table_author_citation(1000)  
