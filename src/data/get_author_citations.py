import requests
import pandas as pd
from progress.bar import IncrementalBar
from collections import defaultdict


ihep_search_arxiv = "https://inspirehep.net/api/arxiv/"
ihep_litterature_search = "https://inspirehep.net/api/literature/"
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
