import urllib, urllib.request
import feedparser
import time
import pandas as pd
import sys

try:
    start_year = sys.argv[1]
    end_year = sys.argv[2]
except IndexError:
    print("Error : you should specify the starting and ending year such as <script.py> <start> <end>")
    sys.exit(1)

def get_data():

    print(f"Querying arxiv (hep-th) articles between {start_year} and {end_year}")

    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'cat:hep-th+AND+submittedDate:[{start_year}01010000+TO+{end_year}01010000]'
    start = 0
    results_per_iteration = 1000
    wait_time = 5                 
    max_results = 10000
    total_steps = max_results/results_per_iteration

    feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
    feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

    df = pd.DataFrame(columns=("title", "abstract", "categories", "created", "arxiv_id", "doi"))

    for i in range(start,max_results,results_per_iteration):

        step = i/results_per_iteration
        query = 'search_query=%s&start=%i&max_results=%i' % (search_query,i,results_per_iteration)
        response = urllib.request.urlopen(base_url+query).read()
        feed = feedparser.parse(response)

        for entry in feed.entries:
            try:
                doi = entry.arxiv_doi
            except AttributeError:
                doi = 'No doi'
            contents = {'title': entry.title,
                        'arxiv_id': entry.id.split('/abs/')[-1].partition("v")[0],
                        'abstract': entry.summary,
                        'created': entry.published[:4],
                        'categories': entry.tags[0]['term'],
                        'doi': doi,
                        }
            df = df.append(contents, ignore_index=True)

        print(f"Finishing round {step} over {total_steps}")
        print ('Sleeping for %i seconds' % wait_time)
        time.sleep(wait_time)

    return df

if __name__ == "__main__":
    df = get_data()
    print(df.head())
    file_path = f"data/full_data_th_{start_year}_{end_year}.pkl"
    df.to_pickle(file_path)
    print("File {file_path} has been created")
        