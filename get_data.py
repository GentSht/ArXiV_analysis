import urllib, urllib.request
import feedparser
import time
import pandas as pd


def get_data():

    base_url = 'http://export.arxiv.org/api/query?'

    search_query = 'cat:hep-ex+AND+submittedDate:[201001010000+TO+201501010000]'
    start = 0
    results_per_iteration = 5
    wait_time = 5                 
    max_results = 10

    feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
    feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

    df = pd.DataFrame(columns=("title", "abstract", "categories", "created", "id", "doi"))

    for i in range(start,max_results,results_per_iteration):

        query = 'search_query=%s&start=%i&max_results=%i' % (search_query,i,results_per_iteration)

        response = urllib.request.urlopen(base_url+query).read()
        feed = feedparser.parse(response)

        for entry in feed.entries:
            
            try:
                doi = entry.arxiv_doi
            except AttributeError:
                doi = 'No doi'

            contents = {'title': entry.title,
                        'id': entry.id.split('/abs/')[-1],
                        'abstract': entry.summary,
                        'created': entry.published,
                        'categories': entry.tags[0]['term'],
                        'doi': doi,
                        }

            df = df.append(contents, ignore_index=True)

        print ('Sleeping for %i seconds' % wait_time)
        time.sleep(wait_time)


    return df

df = get_data()
print(df.tail())

df.to_pickle("d:/genti/Desktop/datasets/arxiv dataset/my_data.pkl")