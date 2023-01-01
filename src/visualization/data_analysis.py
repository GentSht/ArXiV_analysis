import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

year = [str(x+1) for x in range(2009,2022)]

df = pd.read_pickle("data/arxiv_id_total_citation_th.pkl")

df_year = pd.read_pickle('data/arxiv_id_citation_year_th.pkl')

df.columns = ['arxiv_id' if x=='ArXiV ID' else x for x in df.columns]


mean_citation_tot = df['Number of total citations'].mean()
print('Total citation mean :', mean_citation_tot)

mean_per_year = [(y,df_year[y].mean()) for y in year]
print(mean_per_year)

df_sorted = df.sort_values(by=['Number of total citations'], ascending=False)
df_test = df.merge(df_year, on='arxiv_id')

df_test.columns = ['Total' if x == 'Number of total citations' else x for x in df_test.columns]  

print(df_test.head())

df_test.to_pickle("data/final_hep_th.pkl")

cit_11_13 = df_year[['2011','2012','2013']].sum(axis=1).to_list()
print('Percentage of articles (2011-13) with number of citations bigger than average : ', 100*sum(i>mean_citation_tot for i in cit_11_13)/len(df_year))


cit_11_15 = df_year[['2011','2012','2013','2014','2015']].sum(axis=1).to_list()
print('Percentage of articles (2011-15) with number of citations bigger than average : ', 100*sum(i>mean_citation_tot for i in cit_11_15)/len(df_year))


col = df['Number of total citations']

count_infmean = 100*(np.count_nonzero(col <= mean_citation_tot))/len(col)
count_supmean_inf100 = 100*(np.count_nonzero((col>mean_citation_tot) & (col<100))/len(col))
count_sup100 = 100*(np.count_nonzero(col >= 100))/len(col)


print("Between 0 and 32 citations : ", count_infmean)
print("Between 32 and 100 citations : ", count_supmean_inf100)
print("Greater than 100 citations : ", count_sup100)


''' mean_per_year = [(y,df_year[y].mean()) for y in year]
print(mean_per_year)

plt.hist(df_year['2010'],density=True,bins=100,range=(0,300),histtype=u'step')

plt.xlabel('Number of citations')
plt.yscale("log")
plt.xlim(left=0)
plt.title("Distribution of the number of citations for hep-th articles in 2010")
#plt.show()
plt.savefig("data/hep_th_distribution_citation_2010.png") '''

''' df_sorted = df.sort_values(by=['Number of total citations'], ascending=False)
df_ysort = df_year.sort_values(by=['2010','2011','2012','2013'], ascending=False)
df_test = df.merge(df_year, on='arxiv_id')

plt.hist(df['Number of total citations'],density=True,bins=100,range=(0,300),histtype=u'step')

plt.xlabel('Number of citations')
plt.yscale("log")
plt.xlim(left=0)
plt.title("Distribution of the number of citations for hep-th articles (2010-15)")

plt.savefig("data/hep_th_distribution_citation.png") ''' 



