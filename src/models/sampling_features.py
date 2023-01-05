import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys

try:
    start_year = sys.argv[1]
    end_year = sys.argv[2]
except IndexError:
    print("Error : you should specify the starting and ending year such as <script.py> <start> <end>")
    sys.exit(1)

def get_dataset():

    df_citation = pd.read_pickle(f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}.pkl")
    df_full = pd.read_pickle(f"data/full_data_th_{start_year}_{end_year}.pkl")

    df_full = df_full[['arxiv_id','created']]
    df_final = df_citation.merge(df_full,on='arxiv_id') #final dataframe with valid arxiv id, total citations and citations per year

    return df_final

def sampling(df_final):

    col = df_final["Total"].to_list()
    mean = sum(col)/len(col) #mean of the total number of ciations

    df_final['strat_category'] = pd.cut(df_final['Total'], bins=[0,mean,100,np.inf], labels=['A','B','C'],include_lowest=True,right=True) #defining classes
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=45) #unbalanced classifications need stratified sampling

    for train_index, test_index in split.split(df_final, df_final['strat_category']):
        strat_train_set = df_final.loc[train_index]
        strat_test_set = df_final.loc[test_index]

    return strat_train_set,strat_test_set

def get_feature_label(data):
    
    col = data["Total"].to_list()
    labels = data["strat_category"]
    ft = []

    for i, _ in enumerate(col):
        published = int(data.iloc[i]["created"])
        ft.append([data.iloc[i][str(published)]+data.iloc[i][str(published+1)],data.iloc[i][str(published+2)],data.iloc[i][str(published+3)]])

    features = pd.DataFrame(ft, columns=['1y','2y','3y'])
    
    return features, labels

def scaling(train,train_test):
    #maybe not necessary to scale data in our case
    scaler = StandardScaler()
    scaler.fit(train)
    train_test = scaler.transform(train_test)

    return train_test
    
if __name__ == "__main__":

    df_final = get_dataset()
    train_set, test_set = sampling(df_final)
    train_feature, train_label = get_feature_label(train_set.copy())
    test_feature, test_label = get_feature_label(test_set.copy())
