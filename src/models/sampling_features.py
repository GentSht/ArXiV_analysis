import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys

try:
    start_year = sys.argv[1]
    end_year = sys.argv[2]
    met = sys.argv[3]
except IndexError:
    print("Error : you should specify the starting and ending year such as <script.py> <start> <end> <features>")
    print("For <features> you have two choices : author or no_author")
    sys.exit(1)

def filtering_nan_authors(df):
    
    print("The length of the dataframe with all the authors : ", len(df))
    authors = df.author_citations.to_list()
    ind = []
    for i,auth in enumerate(authors):
        if 'NaN' in auth.values():
            ind.append(i)
    
    df = df.drop(ind).reset_index(drop=True)
    print("Dropping the rows with 'NaN' citations : ", len(df))
    return df

def get_dataset_no_author():

    df_citation = pd.read_pickle(f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}.pkl")
    df_full = pd.read_pickle(f"data/full_data_th_{start_year}_{end_year}.pkl")
    df_no_author = df_citation.merge(df_full[['arxiv_id','created']],on='arxiv_id')#final dataframe with valid arxiv id, total citations, 
    #citations per year and creation date
    
    return df_no_author

def get_dataset_author():

    df_citation = pd.read_pickle(f"data/arxiv_id_total_citation_year_th_{start_year}_{end_year}.pkl")
    df_authors = pd.read_pickle(f"data/arxiv_id_author_citations_{start_year}_{end_year}.pkl")
    df_full = pd.read_pickle(f"data/full_data_th_{start_year}_{end_year}.pkl")
    df_int = df_citation.merge(df_full[['arxiv_id','created']],on='arxiv_id')
    df_with_authors = df_int.merge(df_authors[['arxiv_id','author_citations']], on='arxiv_id')#same dataframe but with the author's number of citations before 
    #the first upload of the article

    df_with_authors = filtering_nan_authors(df_with_authors)

    return df_with_authors

def sampling(df_final):

    mean = df_final['Total'].mean()

    df_final['strat_category'] = pd.cut(df_final['Total'], bins=[0,mean,100,np.inf], labels=['A','B','C'],include_lowest=True,right=True) #defining 3 classes
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=45) #unbalanced classifications need stratified sampling

    #creating the training and test sets.
    for train_index, test_index in split.split(df_final, df_final['strat_category']):
        strat_train_set = df_final.loc[train_index]
        strat_test_set = df_final.loc[test_index]

    return strat_train_set,strat_test_set

def get_feature_label(data):
    
    col = data["Total"].to_list()
    labels = data["strat_category"] #the three categories
    ft = []

    for i, _ in enumerate(col):
        published = int(data.iloc[i]["created"])
        ft.append([data.iloc[i][str(published)]+data.iloc[i][str(published+1)],data.iloc[i][str(published+2)],data.iloc[i][str(published+3)]])
        #defining features as the number of citations each year the article is published (0+1,2,3)                                                             

    features = pd.DataFrame(ft, columns=['1y','2y','3y'])
    
    return features, labels

def get_author_features(data):

    #considering the average of author's citations
    col = data["Total"].to_list()
    labels = data["strat_category"] #the three categories
    ft = []

    for i, _ in enumerate(col):
        a = list(data.iloc[i]["author_citations"].values())
        published = int(data.iloc[i]["created"])
        ft.append([data.iloc[i][str(published)]+data.iloc[i][str(published+1)],data.iloc[i][str(published+2)],data.iloc[i][str(published+3)],mean(a)])

    features = pd.DataFrame(ft, columns=['1y','2y','3y','mean_author_citations'])

    return features,labels

def scaling(train,train_test):
    #maybe not necessary to scale data in our case
    scaler = StandardScaler()
    scaler.fit(train)
    train_test = scaler.transform(train_test)

    return train_test

def plot_labels(folder_path,df_final,train_label,test_label):

    #plot the fraction of categories in the training and test sets.
    mean = int(df_final['Total'].mean())
    category = ['A','B','C']
    abs = [f'[0,{mean})',f'[{mean},100]','> 100 citations']
    data_train = []
    data_test = []

    for i,cat in enumerate(category):
        data_train.append(train_label.value_counts()[cat])
        data_test.append(test_label.value_counts()[cat])

    len_train = len(train_label)
    len_test = len(test_label)

    df_bar = pd.DataFrame({f'Training set ({len_train} samples)':data_train,f'Test set ({len_test} samples)':data_test},index=abs)
    ax = df_bar.plot.bar(rot=0, subplots=True,legend=False,color={f'Training set ({len_train} samples)':"blue",f'Test set ({len_test} samples)':"red"})
    
    plt.savefig(f"{folder_path}figures/train_test_stratification_{start_year}_{end_year}_{met}.png")

    train_label.to_excel(f"{folder_path}train_set_{start_year}_{end_year}_{met}.xlsx")
    test_label.to_excel(f"{folder_path}test_set_{start_year}_{end_year}_{met}.xlsx")

if __name__ == "__main__":

    #6.1% 'NaN' author values in final dataset. Don't consider these articles. Dataset will be smaller compared to other method.

    if met == "author":
        df_final_author = get_dataset_author()
        train_set, test_set = sampling(df_final_author)
        train_feature, train_label = get_author_features(train_set.copy())
        test_feature, test_label = get_author_features(test_set.copy())
        plot_labels("reports/",df_final_author,train_label,test_label)
    else:
        df_final_no_author = get_dataset_no_author()
        train_set, test_set = sampling(df_final_no_author)
        train_feature, train_label = get_feature_label(train_set.copy())
        test_feature, test_label = get_feature_label(test_set.copy())
        plot_labels("reports/",df_final_no_author,train_label,test_label)