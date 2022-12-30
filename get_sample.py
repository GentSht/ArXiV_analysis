import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def get_dataset():

    df_citation = pd.read_pickle("d:/genti/Desktop/datasets/arxiv dataset/final_hep_th.pkl")

    df_full = pd.read_pickle("d:/genti/Desktop/datasets/arxiv dataset/full_data_th_2010_2015_10k.pkl")

    df_full["id"] = df_full["id"].map(lambda x: x.partition("v")[0])
    df_full["created"] = df_full["created"].map(lambda x: x[:4])
    df_full.columns = ['arxiv_id' if x=='id' else x for x in df_full.columns]

    df_full = df_full[['arxiv_id','created']]

    df_final = df_citation.merge(df_full,on='arxiv_id')

    return df_final


def get_feature_label(data):

    col = data["Total"].to_list()
    mean = sum(col)/len(col)

    labels = data["strat_category"]

    ft = []

    for i, _ in enumerate(col):

        published = int(data.iloc[i]["created"])
        ft.append([data.iloc[i][str(published)]+data.iloc[i][str(published+1)],data.iloc[i][str(published+2)],data.iloc[i][str(published+3)]])

    features = pd.DataFrame(ft, columns=['1y','2y','3y'])
    
    return features, labels


def sampling():

    col = df_final["Total"].to_list()
    mean = sum(col)/len(col)

    df_final['strat_category'] = pd.cut(df_final['Total'], bins=[0,mean,100,np.inf], labels=['A','B','C'],include_lowest=True,right=True)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=45)

    for train_index, test_index in split.split(df_final, df_final['strat_category']):
        strat_train_set = df_final.loc[train_index]
        strat_test_set = df_final.loc[test_index]

    return strat_train_set,strat_test_set



df_final = get_dataset()
train_set, test_set = sampling()
train_feature, train_label = get_feature_label(train_set.copy())
test_feature, test_label = get_feature_label(test_set.copy())

LM_model = LogisticRegression(penalty='l2', max_iter=500, multi_class='ovr', solver='liblinear')
LM_model.fit(train_feature,train_label)

LM_pred_prob = pd.DataFrame(LM_model.predict_proba(test_feature))
LM_pred = pd.DataFrame(LM_model.predict(test_feature))

LM_final = pd.concat([test_label.reset_index()['strat_category'],LM_pred_prob,LM_pred],axis=1, ignore_index=True)
LM_final.columns = ['Actual','A','B','C','Predicted']

print(LM_final.head())
print(LM_final.tail())

a = f1_score(LM_final['Actual'],LM_final['Predicted'],average='micro')
print(a*100)
