import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import metrics


def get_dataset():

    df_citation = pd.read_pickle("data/final_hep_th.pkl")

    df_full = pd.read_pickle("data/full_data_th_2010_2015_10k.pkl")

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


def log_class():

    LM_model = LogisticRegression(penalty='l1', max_iter=5000, multi_class='ovr', solver='saga')
    LM_model.fit(train_feature,train_label)

    LM_pred_prob = pd.DataFrame(LM_model.predict_proba(test_feature))
    LM_pred = pd.DataFrame(LM_model.predict(test_feature))

    LM_final = pd.concat([test_label.reset_index()['strat_category'],LM_pred_prob,LM_pred],axis=1, ignore_index=True)
    LM_final.columns = ['Actual','A','B','C','Predicted']

    print(LM_final.head())
    print(LM_final.tail())

    a = f1_score(LM_final['Actual'],LM_final['Predicted'],average='micro')
    print(a*100)


def decision_tree():

    dt_model = DecisionTreeClassifier(random_state=0).fit(train_feature,train_label)

    pred_prob = pd.DataFrame(dt_model.predict_proba(test_feature))
    pred = pd.DataFrame(dt_model.predict(test_feature))

    dt_final = pd.concat([test_label.reset_index()['strat_category'],pred_prob,pred],axis=1,ignore_index=True)

    dt_final.columns = ['Actual','A','B','C','Predicted']

    print(dt_final.head())
    print(dt_final.tail())

    a = f1_score(dt_final['Actual'],dt_final['Predicted'],average='micro')
    print(a*100)

def random_forest():

    rf_model = RandomForestClassifier(max_depth=8, random_state=0).fit(train_feature,train_label)
    pred_prob = pd.DataFrame(rf_model.predict_proba(test_feature))
    pred = pd.DataFrame(rf_model.predict(test_feature))

    rf_final = pd.concat([test_label.reset_index()['strat_category'],pred_prob,pred],axis=1,ignore_index=True)
    rf_final.columns = ['Actual','A','B','C','Predicted']

    print(rf_final.head())
    print(rf_final.tail())

    a = f1_score(rf_final['Actual'],rf_final['Predicted'],average='macro')
    print(a*100)

    print(metrics.classification_report(rf_final['Actual'],rf_final['Predicted']))


df_final = get_dataset()
train_set, test_set = sampling()
train_feature, train_label = get_feature_label(train_set.copy())
test_feature, test_label = get_feature_label(test_set.copy())

random_forest()