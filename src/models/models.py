import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *


''' def log_class():
    
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
train_set, test_set = sampling(df_final)
train_feature, train_label = get_feature_label(train_set.copy())
test_feature, test_label = get_feature_label(test_set.copy())  '''
