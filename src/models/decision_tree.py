import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *

def decision_tree():
    
    dt_model = DecisionTreeClassifier(random_state=0)
    dt_model.fit(train_feature,train_label)

    pred_prob = pd.DataFrame(dt_model.predict_proba(test_feature))
    pred = pd.DataFrame(dt_model.predict(test_feature))

    dt_final = pd.concat([test_label.reset_index()['strat_category'],pred_prob,pred],axis=1,ignore_index=True)

    dt_final.columns = ['Actual','A','B','C','Predicted']

    print(dt_final.head())
    print(dt_final.tail())

    a = f1_score(dt_final['Actual'],dt_final['Predicted'],average='micro')
    print(a*100)

    print(metrics.classification_report(dt_final['Actual'],dt_final['Predicted']))


df_final = get_dataset()
train_set, test_set = sampling(df_final)

train_feature, train_label = get_feature_label(train_set.copy())
test_feature, test_label = get_feature_label(test_set.copy())

train_feature = scaling(train_feature)
test_feature = scaling(test_feature)