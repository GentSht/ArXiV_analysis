import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *

def random_forest():
    
    rf_model = RandomForestClassifier(max_depth=8, random_state=0)
    rf_model.fit(train_feature,train_label)

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
test_feature, test_label = get_feature_label(test_set.copy())

train_feature = scaling(train_feature)
test_feature = scaling(test_feature)