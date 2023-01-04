import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *

def random_forest(parameter):
    
    rf_model = RandomForestClassifier(random_state=0,max_depth=parameter['max_depth'],criterion=parameter['criterion'],max_features=parameter['max_features'],
    n_estimators=parameter['n_estimators'])
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


def param_tuning(score):

    max_depth = [1, 5, 10, 50]
    criterion = ['gini','entropy']
    max_features = ['log2','sqrt']
    n_estimators = [100, 150, 200, 250, 300]

    rf_model = RandomForestClassifier(random_state=0)
    parameters = {'max_depth':max_depth,'criterion':criterion,'max_features':max_features,'n_estimators':n_estimators}

    grid_search = GridSearchCV(rf_model,parameters, cv=5,scoring=score)
    grid_search.fit(train_feature,train_label)

    print('Best parameters for a random forest : ', grid_search.best_params_)

    return grid_search.best_params_
    

df_final = get_dataset()
train_set, test_set = sampling(df_final)

train_feature, train_label = get_feature_label(train_set.copy())
test_feature, test_label = get_feature_label(test_set.copy())

parameter = param_tuning('f1_weighted')
random_forest(parameter)