import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *

def random_forest(parameter):
    #same code structure as in log_reg.py
    rf_model = RandomForestClassifier(random_state=0,max_depth=parameter['max_depth'],criterion=parameter['criterion'],max_features=parameter['max_features'],
    n_estimators=parameter['n_estimators'])
    rf_model.fit(train_feature,train_label)

    pred_prob = pd.DataFrame(rf_model.predict_proba(test_feature))
    pred = pd.DataFrame(rf_model.predict(test_feature))

    rf_final = pd.concat([test_label.reset_index()['strat_category'],pred_prob,pred],axis=1,ignore_index=True)
    rf_final.columns = ['Actual','A','B','C','Predicted']

    print(metrics.classification_report(rf_final['Actual'],rf_final['Predicted']))

    return rf_model.__class__.__name__, rf_final


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

def class_report(estimator,df,parameter):
    
    class_dict = {**{'Estimator':estimator,'f1_macro':metrics.f1_score(df['Actual'],df['Predicted'],average='macro')},**parameter}
    report_metric = metrics.classification_report(df['Actual'],df['Predicted'],output_dict=True)
    
    j_dict = {**class_dict,**report_metric}

    with open(f'reports/{estimator}_report.json','w') as jfile:
        json.dump(j_dict,jfile,indent=4)
        print(f'{jfile} has been created')    

if __name__ == '__main__':
    df_final = get_dataset()
    train_set, test_set = sampling(df_final)

    train_feature, train_label = get_feature_label(train_set.copy())
    test_feature, test_label = get_feature_label(test_set.copy())

    print('----------------------Evaluating the random forest----------------------')
    parameter = param_tuning('f1_macro')
    estimator,df = random_forest(parameter)
    class_report(estimator,df,parameter)