import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *


def log_class(parameter):
    
    lm_model = LogisticRegression(multi_class='auto',max_iter=parameter['max_iter'],solver=parameter['solver'])
    lm_model.fit(train_feature,train_label)

    lm_pred_prob = pd.DataFrame(lm_model.predict_proba(test_feature))
    lm_pred = pd.DataFrame(lm_model.predict(test_feature))

    lm_final = pd.concat([test_label.reset_index()['strat_category'],lm_pred_prob,lm_pred],axis=1, ignore_index=True)
    lm_final.columns = ['Actual','A','B','C','Predicted']

    print(metrics.classification_report(lm_final['Actual'],lm_final['Predicted']))

    return lm_model.__class__.__name__, lm_final

def param_tuning(score):
  
    iter = list(np.linspace(1000,10000,10))

    lm_model = LogisticRegression(multi_class='auto')
    parameters = {'solver':['saga','lbfgs','newton-cg'], 'max_iter':iter}

    grid_search = GridSearchCV(lm_model,parameters, cv=5,scoring=score)
    grid_search.fit(train_feature,train_label)

    print('Best parameters for a logistic classification : ', grid_search.best_params_)
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

    print('----------------------Evaluating the logistic regression----------------------')
    parameter = param_tuning('f1_macro')
    estimator, df = log_class(parameter)
    class_report(estimator,df,parameter)