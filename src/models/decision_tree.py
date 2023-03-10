import pandas as pd
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *


def decision_tree(parameter):
    #same code structure as in log_reg.py
    dt_model = DecisionTreeClassifier(random_state=0,max_depth=parameter['max_depth'], min_samples_split=parameter['min_samples_split'],
    min_samples_leaf=parameter['min_samples_leaf'])

    dt_model.fit(train_feature,train_label)

    pred_prob = pd.DataFrame(dt_model.predict_proba(test_feature))
    pred = pd.DataFrame(dt_model.predict(test_feature))

    dt_final = pd.concat([test_label.reset_index()['strat_category'],pred_prob,pred],axis=1,ignore_index=True)
    dt_final.columns = ['Actual','A','B','C','Predicted']

    print(metrics.classification_report(dt_final['Actual'],dt_final['Predicted']))

    return dt_model.__class__.__name__, dt_final

def param_tuning(score):

    max_depth = [x+1 for x in range(32)]
    min_samples_split = [2, 5, 10, 20, 50, 100, 200]
    min_samples_leaf = [1, 4, 7, 10]

    dt_model = DecisionTreeClassifier(random_state=0)
    parameters = {'max_depth':max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}

    grid_search = GridSearchCV(dt_model,parameters, cv=5,scoring=score)
    grid_search.fit(train_feature,train_label)

    print('Best parameters for the decision tree : ', grid_search.best_params_)

    return grid_search.best_params_

def class_report(estimator,df,parameter):

    class_dict = {**{'Estimator':estimator,'f1_macro':metrics.f1_score(df['Actual'],df['Predicted'],average='macro')},**parameter}
    report_metric = metrics.classification_report(df['Actual'],df['Predicted'],output_dict=True)
    
    j_dict = {**class_dict,**report_metric}

    with open(f'reports/{estimator}_{met}_report.json','w') as jfile:
        json.dump(j_dict,jfile,indent=4)
        print(f'{jfile} has been created')
    

if __name__ == '__main__':

    if met == "no_author":
        df_final = get_dataset_no_author()
        train_set, test_set = sampling(df_final)

        train_feature, train_label = get_feature_label(train_set.copy())
        test_feature, test_label = get_feature_label(test_set.copy())

        print('----------------------Evaluating the decision tree----------------------')
        parameter = param_tuning('f1_macro')
        estimator,df = decision_tree(parameter)
        class_report(estimator,df,parameter)
    else:
        df_final = get_dataset_author()
        train_set, test_set = sampling(df_final)

        train_feature, train_label = get_author_features(train_set.copy())
        test_feature, test_label = get_author_features(test_set.copy())

        print('----------------------Evaluating the decision tree----------------------')
        parameter = param_tuning('f1_macro')
        estimator,df = decision_tree(parameter)
        class_report(estimator,df,parameter)
