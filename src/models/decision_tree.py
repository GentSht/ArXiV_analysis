import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *

def decision_tree(parameter):
    
    dt_model = DecisionTreeClassifier(random_state=0,max_depth=parameter['max_depth'], min_samples_split=parameter['min_samples_split'],
    min_samples_leaf=parameter['min_samples_leaf'])

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


df_final = get_dataset()
train_set, test_set = sampling(df_final)

train_feature, train_label = get_feature_label(train_set.copy())
test_feature, test_label = get_feature_label(test_set.copy())

parameter = param_tuning('f1_weighted')
decision_tree(parameter)