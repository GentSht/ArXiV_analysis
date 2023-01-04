import pandas as pd
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

    print(lm_final.head())
    print(lm_final.tail())

    a = f1_score(lm_final['Actual'],lm_final['Predicted'],average='micro')
    print(a*100)

    print(metrics.classification_report(lm_final['Actual'],lm_final['Predicted']))


def param_tuning():
    #iter:100,solver:sag
    iter = list(np.linspace(100,10000,20))

    lm_model = LogisticRegression(multi_class='auto')
    parameters = {'solver':['sag','saga','lbfgs','newton-cg'], 'max_iter':iter}

    grid_search = GridSearchCV(lm_model,parameters, cv=5,scoring='f1_micro')
    grid_search.fit(train_feature,train_label)

    return grid_search.best_params_


df_final = get_dataset()
train_set, test_set = sampling(df_final)

train_feature, train_label = get_feature_label(train_set.copy())
test_feature, test_label = get_feature_label(test_set.copy())

train_feature = scaling(train_feature)
test_feature = scaling(test_feature)

parameter = param_tuning()
log_class(parameter)