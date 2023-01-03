import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import metrics
from sampling_features import *



def log_class():
    
    LM_model = LogisticRegression(penalty = 'l2', multi_class='auto', solver='sag',max_iter=100)
    LM_model.fit(train_feature,train_label)

    LM_pred_prob = pd.DataFrame(LM_model.predict_proba(test_feature))
    LM_pred = pd.DataFrame(LM_model.predict(test_feature))

    LM_final = pd.concat([test_label.reset_index()['strat_category'],LM_pred_prob,LM_pred],axis=1, ignore_index=True)
    LM_final.columns = ['Actual','A','B','C','Predicted']

    print(LM_final.head())
    print(LM_final.tail())

    a = f1_score(LM_final['Actual'],LM_final['Predicted'],average='micro')
    print(a*100)


def param_tuning():
    #iter:100,solver:sag
    iter = list(np.linspace(100,10000,20))

    LM_model = LogisticRegression(multi_class='auto')
    parameters = {'solver':['sag','saga','lbfgs','newton-cg'], 'max_iter':iter}

    
    grid_search = GridSearchCV(LM_model,parameters, cv=5,scoring='f1_micro')
    grid_search.fit(train_feature,train_label)

    print(grid_search.best_params_)


df_final = get_dataset()
train_set, test_set = sampling(df_final)
train_feature, train_label = get_feature_label(train_set.copy())
test_feature, test_label = get_feature_label(test_set.copy())
train_feature = scaling(train_feature)
test_feature = scaling(test_feature)
log_class()