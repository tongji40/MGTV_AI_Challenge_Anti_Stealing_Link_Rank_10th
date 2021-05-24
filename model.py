# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:50:52 2021

@author: tongj
"""

import os
import gc
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import multiprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# LGBT
parameters =  { 'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'n_jobs': -1,
                'verbose': -1
                }

num_cores = int(multiprocessing.cpu_count() * 0.8)

def read_feature(feature_class, feature_name):
    '''
    feature_name = 'http_referer.Id'
    feature.tail()
    '''
    try:
        feature = pd.read_feather(f'G:/feature/{feature_class}/{feature_name}')
        return feature
    except:
        pass
    
def get_categorical_feature(df):
    categorical_feature = [col for col in df.columns if col not in ['n', 'auth_ret']]    
    categorical_feature = [col for col in categorical_feature if 'count' not in col]    
    categorical_feature = [col for col in categorical_feature if 'diff' not in col]     
    categorical_feature = [col for col in categorical_feature if 'Encoder' not in col]         
    return categorical_feature
    
def cal_f1_score(y_train, y_test, y_test_pred):

    y_pred = pd.Series(y_test_pred)
    threshold = pd.Series(y_pred).quantile(len(y_train[y_train == 0]) / y_train.shape[0])
    y_pred = y_pred.map(lambda x: 1 if x > threshold else 0)
    f1 = round(f1_score(y_test, y_pred), 4) 
    return f1

def model_offline(df):
    '''
    df_label = pd.read_feather('data/ab-label.feather')  
    feature_list = os.listdir('G:/feature/diff/') 
    feature_list = [x for x in feature_list if 'diffnext' in x]
    
    df = Parallel(n_jobs = num_cores)(delayed(read_feature)('diff', i) for i in feature_list)
    df = pd.concat(df, axis = 1)
    df = pd.concat([df_label, df], axis = 1)  
    
    del df_label
    gc.collect() 
    '''
    
    # df = data.copy()
    # temp = df.head(10)
    df = df[df['auth_ret'] != -1].reset_index(drop=True)
    feature = [col for col in df.columns if col not in ['n','auth_ret']]    
    
    X_train = df[df['time_day'] < '2021-04-05']
    X_val  = df[(df['time_day'] >= '2021-04-05') & (df['time_day'] < '2021-04-09')]    
    X_test = df[df['time_day']  >= '2021-04-09']    
    
    categorical_feature = get_categorical_feature(df)
    for i in categorical_feature:
        X_train[i] = X_train[i].astype('category')
        X_val[i] = X_val[i].astype('category')            
        X_test[i] = X_test[i].astype('category')
    
    y_train = X_train['auth_ret']
    X_train = X_train[feature]
    
    y_val = X_val['auth_ret']
    X_val = X_val[feature]
    
    y_test = X_test['auth_ret']
    X_test = X_test[feature]
    
    lgb_train = lgb.Dataset(X_train, label = y_train)
    lgb_val  = lgb.Dataset(X_val, label = y_val)
    
    del df
    gc.collect()
    
    lgb_model = lgb.train(
        parameters,
        lgb_train,
        valid_sets = lgb_val,
        categorical_feature = categorical_feature,
        num_boost_round = 500,        
        early_stopping_rounds = 50,
        verbose_eval = 50
    )
    # lgb_online_diffnext.pkl [708]	valid_0's auc: 0.848873 val f1 score: 0.6249
    
    # 模型存储
    # joblib.dump(lgb_model, 'model/lgb_one_fold.pkl')
    
    # 模型加载
    # lgb_model = joblib.load('model/lgb_one_fold.pkl')

    y_val_pred  = lgb_model.predict(X_val, num_iteration = lgb_model.best_iteration)    
    print('val f1 score:', cal_f1_score(y_train, y_val, y_val_pred)) 
    # val f1 score: 0.5986

    y_test_pred  = lgb_model.predict(X_test, num_iteration = lgb_model.best_iteration)    
    print('test f1 score:', cal_f1_score(y_train, y_test, y_test_pred)) 
    # test f1 score: 0.5749
    
    df_feature_importance = pd.DataFrame({'column': X_train.columns,
                                          'importance': lgb_model.feature_importance()})
    df_feature_importance.sort_values(by = 'importance', ascending = False, inplace = True)
    df_feature_importance.reset_index(drop = True, inplace = True)
    print(df_feature_importance.head(10))
    # df_feature_importance.to_csv('offline_feature_importance.csv', index = False)    
    return df_feature_importance

def model_online(data, num):
    '''
    df_label = pd.read_feather('data/ab-label.feather')  
    feature_list = os.listdir('G:/feature/diff/') 
    feature_list = [x for x in feature_list if 'diffnext' in x]
    
    df = Parallel(n_jobs = num_cores)(delayed(read_feature)('diff', i) for i in feature_list)
    df = pd.concat(df, axis = 1)
    df = pd.concat([df_label, df], axis = 1)  
    temp = X_test.head()
    '''
    df = data.copy()
    
    categorical_feature = get_categorical_feature(df)
    categorical_feature.remove('time_day')
    for i in categorical_feature:
        df[i] = df[i].astype('category')
            
    feature = [col for col in df.columns if col not in ['n', 'auth_ret']]
    
    X_train = df[df['auth_ret'] != -1].reset_index(drop=True)
    X_test = df[df['auth_ret'] == -1].reset_index(drop=True)
    
    X_val = X_train[X_train['time_day']  >= '2021-04-09']    
    X_train = X_train[X_train['time_day'] < '2021-04-09']
    X_train['time_day'] = X_train['time_day'].astype('category')
    X_val['time_day'] = X_val['time_day'].astype('category')    
    X_test['time_day'] = X_test['time_day'].astype('category')   
    
    y_train = X_train['auth_ret']
    X_train = X_train[feature]
    
    y_val = X_val['auth_ret']
    X_val = X_val[feature]

    X_test = X_test[feature]
     
    lgb_train = lgb.Dataset(X_train, label = y_train)
    lgb_val  = lgb.Dataset(X_val, label = y_val)

    lgb_model = lgb.train(
        parameters,
        lgb_train,
        valid_sets = lgb_val,
        num_boost_round = 5000,        
        early_stopping_rounds = 50,
        verbose_eval = 50
    )

    # 模型存储
    joblib.dump(lgb_model, f'model/lgb_online_top{num}.pkl')
    # lgb_online_diffnext.pkl [788]	valid_0's auc: 0.858079 val f1 score: 0.6249
    
    # 模型加载
    # lgb_model = joblib.load('model/lgb_one_fold.pkl')

    y_val_pred  = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)   
    val_f1_score = cal_f1_score(y_train, y_val, y_val_pred)
    print('val f1 score:', val_f1_score)

    df_feature_importance = pd.DataFrame({'column': X_train.columns,
                                          'importance': lgb_model.feature_importance()})
    df_feature_importance.sort_values(by='importance', ascending = False, inplace = True)
    df_feature_importance.reset_index(drop = True, inplace = True)
    print(df_feature_importance.head(10))
    # df_feature_importance.to_csv('online_diffnext_feature.csv', index = False)     
    
    # submission    
    y_test_pred  = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) 
    
    result = pd.DataFrame(y_test_pred, columns = ['y_pred'])
    result['n'] = df[df['auth_ret'] == -1]['n'].values
    result.sort_values(by = 'n', inplace = True)
    
    threshold = pd.Series(result['y_pred']).quantile(2564036 / 3190634)
    result['y_label'] = result['y_pred'].map(lambda x: 1 if x > threshold else 0)
    result['y_label'] = result['y_label'].astype(int)  
    
    print(result['y_label'].value_counts())
    result['y_label'].to_csv(f"submission/b-submission-top{num}-{val_f1_score}.csv", index=False, header=None)
    result['y_pred'].to_csv(f"submission/b-prob-top{num}.csv", index=False)
    
    # b = pd.read_csv("submission/b-submission.csv", header=None)
    return df_feature_importance

def model_k_fold(data):
    df = data.copy()
    # temp = df.head()

    categorical_feature = get_categorical_feature(df)
    for i in categorical_feature:
        df[i] = df[i].astype('category')
    # df['auth_ret'] = df['auth_ret'].astype(int)    
    
    feature = [col for col in df.columns if col not in ['n', 'auth_ret']]    

    skf = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = True)
    
    X_train = df[df['auth_ret'] != -1].reset_index(drop=True)
    X_test = df[df['auth_ret'] == -1].reset_index(drop=True)

    y = X_train['auth_ret']
    X = X_train[feature]
    X_test = X_test[feature]
    
    val_pred = np.zeros(X.shape[0])
    y_test_pred  = np.zeros(X_test.shape[0])
   
    for index, (train_index, val_index) in enumerate(skf.split(X, y)):
        print('K-fold:', index + 1)
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val, y_val = X.iloc[val_index], y.iloc[val_index]
        
        lgb_train = lgb.Dataset(X_train, label = y_train)
        lgb_val = lgb.Dataset(X_val, label = y_val)
        
        lgb_model = lgb.train(
            parameters,
            lgb_train,
            valid_sets = lgb_val,
            num_boost_round = 5000,        
            early_stopping_rounds = 50,
            verbose_eval = 50
        )
        
        # 模型存储        
        # joblib.dump(lgb_model, f'model/lgb_kfold_{index}.pkl')   
        
        y_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        val_pred[val_index] += y_val_pred
        
        prediction = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)   
        y_test_pred += prediction / 5
        
    # lgb_online_diffnext [1245]	valid_0's auc: 0.858183 val f1 score: 0.6246
    print('val f1 score:', cal_f1_score(y, y, val_pred))

    df_feature_importance = pd.DataFrame({'column': X_train.columns,
                                          'importance': lgb_model.feature_importance()})
    df_feature_importance.sort_values(by='importance', ascending = False, inplace = True)
    df_feature_importance.reset_index(drop = True, inplace = True)
    print(df_feature_importance.head(10))
    df_feature_importance.to_csv('k_fold_feature_importance.csv', index = False)    
    
    result = pd.DataFrame(y_test_pred, columns = ['y_pred'])
    result['n'] = df[df['auth_ret'] == -1]['n'].values 
    result.sort_values(by = 'n', inplace = True)
    
    threshold = pd.Series(result['y_pred']).quantile(2564036 / 3190634)
    result['y_label'] = result['y_pred'].map(lambda x: 1 if x > threshold else 0)
    result['y_label'] = result['y_label'].astype(int)  

    result['y_label'].to_csv("submission/b-kfold-submission.csv", index=False, header=None)
    result['y_pred'].to_csv("submission/b-kfold-prob.csv", index=False)
    
    return df_feature_importance

def get_feature_importance():
    # 查看特征f1得分
    file_list = os.listdir('feature_score/')
    result = []
    for file in file_list:
        result.append(pd.read_csv(f'feature_score/{file}'))
    result = pd.concat(result)
    result.sort_values(by = 'f1', ascending = False, inplace = True)
    result.reset_index(drop = True, inplace = True)    
    return result