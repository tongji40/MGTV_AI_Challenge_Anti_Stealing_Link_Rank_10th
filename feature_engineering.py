# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:38:14 2021

@author: tongj
"""

import os
import time
import pandas as pd
import lightgbm as lgb
import category_encoders as ce
import multiprocessing
from joblib import Parallel, delayed
from itertools import chain, combinations
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import model

data_dir = 'data/'
num_cores = int(multiprocessing.cpu_count() * 0.8)

def train_sample():
    df = pd.read_feather(os.path.join(data_dir, 'a-label.feather'))
    df = df[df['auth_ret'] != -1].reset_index(drop=True)    
    feature = [col for col in df.columns if col not in ['auth_ret']]   
    X_train, X_test, y_train, y_test = train_test_split(df[feature], 
                                                        df['auth_ret'], 
                                                        test_size=0.3, 
                                                        random_state=0)
    df_sample = pd.concat([X_test, y_test], axis = 1)
    df_sample.reset_index(drop = True, inplace = True)
    # temp = df_sample.head(10000)
    df_sample.to_feather(os.path.join(data_dir, 'a-sample.feather'))
    
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
        
def raw_feature():
    df = pd.read_feather(os.path.join(data_dir, 'ab-label.feather'))
    # temp = df.tail()

    df_feature_importance = model.model_offline(df)
    
    # 保存特征重要性
    df_feature_importance.to_csv('feature_score/raw.feature.csv', index = False)        

def group_feature():

    df = pd.read_feather(os.path.join(data_dir, 'ab-label-str.feather'))
    for i in df.columns:
        feature = df[[i]]
        feature.reset_index(drop = True, inplace = True)
        feature.to_feather(f'G:/feature/group/{i}.feather')   
        
    # temp = df.head()
    # df['n'].nunique()
    # temp.info()
    # temp = pd.read_feather(f'G:/feature/group/{feature_name}.feather')

    categorical_feature =   ['remote_addr', 
                             'file', 
                             'http_referer',
                             'http_user_agent', 
                             'http_x_forwarded_for',
                             'pno', 
                             'Id']
    
    # for i in categorical_feature:
    #     df[i] = df[i].astype(str)
        
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    # group_feature = list(enumerate(powerset(categorical_feature)))[8:]
    group_feature = list(enumerate(powerset(categorical_feature)))    
    group_feature = [list(x[1]) for x in group_feature]
    # group_feature = [x for x in group_feature if len(x) < 5]
    pass_feature = os.listdir('G:/feature/group/')
    pass_feature = [x.replace('.feather', '').split('.') for x in pass_feature]
    group_feature = [x for x in group_feature if x not in pass_feature]

    # cols = group_feature[-1]
    def save_group_feature(cols):
        try:
            feature_name = '.'.join(cols)
            feature = df[cols[0]].str.cat(df[cols[1:]] , sep='.')
            le = preprocessing.LabelEncoder()
            feature = pd.DataFrame(le.fit_transform(feature.values), columns = [feature_name])
            feature[feature_name] = feature[feature_name].astype(str)   
            feature.reset_index(drop = True, inplace = True)
            feature.to_feather(f'G:/feature/group/{feature_name}.feather')            
        except:
            pass
                    
    Parallel(n_jobs = num_cores)(delayed(save_group_feature)(i) for i in group_feature)

    # 特征重要性
    df = pd.read_feather(os.path.join(data_dir, 'ab-group.feather'))

    print('\nmodel: group.')
    # feature_list = [x for x in feature_list if 'group.' in x]
    # df = pd.concat([df_label, df], axis = 1)  
    # temp = df.head()  
    # df.columns
 
    df_feature_importance = model.model_offline(df)
    df_feature_importance.to_csv('feature_score/group.feature.csv', index = False)          

def count_feature():
    '''
    [250]	valid_0's auc: 0.843612
    val f1 score: 0.5721
    test f1 score: 0.5717
                     column  importance
    0                  file        3098
    1       http_user_agent         845
    2                  time         741
    3  http_x_forwarded_for         280
    4                   pno         195
    5       body_bytes_sent         182
    6          request_time         165
    7          http_referer         148
    8           remote_addr         143
    9                    Id         101
    '''
    df = pd.read_feather(os.path.join(data_dir, 'ab-group.feather'))
    # temp = df.head()
    # df.columns

    feature_list = [col for col in df.columns if col not in ['n', 'auth_ret']]

    pass_feature = os.listdir('G:/feature/count/')
    pass_feature = [x.replace('.feather', '') for x in pass_feature]

    # cols = feature[-1]
    def save_feature(cols):
        try:
            feature_name = 'count.' + cols
            # if feature_name not in pass_feature:
            feature = df[cols].map(df[cols].value_counts()) 
            feature = pd.DataFrame(feature.values, columns = [feature_name])                
            feature.reset_index(drop = True, inplace = True)
            feature.to_feather(f'G:/feature/count/{feature_name}.feather')            
        except:
            pass
                    
    Parallel(n_jobs = num_cores)(delayed(save_feature)(i) for i in feature_list)
    

    # 特征重要性
    df_label = pd.read_feather(os.path.join(data_dir, 'ab-label.feather'))

    print('\nmodel: count.')
    feature_list = os.listdir('G:/feature/count/') 
    feature_list = [x for x in feature_list if 'cumcount.' not in x]
    feature_list = [x for x in feature_list if 'recumcount.' not in x]
    
    df = Parallel(n_jobs = num_cores)(delayed(read_feature)('count', i) for i in feature_list)
    df = pd.concat(df, axis = 1)
    df = pd.concat([df_label, df], axis = 1)  
    # temp = df.head()   
 
    df_feature_importance = model.model_offline(df)
    df_feature_importance.to_csv('feature_score/count.feature.csv', index = False)          
      
   
def count_time_feature():

    df = pd.read_feather(os.path.join(data_dir, 'ab-group.feather'))    
    # temp = feature.head()
    # df.columns
    # for i in df.columns:
        # df[i] = df[i].astype(str)

    time_feature = ['time', 'time_day', 'time_hour', 'time_minute', 'time_second']
    df_time = df[time_feature]
    categorical_feature = [col for col in df.columns if col not in time_feature + ['n', 'auth_ret']]
    pass_feature = os.listdir('G:/feature/count/')
    pass_feature = [x.replace('.feather', '') for x in pass_feature]
    
    # feature_class = 'day'
    # cols = 'time_day|remote_addr'
    def save_feature(feature_class, cols):
        try:
            cols = cols.split('|')
            feature_name = 'count' + feature_class + '.' +  cols[-1]
            if feature_name not in pass_feature:
                df_count = df_time.copy()
                feature_list = [x for x in cols if x not in time_feature]
                for i in feature_list:
                    df_count[i] = pd.read_feather(f'G:/feature/group/{i}.feather')    
                df_count[feature_name] = df_count[cols[0]].str.cat(df_count[cols[1:]] , sep='.')
                feature = df_count[feature_name].map(df_count[feature_name].value_counts())
                feature = pd.DataFrame(feature, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/count/{feature_name}.feather')            
        except:
            pass

    feature = ['time_day|' + col for col in categorical_feature]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('day', i) for i in feature)
    
    feature = ['time_day|time_hour|' + col for col in categorical_feature]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('hour', i) for i in feature)    

    feature = ['time_day|time_hour|time_minute|' + col for col in categorical_feature]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('minute', i) for i in feature)    

    feature = ['time_day|time_hour|time_minute|time_second|' + col for col in categorical_feature]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('second', i) for i in feature)    

    # 特征重要性
    df_label = pd.read_feather(os.path.join(data_dir, 'ab-label.feather'))
    
    for i in ['countday.', 'counthour.', 'countminute.','countsecond.']:
        print('\nmodel:', i)
        feature_list = os.listdir('G:/feature/count/') 
        feature_list = [x for x in feature_list if i in x]
        
        df = Parallel(n_jobs = num_cores)(delayed(read_feature)('count', i) for i in feature_list)
        df = pd.concat(df, axis = 1)
        df = pd.concat([df_label, df], axis = 1)  
        # temp = df.head()   
     
        df_feature_importance = model.model_offline(df)
        df_feature_importance.to_csv(f'feature_score/{i}feature.csv', index = False)          

def cumcount_time_feature():

    df = pd.read_feather(os.path.join(data_dir, 'ab-group.feather'))    
    # temp = df.head(100)
    # df.columns
    # for i in df.columns:
        # df[i] = df[i].astype(str)

    time_feature = ['time', 'time_day', 'time_hour', 'time_minute', 'time_second']
    # df_time = df[time_feature]
    # df_time.drop('time_second', axis = 1, inplace = True)
    categorical_feature = [col for col in df.columns if col not in time_feature + ['n', 'auth_ret']]
    pass_feature = os.listdir('G:/feature/count/')
    pass_feature = [x.replace('.feather', '') for x in pass_feature]
    
    # feature_class = 'hour'
    # cols = 'time_day|time_hour|remote_addr'
    # t1 = feature.head(1000)
    # t2 = feature.head(1000)
    # test = pd.concat([t1, t2], axis = 1)
    def save_feature(feature_class, cols):
        try:
            cols = cols.split('|')
            df_cum = df[cols]
            feature_name = 'cumcount' + feature_class + '.' +  cols[-1]
            if feature_name not in pass_feature:
                feature = df_cum.groupby(cols).cumcount()
                feature = pd.DataFrame(feature, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/count/{feature_name}.feather')  
                
            feature_name = 'recumcount' + feature_class + '.' +  cols[-1]
            if feature_name not in pass_feature:
                feature = df_cum.groupby(cols).cumcount(ascending = False)  
                feature = pd.DataFrame(feature, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/count/{feature_name}.feather')                                
        except:
            pass

    feature = ['time_day|' + col for col in categorical_feature]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('day', i) for i in feature)
    
    feature = ['time_day|time_hour|' + col for col in categorical_feature]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('hour', i) for i in feature)    

    feature = ['time_day|time_hour|time_minute|' + col for col in categorical_feature]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('minute', i) for i in feature)    


    # 特征重要性
    df_label = pd.read_feather(os.path.join(data_dir, 'ab-label.feather'))
    
    for i in ['recumcountday.', 'recumcounthour.', 'recumcountminute.']:
        print('\nmodel:', i)
        feature_list = os.listdir('G:/feature/count/') 
        feature_list = [x for x in feature_list if i in x]
        # feature_list = [x for x in feature_list if 'recumcount' not in x]        
        df = Parallel(n_jobs = num_cores)(delayed(read_feature)('count', i) for i in feature_list)
        df = pd.concat(df, axis = 1)
        df = pd.concat([df_label, df], axis = 1)  
        # temp = df.head()   
     
        df_feature_importance = model.model_offline(df)
        df_feature_importance.to_csv(f'feature_score/{i}feature.csv', index = False)          

def cumcount_feature():

    df = pd.read_feather(os.path.join(data_dir, 'ab-group.feather'))    

    time_feature = ['time', 'time_day', 'time_hour', 'time_minute', 'time_second']
    categorical_feature = [col for col in df.columns if col not in time_feature + ['n', 'auth_ret']]
    pass_feature = os.listdir('G:/feature/count/')
    pass_feature = [x.replace('.feather', '') for x in pass_feature]

    def save_feature(cols):
        try:
            feature_name = 'cumcount.' +  cols
            if feature_name not in pass_feature:
                feature = df.groupby(cols).cumcount()  
                feature = pd.DataFrame(feature, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/count/{feature_name}.feather')   

            feature_name = 'recumcount.' +  cols
            if feature_name not in pass_feature:
                feature = df.groupby(cols).cumcount(ascending = False)  
                feature = pd.DataFrame(feature, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/count/{feature_name}.feather')           
        except:
            pass

    Parallel(n_jobs = num_cores)(delayed(save_feature)(i) for i in categorical_feature)


    # 特征重要性
    df_label = pd.read_feather(os.path.join(data_dir, 'ab-label.feather'))
    
    i = 'cumcount.'
    for i in ['cumcount.', 'recumcount.']:
        feature_list = os.listdir('G:/feature/count/') 
        feature_list = [x for x in feature_list if i in x]
        feature_list = [x for x in feature_list if 'recumcount.' not in x]  
        
        df = Parallel(n_jobs = num_cores)(delayed(read_feature)('count', i) for i in feature_list)
        df = pd.concat(df, axis = 1)
        df = pd.concat([df_label, df], axis = 1)  
        # temp = df.head()   
        
        print('\nmodel:', i)
        df_feature_importance = model.model_offline(df)
        df_feature_importance.to_csv(f'feature_score/{i}feature.csv', index = False)          


def cumcountratio_feature():

    df = pd.read_feather(os.path.join(data_dir, 'ab-group.feather'))    
    # temp =feature.head(10000)
    time_feature = ['time', 'time_day', 'time_hour', 'time_minute', 'time_second']
    feature_list = [col for col in df.columns if col not in time_feature + ['n', 'auth_ret']]

    # feature_class = 'day'
    # cols = 'time_day|remote_addr'
    def save_feature(feature_class, cols):
        try:
            cols = cols.split('|')            
            feature_name = 'cumcount' + feature_class + '.' + cols[-1]
            feature_cumcount = pd.read_feather(f'G:/feature/count/{feature_name}.feather')
            
            feature_name = 'count' + feature_class + '.' + cols[-1]
            feature_count = pd.read_feather(f'G:/feature/count/{feature_name}.feather')

            feature_name = 'cumcountratio' + feature_class + '.' + cols[-1]
            feature = feature_cumcount.iloc[:, 0] / (feature_count.iloc[:, 0] + 1)
            feature.fillna(99999, inplace = True)
            feature = pd.DataFrame(feature, columns = [feature_name])
            feature.reset_index(drop = True, inplace = True)
            feature.to_feather(f'G:/feature/count/{feature_name}.feather')                                
        except:
            pass

    feature = ['time_day|' + col for col in feature_list]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('day', i) for i in feature)

    feature = ['time_day|time_hour|time_minute|' + col for col in feature_list]    
    Parallel(n_jobs = num_cores)(delayed(save_feature)('minute', i) for i in feature)    


    # 特征重要性
    df_label = pd.read_feather(os.path.join(data_dir, 'ab-label.feather'))
    
    for i in ['cumcountratioday.', 'cumcountratiominute.']:
        print('\nmodel:', i)
        feature_list = os.listdir('G:/feature/count/') 
        feature_list = [x for x in feature_list if i in x]
     
        df = Parallel(n_jobs = num_cores)(delayed(read_feature)('count', i) for i in feature_list)
        df = pd.concat(df, axis = 1)
        df = pd.concat([df_label, df], axis = 1)  
        # temp = df.head()   
     
        df_feature_importance = model.model_offline(df)
        df_feature_importance.to_csv(f'feature_score/{i}feature.csv', index = False)          


def diff_feature():

    df = pd.read_feather(os.path.join(data_dir, 'ab-group.feather'))
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    # temp = feature.head(1000)
    # temp = pd.read_feather('G:/feature/diff/diffprevious.remote_addr.pno.Id.feather')
    # temp = temp.head(10000)
    # df.columns
 
    time_feature = ['time', 'time_day', 'time_hour', 'time_minute', 'time_second']
    categorical_feature = [col for col in df.columns if col not in time_feature + ['n', 'auth_ret']]
    pass_feature = os.listdir('G:/feature/diff/')
    pass_feature = [x.replace('.feather', '') for x in pass_feature]
    
    # feature_class = 'tenclick'
    # cols = categorical_feature[0]
    def save_feature(feature_class, cols):
        try:
            feature_name = 'diff' + feature_class + '.' + cols
            if feature_name not in pass_feature:            
                feature = df.groupby(['time_day', cols])['time'].shift(-1) - df['time']
                feature = feature.dt.seconds
                feature.fillna(999999, inplace = True)
                feature = pd.DataFrame(feature.values, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/diff/{feature_name}.feather')            
        except:
            pass

    Parallel(n_jobs = num_cores)(delayed(save_feature)('next', i) for i in categorical_feature)

    def save_feature(feature_class, cols):
        try:
            feature_name = 'diff' + feature_class + '.' + cols
            if feature_name not in pass_feature:            
                feature = df.groupby(['time_day', cols])['time'].shift(1) - df['time']
                feature = feature.dt.seconds
                feature.fillna(999999, inplace = True)
                feature = pd.DataFrame(feature.values, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/diff/{feature_name}.feather')            
        except:
            pass

    Parallel(n_jobs = num_cores)(delayed(save_feature)('previous', i) for i in categorical_feature)

    
    def save_feature(feature_class, cols):
        try:
            feature_name = 'diff' + feature_class + '.' + cols
            if feature_name not in pass_feature:            
                feature = df.groupby(['time_day', cols])['time'].shift(-2) - df['time']
                feature = feature.dt.seconds
                feature.fillna(999999, inplace = True)
                feature = pd.DataFrame(feature.values, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/diff/{feature_name}.feather')            
        except:
            pass

    Parallel(n_jobs = num_cores)(delayed(save_feature)('next_2', i) for i in categorical_feature)

    def save_feature(feature_class, cols):
        try:
            feature_name = 'diff' + feature_class + '.' + cols
            if feature_name not in pass_feature:            
                feature = df.groupby(['time_day', cols])['time'].shift(2) - df['time']
                feature = feature.dt.seconds
                feature.fillna(999999, inplace = True)
                feature = pd.DataFrame(feature.values, columns = [feature_name])
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/diff/{feature_name}.feather')            
        except:
            pass

    Parallel(n_jobs = num_cores)(delayed(save_feature)('previous_2', i) for i in categorical_feature)


    # 特征重要性
    df_label = pd.read_feather(os.path.join(data_dir, 'ab-label.feather'))
    
    # for i in ['diffnext.', 'diffprevious.', 'diffnext_5.', 'diffprevious_5.']:
    for i in ['diffnext_2.', 'diffprevious_2.']:        
        print('\nmodel:', i)
        feature_list = os.listdir('G:/feature/diff/') 
        feature_list = [x for x in feature_list if i in x]
        # feature_list = [x for x in feature_list if 'recumcount' not in x]        
        df = Parallel(n_jobs = num_cores)(delayed(read_feature)('diff', i) for i in feature_list)
        df = pd.concat(df, axis = 1)
        df = pd.concat([df_label, df], axis = 1)  
        # temp = df.head()   
     
        df_feature_importance = model.model_offline(df)
        df_feature_importance.to_csv(f'feature_score/{i}feature.csv', index = False)          
       
        
def category_encoders_feature():
    df = pd.read_feather(os.path.join(data_dir, 'ab-group.feather'))
    # temp = feature.head()
    
    # offline
    df = df[df['auth_ret'] != -1].reset_index(drop=True)
    feature_list = [col for col in df.columns if col not in ['n','auth_ret']]    
    pass_feature = os.listdir('G:/feature/encoder/')
    pass_feature = [x.replace('.feather', '') for x in pass_feature]
    
    X_train = df[df['time_day'] < '2021-04-05']
    X_val  = df[(df['time_day'] >= '2021-04-05') & (df['time_day'] < '2021-04-09')]    
    X_test = df[df['time_day']  >= '2021-04-09']    

    y_train = X_train['auth_ret']
    X_train = X_train[feature_list]
    
    X_val = X_val[feature_list]

    X_test = X_test[feature_list]   

    # cols = feature_list[0]
    def save_feature(cols):
        try:
            feature_name = 'LeaveOneOutEncoder.' + cols
            if feature_name not in pass_feature:            
                encoder = ce.LeaveOneOutEncoder()        
                encoder.cols = [cols]
                encoder.fit(X_train[cols], y_train)
                X_train_encoder = encoder.transform(X_train[cols])
                X_val_encoder = encoder.transform(X_val[cols])            
                X_test_encoder = encoder.transform(X_test[cols])  
                feature = pd.concat([X_train_encoder, X_val_encoder, X_test_encoder])
                feature.columns = [feature_name]
                feature.reset_index(drop = True, inplace = True)
                feature.to_feather(f'G:/feature/encoder/{feature_name}.feather')            
        except:
            pass
        
    Parallel(n_jobs = num_cores)(delayed(save_feature)(i) for i in feature_list)

    # 特征重要性
    df_label = pd.read_feather(os.path.join(data_dir, 'ab-label.feather'))
    df_label = df_label[df_label['auth_ret'] != -1].reset_index(drop=True)   
    
    feature_list = os.listdir('G:/feature/encoder/') 
    feature_list = [x for x in feature_list if 'LeaveOneOutEncoder' in x]
    
    df = Parallel(n_jobs = num_cores)(delayed(read_feature)('encoder', i) for i in feature_list)
    df = pd.concat(df, axis = 1)
    df = pd.concat([df_label, df], axis = 1)  
    # temp = df.head()   
    
    print('\nmodel: LeaveOneOutEncoder.') 
    df_feature_importance = model.model_offline(df)
    df_feature_importance.to_csv('feature_score/LeaveOneOutEncoder.feature.csv', index = False)          

    # online
       
def nunique_feature():
    data  = pd.read_csv('data/a-sample-combination.csv')   
    
    y = data['auth_ret']  
    df = data.copy()
    
    # 类别变量
    categorical_feature = [col for col in df.columns if col not in ['auth_ret']]
    categorical_feature = [col for col in categorical_feature if 'combination.' not in col]
    
    for i in categorical_feature:
        count_feature_name = 'count.' + i
        df[count_feature_name] = df[i].map(df[i].value_counts())        
        for j in categorical_feature: 
            if i != j:
                # nuique
                nunique_feature_name = 'nunique.' + i + '.' + j
                df[nunique_feature_name] = df.groupby(i)[j].transform('nunique').values
                f1 = feature_f1_score(y, nunique_feature_name, df)
                result = pd.DataFrame([[nunique_feature_name, f1]], columns = ['feature', 'f1'])
                result.to_csv(f'feature_score/{nunique_feature_name}.csv', index = False)        

                # count / nuique
                feature_name = 'count_nunique.' + i + '.' + j
                df[feature_name] = df[count_feature_name] / df[nunique_feature_name]
                f1 = feature_f1_score(y, feature_name, df)
                result = pd.DataFrame([[feature_name, f1]], columns = ['feature', 'f1'])
                result.to_csv(f'feature_score/{feature_name}.csv', index = False)        
                df.drop(nunique_feature_name, axis = 1, inplace = True)                
                df.drop(feature_name, axis = 1, inplace = True)
        df.drop(count_feature_name, axis = 1, inplace = True)
        
    # 查看特征f1得分
    result = get_feature_importance()
    result = result[result['feature'].str.startswith('label.')]

if __name__ == '__main__':
    raw_feature()
    group_feature()
    count_feature()
    count_time_feature()
    cumcount_time_feature()
    cumcount_feature()
    cumcountratio_feature()
    diff_feature()
    category_encoders_feature()
