# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:25:58 2021

@author: tongj
"""

import os
import pandas as pd
from sklearn import preprocessing
from pandas_profiling import ProfileReport

import warnings
warnings.filterwarnings("ignore")

data_dir = 'data/'
# 626599 / 3190634
# 0.3283

def txt_to_feather(file = None, train = True):
    test_cols = ['remote_addr', 'time_iso8601', 'status', 'file', 
                  'body_bytes_sent','request_time', 'uuid', 'http_referer', 
                  'http_user_agent', 'http_x_forwarded_for', 'pno', 'Id']
    train_cols = test_cols + ['auth_ret']
    
    f = open(f'data/{file}.txt', 'r')
    df = f.read()
    f.close()
    df = df.split('\n')
    df = [eval(x) for x in df[:-1]]    
    
    if train:
        df = pd.DataFrame(df, columns = train_cols)
    else:
        df = pd.DataFrame(df, columns = test_cols)  
        df['auth_ret'] = '-1' 
        
    df['auth_ret'] = df['auth_ret'].astype(int)
    df.reset_index(drop = True, inplace = True)
    df.to_feather(f'data/{file}.feather')        
    
def save_feather():
    txt_to_feather(file = 'a', train = True)
    txt_to_feather(file = 'a-test', train = False)
    txt_to_feather(file = 'b', train = False)
    
    df_a = pd.read_feather(os.path.join(data_dir, 'a.feather'))
    df_b = pd.read_feather(os.path.join(data_dir, 'b.feather'))    
    data = pd.concat([df_a, df_b], axis = 0)   
    data.reset_index(drop = True, inplace = True)
    data.to_feather('data/ab.feather')
    
    # test = df_b.head()
    # temp['auth_ret']
    
def EDA_html(train_df):
    profile = ProfileReport(train_df,
                            title='Fast Report for EDA',
                            html={'style':{'full_width':True}})
    
    profile.to_file("pf_report.html")
    
def main():
    '''
    a
    	index	time_day
    6	2021-03-22	1287627
    7	2021-03-24	1094114
    8	2021-03-26	842596
    9	2021-03-29	719464
    5	2021-04-01	1632399
    2	2021-04-03	1767317
    4	2021-04-05	1734000
    3	2021-04-07	1744110
    0	2021-04-09	1845708
    1	2021-04-10	1809955

    a-test
    	index	time_day
    0	2021-04-17	896919
    1	2021-04-18	837104
    2	2021-04-19	835990
    3	2021-04-20	588987
    
    b
    	index	time_day
    3	2021-04-26	660132
    2	2021-04-28	801261
    1	2021-04-29	822166
    0	2021-04-30	907075
    '''
    df = pd.read_feather(os.path.join(data_dir, 'ab.feather'))
    df['n'] = range(len(df))   
    df.drop(['uuid', 'status'], axis = 1, inplace = True)    
    # temp = df.tail(1000)
    # df.columns
    # df['auth_ret'].value_counts()
    # temp.info()
    
    df.sort_values(by = 'time_iso8601', inplace = True)
    df['time'] = df['time_iso8601'].apply(lambda x: x[:10] + ' ' + x[11:19])       
    df['time_day'] = df['time_iso8601'].apply(lambda x: x[:10])    
    df['time_hour'] = df['time_iso8601'].apply(lambda x: x[11:13])  
    df['time_minute'] = df['time_iso8601'].apply(lambda x: x[14:16])         
    df['time_second'] = df['time_iso8601'].apply(lambda x: x[17:19])   
    df.drop('time_iso8601', axis = 1, inplace = True)      
    
    # temp = df['time_day'].value_counts().reset_index()

    # 类别变量
    categorical_feature =   ['remote_addr',
                             'file',
                             'http_referer',
                             'http_user_agent',
                             'http_x_forwarded_for',
                             'pno',
                             'Id']
    
    for i in categorical_feature:
        le = preprocessing.LabelEncoder()
        le.fit(df[i].values)
        df[i] = le.transform(df[i].values) 
        
    str_feature = [col for col in df.columns if col not in ['n', 'auth_ret']]  
    for i in str_feature:
        df[i] = df[i].astype(str)
        
    df.reset_index(drop = True, inplace = True)
    df.to_feather(os.path.join(data_dir, 'ab-label-str.feather'))
        
    df.reset_index(drop = True, inplace = True)
    df.to_feather(os.path.join(data_dir, 'ab-label.feather'))

if __name__ == '__main__':
    main()