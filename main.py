# -*- coding: utf-8 -*-
"""
Created on Sat May  8 23:31:41 2021

@author: tongj
"""

import os
import gc
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import model

data_dir = 'data/'
num_cores = int(multiprocessing.cpu_count() * 0.8)

def read_top_feature(raw_feature, num = 30):
    drop_feature = ['raw', 'group', 'cumcounthour', 'recumcounthour', 
                    'CatBoostEncoder', 'LeaveOneOutEncoder']
    
    feature_f1 = pd.read_csv('feature_class_f1.csv')
    feature_f1 = feature_f1[~feature_f1['feature_class'].isin(drop_feature)]
    feature_f1.sort_values(by = 'f1', ascending = False, inplace = True)
    
    feature_list = []
    for i in feature_f1['feature_class']:
        feature = pd.read_csv(f'feature_score/{i}.feature.csv')
        feature = feature[~feature['column'].isin(raw_feature)]
        feature_list.append(feature)
        # feature = feature['column'].tolist()[:15]
        # feature = feature['column'].tolist()        
        # feature_list += feature
    feature_list = pd.concat(feature_list)
    feature_list.sort_values(by = 'importance', ascending = False, inplace = True)
    feature_list = feature_list['column'].tolist()[:num]
    return feature_list
    
def read_feature(feature_name):
    '''
    feature_name = 'http_referer.Id'
    feature.tail()
    '''
    try:
        if 'count' in feature_name:  
            feature = pd.read_feather(f'G:/feature/count/{feature_name}.feather')
        elif 'diff' in feature_name: 
            feature = pd.read_feather(f'G:/feature/diff/{feature_name}.feather')
        return feature
    except:
        pass
    
def main():
    
    # 读取数据
    df_label = pd.read_feather('data/ab-label.feather')  
    raw_feature = list(df_label.columns)
    
    num = 250
    for num in [50, 100, 150, 200, 250]:
        
        num = 150
        print(f'top{num}.online')
        feature_list = read_top_feature(raw_feature, num)
        # feature_list = [x for x in feature_list if x not in raw_feature]
        
        df = Parallel(n_jobs = num_cores)(delayed(read_feature)(i) for i in feature_list)
        df = pd.concat(df, axis = 1)
        df = pd.concat([df_label, df], axis = 1)
            
        ### 模型预测
        # df_feature_importance = model.model_offline(df)   
        
        df_feature_importance = model.model_online(df, num)
        
        # df_feature_importance = model.model_k_fold(df)
        
        del df
        gc.collect() 
        
        df_feature_importance.to_csv(f'top{num}_feature_importance.csv', index = False)     

if __name__ == '__main__':
    main()        