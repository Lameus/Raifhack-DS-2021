# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:43:56 2021

@author: 1vany
"""
import pandas as pd
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt 


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def preprocessing(train_path, test_path):
    # Загрузка данных
    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # Признаки с пропусками
    features_with_nan = [feature for feature in df. columns if df[feature].isnull().sum()>1]
    features_with_nan.remove('floor')
    # Объединение наборов данных для заполнения пропусков
    support_df = pd.concat([df, test_df], ignore_index=True)
    support_df = support_df.drop(['floor', 'per_square_meter_price'], axis=1)
    numerical_feat=[features for features in support_df.columns if support_df[features].dtypes!='O']
    # Вспомогательный набор данных без признаков с пропусками
    df_dropped = support_df.drop(features_with_nan, axis = 1)
    cat_features = set(df_dropped.columns) - set(numerical_feat)
    df_dropped = df_dropped.drop(cat_features, axis = 1)
    # Масштабирование набора данных перед кластеризацией
    scaler = StandardScaler()
    support_scaled = scaler.fit_transform(df_dropped)
    # Кластеризация
    clust = KMeans()
    clst_labels = clust.fit_predict(support_scaled)
    labels_set = list(set(clst_labels))
    # Добавление столбца соответствующего принадлежности к набору аналогов
    support_df['cluster'] = clst_labels
    # Численные признаки с пропусками
    features_nan_num = []
    for i in features_with_nan:
        if i in numerical_feat:
            features_nan_num.append(i)
    # Заполнение пропусков средними значениями аналогов
    for drop in features_nan_num:
        sub_df = support_df[support_df[drop].isnull()]
        for clust in sub_df.groupby('cluster')['cluster'].unique():
            similar_val_list = support_df[support_df['cluster'] == clust[0]][drop]
            if len(similar_val_list) > 1:
                missed_val = similar_val_list.dropna().mean()
                support_df.loc[similar_val_list[similar_val_list.isnull()].index, drop] = missed_val
    support_df = support_df.drop('id', axis=1)
    # Выделим из даты значения года и месяца
    support_df['year'] = pd.DatetimeIndex(support_df['date']).year
    support_df['month'] = pd.DatetimeIndex(support_df['date']).month
    support_df = support_df.drop('date', axis=1)
    
    # Хэширование названий городов и регионов
    dct = {}
    for s in ['city', 'osm_city_nearest_name', 'region']:
        for i in set(support_df[s]):
            d1 = {i : hash(i)%10000}
            dct.update(d1)
    for i in ['city', 'osm_city_nearest_name', 'region']:
        support_df[i] = support_df[i].map(dct)
    
    # Перевод номеров улиц в числа
    support_df['street'] = support_df['street'].str[1:]
    support_df['street'] = pd.to_numeric(support_df['street'])
    support_df.fillna(0)
    support_df['street'].replace({0:support_df['street'].median()})
    train_new = support_df[:df.shape[0]]
    test_new = support_df[df.shape[0]:]
    
    return train_new, test_new, df['per_square_meter_price']

def main():
    train_new, test_new, y = preprocessing('train.csv', 'test.csv')

if __name__ == '__main__':
    main()