# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:12:57 2021

@author: 1vany
"""
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from preprocessing import preprocessing
import joblib

def train(train_new, y):
    train_df, valid_df, y_train, y_valid = train_test_split(train_new, y, test_size=0.7, random_state=42) 
    # Настройка гиперпараметров
    parameters = {'depth' : [3, 5, 7], 'learning_rate': [0.03, 0.05, .09]}
    ctb_fixed = CatBoostRegressor()
    ctb_grid = GridSearchCV(ctb_fixed,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)
    ctb_grid.fit(train_df, y_train)
    
    joblib.dump(ctb_grid, "model.pkl") #сохранение модели
    
def main():
    train_new, _, y= preprocessing('train.csv', 'test.csv')
    train(train_new, y)
    
if __name__ == '__main__':
    main()