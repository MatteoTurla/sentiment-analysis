import optuna.integration.lightgbm as lgb

from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
from pathlib import Path
import os

from optuna import logging

logging.set_verbosity(logging.ERROR)

class SentimentClassifier:
    
    def __init__(self):
        self.model = None
        
    def fit(self, X, target):
        dtrain = lgb.Dataset(X, label=target)
        
        params = {
            "objective": "regression",
            "metric": "l2",
            "boosting_type": "gbdt",
            "verbosity": -1
        }

        tuner = lgb.LightGBMTunerCV(
            params,
            dtrain,
            #time_budget = 10,
            return_cvbooster = True,
            folds=KFold(n_splits=3),
            callbacks=[log_evaluation(100), early_stopping(50)],
            verbose_eval = False
        )

        tuner.run()

        print("Best score:", tuner.best_score)
        best_params = tuner.best_params
        print("Best params:", best_params)
        print("  Params: ")
        for key, value in best_params.items():
            print("    {}: {}".format(key, value))
            
        self.model = tuner.get_best_booster().boosters
    
    
    def save(self, directory):
        print("saving model")
        for i, booster in enumerate(self.model):
            booster.save_model(Path(directory)/f'boosters_{i}.txt')
            
        
    def predict(self, X):
        Y = []
        for b in self.model:
            y = b.predict(X)
            Y.append(y)
        return np.mean(Y, axis=0)
    
    def load(self, directory):
        print("loading model")
        self.model = []
        fns = os.listdir(directory)
        
        for fn in fns:
            self.model.append(lgb.Booster(model_file=Path(directory)/fn))
        
        
    