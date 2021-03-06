import copy
from collections import OrderedDict

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

class MultiModel:
    # constants
    num_models : int
    features : OrderedDict
    # models
    small_models : OrderedDict
    final_model : object

    def __init__(self, features):
        self.features = copy.copy(features)
        self.num_models = len(self.features.keys())
    
    def __init_models(self):
        """Init untrained models"""
        self.small_models = OrderedDict()
        for feature in self.features.keys():
            self.small_models[feature] = make_pipeline(StandardScaler(), SVC(gamma = "auto", probability=True, random_state=0))
        self.final_model = RandomForestClassifier(n_estimators = 1000, random_state=0)

    def __fit_small_models(self, X : list, y : list):
        """Training small models (first layer of models)"""
        d = OrderedDict()
        for k,v in self.features.items():
            d[k] = np.array(list(map(lambda item: v(item), X)))
            self.small_models[k].fit(d[k], y)
        return d

    def __transform_seq_in_X_for_final_model(self, seq : str) -> np.array:
        """Transform results from small models to final features vector for one sequence"""
        transform_func = lambda f: f(seq)
        predict_func = lambda item: self.small_models[item[0]].predict_proba([item[1]])[0][1]

        temp_values = list(map(transform_func, self.features.values()))
        temp =  OrderedDict(
            [(k, v) for k, v in zip(self.features.keys(), temp_values)])
        res = list(map(predict_func, temp.items()))
        return np.array(res)
    
    def transform(self, seqs : list[str]) -> np.array:
        """Transform list of sequences to final features vector"""
        return np.array(list(map(self.__transform_seq_in_X_for_final_model, seqs)))

    def fit(self, X : list, y : list):
        """Training all models"""
        self.__init_models()
        
        transformed_X = self.__fit_small_models(X, y)
        predict_func = lambda item: self.small_models[item[0]].predict_proba(item[1])[:,1]
        
        predicted_X = list(map(predict_func, transformed_X.items()))
        del transformed_X
        predicted_X = OrderedDict([(k, v) for k, v in zip(self.features.keys(), predicted_X)])
        
        X_final = np.array(list(predicted_X.values())).T
        self.final_model.fit(X_final, y)
    
    def predict(self, X : list) -> list:
        """Return model predictions for list of sequences"""
        X_final = self.transform(X)
        return self.final_model.predict(X_final)

    def predict_proba(self, X : list) -> list:
        """Return model confidence for list of sequences"""
        X_final = self.transform(X)
        return self.final_model.predict_proba(X_final)
    
    def score(self, X : list, y : list):
        """Return accuracy of predictions on given dataset"""
        X_final = self.transform(X)
        return self.final_model.score(X_final, y)