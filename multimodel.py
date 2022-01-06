from collections import OrderedDict, Counter
import itertools

import numpy as np
import nltk

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

class MultiModel():
    # constants
    num_models : int
    keys : str
    features : OrderedDict
    bigrams_from_keys : tuple
    # models
    small_models : OrderedDict
    final_model : object

    def __init__(self):
        self.__init_constants()
    
    def __init_constants(self):

        self.keys = "ABCDEFGHIKLMNPQRSTVWYZ"
        self.keys = ''.join(sorted(tuple(self.keys))) # alphabet sort
        self.keys = self.keys.replace('Z', '').replace('B', '') # B and Z probably unnecessary

        self.bigrams_from_keys = tuple(itertools.product(self.keys, repeat=2))

        self.features = OrderedDict()
        self.features["aac"] = self.get_aac
        self.features["dpc"] = self.get_dpc
        self.features["qso"] = self.get_qso
    
        self.num_models = len(self.features.keys())
    
    def __init_models(self):
        self.small_models = OrderedDict()
        for feature in self.features.keys():
            self.small_models[feature] = make_pipeline(StandardScaler(), SVC(gamma = "auto", probability=True, random_state=0))
        self.final_model = DecisionTreeClassifier(random_state=0)

    def __fit_small_models(self, X : list, y : list):
        d = OrderedDict()
        for k,v in self.features.items():
            d[k] = np.array(list(map(lambda item: v(item), X)))
            self.small_models[k].fit(d[k], y)
        return d

    def __transform_seq_in_X_for_final_model(self, seq : str) -> np.array:
        transform_func = lambda f: f(seq)
        predict_func = lambda item: self.small_models[item[0]].predict_proba([item[1]])[0][1]

        temp_values = list(map(transform_func, self.features.values()))
        temp =  OrderedDict(
            [(k, v) for k, v in zip(self.features.keys(), temp_values)])
        res = list(map(predict_func, temp.items()))
        return np.array(res)

    def fit(self, X : list, y : list):
        self.__init_models()
        transformed_X = self.__fit_small_models(X, y)
        predict_func = lambda item: self.small_models[item[0]].predict_proba(item[1])[:,1]

        predicted_X = list(map(predict_func, transformed_X.items()))
        del transformed_X
        predicted_X = OrderedDict([(k, v) for k, v in zip(self.features.keys(), predicted_X)])
        X_final = np.array(list(predicted_X.values())).T
        self.final_model.fit(X_final, y)
            

    def predict(self, X : list) -> list:
        X_final = list(map(self.__transform_seq_in_X_for_final_model, X))
        X_final = np.array(X_final)
        return self.final_model.predict(X_final)
    
    def score(self, X : list, y : list):
        X_final = list(map(self.__transform_seq_in_X_for_final_model, X))
        X_final = np.array(X_final)
        return self.final_model.score(X_final, y)


    def get_aac(self, seq : str) -> np.array:
        n = len(seq)
        d = OrderedDict.fromkeys(self.keys, value=0)
        it = OrderedDict(Counter(seq))
        for k, v in it.items():
            d[k] = v / n
        return np.array(list(d.values()))

    def get_dpc(self, seq : str) -> np.array:
        n = len(seq)
        d = OrderedDict.fromkeys(self.bigrams_from_keys, value=0)
        it = OrderedDict(Counter(nltk.bigrams(seq)))
        for k, v in it.items():
            d[k] = v / n
        return np.array(list(d.values()))
    
    def get_qso(self, seq : str, maxlag : int = 5, weight : float = 0.1) -> np.array:
        aac, n = self.get_aac(seq), len(self.keys)
        sum_aac = sum(aac)
        d = np.zeros((n,n))
        for i, j in itertools.product(range(n), repeat=2):
            d[i,j] = (aac[j] - aac[i])**2
        r = np.zeros((maxlag,))
        for i in range(maxlag):
            r[i] = 0
            for j in range(n - maxlag):
                r[i] += d[j,i+j]**2 
        sum_r = sum(r)
        temp = sum_aac + weight*sum_r
        qso = aac / temp
        return qso