import itertools
from collections import OrderedDict, Counter

import numpy as np
import nltk

class FeaturesCalculator:
    keys : str
    bigrams_from_keys : tuple
    features : OrderedDict

    def __init__(self):
        self.__init_constants()
        self.__init_features()

    def __init_constants(self):
        self.keys = "ABCDEFGHIKLMNPQRSTVWYZ"
        self.keys = ''.join(sorted(tuple(self.keys))) # alphabet sort
        self.keys = self.keys.replace('Z', '').replace('B', '') # B and Z probably unnecessary

        self.bigrams_from_keys = tuple(itertools.product(self.keys, repeat=2))

    def __init_features(self):
        self.features = OrderedDict()
        self.features["aac"] = self.get_aac
        self.features["dpc"] = self.get_dpc
        self.features["qso"] = self.get_qso

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