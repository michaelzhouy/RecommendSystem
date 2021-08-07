# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 8:55 上午
# @Author  : Michael Zhouy
import pandas as pd
from ItemCF import ItemCF
import joblib

data = pd.read_csv('../data/ml-100k/ua.base', sep='\\t', header=None,
                   names=["userId", "movieId", "rating", "timestamp"], engine='python')
data.drop(columns=["timestamp"], inplace=True)
clf = ItemCF(method='base', alpha=0.6, normalized=True)
clf.fit(data)
joblib.dump(clf, './itemCF.m')

clf = joblib.load('./itemCF.m')
print(clf.recommendProducts(1, 30, 30))
