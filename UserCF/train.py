# -*- coding: utf-8 -*-
# @Time    : 2021/8/5 8:46 上午
# @Author  : Michael Zhouy
import pandas as pd
from UserCF import UserCF
import joblib


data = pd.read_csv('../../RecommendSystem/data/ml-100k/ua.base', sep='\\t', header=None, names=["userId", "movieId", "rating", "timestamp"],engine='python')
data.drop(columns=["timestamp"], inplace=True)
clf = UserCF(method='base')
clf.fit(data)
joblib.dump(clf, './userCf.m')

clf = joblib.load('./userCf.m')
print(clf.recommendProducts(1, 15, 10))
