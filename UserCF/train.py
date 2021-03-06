# -*- coding: utf-8 -*-
# @Time    : 2021/8/5 8:46 上午
# @Author  : Michael Zhouy
import pandas as pd
from UserCF import UserCF
import joblib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--method", type=str, default="base", help="method for calculate similarity")
parser.add_argument("--modelPath", type=str, default="./userCf.m", help="model path")
a = parser.parse_args()


if __name__ == "__main__":
    if a.mode == 'train':
        data = pd.read_csv('../data/ml-100k/ua.base', sep='\\t', header=None, names=["userId", "movieId", "rating", "timestamp"],engine='python')
        data.drop(columns=["timestamp"], inplace=True)
        clf = UserCF(method='base')
        clf.fit(data)
        joblib.dump(clf, a.modelPath)
    elif a.mode == 'test':
        clf = joblib.load('./userCf.m')
        print(clf.recommendProducts(1, 15, 10))
