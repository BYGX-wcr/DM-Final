# -*- coding:utf-8 -*-

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import scipy as sp

import preprocess as prep
import summary as summary

def train_AdaMulNB(data, labels):
    ada_mnb = AdaBoostClassifier(base_estimator=MultinomialNB(alpha=2.0, fit_prior=False), n_estimators=60, algorithm='SAMME.R')
    ada_mnb.fit(data, labels)

    return ada_mnb

def predict_AdaMulNB(data, model):
    return model.predict(data)

def score_AdaMulNB(data, labels, model):
    return model.score(data, labels)

def experiment(train_dataset, test_dataset, train_labels, test_labels=None):
    total_dataset = train_dataset + test_dataset
    seg_dataset = prep.seg_words(total_dataset)
    seg_dataset = prep.eliminate_noise(seg_dataset, "，。、\t “”；")
    vec_dataset = prep.tfidf(seg_dataset)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]

    AdaMulNB_model = train_AdaMulNB(vec_train_dataset, train_labels)
    res = predict_AdaMulNB(vec_test_dataset, AdaMulNB_model)
    if test_labels != None:
        print("accuracy: {0}".format(score_AdaMulNB(vec_test_dataset, test_labels, AdaMulNB_model)))

    return res

if __name__ == "__main__":
    dataset, labels, test_dataset, class_num = prep.load_raw_data()

    train_dataset = dataset
    predict_dataset = test_dataset
    train_labels = labels
    predict_labels = None

    res = experiment(train_dataset, predict_dataset, train_labels, predict_labels)
    summary.save_result(res, "./dataset/submission9.csv")
