# -*- coding:utf-8 -*-

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import scipy as sp

import preprocess as prep
import summary as summary

def train_EnsClf(data, labels):
    clf1 = MultinomialNB(alpha=2.0, fit_prior=False)
    clf2 = LinearSVC(dual=False)
    clf3 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
    clf4 = RidgeClassifier(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=1, solver='auto', tol=0.001)
    vc = VotingClassifier(estimators=[('mnb', clf1), ('svm', clf2), ('lr', clf3), ('rc', clf4)], voting='soft')
    vc.fit(data, labels)

    return vc

def predict_EnsClf(data, model):
    return model.predict(data)

def score_EnsClf(data, labels, model):
    return model.score(data, labels)

def experiment(train_dataset, test_dataset, train_labels, test_labels=None):
    total_dataset = train_dataset + test_dataset
    seg_dataset = prep.seg_words(total_dataset)
    seg_dataset = prep.eliminate_noise(seg_dataset, "，。、\t “”；")
    vec_dataset = prep.tfidf(seg_dataset)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]

    EnsClf_model = train_EnsClf(vec_train_dataset, train_labels)
    res = predict_EnsClf(vec_test_dataset, EnsClf_model)
    if test_labels != None:
        print("accuracy: {0}".format(score_EnsClf(vec_test_dataset, test_labels, EnsClf_model)))

    return res

if __name__ == "__main__":
    dataset, labels, test_dataset, class_num = prep.load_raw_data()

    train_dataset = dataset
    predict_dataset = test_dataset
    train_labels = labels
    predict_labels = None

    res = experiment(train_dataset, predict_dataset, train_labels, predict_labels)
    summary.save_result(res, "./dataset/submission13.csv")