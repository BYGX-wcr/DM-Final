# -*- coding:utf-8 -*-

import numpy as np
import scipy as sp
from sklearn.naive_bayes import MultinomialNB

import preprocess as prep
import summary as summary

def train_MulNB(data, labels, class_prior=None):
    mnb = MultinomialNB(alpha=2.0, fit_prior=True, class_prior=class_prior)
    mnb.fit(data, labels)

    print("Training of MultinomialNB model finished!")
    return mnb

def predict_MulNB(data, model):
    return model.predict(data)

def score_MulNB(data, labels, model):
    return model.score(data, labels)

def experiment(train_dataset, test_dataset, train_labels, test_labels=None, model_file=None):
    total_dataset = train_dataset + test_dataset
    seg_dataset = prep.seg_words(total_dataset)
    seg_dataset = prep.eliminate_noise(seg_dataset, "，。、\t “”；")
    if model_file == None:
        prep.train_word2vec_model(seg_dataset, output_path="./dataset/word2vec.model")
        model_file = "./dataset/word2vec.model"
    vec_dataset = prep.word_to_vec(seg_dataset, input_path=model_file)
    vec_dataset = prep.max_min_normalize(vec_dataset, max=10, min=0)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]

    MulNB_model = train_MulNB(vec_train_dataset, train_labels)
    res = predict_MulNB(vec_test_dataset, MulNB_model)
    if test_labels != None:
        print("accuracy: {0}".format(score_MulNB(vec_test_dataset, test_labels, MulNB_model)))

    return res

def experiment_tfidf(train_dataset, test_dataset, train_labels, test_labels=None):
    total_dataset = train_dataset + test_dataset
    seg_dataset = prep.seg_words(total_dataset)
    seg_dataset = prep.eliminate_noise(seg_dataset, "，。、\t “”；")
    vec_dataset = prep.tfidf(seg_dataset)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]

    MulNB_model = train_MulNB(vec_train_dataset, train_labels)
    res = predict_MulNB(vec_test_dataset, MulNB_model)
    if test_labels != None:
        print("accuracy: {0}".format(score_MulNB(vec_test_dataset, test_labels, MulNB_model)))

    return res

if __name__ == "__main__":
    dataset, labels, test_dataset, class_num = prep.load_raw_data()

    # train_dataset = dataset[1:700000]
    # predict_dataset = dataset[700000:]
    # train_labels = labels[1:700000]
    # predict_labels = labels[700000:]
    train_dataset = dataset
    predict_dataset = test_dataset
    train_labels = labels
    predict_labels = None
    model_file = "./dataset/all_word2vec.model"

    res = experiment_tfidf(train_dataset, predict_dataset, train_labels, predict_labels)
    summary.save_result(res, "./dataset/submission5.csv")