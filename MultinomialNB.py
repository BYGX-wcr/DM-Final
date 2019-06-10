# -*- coding:utf-8 -*-

import numpy as np
from sklearn.naive_bayes import MultinomialNB

import preprocess as prep

def train_MulNB(data, labels, class_prior=None):
    mnb = MultinomialNB(alpha=2.0, fit_prior=True, class_prior=class_prior)
    mnb.fit(data, labels)

    return mnb

def predict_MulNB(data, model):
    return model.predict(data)

def score_MulNB(data, labels, model):
    return model.score(data, labels)

def experiment(train_dataset, test_dataset, train_labels, test_labels=None, model_file=None):
    total_dataset = train_dataset + test_dataset
    total_dataset = prep.eliminate_noise(total_dataset, "，。\t “”；")
    seg_dataset = prep.seg_words(total_dataset)
    if model_file == None:
        prep.train_word2vec_model(seg_dataset, output_path="./dataset/word2vec.model")
        model_file = "./dataset/word2vec.model"
    vec_dataset = prep.word_to_vec(seg_dataset, input_path=model_file)
    vec_dataset = prep.max_min_normalize(dataset, max=1, min=0)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]

    MulNB_model = train_MulNB(vec_train_dataset, train_labels)
    res = predict_MulNB(vec_test_dataset, MulNB_model)
    if test_labels != None:
        print("accuracy: {0}".format(score_MulNB(test_dataset, test_labels, MulNB_model)))

    return res

if __name__ == "__main__":
    dataset, labels, test_dataset, class_num = prep.load_raw_data()

    train_dataset = dataset[0:700000]
    predict_dataset = dataset[700000:]
    train_labels = labels[0:700000]
    predict_labels = labels[700000:]

    res = experiment(train_dataset, predict_dataset, train_labels, predict_labels, model_file="./dataset/word2vec.model")
