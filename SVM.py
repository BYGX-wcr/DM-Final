# -*- coding:utf-8 -*-

from sklearn.svm import LinearSVC
import numpy as np

import preprocess as prep
import summary as summary

def train_SVM(data, labels):
    svm = LinearSVC(dual=False)
    svm.fit(data, labels)

    return svm

def predict_SVM(data, model):
    return model.predict(data)

def score_SVM(data, labels, model):
    return model.score(data, labels)

def experiment(train_dataset, test_dataset, train_labels, test_labels=None, model_file=None):
    total_dataset = train_dataset + test_dataset
    seg_dataset = prep.seg_words(total_dataset)
    seg_dataset = prep.eliminate_noise(seg_dataset, "，。、\t “”；")
    if model_file == None:
        prep.train_word2vec_model(seg_dataset, output_path="./dataset/word2vec.model")
        model_file = "./dataset/word2vec.model"
    vec_dataset = prep.word_to_vec_highdim(seg_dataset, input_path=model_file)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]

    SVM_model = train_SVM(vec_train_dataset, train_labels)
    res = predict_SVM(vec_test_dataset, SVM_model)
    if test_labels != None:
        print("accuracy: {0}".format(score_SVM(vec_test_dataset, test_labels, SVM_model)))

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

    res = experiment(train_dataset, predict_dataset, train_labels, predict_labels, model_file="./dataset/all_word2vec.model")
    summary.save_result(res, "./dataset/submission6.csv")