# -*- coding:utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import preprocess as prep
import summary as summary

def cosine_metric(X, Y):
    x = np.reshape(X, (1, -1))
    y = np.reshape(Y, (1, -1))
    return 1 - cosine_similarity(x, y)

def train_KNN(data, labels, n=1):
    knn = KNeighborsClassifier(n_neighbors=n, weights='uniform', n_jobs=-1, metric=cosine_metric)
    knn.fit(data, labels)

    return knn

def predict_KNN(data, model):
    return model.predict(data)

def score_KNN(data, labels, model):
    return model.score(data, labels)

def experiment(train_dataset, test_dataset, train_labels, test_labels=None, model_file=None):
    total_dataset = train_dataset + test_dataset
    total_dataset = prep.eliminate_noise(total_dataset, "，。\t “”；")
    seg_dataset = prep.seg_words(total_dataset)
    if model_file == None:
        prep.train_word2vec_model(seg_dataset, output_path="./dataset/word2vec.model")
        model_file = "./dataset/word2vec.model"
    vec_dataset = prep.word_to_vec(seg_dataset, input_path=model_file)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]

    KNN_model = train_KNN(vec_train_dataset, train_labels)
    res = predict_KNN(vec_test_dataset, KNN_model)
    if test_labels != None:
        print("accuracy: {0}".format(score_KNN(vec_test_dataset, test_labels, KNN_model)))

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
    summary.save_result(res, "./dataset/submission2.csv")