# -*- coding:utf-8 -*-

import numpy as np
import scipy as sp
import sys
import jieba
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

def load_raw_data(output_path=None):
    data_file = open("./dataset/train.data", encoding="utf-8-sig")
    test_file = open("./dataset/test.data", encoding="utf-8-sig")
    label_file = open("./dataset/train.solution", encoding="utf-8-sig")
    emoji_file = open("./dataset/emoji.data", encoding="utf-8-sig")

    train_data = data_file.readlines()
    test_data = test_file.readlines()
    train_label = label_file.readlines()
    emoji = emoji_file.readlines()

    emoji_map = {}
    dataset = []
    test_dataset = []
    labels = []

    #construct emoji map: string -> id
    for line in emoji:
        content = str(line).split()
        emoji_map[content[1]] = content[0]

    #combine files to construct dataset and labels
    if len(train_data) != len(train_label):
        print("Invalid data, may be stained")
    for lineno in range(len(train_data)):
        emoji_name = str(train_label[lineno]).rstrip('\n').rstrip('}').lstrip('{')
        dataset.append(train_data[lineno].strip())
        labels.append(int(emoji_map[emoji_name]))

    #write the labeled_dataset into disk file
    if output_path != None:
        output_file = open(output_path, "w", encoding="utf-8")
        for i in range(len(dataset)):
            output_file.write(str(labels[i]) + " " + dataset[i] + "\n")
        
        output_file.close()

    #load test data
    for line in test_data:
        sentence = line.split("\t", 1)
        test_dataset.append(sentence[1])

    data_file.close()
    test_file.close()
    label_file.close()
    emoji_file.close()
    return dataset, labels, test_dataset, len(emoji_map.keys())

def class_statistic(labels, class_num):
    counters = {}
    for i in range(class_num):
        counters[i] = 0

    #count instances for every class
    for l in labels:
        counters[l] = counters[l] + 1

    return counters

def eliminate_noise(seg_dataset, noise_set):
    new_dataset = []
    for sentence in seg_dataset:
        new_sentence = []
        for word in sentence:
            for char in noise_set:
                word = word.replace(char, "")

            if len(word) != 0:
                new_sentence.append(word)
        
        new_dataset.append(new_sentence)

    print("Eliminating noise finished!")    
    return new_dataset

def seg_words(dataset, output_path=None):
    new_dataset = []
    for instance in dataset:
        words = list(jieba.cut(instance))
        new_dataset.append(words)

    print("Segmenting words finished!")
    return new_dataset

def tfidf(seg_dataset):
    task_dict = Dictionary(seg_dataset)
    corpus = [task_dict.doc2bow(line) for line in seg_dataset]
    model = TfidfModel(corpus)

    data = []
    indices = []
    indptr = []
    counter = 0
    for sentence in corpus:
        sparse_vec = model[sentence]
        indptr.append(len(data))
        for i in range(len(sparse_vec)):
            indices.append(sparse_vec[i][0])
            data.append(sparse_vec[i][1])
        counter += 1
    indptr.append(len(data))

    print("TfIdf preprocess finished!")
    return sp.sparse.csr_matrix((data, indices, indptr))

def train_MulNB(data, labels, class_prior=None):
    mnb = MultinomialNB(alpha=2.0, fit_prior=False, class_prior=class_prior)
    mnb.fit(data, labels)

    print("Training of MultinomialNB model finished!")
    return mnb

def predict_MulNB(data, model):
    return model.predict(data)

def experiment_tfidf(train_dataset, test_dataset, train_labels, model_file=None):
    total_dataset = train_dataset + test_dataset
    seg_dataset = seg_words(total_dataset)
    seg_dataset = eliminate_noise(seg_dataset, "，。、\t “”；")
    vec_dataset = tfidf(seg_dataset)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]

    MulNB_model = None
    if model_file == None:
        MulNB_model = train_MulNB(vec_train_dataset, train_labels)
        joblib.dump(MulNB_model, "MultinomialNB.model")
    else:
        MulNB_model = joblib.load(model_file)
    
    res = predict_MulNB(vec_test_dataset, MulNB_model)

    return res

def save_result(res, output_path):
    output_file = open(output_path, 'w', encoding='utf-8')
    output_file.write('ID,Expected\n')

    for index in range(len(res)):
        output_file.write('{0},{1}\n'.format(index,res[index]))

    output_file.close()
    print("Result saved to {0}\n".format(output_path))

if __name__ == "__main__":
    dataset, labels, test_dataset, class_num = load_raw_data()

    train_dataset = dataset
    predict_dataset = test_dataset
    train_labels = labels
    model_file = None
    if len(sys.argv) > 1:
        model_file = sys.argv[1]

    res = experiment_tfidf(train_dataset, predict_dataset, train_labels, model_file=model_file)
    save_result(res, "./best-private-score-submission.csv")
