# -*- coding:utf-8 -*-

import jieba
from gensim.models import Word2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import scipy as sp

'''
The python script used to load and preprocess dataset
All dataset will be stored as list([sentence, label]) and output as list([label, sentence])
'''

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

def max_min_normalize(dataset, max, min):
    dataset = np.array(dataset)
    minv = dataset.min(axis=0)
    maxv = dataset.max(axis=0)

    for row in dataset:
        for colno in range(len(row)):
            col = row[colno]
            temp = col
            col = ((col - minv[colno]) / (maxv[colno] - minv[colno])) * (max - min) + min
            if col < min or col > max:
                print("Normalization Error!")
                print(str(temp) + ',' + str(col) + ',' + str(minv[colno]) + ',' + str(maxv[colno]))
                exit()
            row[colno] = col

    print("Max min normalization finished!")
    return dataset

def train_word2vec_model(seg_dataset, output_path):
    model = Word2Vec(seg_dataset, workers=4, window=4, size=100, min_count=0, sg=0, hs=0)
    model.save(output_path)
    print("Training of Word2Vec model finished, saved to %s" % output_path)

def word_to_vec(seg_dataset, input_path):
    # load the model
    model = Word2Vec.load(input_path)

    #construct new dataset
    new_dataset = []
    for sentence in seg_dataset:
        #construct the word vectors for a sentence, supplement missing tails
        word_vec = np.zeros((100))
        counter = 0
        for word in sentence:
            word_vec = word_vec + model.wv[word]
            counter = counter + 1

        if counter != 0: #compute average value
            for x in np.nditer(word_vec, op_flags=['readwrite']):
                x[...] = x / counter
        new_dataset.append(word_vec)

    print("Word2vec preprocess finished!")
    return new_dataset

def word_to_vec_highdim(seg_dataset, input_path):
    # load the model
    model = Word2Vec.load(input_path)

    max_len = -1
    for sentence in seg_dataset:
        max_len = max(len(sentence), max_len)

    #construct new dataset
    new_dataset = []
    for sentence in seg_dataset:
        #construct the word vectors for a sentence, supplement missing tails
        word_vec = []
        counter = 0
        for word in sentence:
            word_vec = word_vec + list(model.wv[word])
            counter = counter + 1

        if counter < max_len:
            word_vec = word_vec + [0]*(100 * (max_len - counter))
        new_dataset.append(np.array(word_vec))

    print("Word2vec preprocess finished!")
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
    return sp.sparse.csr_matrix((data, indices, indptr)).toarray()

if __name__ == "__main__":
    dataset, labels, test_dataset, class_num = load_raw_data()

    total_dataset = dataset + test_dataset

    seg_dataset = seg_words(total_dataset)

    seg_dataset = eliminate_noise(seg_dataset, "，。、\t “”；")

    train_word2vec_model(seg_dataset, output_path="./dataset/all_word2vec.model")
