# -*- coding:utf-8 -*-

import jieba
from gensim.models import Word2Vec
import numpy as np

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
        sentence = line.split(" ", 1)
        test_dataset.append(sentence)

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

def eliminate_noise(dataset, noise_set):
    new_dataset = []
    for sentence in dataset:
        for char in noise_set:
            sentence = sentence.strip(char)
        
        new_sentence = sentence
        new_dataset.append(new_sentence)
    
    return new_dataset

def seg_words(dataset, output_path=None):
    new_dataset = []
    for instance in dataset:
        words = list(jieba.cut(instance))
        new_instance = words
        new_dataset.append(new_instance)

    return new_dataset

def word_to_vec(seg_dataset, input_path=None, output_path=None):
    # load or train the model
    if input_path == None:
        model = Word2Vec(seg_dataset, workers=4, window=4, size=100, min_count=0, sg=0, hs=0)
    else:
        model = Word2Vec.load(input_path)

    # check whether to save the model
    if output_path != None:
        model.save(output_path)

    #compute the max length of each sentence
    max_len = 0
    for sentence in seg_dataset:
        if len(sentence) > max_len:
            max_len = len(sentence)

    #construct new dataset
    new_dataset = []
    for i in range(len(seg_dataset)):
        #construct the word vectors for a sentence, supplement missing tails
        word_vecs = []
        for word in seg_dataset[i]:
            word_vecs.append(model.wv[word])
        for j in range(len(seg_dataset[i]), max_len):
            word_vecs.append(np.array([0] * 100))

        new_dataset.append(word_vecs)

    return new_dataset

if __name__ == "__main__":
    dataset, labels, test_dataset, class_num = load_raw_data()

    dataset = eliminate_noise(dataset, "，。\t  “”；")

    seg_dataset = seg_words(dataset)

    vec_dataset = word_to_vec(seg_dataset, output_path="./dataset/word2vec.model")
