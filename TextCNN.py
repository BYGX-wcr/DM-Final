
# -*- coding:utf-8 -*-

import preprocess as prep
import summary

from keras import Input
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model


def textcnn(train_dataset, test_dataset, train_labels, model_file=None, output_path=None):
    """ TextCNN: 1. embedding, 2.convolution layer, 3.max-pooling, 4.softmax layer. """

    vec_dim = 100

    # Input layer
    x_input = Input(shape=(vec_dim, 1, ))
    print("x_input.shape: %s" % str(x_input.shape))  # (?, 60)

    # # Embedding layer
    # x_emb = Embedding(input_dim=vec_dim, output_dim=vec_dim, input_length=vec_dim)(x_input)
    # print("x_emb.shape: %s" % str(x_emb.shape))  # (?, 60, 300)

    # Conv & MaxPool layer
    pool_output = []
    kernel_sizes = [2, 3, 4]
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=2, kernel_size=kernel_size, strides=1, activation='tanh')(x_input)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
        print("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(c.shape), str(p.shape)))
    pool_output = concatenate([p for p in pool_output])
    print("pool_output.shape: %s" % str(pool_output.shape))  # (?, 1, 6)

    # Flatten & Dense layer
    x_flatten = Flatten()(pool_output)  # (?, 6)
    y = Dense(class_num, activation='softmax')(x_flatten)  # (?, 2)
    print("y.shape: %s \n" % str(y.shape))

    model = Model(inputs=[x_input], outputs=[y])
    if output_path:
        plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=False)
    model.summary()

    total_dataset = train_dataset + test_dataset
    seg_dataset = prep.seg_words(total_dataset)
    seg_dataset = prep.eliminate_noise(seg_dataset, "，。、\t “”；")
    if model_file == None:
        prep.train_word2vec_model(seg_dataset, output_path="./dataset/word2vec.model")
        model_file = "./dataset/word2vec.model"
    vec_dataset, vec_dim = prep.word_to_vec(seg_dataset, input_path=model_file)

    vec_train_dataset = vec_dataset[0:len(train_dataset)]
    vec_test_dataset = vec_dataset[len(train_dataset):]
    
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    for i in range(len(vec_train_dataset)):
        model.fit(vec_train_dataset[i], train_labels[i], batch_size=100, epochs=10, shuffle=True, verbose=1, validation_split=0.2)
    res = model.predict(vec_test_dataset, batch_size=100)
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

    res = textcnn(train_dataset, predict_dataset, train_labels, model_file=model_file)
    summary.save_result(res, "./dataset/submission12.csv")
