import numpy as np
from sklearn.naive_bayes import BernoulliNB

def train_BerNB(data, labels, threshold=0.0, class_prior=None):
    bnb = BernoulliNB(alpha=2.0, binarize=threshold, fit_prior=True,class_prior=class_prior)
    bnb.fit(data, labels)

    return bnb

def predict_BerNB(data, model):
    return model.predict(data)

def score_BerNB(data, labels, model):
    return model.score(data, labels)

if __name__ == "__main__":
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]])
    y = np.array([1, 1, 2, 2])
    test_data = np.array([[1, 1, 0], [1, 0, 1]])
    test_label = np.array([1, 2])

    BerNB_model = train_BerNB(X, y)
    res = predict_BerNB(test_data, BerNB_model)
    score = score_BerNB(test_data, test_label, BerNB_model)
    print(res)
    print(score)