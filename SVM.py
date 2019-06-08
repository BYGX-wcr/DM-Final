from sklearn.svm import LinearSVC
import numpy as np

def train_SVM(data, labels):
    svm = LinearSVC(dual=False)
    svm.fit(data, labels)

    return svm

def predict_SVM(data, model):
    return model.predict(data)

def score_SVM(data, labels, model):
    return model.score(data, labels)

if __name__ == "__main__":
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]])
    y = np.array([1, 1, 2, 2])
    test_data = np.array([[1, 1, 0], [1, 0, 1]])
    test_label = np.array([1, 2])

    SVM_model = train_SVM(X, y)
    res = predict_SVM(test_data, SVM_model)
    score = score_SVM(test_data, test_label, SVM_model)
    print(res)
    print(score)