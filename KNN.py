from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

if __name__ == "__main__":
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]])
    y = np.array([1, 1, 2, 2])
    test_data = np.array([[1, 1, 0], [1, 0, 1]])
    test_label = np.array([1, 2])

    KNN_model = train_KNN(X, y, n=2)
    res = predict_KNN(test_data, KNN_model)
    score = score_KNN(test_data, test_label, KNN_model)
    print(res)
    print(score)