import chord
import pickle
import numpy as np

class LogisticRegression(object):
    def __init__(num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1), dtype=np.float64)

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def sigmoid_pr(x):
        return DNN.sigmoid(x)*(1-DNN.sigmoid(x))

    def train(self, X, Y, num_epoch):
        for i in range(num_epoch):
            err = 0
            for j in range(X.shape[0]):
                y_cap = X[i].dot(self.W)
                err = 

    def predict(self, X):




def train_logistic():
    X = pickle.load(open('../data/interim/dnn_chroma.p', 'rb'))
    y = pickle.load(open('../data/hmmdata/hmm_output.p', 'rb'))

    num_data = int(len(X) * 0.8)
    num_test = len(X) - num_data

    model = LogisticRegression()

    for i in range(num_data):
        n = min(X[i].shape[0], len(y[i][1])-7)
        new_y = []
        for j in range(n):
            new_y.append(chord.chord_to_category(chord.get_majminchord(y[i][1][j])))
        model.train(X[i][:n], new_y)

    new_X = X[num_data]
    n = min(X[num_data].shape[0], len(y[num_data][1])-7)
    new_y = []
    for j in range(n):
        new_y.append(chord.chord_to_category(chord.get_majminchord(y[num_data][1][j])))

    for i in range(1, num_test):
        n = min(X[i+num_data].shape[0], len(y[i+num_data][1])-7)

        new_X = np.concatenate((new_X, X[i+num_data][:n]), axis=0)
        for j in range(n):
            new_y.append(chord.chord_to_category(chord.get_majminchord(y[i+num_data][1][j])))
 
    print(model.predict(new_X, new_y))

if __name__ == '__main__':
    train_logistic()
