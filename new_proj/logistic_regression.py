import chord
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

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
        model.fit(X[i][:n], new_y)

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
 
    print(model.score(new_X, new_y))

if __name__ == '__main__':
    train_logistic()
