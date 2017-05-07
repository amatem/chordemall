import chord
import pickle
import numpy as np
import mir_eval
import math
import matplotlib.pyplot as plt
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

    #print(model.score(new_X, new_y))
    true_pred = []
    all_pred = []
    num_categories = 25
    conf = np.zeros((num_categories, num_categories), dtype=np.float64)
    #for i in range(len(X)):
    for i in range(5):
        n = min(X[i].shape[0], len(y[i][1])-7)
        new_y = []
        for j in range(n):
            new_y.append(chord.chord_to_category(chord.get_majminchord(y[i][1][j])))
        y_cap = model.predict(X[i][:n])
        err = mir_eval.chord.mirex([chord.category_to_chord(new_y[j]) for j in range(len(new_y))],
                                   [chord.category_to_chord(y_cap[j]) for j in range(len(y_cap))])
        true_pred.append(np.sum(err))
        all_pred.append(len(err))
        for j in range(n):
            conf[new_y[j]-1][y_cap[j]-1] += 1
        
        print("Song#{}".format(i))

    acc = float(sum(true_pred))/sum(all_pred)
    std = math.sqrt((1./sum(all_pred))*sum([(true_pred[i] - acc)**2 for i in range(len(true_pred))]))
    print("ACCURACY: {} - STD: {}".format(acc, std))

    for i in range(len(conf)):
        if sum(conf[i]) != 0:
            conf[i] = [float(conf[i][j]) / sum(conf[i]) for j in range(len(conf[i]))]

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(conf, cmap=plt.cm.Blues, alpha=0.8)
    ax.set_yticks(np.arange(len(conf))+0.5, minor=False)
    ax.set_xticks(np.arange(len(conf[0]))+0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels([chord.category_to_chord(i) for i in range(1, 26)])
    ax.set_yticklabels([chord.category_to_chord(i) for i in range(1, 26)])
    plt.xticks(rotation=90)

    plt.show()

if __name__ == '__main__':
    train_logistic()
