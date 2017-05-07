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
                delta = []
                batch_x = (batch_x - self.mean)/self.std;
                X_new.append(np.concatenate((np.ones((batch_x.shape[0],1)),
                                              batch_x), axis=1))
                new_x = np.concatenate((np.ones((V_batch[l].shape[0],1)), new_x),
                                        axis=1)

                self.W = self.W - (1./self.batch_size)*self.learning_rate*delta

    def predict(self, X):
        delta = []
        batch_x = (batch_x - self.mean)/self.std;
        X_new.append(np.concatenate((np.ones((batch_x.shape[0],1)),
                                    batch_x), axis=1))
        new_y = np.concatenate((np.ones((V_batch[l].shape[0],1)), new_x),
                                        axis=1)
        return new_y

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
