import librosa
import numpy as np
import pickle
import random

class DNN(object):
    num_outputs = 12
    dropout_rate = 0.5

    def __init__(self, num_features, num_layers, num_nodes):
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.W = []
        self.train_err = []
        self.test_err = []
        self.W.append(np.random.rand(num_nodes, num_features+1))
        for i in range(0, num_layers-1):
            self.W.append(np.random.rand(num_nodes, num_nodes+1))
        self.W.append(np.random.rand(DNN.num_outputs, num_nodes+1))

    def transform_song(self, D, Y):
        X = np.zeros((min(len(Y[0])-7, D.shape[1]-14), D.shape[0]*15), np.float)
        new_y = np.zeros((min(len(Y[0])-7, D.shape[1]-14), 12), np.float)
        for i in range(7, min(len(Y[0]), D.shape[1]-7)):
            X[i-7] = D[..., i-7:i+8].T.flatten()
            new_y[i-7] = Y[1][i]
        return (X, new_y)

    def next_batch(self, X, Y):
        ind = numpy.random.shuffe(range(X.shape[0]))
        for i in range(0, X.shape[0]-512, 512):
            return (X[i:i+512], Y[i:i+512])


    def train(self, X, Y, num_epoch):
        for i in range(num_epoch):
            for batch_x, batch_y in self.next_batch(X, Y, num_epoch):


    def plot_learning_rate():
        pass

def merge_dataset():
    dnn = DNN(176*15, 3, 512)
    print('Loading data...')
    raw_X = pickle.load(open('dpp_input.p', 'rb'))
    raw_Y = pickle.load(open('dpp_output.p', 'rb'))
    print('Data loaded...')
    (X, Y) = dnn.transform_song(raw_X[0], raw_Y[0])
    for i in range(1, len(raw_X)):
        (new_X, new_Y) = dnn.transform_song(raw_X[i], raw_Y[i])
        X = np.concatenate((X, new_X), axis=0)
        Y = np.concatenate((Y, new_Y), axis=0)
        print('Song #{}'.format(i))

    print('X: {} - Y: {}'.format(X.shape, Y.shape))
    pickle.dump(X, open('merged_input.p', 'wb'))
    pickle.dump(Y, open('merged_output.p', 'wb'))

if __name__ == '__main__':
    merge_dataset()
