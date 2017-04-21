import librosa
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

class DNN(object):
    dropout_rate = 0.5

    def __init__(self, num_features, learning_rate=0.0005, num_outputs = 12,
                 num_layers=3, num_nodes=512, batch_size=512, offset=7):
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.offset = offset
        self.W = []
        self.train_err = []
        self.test_err = []
        init_mult = (1./2000.)
        self.W.append(init_mult*np.random.rand(num_nodes, num_features+1))
        for i in range(0, num_layers-1):
            self.W.append(init_mult*np.random.rand(num_nodes, num_nodes+1))
        self.W.append(init_mult*np.random.rand(num_outputs, num_nodes+1))
        self.activation = []
        self.activation_pr = []
        for i in range(0, num_layers):
            self.activation.append(DNN.relu)
            self.activation_pr.append(DNN.relu_pr)

        self.activation.append(DNN.sigmoid)
        self.activation_pr.append(DNN.sigmoid_pr)

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def relu(x):
        return x*(x>0);

    @staticmethod
    def sigmoid_pr(x):
        return DNN.sigmoid(x)*(1-DNN.sigmoid(x))

    @staticmethod
    def relu_pr(x):
        return x>0;

    @staticmethod
    def loss(X_cap, X):
        return (1./12.)*np.sum(-X*np.log(X_cap)-(1-X)*np.log(1-X_cap))

    @staticmethod
    def loss_pr(X_cap, X):
        #print("CAP: {} - REAL: {}".format(X_cap, X))
        return (1./12.)*((1-X)/(1-X_cap) - X/X_cap)

    def gen_data_mappings(self, raw_X, raw_y):
        self.sample_sizes = []
        self.num_data = 0;
        for i in range(len(raw_X)):
            n = min(raw_X[i].shape[1]-2*self.offset, len(raw_y[i][0])-self.offset)
            self.sample_sizes.append(n)
            self.num_data += n

    def nth_instance(self, n):
        songnum = 0
        while True:
            if n - self.sample_sizes[songnum] < 0:
                break
            n -= self.sample_sizes[songnum]
            songnum += 1
        return (songnum, n)

    def transform_song(self, D, Y):
        X = np.zeros((min(len(Y[0])-7, D.shape[1]-14), D.shape[0]*15), np.float)
        new_y = np.zeros((min(len(Y[0])-7, D.shape[1]-14), 12), np.float)
        for i in range(7, min(len(Y[0]), D.shape[1]-7)):
            X[i-7] = D[..., i-7:i+8].T.flatten()
            new_y[i-7] = Y[1][i]
        return (X, new_y)

    def next_batch(self, X, y):
        shuffled = range(self.num_data)
        np.random.shuffle(shuffled)
        for i in range(0, self.num_data - self.batch_size, self.batch_size):
            rev_x = np.zeros((self.batch_size, self.num_features), dtype=np.float)
            rev_y = np.zeros((self.batch_size, self.num_outputs), dtype=np.float)
            for j in range(self.batch_size):
                s, ind = self.nth_instance(shuffled[i+j])
                rev_x[j] = X[s][...,ind:ind+2*self.offset+1].T.flatten()
                rev_y[j] = y[s][1][ind+self.offset]
            yield (rev_x, rev_y)

    def train(self, X, Y, num_epoch, mean, std):
        print('Total passes: {}'.format(self.num_data/512))
        p = 1
        for i in range(num_epoch):
            for batch_x, batch_y in self.next_batch(X, Y):
                print("PASS #{}".format(p))
                p += 1
                X_batch = []
                V_batch = []
                delta = []
                # Feed forward
                batch_x = (batch_x - mean)/std;
                X_batch.append(np.concatenate((np.ones((batch_x.shape[0],1)),
                                              batch_x), axis=1))
                for l in range(self.num_layers + 1):
                    V_batch.append(X_batch[l].dot(self.W[l].T))
                    new_x = self.activation[l](V_batch[l])
                    if l != self.num_layers:
                        new_x = np.concatenate((np.ones((V_batch[l].shape[0],1)), new_x),
                                               axis=1)
                    X_batch.append(new_x)

                # Backprop
                delta.append(
                    self.activation_pr[-1](V_batch[-1]) * DNN.loss_pr(X_batch[-1], batch_y))
                for l in reversed(range(self.num_layers)):
                    delta.append(self.activation_pr[l](V_batch[l])*delta[-1].dot(
                        self.W[l+1][:,1:]))
                delta = delta[::-1]

                # Update Weights
                for l in range(self.num_layers+1):
                    self.W[l] = self.W[l] - (1./self.batch_size)*self.learning_rate*delta[l].T.dot(X_batch[l])

                # Calculate the errors
                self.train_err.append(DNN.loss(X_batch[-1], batch_y))

def merge_dataset():
    raw_X = pickle.load(open('data/interim/dpp_input.p', 'rb'))
    raw_Y = pickle.load(open('data/interim/dpp_output.p', 'rb'))
    print('Data loaded...')
    dnn = DNN(178*15, learning_rate = 0.0008)
    dnn.gen_data_mappings(raw_X, raw_Y)
    mean = np.zeros((1,178*15))
    std = np.zeros((1,178*15))
    for batch_x, batch_y in dnn.next_batch(raw_X, raw_Y):
        mean += np.sum(batch_x, axis=0)
    mean /= dnn.num_data
    for batch_x, batch_y in dnn.next_batch(raw_X, raw_Y):
        std += np.sum((batch_x-mean)**2, axis=0)
    std = np.sqrt((1./dnn.num_data)*std)
    #print("MEAN: {} - STD: {}", mean, std)
    print('Network initial config completed...')
    dnn.train(raw_X, raw_Y, 50, mean, std)
    print(dnn.train_err)
    plt.plot(dnn.train_err)
    plt.show()


if __name__ == '__main__':
    merge_dataset()
