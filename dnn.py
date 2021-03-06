import librosa
import numpy as np
import pickle
import copy
import random
import json
import librosa.display
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
        self.M = []
        self.V = []
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 10e-8
        self.train_err = []
        self.test_err = []

        init_mult = (2./(num_features + num_outputs))
        self.M.append(np.zeros((num_nodes, num_features+1), dtype=np.float64))
        self.V.append(np.zeros((num_nodes, num_features+1), dtype=np.float64))
        self.W.append(init_mult*np.random.randn(num_nodes, num_features+1)
                      .astype(np.float64))
        for i in range(0, num_layers-1):
            self.W.append(init_mult*np.random.rand(num_nodes, num_nodes+1)
                          .astype(np.float64))
            self.M.append(np.zeros((num_nodes, num_nodes+1), dtype=np.float64))
            self.V.append(np.zeros((num_nodes, num_nodes+1), dtype=np.float64))
        self.W.append(init_mult*np.random.randn(num_outputs, num_nodes+1)
                      .astype(np.float64))
        self.M.append(np.zeros((num_outputs, num_nodes+1), dtype=np.float64))
        self.V.append(np.zeros((num_outputs, num_nodes+1), dtype=np.float64))

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
        self.num_test = int(self.num_data * 0.2)
        self.num_data -= self.num_test
        self.mean = np.zeros((1,178*15))
        self.std = np.zeros((1,178*15))
        for batch_x, batch_y in self.next_batch(raw_X, raw_y):
            self.mean += np.sum(batch_x, axis=0)
        self.mean /= self.num_data
        for batch_x, batch_y in self.next_batch(raw_X, raw_y):
            self.std += np.sum((batch_x-self.mean)**2, axis=0)
        self.std = np.sqrt((1./self.num_data)*self.std)

    def nth_instance(self, n):
        songnum = 0
        while True:
            if n - self.sample_sizes[songnum] < 0:
                break
            n -= self.sample_sizes[songnum]
            songnum += 1
        return (songnum, n)

    def get_test_data(self, X, y):
        shuffled = range(self.num_data, self.num_data+self.num_test)
        np.random.shuffle(shuffled)
        data_size = 3000
        if(self.num_test < 3000):
            data_size = self.num_test
        rev_x = np.zeros((data_size, self.num_features), dtype=np.float64)
        rev_y = np.zeros((data_size, self.num_outputs), dtype=np.float64)
        for i in range(data_size):
            s, ind = self.nth_instance(shuffled[i])
            rev_x[i] = X[s][...,ind:ind+2*self.offset+1].T.flatten()
            rev_y[i] = y[s][1][ind+self.offset]
        return rev_x, rev_y

    def get_train_sampled(self, X, y):
        shuffled = range(self.num_data)
        np.random.shuffle(shuffled)
        #data_size = int(self.num_data * 0.1)
        data_size = int(3000)
        if self.num_data < 3000:
            data_size = self.num_data
        rev_x = np.zeros((data_size, self.num_features), dtype=np.float64)
        rev_y = np.zeros((data_size, self.num_outputs), dtype=np.float64)
        for i in range(data_size):
            s, ind = self.nth_instance(shuffled[i])
            rev_x[i] = X[s][...,ind:ind+2*self.offset+1].T.flatten()
            rev_y[i] = y[s][1][ind+self.offset]
        return rev_x, rev_y

    def next_batch(self, X, y):
        shuffled = range(self.num_data)
        np.random.shuffle(shuffled)
        for i in range(0, self.num_data - self.batch_size, self.batch_size):
            rev_x = np.zeros((self.batch_size, self.num_features), dtype=np.float64)
            rev_y = np.zeros((self.batch_size, self.num_outputs), dtype=np.float64)
            for j in range(self.batch_size):
                s, ind = self.nth_instance(shuffled[i+j])
                rev_x[j] = X[s][...,ind:ind+2*self.offset+1].T.flatten()
                rev_y[j] = y[s][1][ind+self.offset]
            yield (rev_x, rev_y)

    def train(self, X, Y, num_epoch):
        p = 1
        loss_train, loss_test = ([], [])
        for i in range(num_epoch):
            print("EPOCH #{}".format(i))
            for batch_x, batch_y in self.next_batch(X, Y):
                p += 1
                X_batch = []
                V_batch = []
                delta = []
                # Feed forward
                batch_x = (batch_x - self.mean)/self.std;
                X_batch.append(np.concatenate((np.ones((batch_x.shape[0],1)),
                                              batch_x), axis=1))
                for l in range(self.num_layers + 1):
                    V_batch.append(X_batch[l].dot(self.W[l].T))
                    new_x = self.activation[l](V_batch[l])
                    if l != self.num_layers:
                        new_x = np.concatenate((np.ones((V_batch[l].shape[0],1)), new_x),
                                               axis=1)
                    print new_x.flags
                    read_input()
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
                    self.M[l] = (self.beta_1)*self.M[l]+(1.-self.beta_1)*delta[l].T.dot(X_batch[l])
                    self.V[l] = (self.beta_2)*self.V[l]+(1.-self.beta_2)*(delta[l].T.dot(X_batch[l]))**2
                    M_cap = self.M[l]/(1-self.beta_1**p)
                    V_cap = self.V[l]/(1-self.beta_2**p)
                    self.W[l] = self.W[l] - (1./self.batch_size)*self.learning_rate*(1./np.sqrt(V_cap))*M_cap

            if (i + 1)%10 == 0:
                ltrain, ltest = self.get_training_test_loss(X, Y)
                loss_train.append(ltrain)
                loss_test.append(ltest)
        return loss_train, loss_test

    def feedforward(self, X, W):
        X_batch = []
        V_batch = []
        # Feed forward
        X = (X - self.mean)/self.std;
        X_batch = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        for l in range(self.num_layers + 1):
            V_batch = X_batch.dot(W[l].T)
            new_x = self.activation[l](V_batch)
            if l != self.num_layers:
                new_x = np.concatenate((np.ones((V_batch.shape[0],1)), new_x),
                                        axis=1)
            X_batch = new_x

        return X_batch

    def get_training_test_loss(self, X, Y):
        (X_test, Y_test) = self.get_test_data(X, Y)
        (X_train, Y_train) = self.get_train_sampled(X, Y)
        err_train = []
        err_test = []
        Y_train_cap = self.feedforward(X_train, self.W)
        Y_test_cap = self.feedforward(X_test, self.W)
        err_train = DNN.loss(Y_train_cap, Y_train)
        err_test = DNN.loss(Y_test_cap, Y_test)
        return (err_train, err_test)

    def get_chroma(self, X, Y, song_num):
        song_x = np.zeros((self.sample_sizes[song_num], self.num_features), dtype=np.float)
        song_y = np.zeros((self.sample_sizes[song_num], self.num_outputs), dtype=np.float)
        prev = sum(self.sample_sizes[:song_num])
        for i in range(self.sample_sizes[song_num]):
            s, ind = song_num, i
            song_x[i] = X[s][...,ind:ind+2*self.offset+1].T.flatten()
            song_y[i] = Y[s][1][ind+self.offset]

        return self.feedforward(song_x, self.W), song_y

    def save_config(self, fpath):
        pickle.dump(self.W, open(fpath, 'wb'))

    def load_config(self, fpath):
        self.W = pickle.load(open(fpath, 'rb'))

def train_beatles(num_batches, flag):
    raw_X = pickle.load(open('data/interim/dpp_input.p', 'rb'))
    raw_Y = pickle.load(open('data/interim/dpp_output.p', 'rb'))
    print('Data loaded...')
    dnn = DNN(178*15, learning_rate = 0.0007)
    dnn.gen_data_mappings(raw_X, raw_Y)
    if flag:
        dnn.load_config('data/interim/dnn_config.p')
        print('Config loaded...')

    for batch_num in range(num_batches):
        print("########### BATCH NUM: {}".format(batch_num))
        #print("MEAN: {} - STD: {}", mean, std)
        dnn.train(raw_X, raw_Y, 100)
        dnn.save_config('data/interim/dnn_config_{}.p'.format(batch_num))

def train_toy(num_batches, is_debug):
    toy_X = pickle.load(open('data/interim/toy_input.p', 'rb'))
    toy_Y = pickle.load(open('data/interim/toy_output.p', 'rb'))
    print("Data loaded...")
    dnn = DNN(178*15, learning_rate = 0.0007)
    dnn.gen_data_mappings(toy_X, toy_Y)
    loss_tr = []
    loss_te = []
    print("DNN configured...")
    for batch_num in range(num_batches):
        print('Batch num #{}'.format(batch_num))
        loss_train, loss_test = dnn.train(toy_X, toy_Y, 100)
        loss_tr += loss_train
        loss_te += loss_test
    print('DNN trained')
    print loss_tr
    if is_debug:
        open('toy_loss.json', 'w').write(json.dumps((loss_tr, loss_te)))
        plot_err('toy_loss.json')

def plot_err(fpath):
    (loss_train, loss_test) = json.loads(open(fpath, 'r').read())
    print(loss_train)
    print(loss_test)
    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.show()

def gen_toy_dataset():
    raw_X = pickle.load(open('data/interim/dpp_input.p', 'rb'))
    raw_Y = pickle.load(open('data/interim/dpp_output.p', 'rb'))
    print('Data loaded...')
    toy_X = raw_X[1:3]
    toy_Y = raw_Y[1:3]
    pickle.dump(toy_X, open('data/interim/toy_input.p', 'wb'))
    pickle.dump(toy_Y, open('data/interim/toy_output.p', 'wb'))

def gen_chromagram_beatles():
    raw_X = pickle.load(open('data/interim/dpp_input.p', 'rb'))
    raw_Y = pickle.load(open('data/interim/dpp_output.p', 'rb'))
    print('Data loaded...')
    dnn = DNN(178*15, learning_rate = 0.0007)
    dnn.gen_data_mappings(raw_X, raw_Y)
    dnn.load_config('data/interim/dnn_config.json')
    print('Config loaded...')
    chroma_X, chroma_Y = [], [];
    for i in range(len(raw_X)):
        c_x, c_y = dnn.get_chroma(raw_X, raw_Y, i)
        chroma_X.append(c_x)
        chroma_Y.append(c_y)
        print("Song #{}".format(i))
    pickle.dump(chroma_X, open('data/interim/dnn_c_inp.p', 'wb'))
    pickle.dump(chroma_Y, open('data/interim/dnn_c_out.p', 'wb'))

def config_debug():
    raw_X = pickle.load(open('data/interim/dpp_input.p', 'rb'))
    raw_Y = pickle.load(open('data/interim/dpp_output.p', 'rb'))
    print('Data loaded...')
    dnn = DNN(178*15, learning_rate = 0.0007)
    dnn.gen_data_mappings(raw_X, raw_Y)
    dnn.load_config('data/interim/dnn_config.json')
    print('Config loaded...')

if __name__ == '__main__':
    np.random.seed(0)
    #config_debug()
    #gen_toy_dataset()
    #plot_err('toy_loss.json')
    #train_toy(1, False)
    train_beatles(3, False)
    #plot_err()
    #gen_chromagram_beatles()
