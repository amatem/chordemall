import reader
import librosa
import numpy as np
np.random.seed(0)
import pickle
import librosa.display
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping, History
import keras
import pprint
import chord
import mir_eval
import math
np.set_printoptions(threshold=np.inf)
pp = pprint.PrettyPrinter(indent=3)

DEBUG = False
model = None

class Dataset(object):
    def __init__(self, raw_X, raw_Y):
        self.num_features = 12
        self.num_outputs = 25
        self.offset=7
        self.window = 5
        self.batch_size = 64
        self.raw_X = raw_X
        self.raw_Y = raw_Y
        self.sample_sizes = []
        self.num_data = 0;
        for i in range(len(raw_X)):
            n = min(raw_X[i].shape[0], len(raw_Y[i][0])-7)
            self.sample_sizes.append(n)
            self.num_data += n
        self.num_test = int(self.num_data * 0.2)
        self.num_data -= self.num_test

    def nth_instance(self, n):
        songnum = 0
        while True:
            if n - self.sample_sizes[songnum] < 0:
                break
            n -= self.sample_sizes[songnum]
            songnum += 1
        return (songnum, n)

    def get_num_iters(self):
        return int((self.num_data - self.batch_size)/self.batch_size)

    def get_num_valid(self):
        return int((self.num_test - self.batch_size)/self.batch_size)

    def get_song(self, song_num):
        rev_x = np.zeros((self.sample_sizes[song_num] - self.sample_sizes[song_num]%self.batch_size,
                          self.window, self.num_features), dtype=np.float64)
        rev_y = []
        for j in range(self.sample_sizes[song_num] - self.sample_sizes[song_num]%self.batch_size):
            for k in range(max(0, j-self.window+1), j + 1):
                rev_x[j][k-j+self.window-1] = self.raw_X[song_num][k]
            rev_y.append(chord.get_majminchord(self.raw_Y[song_num][1][j+7]))

        return rev_x, rev_y
 
    def get_batch_from_shuffled(self, shuffled, i):
        rev_x = np.zeros((self.batch_size, self.window, self.num_features), dtype=np.float64)
        rev_y = np.zeros((self.batch_size, self.num_outputs), dtype=np.float64)
        for j in range(self.batch_size):
            s, ind = self.nth_instance(shuffled[i+j])
            for k in range(max(0, ind-self.window+1), ind + 1):
                rev_x[j][k-ind+self.window-1] = self.raw_X[s][k]
            cat = chord.chord_to_category(chord.get_majminchord(self.raw_Y[s][1][ind+7])) - 1
            rev_y[j][cat] = 1
        return rev_x, rev_y

    def next_batch(self):
        shuffled = range(self.num_data)
        #np.random.shuffle(shuffled)
        for i in range(0, self.num_data - self.batch_size, self.batch_size):
            yield self.get_batch_from_shuffled(shuffled, i)

    def next_validation(self):
        shuffled = range(self.num_data, self.num_data+self.num_test)
        #np.random.shuffle(shuffled)
        for i in range(0, self.num_test - self.batch_size, self.batch_size):
            rev_x, rev_y = self.get_batch_from_shuffled(shuffled, i)
            yield rev_x, rev_y

    def next_batch_inf(self):
        while True:
            for batch_x, batch_y in self.next_batch():
                yield (batch_x, batch_y)

    def next_validation_inf(self):
        while True:
            for batch_x, batch_y in self.next_validation():
                yield (batch_x, batch_y)

def train_network(num_features, learning_rate=0.008, num_outputs=25, num_nodes=128):
    raw_X = pickle.load(open('../data/interim/dnn_chroma.p', 'rb'))
    raw_Y = pickle.load(open('../data/hmmdata/hmm_output.p', 'rb'))
    print('Data loaded...')
    D = Dataset(raw_X, raw_Y)
    print(D.num_data, D.batch_size)
    print(D.get_num_iters())
    print('Data generator created...')

    model = Sequential()
    #model.add(LSTM(num_nodes, input_shape=(7, num_features), return_sequences=True))
    model.add(LSTM(num_nodes, input_shape=(D.window, num_features), dropout=0.5, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(num_nodes, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(num_outputs))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    hist = model.fit_generator(D.next_batch_inf(), epochs=20, steps_per_epoch=D.get_num_iters(),
                                validation_data=D.next_validation_inf(),
                                validation_steps=D.get_num_valid(),
                                callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    model.save('rnn_model.h5')
    pp.pprint(hist.history)
    pickle.dump(hist.history, open('rnn_loss.p', 'wb'))

def predict_beatles():
    raw_X = pickle.load(open('../data/interim/dnn_chroma.p', 'rb'))
    raw_Y = pickle.load(open('../data/hmmdata/hmm_output.p', 'rb'))
    print('Data loaded...')
    D = Dataset(raw_X, raw_Y)
    print(D.num_data, D.batch_size)
    print(D.get_num_iters())
    print('Data generator created...')

    true_pred = []
    all_pred = []
    num_categories = 25
    conf = np.zeros((num_categories, num_categories), dtype=np.float64)

    model = load_model('rnn_model.h5')
    #for song_num in range(len(raw_X)):
    for song_num in range(5):
        #song_num = -1
        song_x, song_y = D.get_song(song_num)
        y_cap_c = model.predict(song_x, batch_size=64)
        y_cap = np.argmax(y_cap_c, axis=1)
        model.reset_states()
        y_cap = [chord.category_to_chord(it+1) for it in y_cap]
        err = mir_eval.chord.mirex(song_y, y_cap)
        print("Song #{}".format(song_num))
        true_pred.append(np.sum(err))
        all_pred.append(len(err))
        for j in range(len(y_cap)):
            conf[chord.chord_to_category(song_y[j])-1][chord.chord_to_category(y_cap[j])-1] += 1

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


def test():
    hist = pickle.load(open('rnn_loss.p', 'rb'))
    pp.pprint(hist)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean Categorical Cross Entropy Error(Loss)')
    plt.title('Loss vs. Epochs')
    plt.show()
    plt.show()

if __name__ == '__main__':
    #plot_chromagram()
    #train_network(12)
    test()
    #predict_beatles()
