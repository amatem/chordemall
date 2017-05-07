import reader
import librosa
import numpy as np
np.random.seed(0)
import pickle
import librosa.display
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping
import keras
import pprint
import chord
np.set_printoptions(threshold=np.inf)
pp = pprint.PrettyPrinter(indent=3)

DEBUG = False
model = None

class Dataset(object):
    def __init__(self, raw_X, raw_Y):
        self.num_features = 12
        self.num_outputs = 25
        self.offset=7
        self.batch_size = 32
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

    def get_batch_from_shuffled(self, shuffled, i):
        rev_x = np.zeros((self.batch_size, self.offset, self.num_features), dtype=np.float64)
        rev_y = np.zeros((self.batch_size, self.num_outputs), dtype=np.float64)
        for j in range(self.batch_size):
            s, ind = self.nth_instance(shuffled[i+j])
            for k in range(max(0, ind-6), ind + 1):
                rev_x[j][k-ind+6] = self.raw_X[s][k]
            cat = chord.chord_to_category(chord.get_majminchord(self.raw_Y[s][1][k+7])) - 1
            rev_y[j][cat] = 1
        return rev_x, rev_y

    def next_batch(self):
        shuffled = range(self.num_data)
        np.random.shuffle(shuffled)
        for i in range(0, self.num_data - self.batch_size, self.batch_size):
            yield self.get_batch_from_shuffled(shuffled, i)

    def next_validation(self):
        shuffled = range(self.num_data, self.num_data+self.num_test)
        np.random.shuffle(shuffled)
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

def train_network(num_features, learning_rate=0.008, num_outputs=25, num_nodes=50):
    raw_X = pickle.load(open('../data/interim/dnn_chroma.p', 'rb'))
    raw_Y = pickle.load(open('../data/hmmdata/hmm_output.p', 'rb'))
    print('Data loaded...')
    D = Dataset(raw_X, raw_Y)
    print(D.num_data, D.batch_size)
    print(D.get_num_iters())
    print('Data generator created...')

    model = Sequential()
    #model.add(LSTM(num_nodes, input_shape=(7, num_features), return_sequences=True))
    model.add(LSTM(num_nodes, input_shape=(7, num_features)))
    #model.add(LSTM(num_nodes))
    model.add(Dense(num_outputs))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    hist = model.fit_generator(D.next_batch_inf(), steps_per_epoch=D.get_num_iters(),
                               epochs=30, validation_data=D.next_validation_inf(),
                               validation_steps=D.get_num_valid(),
                               callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    pp.pprint(hist.history)
    pickle.dump(hist.history, open('rnn_loss.p', 'wb'))
    model.save('rnn_model.h5')

def gen_beatles_dataset():
    raw_X = pickle.load(open('../data/interim/dnn_chroma.p', 'rb'))
    raw_Y = pickle.load(open('../data/hmmdata/hmm_output.p', 'rb'))
    print('Data loaded...')
    D = Dataset(raw_X, raw_Y)
    print(D.num_data, D.batch_size)
    print(D.get_num_iters())
    print('Data generator created...')

    model = load_model('rnn_model.h5')
    for song_num in range(len(raw_X)):
        song_num = -1
        song_x, song_y = D.get_song(song_num)
        y_cap = model.predict(song_x)
        print(y_cap)
        plot_chromagram(raw_X, chroma, song_y, song_num)
        new_X.append(chroma)
    pickle.dump(new_X, open('../data/interim/dnn_chroma.p', 'wb'))

def test():
    hist = pickle.load(open('rnn_loss.p', 'rb'))
    pp.pprint(hist)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.show()

if __name__ == '__main__':
    #plot_chromagram()
    train_network(12)
    #test()
    #gen_beatles_dataset()
