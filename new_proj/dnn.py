import reader
import librosa
import numpy as np
np.random.seed(0)
import pickle
import librosa.display
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import keras
import pprint
from functools import partial
np.set_printoptions(threshold=np.inf)
pp = pprint.PrettyPrinter(indent=3)

model = None

class Dataset(object):
    def __init__(self, raw_X, raw_Y):
        self.num_features = 178*15
        self.num_outputs = 12
        self.offset=7
        self.batch_size = 512
        self.raw_X = raw_X
        self.raw_Y = raw_Y
        self.sample_sizes = []
        self.num_data = 0;
        for i in range(len(raw_X)):
            n = min(raw_X[i].shape[1]-2*self.offset, len(raw_Y[i][0])-self.offset)
            self.sample_sizes.append(n)
            self.num_data += n
        self.num_test = int(self.num_data * 0.2)
        self.num_data -= self.num_test
        self.mean = np.zeros((1,178*15))
        self.std = np.zeros((1,178*15))
        for batch_x, batch_y in self.next_batch():
            self.mean += np.sum(batch_x, axis=0)
        self.mean /= self.num_data
        for batch_x, batch_y in self.next_batch():
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

    def get_num_iters(self):
        return int((self.num_data - self.batch_size)/self.batch_size)

    def get_num_valid(self):
        return int((self.num_test - self.batch_size)/self.batch_size)

    def next_batch(self):
        shuffled = range(self.num_data)
        np.random.shuffle(shuffled)
        for i in range(0, self.num_data - self.batch_size, self.batch_size):
            rev_x = np.zeros((self.batch_size, self.num_features), dtype=np.float64)
            rev_y = np.zeros((self.batch_size, self.num_outputs), dtype=np.float64)
            for j in range(self.batch_size):
                s, ind = self.nth_instance(shuffled[i+j])
                rev_x[j] = self.raw_X[s][...,ind:ind+2*self.offset+1].T.flatten()
                rev_y[j] = self.raw_Y[s][1][ind+self.offset]
            yield (rev_x, rev_y)

    def next_validation(self):
        shuffled = range(self.num_data, self.num_data+self.num_test)
        np.random.shuffle(shuffled)
        for i in range(0, self.num_test - self.batch_size, self.batch_size):
            rev_x = np.zeros((self.batch_size, self.num_features), dtype=np.float64)
            rev_y = np.zeros((self.batch_size, self.num_outputs), dtype=np.float64)
            for j in range(self.batch_size):
                s, ind = self.nth_instance(shuffled[i+j])
                rev_x[j] = self.raw_X[s][...,ind:ind+2*self.offset+1].T.flatten()
                rev_y[j] = self.raw_Y[s][1][ind+self.offset]
            yield rev_x, rev_y

    def next_normalized(self):
        while True:
            for batch_x, batch_y in self.next_batch():
                for i in range(batch_x.shape[0]):
                    batch_x[i] = (batch_x[i] - self.mean) / self.std

                yield (batch_x, batch_y)

    def next_validation_norm(self):
        while True:
            for batch_x, batch_y in self.next_validation():
                for i in range(batch_x.shape[0]):
                    batch_x[i] = (batch_x[i] - self.mean) / self.std

                yield (batch_x, batch_y)

def train_network(num_features, learning_rate=0.008, num_outputs=12, num_nodes=100):
    raw_X = pickle.load(open('../data/interim/dpp_input.p', 'rb'))
    raw_Y = pickle.load(open('../data/interim/dpp_output.p', 'rb'))
    print('Data loaded...')
    D = Dataset(raw_X, raw_Y)
    print(D.num_data, D.batch_size)
    print(D.get_num_iters())
    print('Data generator created...')

    model = Sequential()
    model.add(Dense(num_nodes, input_shape=(num_features,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_nodes))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_nodes))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_outputs))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    hist = model.fit_generator(D.next_normalized(), steps_per_epoch=D.get_num_iters(),
                               epochs=30, validation_data=D.next_validation_norm(),
                               validation_steps=D.get_num_valid(),
                               callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    pp.pprint(hist.history)
    pickle.dump(hist.history, open('loss.p', 'wb'))
    model.save('dnn_model.h5')

def gen_beatles_dataset():
    raw_X = pickle.load(open('../data/interim/dpp_input.p', 'rb'))
    raw_Y = pickle.load(open('../data/interim/dpp_output.p', 'rb'))
    print('Data loaded...')
    D = Dataset(raw_X, raw_Y)
    print(D.num_data, D.batch_size)
    print(D.get_num_iters())

    model = load_model('dnn_model.h5')
    new_X = []
    for song_num in range(len(raw_X)):
        print('Song NUM:{}'.format(song_num))
        song_x = np.zeros((D.sample_sizes[song_num], D.num_features), dtype=np.float)
        song_y = np.zeros((D.sample_sizes[song_num], D.num_outputs), dtype=np.float)
        prev = sum(D.sample_sizes[:song_num])
        for i in range(D.sample_sizes[song_num]):
            s, ind = song_num, i
            song_x[i] = raw_X[s][...,ind:ind+2*D.offset+1].T.flatten()
            song_y[i] = raw_Y[s][1][ind+D.offset]

        chroma = model.predict(song_x)
        #plot_chromagram(raw_X, chroma, song_y, song_num)
        new_X.append(chroma)
        print(chroma.shape)
    print len(new_X)
    pickle.dump(new_X, open('../data/interim/dnn_chroma.p', 'wb'))

def plot_chromagram(raw_X, chroma, song_y, song_num):
    plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    librosa.display.specshow(raw_X[song_num])
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 1, 2, sharex=ax1)
    librosa.display.specshow(chroma.T, y_axis='chroma')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 1, 3, sharex=ax1)
    librosa.display.specshow(song_y.T, y_axis='chroma')
    plt.colorbar()
    plt.tight_layout()

    plt.show()

def test():
    hist = pickle.load(open('loss.p', 'rb'))
    pp.pprint(hist)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.show()

if __name__ == '__main__':
    #plot_chromagram()
    #train_network(178*15)
    #test()
    gen_beatles_dataset()
