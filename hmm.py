import librosa
import numpy as np
import copy
import mir_eval
import os
from os import path
import labreader
import pickle
import matplotlib.pyplot as plt
import pprint
np.set_printoptions(suppress=True)
pp = pprint.PrettyPrinter(indent=2)
#DEBUG = True
DEBUG = False
#FILENAME = '/Volumes/CORSAIR/input/The Beatles/A Hard Day\'s Night/02 I Should Have Known Better.mp3'

def har_per_sep(alpha,param,kmax,D):
    S=np.abs(D)
    pow_spec=(S**(2*param))
    H=pow_spec/2;
    P=pow_spec/2;
    delta_k=0;
    temp_h=H;
    temp_p=P;
    H_old=H[:];
    P_old=P[:]
    for k in range(kmax):
    	for h in range(1,len(pow_spec)-1):
    		for i in range(1,len(pow_spec[0])-1):
    			delta_k=alpha*(H_old[h][i-1]-2*H_old[h][i]+H_old[h][i+1])/4-(1-alpha)*(P_old[h-1][i]-2*P_old[h][i]+P_old[h+1][i])/4
    			H[h][i]=min(max(H_old[h][i]+delta_k,0),pow_spec[h][i])
    			P[h][i]=pow_spec[h][i]-H[h][i]
    	if(k==kmax-2):
    		temp_h=copy.copy(H)
    		temp_p=copy.copy(P)
    	H_old=H[:]
    	P_old=P[:]
    	print k;
    for h in range(len(pow_spec)):
    	for i in range(len(pow_spec[0])):
    		if(temp_h[h][i]<temp_p[h][i]):
    			H[h][i]=0
    			P[h][i]=pow_spec[h][i]
    		else:
    			P[h][i]=0
    			H[h][i]=pow_spec[h][i]
    y_harm=librosa.istft(H);
    y_per=librosa.istft(P);
    return y_harm, y_per

def read_folder(folder):
    res = {}
    for f in os.listdir(folder):
        if f.endswith('.lab'):
        	res[f] = mir_eval.io.load_labeled_intervals(path.join(folder, f))
    return res

chord_map = {}
rev_chord_map = {}
cnt = 1
for qual in ['maj', 'min']:
    for root in ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']:
        chord_map['{}:{}'.format(root,qual)] = cnt
        rev_chord_map[cnt] = '{}:{}'.format(root,qual)
        cnt =cnt+ 1
    chord_map['{}:{}'.format('D#', qual)] = chord_map['{}:{}'.format('Eb', qual)]
    chord_map['{}:{}'.format('Db', qual)] = chord_map['{}:{}'.format('C#', qual)]
    chord_map['{}:{}'.format('Gb', qual)] = chord_map['{}:{}'.format('F#', qual)]
    chord_map['{}:{}'.format('Ab', qual)] = chord_map['{}:{}'.format('G#', qual)]
chord_map['N']=cnt
rev_chord_map[cnt] = 'N'

def chord_to_category(chord):
	return chord_map[chord]

def category_to_chord(cat):
    return rev_chord_map[cat]

def is_subset(bitmap, ground):
    for i in range(len(bitmap)):
        if bitmap[i] == 0 and ground[i] == 1:
            return False
    return True 

QUALITIES = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'maj9':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '9':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'min11':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '11':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#11':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj13':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min13':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b13':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '1':       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

def get_majminchord(chord):
    root, quality, bass = mir_eval.chord.encode(chord)
    root_char = mir_eval.chord.split(chord)[0]
    if root_char == 'N':
        return root_char
    if is_subset(quality, QUALITIES['maj']):
       return "{}:{}".format(root_char, 'maj')
    if is_subset(quality, QUALITIES['min']):
       return "{}:{}".format(root_char, 'min')
    return 'N'

def generate_features(path):
    y, sr = librosa.core.load(path)
    y = librosa.core.to_mono(y)
    D = librosa.stft(y,8192, hop_length=2205)
    D_harm, D_perc = har_per_sep(D)
    chroma=librosa.feature.chroma_stft(S=np.abs(D_harm),sr=sr)
    return chroma

def extract_chord(path):
    (intervals, labels) = mir_eval.io.load_labeled_intervals(path)
    (sample_times, sample_labels) = mir_eval.util.intervals_to_samples(intervals, labels, sample_size=2205./22050)
    return (sample_times, sample_labels)


def generate_dataset(input_output_map, input_folder, output_folder):
    inp_data = []
    out_data = []
    for album in os.listdir(input_folder):
        if path.isdir(path.join(input_folder, album)):
            if not album in input_output_map:
                continue
            print('Generating data for album {}'.format(album))
            songs_inp = []
            songs_out = []
            for song in os.listdir(path.join(input_folder, album)):
                songs_inp.append(song);
            for song in os.listdir(path.join(output_folder, input_output_map[album])):
                if song.endswith('.lab'):
                    songs_out.append(song)
            songs_inp.sort()
            songs_out.sort()
            for i in range(len(songs_inp)):
                print('Generating data for song: {}'.format(songs_inp[i]))
                inp_data.append(generate_features(path.join(input_folder, album, songs_inp[i])))
                out_data.append(extract_chord(path.join(output_folder, input_output_map[album], songs_out[i])))

    pickle.dump(inp_data, open('data/hmmdata/hmm_input.p', 'wb'))
    pickle.dump(out_data, open('data/hmmdata/hmm_output.p', 'wb'))

def gen_beatles_dataset():
    imap = {
        'Please Please Me': '01_-_Please_Please_Me',
        'With The Beatles': '02_-_With_the_Beatles',
        'A Hard Day\'s Night': '03_-_A_Hard_Day\'s_Night',
        'Beatles For Sale': '04_-_Beatles_for_Sale',
        'Help!': '05_-_Help!',
        'Rubber Soul': '06_-_Rubber_Soul',
        'Revolver': '07_-_Revolver',
        'Sgt. Pepper\'s Lonely Hearts Club Band': '08_-_Sgt._Pepper\'s_Lonely_Hearts_Club_Band',
        'Magical Mystery Tour': '09_-_Magical_Mystery_Tour',
        'TheBeatles1': '10CD1_-_The_Beatles',
        'TheBeatles2': '10CD2_-_The_Beatles',
        'Abbey Road': '11_-_Abbey_Road',
        'Let It Be': '12_-_Let_It_Be',
    }

    out_folder = './data/beatles/chordlab/The Beatles'
    inp_folder = './input/The Beatles'

    generate_dataset(imap, inp_folder, out_folder)

class HMM:
    def __init__(self, num_features, num_categories):
        self.num_features = num_features
        self.num_categories = num_categories
        self.initial = np.zeros(num_categories, dtype=np.float64)
        self.transition = np.zeros((num_categories, num_categories), dtype=np.float64)
        self.emission_mean = np.zeros((num_categories, num_features), dtype=np.float64)
        self.emission_cnt = np.zeros(num_categories, dtype=np.float64)
        self.emission_var = [np.zeros((num_features, num_features), dtype=np.float64) for i in range(num_categories)]
        self.emission_var_inv = [np.zeros((num_features, num_features), dtype=np.float64) for i in range(num_categories)]

    def train(self, X, Y):
        ## Initial emissions
        for i in range(len(Y)):
            cat = chord_to_category(get_majminchord(Y[i][1][0])) - 1
            self.initial[cat] += 1
        self.initial /= np.sum(self.initial)
        print('Initial calculated...')

        ## Transmission
        for i in range(len(Y)):
            for c in range(1, len(Y[i][1])):
                cat1 = chord_to_category(get_majminchord(Y[i][1][c-1])) - 1
                cat2 = chord_to_category(get_majminchord(Y[i][1][c])) - 1
                self.transition[cat1][cat2] += 1
        for i in range(self.num_categories):
            self.transition[i] = self.transition[i]/np.sum(self.transition[i])

        print('Transition calculated...')

        ## Emission
        for i in range(len(X)):
            n = min(X[i].shape[0], len(Y[i][1]))
            for j in range(n):
                cat = chord_to_category(get_majminchord(Y[i][1][j])) - 1
                self.emission_mean[cat] = self.emission_mean[cat] + X[i][j]
                self.emission_cnt[cat] = self.emission_cnt[cat] + 1

        for i in range(self.num_categories):
            self.emission_mean[i] = self.emission_mean[i]/self.emission_cnt[i]

        for i in range(len(X)):
            n = min(X[i].shape[0], len(Y[i][1]))
            for j in range(n):
                cat = chord_to_category(get_majminchord(Y[i][1][j])) - 1
                v = X[i][j].T - self.emission_mean[cat].T
                #print("v:")
                #print(v)
                #print("COV:")
                #print(v[:,np.newaxis].dot(v[:,np.newaxis].T))
                #raw_input('devam:')
                self.emission_var[cat] = self.emission_var[cat] + v[:,np.newaxis].dot(v[:,np.newaxis].T)

        for i in range(self.num_categories):
            self.emission_var[i] = self.emission_var[i]/self.emission_cnt[i]
            self.emission_var_inv[i] = np.linalg.pinv(self.emission_var[i])

    def log_l(self, chroma, cat):
        val = chroma - self.emission_mean[cat].T 
        return -1./2.*(val[:,np.newaxis].T.dot(self.emission_var_inv[cat].dot(val[:,np.newaxis]))+self.emission_var_det[cat]+self.num_features*np.log(2*np.pi))

    def test(self, X, songnum):
        for i in range(self.num_categories):
            self.transition[i][i] = 0
        rev = ['' for i in range(X[songnum].shape[0])]
        T1 = np.zeros((self.num_categories, X[songnum].shape[0]), dtype=np.float64)
        T2 = np.zeros((self.num_categories, X[songnum].shape[0]), dtype=np.int)
        for i in range(self.num_categories):
            T1[i][0] = np.log(self.initial[i])+self.log_l(X[songnum][0].T, i)

        for i in range(X[songnum].shape[0]):
            for j in range(self.num_categories):
                best_k = -1
                for k in range(self.num_categories):
                    if best_k == -1 or T1[k][i-1]+np.log(self.transition[k][j]) > T1[j][i]:
                        best_k = k
                        T1[j][i] = T1[k][i-1]+np.log(self.transition[k][j])
                T2[j][i] = best_k
                T1[j][i] = T1[j][i] + self.log_l(X[songnum][i].T, j)


        last_Z = -1
        for k in range(self.num_categories):
            if last_Z == -1 or T1[k][-1] > T1[last_Z][-1]:
                last_Z = k
        rev[-1] = category_to_chord(last_Z+1)
        for i in reversed(range(X[songnum].shape[0]-1)):
            last_Z = T2[last_Z][i+1]
            rev[i] = category_to_chord(last_Z+1)

        if DEBUG:
            print("T1")
            print(T1)
            print("T2")
            print(T2)

        return rev

    def save_config(self, path):
        pickle.dump((self.initial, self.transition,
                     self.emission_mean, self.emission_cnt,
                     self.emission_var, self.emission_var_inv),
                    open(path, 'wb'))

    def load_config(self, path):
        (self.initial, self.transition, self.emission_mean, self.emission_cnt, self.emission_var, self.emission_var_inv) = pickle.load(open(path, 'rb'))
        self.emission_var_det = []
        for i in range(self.num_categories):
            eig_vals = np.linalg.eigvals(self.emission_var[i])
            self.emission_var_det.append(np.sum(np.log(eig_vals[np.abs(eig_vals) > 1e-12])))


def train_beatles_hmm():
    raw_X = pickle.load(open('data/hmmdata/hmm_input.p', 'rb'))
    raw_Y = pickle.load(open('data/hmmdata/hmm_output.p', 'rb'))
    print("Data loaded...")
    hmm = HMM(12, 25)
    hmm.train(raw_X, raw_Y)
    print("Training completed...")
    hmm.save_config('data/hmmdata/hmm_config.p')

def transform_to_hmm_format(Y, Y_norm):
    Y_new = []
    for i in range(len(Y)):
        Y_new.append((None, Y_norm[i][1][7:7+Y[i].shape[0]]))

    return Y_new

def train_beatles_dnn():
    raw_X = pickle.load(open('data/interim/dnn_c_inp.p', 'rb'))
    raw_Y = pickle.load(open('data/interim/dnn_c_out.p', 'rb'))
    real_Y = pickle.load(open('data/hmmdata/hmm_output.p', 'rb'))
    raw_Y = transform_to_hmm_format(raw_Y, real_Y)
    print("Data loaded...")
    hmm = HMM(12, 25)
    hmm.train(raw_X, raw_Y)
    print("Training completed...")
    hmm.save_config('data/hmmdata/hmm_config.p')

def print_chord(d):
    pp.pprint([(category_to_chord(i+1), d[i]) for i in range(len(d))])

def test_beatles_hmm():
    hmm = HMM(12, 25)
    hmm.load_config('data/hmmdata/hmm_config.p')
    print("Config loaded...")
    raw_X = pickle.load(open('data/interim/dnn_c_inp.p', 'rb'))
    raw_Y = pickle.load(open('data/interim/dnn_c_out.p', 'rb'))
    real_Y = pickle.load(open('data/hmmdata/hmm_output.p', 'rb'))
    raw_Y = transform_to_hmm_format(raw_Y, real_Y)
    print("Data loaded...")
    true_pred = 0
    all_pred = 0
    conf = np.zeros((hmm.num_categories, hmm.num_categories), dtype=np.float64)
    #for i in range(len(raw_X)):
    for i in range(1):
        y_cap = hmm.test(raw_X, i)
        if DEBUG:
            print("initial")
            print_chord(hmm.initial)
            print("transition")
            print_chord(hmm.transition)
            print("emission")
            print_chord(hmm.emission_mean)
            print_chord(hmm.emission_var)
            print("raw_Y")
            pp.pprint(raw_Y[0][1])
            print("y_cap")
            pp.pprint(y_cap)
        n = min(len(y_cap), len(raw_Y[0][1]))
        err = mir_eval.chord.mirex(raw_Y[0][1][:n], y_cap[:n])
        for j in range(n):
            conf[chord_to_category(get_majminchord(raw_Y[i][1][j]))-1][chord_to_category(y_cap[j])-1] += 1
        true_pred += np.sum(err)
        all_pred += len(err)
        print("Song#{}".format(i))
        print y_cap
        print(raw_Y[0][1])

    print("True Predicted: {}\nData Size: {}\nMisclassification Rate:{}"
          .format(true_pred,
                  all_pred,
                  float(all_pred-true_pred)/all_pred))
    print('Confidence Matrix:')
    print(conf)


if __name__ == '__main__':
    #gen_beatles_dataset()
    #train_beatles_dnn()
    test_beatles_hmm()
    #main()
