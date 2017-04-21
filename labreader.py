import os
from os import path
import pickle
import pprint
import mir_eval
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
np.set_printoptions(threshold=np.inf)

def read_lab_file(path):
    rev = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            raw = line.split()
            if len(raw) > 3:
                print("Irregular line found: {}".format(raw))
            rev.append((float(raw[0]), float(raw[1]), raw[2]))

    return rev

def read_folder(folder):
    res = {}
    for f in os.listdir(folder):
        if f.endswith('.lab'):
            res[f] = read_lab_file(path.join(folder, f))
    return res

def read_folder_folder(folder):
    res = {}
    for f in os.listdir(folder):
        if path.isdir(path.join(folder, f)):
            for f1 in os.listdir(path.join(folder, f)):
                if f1.endswith('.lab'):
                    res[path.join(f,f1)] = read_lab_file(path.join(folder, f, f1))
    return res

def extract_bitmaps(path):
    (intervals, labels) = mir_eval.io.load_labeled_intervals(path)
    (sample_times, sample_labels) = mir_eval.util.intervals_to_samples(intervals, labels)
    (root_num, bitmap, bass) = mir_eval.chord.encode_many(sample_labels)
    abs_bitmaps = mir_eval.chord.rotate_bitmaps_to_roots(bitmap, root_num)
    return (sample_times, abs_bitmaps)

def generate_features(path):
    y, sr = librosa.core.load(path)
    y = librosa.core.to_mono(y)
    print('SAMPLING RATE: {}'.format(sr))
    hop_length = int(sr/10.)
    D = librosa.core.stft(y, 8192, hop_length)
    filt = librosa.filters.mel(sr, 8192, 178, 30, 5500)
    return np.log(1 + filt.dot(np.abs(D)**2))


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
                out_data.append(extract_bitmaps(path.join(output_folder, input_output_map[album], songs_out[i])))

    pickle.dump(inp_data, open('dpp_input.p', 'wb'))
    pickle.dump(out_data, open('dpp_output.p', 'wb'))

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

chord_map = {}
cnt = 1
for qual in ['maj', 'min']:
    for root in ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']:
        chord_map['{}:{}'.format(root,qual)] = cnt
        cnt += 1
    chord_map['N'] = cnt
    cnt += 1

def chord_to_category(chord):
    return chord_map[chord]

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

def calc_chord_freqs():
    data = read_folder_folder('/Users/neumann/bilkent/eee485/project/chordemall/data/beatles/chordlab/The Beatles/')
    pp = pprint.PrettyPrinter(indent=3)
    alph = {}
    for key in data:
        print('FILENAME: {}'.format(key))
        #pp.pprint(data[key])
        for chord in data[key]:
            if not chord[2] in alph:
                alph[chord[2]] = 1;
            else:
                alph[chord[2]] += 1;
    print("ALPHABETSIZE: {}".format(len(alph)))
    pp.pprint([(key, get_majminchord(key), alph[key]) for key in alph])

if __name__ == '__main__':
    calc_chord_freqs()
    #gen_beatles_dataset()
