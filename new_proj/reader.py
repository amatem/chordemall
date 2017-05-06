import os
from os import path
import pickle
import mir_eval

def get_sample_labels(path):
    (intervals, labels) = mir_eval.io.load_labeled_intervals(path)
    (sample_times, sample_labels) = mir_eval.util.intervals_to_samples(intervals, labels, sample_size=0.01)
    return sample_labels

def read_folder(folder, func):
    res = {}
    for f in os.listdir(folder):
        if f.endswith('.lab'):
            res[f] = func(path.join(folder, f))
    return res

def read_folder_folder(folder, func):
    res = {}
    for f in os.listdir(folder):
        if path.isdir(path.join(folder, f)):
            for f1 in os.listdir(path.join(folder, f)):
                if f1.endswith('.lab'):
                    res[path.join(f,f1)] = func(path.join(folder, f, f1))
    return res

def generate_dataset(input_output_map, input_folder, output_folder, inp_func, out_func, inp_file, out_file):
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
                inp_data.append(inp_func(path.join(input_folder, album, songs_inp[i])))
                out_data.append(out_func(path.join(output_folder, input_output_map[album], songs_out[i])))

    pickle.dump(inp_data, open(inp_file, 'wb'))
    pickle.dump(out_data, open(out_file, 'wb'))

def gen_beatles_dataset(inp_func, out_func, inp_file, out_file):
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

    out_folder = '../data/beatles/chordlab/The Beatles'
    inp_folder = '../input/The Beatles'

    generate_dataset(imap, inp_folder, out_folder, inp_func, inp_file, out_file)
