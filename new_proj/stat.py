import reader
import chord
import pprint
import matplotlib.pyplot as plt
import numpy as np
import mir_eval
import librosa
import librosa.display
pp = pprint.PrettyPrinter(indent = 3)

def get_sample_labels(path):
    (intervals, labels) = mir_eval.io.load_labeled_intervals(path)
    (sample_times, sample_labels) = mir_eval.util.intervals_to_samples(intervals, labels, sample_size=0.01)
    return sample_labels

def print_chord_freqs():
    data = reader.read_folder_folder('/Users/neumann/bilkent/eee485/project/chordemall/data/beatles/chordlab/The Beatles/', get_sample_labels)
    freqs = {}
    for key in data:
        print('FILENAME: {}'.format(key))
        #pp.pprint(data[key])
        for ch in data[key]:
            if not chord.get_majminchord(ch):
                continue
            chord_new = chord.get_majminchord(ch)
            if not chord_new in freqs:
                freqs[chord_new] = 1
            else:
                freqs[chord_new] += 1
    print('Number of different chords: {}'.format(len(freqs)))
    pp.pprint(freqs)

def get_change_intervals(path):
    (intervals, labels) = mir_eval.io.load_labeled_intervals(path)
    print(path)
    rev = []
    for i in intervals:
        rev.append(i[1] - i[0])
    return rev

def note_change_histogram():
    data = reader.read_folder_folder('/Users/neumann/bilkent/eee485/project/chordemall/data/beatles/chordlab/The Beatles/', get_change_intervals)
    vals = []
    for key in data:
        print('SONGNAME: {}'.format(key))
        vals = vals + data[key]

    n, bins, patches = plt.hist(vals, 100, (0, 10))
    plt.grid(True)
    plt.xticks(np.arange(-1, 10, 0.1))
    plt.show()

def onset_detection_test():
    out_file = '../data/beatles/seglab/The Beatles/01_-_Please_Please_Me/01_-_I_Saw_Her_Standing_There.lab'
    inp_file = '../input/The Beatles/Please Please Me/01 I Saw Her Standing There.mp3'
    y, sr = librosa.core.load(inp_file)
    y = librosa.core.to_mono(y)
    print('SAMPLING RATE: {}'.format(sr))
    o_env = librosa.onset.onset_strength(y, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    print("FOUND ONSETS")
    print(times[onset_frames])
    print("GROUND INTERVALS")
    (intervals, labels) = mir_eval.io.load_labeled_intervals(out_file)
    print(intervals)
    D = librosa.stft(y)
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             x_axis='time', y_axis='log')
    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(times, o_env, label='Onset strength')
    plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
               linestyle='--', label='Onsets')
    plt.vlines(np.asarray(intervals)[..., 0], 0, o_env.max(), color='b', alpha=0.9,
               linestyle='--', label='Ground')
    plt.axis('tight')
    plt.legend(frameon=True, framealpha=0.75)
    plt.show()


if __name__ == '__main__':
    onset_detection_test()
    #note_change_histogram()
    #print_chord_freqs()
