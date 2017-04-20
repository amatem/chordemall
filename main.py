import librosa

FILENAME = 'seven.wav'

def main():
    y, sr = librosa.load(FILENAME)
    D = librosa.stft(y)
    D_harm, D_perc = librosa.decompose.hpss(D)
    y_harm = librosa.istft(D_harm)
    y_perc = librosa.istft(D_perc)
    librosa.output.write_wav('seven_harm.wav', y_harm, sr)
    librosa.output.write_wav('seven_perc.wav', y_perc, sr)

if __name__ == '__main__':
    main()
