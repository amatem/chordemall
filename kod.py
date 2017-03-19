from pylab import *
from scipy.io import wavfile

set_printoptions(threshold=np.inf)

def read_data(fn):
    sample_freq, snd = wavfile.read(fn)
    if(snd.dtype == dtype('int16')):
        snd = snd / (2.**15)
    else:
        snd = snd / (2.**31)
    
    if len(snd.shape) > 1:
        c1 = np.mean(snd, axis=1)
    else:
        c1 = snd
    return sample_freq, c1

def plot_amplitude(y, sample_freq):
    x = arange(0, len(y), 1.)
    x = x / sample_freq
    x = x * 1000
    figure(1)
    subplot(211)
    plot(x, y) 
    xlabel('Time (ms)')
    ylabel('Amplitude')

def dummy_sins():
    time1 = np.arange(0,5,0.0001)
    time = np.arange(0,15,0.0001)
    data1=np.sin(2*np.pi*300*time1)
    data2=np.sin(2*np.pi*600*time1)
    data3=np.sin(2*np.pi*900*time1)
    data=np.append(data1,data2 )
    data=np.append(data,data3)
    return 1./0.0001, data

def plot_spectogram(y, sample_freq):
    subplot(212)
    Pxx, freqs, bins, im = specgram(y, NFFT=1024, Fs=sample_freq, mode='magnitude')
    xlabel('Time (s)')
    ylabel('Frequency ()')
    colorbar(im, )

def main():
    # sample_freq, y = read_data('stairway.wav')
    sample_freq, y = dummy_sins()
    plot_amplitude(y, sample_freq)
    plot_spectogram(y, sample_freq)
    show()

if __name__ == '__main__':
    main()
