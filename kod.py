from pylab import *
from scipy.io import wavfile
import scipy

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

def stft(x, fs, frames, hops):
    w = scipy.hanning(frames)
    X = scipy.array([scipy.fft(w*x[i:i+frames]) 
        for i in range(0, len(x) - frames, hops)])
    return X

def istft(X, sz, fs, frames, hops):
    x = scipy.zeros(sz)
    frames = X.shape[1] 
    for n, i in enumarate(range(0, sz - frames, hops)):
        x[i:i+frames] += scipy.real(scipy.iftt(X[n]))
    return x

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

def plot_freq(y, sample_freq, frames, hops):
    N = 1024
    T = 1. / sample_freq
    xf = linspace(0.0, 1.0/(2*T), N/2)
    subplot(212)
    print y[1]
    plot(xf, 2.0/N * abs(y[1][0:N/2]))
    grid()

def main():
    # sample_freq, y = read_data('stairway.wav')
    sample_freq, y = dummy_sins()
    plot_amplitude(y, sample_freq)
    X = stft(y, sample_freq, 1024, 512)
    plot_freq(X, sample_freq, 1024, 512)

    #plot_spectogram(y, sample_freq)
    show()

if __name__ == '__main__':
    main()
