import librosa
import numpy as np
import copy

FILENAME = 'seven.wav'

def main():
    y, sr = librosa.load(FILENAME)
    D = librosa.stft(y,hop_length=512,win_length=2048)
    har_per_sep(0.3,0.3,200,D)

    #thefile = open('test.txt', 'w')
    #for item in pow_spec:
    #	thefile.write("%s\n" % item)
    '''
    print pow_spec
    D_harm, D_perc = librosa.decompose.hpss(D)
    y_harm = librosa.istft(D_harm)
    y_perc = librosa.istft(D_perc)
    librosa.output.write_wav('seven_harm.wav', y_harm, sr)
    librosa.output.write_wav('seven_perc.wav', y_perc, sr)
    '''
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
    librosa.output.write_wav('seven_harm.wav', y_harm, sr)
    librosa.output.write_wav('seven_perc.wav', y_per, sr)


if __name__ == '__main__':
    main()
