function [stat,delta,double_delta,stft,power_spec]=extract_lfcc(speech,Fs,Window_Length,NFFT,No_Filter)
% Function for computing LFCC features
% Usage: [stat,delta,double_delta]=extract_lfcc(file_path,Fs,Window_Length,No_Filter)
%
% Input: file_path=Path of the speech file
%        Fs=Sampling frequency in Hz
%        Window_Length=Window length in ms
%        NFFT=No of FFT bins
%        No_Filter=No of filter
%
%Output: stat=Static LFCC (Size: NxNo_Filter where N is the number of frames)
%        delta=Delta LFCC (Size: NxNo_Filter where N is the number of frames)
%        double_delta=Double Delta LFCC (Size: NxNo_Filter where N is the number of frames)
%        stft
%
%        Written by Md Sahidullah at School of Computing, University of
%        Eastern Finland (email: sahid@cs.uef.fi)
%
%        Implementation details are available in the following paper:
%        M. Sahidullah, T. Kinnunen, C. Hanil�i, �A comparison of features
%        for synthetic speech detection�, Proc. Interspeech 2015,
%        pp. 2087--2091, Dresden, Germany, September 2015.
%speech=readwav(file_path,'s',-1);
%rng('default');
%speech=speech+randn(size(speech))*eps;                           %dithering
%-------------------------- PRE-EMPHASIS ----------------------------------
speech = filter( [1 -0.97], 1, speech);
%---------------------------FRAMING & WINDOWING----------------------------
frame_length_inSample=(Fs/1000)*Window_Length;
framedspeech=buffer(speech,frame_length_inSample,frame_length_inSample/2,'nodelay')';
w=hamming(frame_length_inSample);
y_framed=framedspeech.*repmat(w',size(framedspeech,1),1);
%--------------------------------------------------------------------------
f=(Fs/2)*linspace(0,1,NFFT/2+1);
filbandwidthsf=linspace(min(f),max(f),No_Filter+2);
stft = complex(fft(y_framed',NFFT)); % ensure result is complex even if y_framed is all zeroes
power_spec=(abs(stft)).^2;
power_spec=power_spec(1:(NFFT/2)+1,:)';
filterbank=zeros((NFFT/2)+1,No_Filter);
for i=1:No_Filter
    filterbank(:,i)=trimf(f,[filbandwidthsf(i),filbandwidthsf(i+1),...
        filbandwidthsf(i+2)]);
end
filbanksum=power_spec*filterbank(1:end,:);
%-------------------------Calculate Static Cepstral------------------------
t=dct(log10(filbanksum'+eps));
t=(t(1:No_Filter,:));
stat=t';
delta=deltas(stat',3)';
double_delta=deltas(delta',3)';
%--------------------------------------------------------------------------
