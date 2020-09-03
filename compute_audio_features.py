import librosa, librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy

import config

os.makedirs(config.path_to_plots,exist_ok=True)




file = os.path.join(config.path_to_examples,'happy_female.wav')

# load audio file with Librosa
signal, sample_rate = librosa.load(file, sr=16000)

# WAVEFORM
# display waveform
plt.figure(figsize=config.FIG_SIZE)
plt.grid(True)
librosa.display.waveplot(signal, sample_rate, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.savefig(os.path.join(config.path_to_plots,'waveform.png'))



# FFT -> power spectrum
# perform Fourier transform
dft = scipy.fft.fft(signal)

# calculate abs values on complex numbers to get magnitude
spectrum = np.abs(dft)

# create frequency variable
f = np.linspace(0, sample_rate, len(spectrum))

# take half of the spectrum and frequency (conjugate symmetry property)
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

# plot spectrum
plt.figure(figsize=config.FIG_SIZE)
plt.grid(True)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.savefig(os.path.join(config.path_to_plots,'spectram.png'))



# STFT -> spectrogram
hop_length = 256 # in num. of samples
n_fft = 512 # window in num. of samples

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sample_rate
n_fft_duration = float(n_fft)/sample_rate

print("STFT window duration is: {}s".format(n_fft_duration))
print("STFT hop length duration is: {}s".format(hop_length_duration))


# perform stft
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

# calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)

# display spectrogram
plt.figure(figsize=config.FIG_SIZE)
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("Spectrogram")
plt.savefig(os.path.join(config.path_to_plots,'spectrogram.png'))


# apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=config.FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.savefig(os.path.join(config.path_to_plots,'log_spectrogram.png'))

#Log_MEL_Filter_Bank_Energies (Log_mel_spectrogram)
mel_spectrogram = librosa.feature.melspectrogram(signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=128)
log_mel_spectrogram= librosa.power_to_db(mel_spectrogram)


plt.figure(figsize=config.FIG_SIZE)
librosa.display.specshow(log_mel_spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel-Spectrogram (dB)")
plt.savefig(os.path.join(config.path_to_plots,'log_Mel_spectrogram.png'))




# MFCCs
# extract 13 MFCCs
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# display MFCCs
plt.figure(figsize=config.FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")
plt.savefig(os.path.join(config.path_to_plots,'mfcc.png'))

# show plots
# plt.show()

# show shapes of different features


print('waveform:'+ str(signal.shape))
print('spectrum:'+ str(left_spectrum.shape))
print('spectrogram:'+ str(spectrogram.T.shape))
print('log_spectrogram:'+ str(log_spectrogram.T.shape))
print('log_Mel_spectrogram:'+ str(log_spectrogram.T.shape))
print('MFCCs:'+ str(MFCCs.T.shape))