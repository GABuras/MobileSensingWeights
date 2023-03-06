# import librosa
# import numpy
# import skimage.io

# def scale_minmax(X, min=0.0, max=1.0):
#     X_std = (X - X.min()) / (X.max() - X.min())
#     X_scaled = X_std * (max - min) + min
#     return X_scaled

# def spectrogram_image(y, sr, out, hop_length, n_mels):
#     # use log-melspectrogram
#     mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
#                                             n_fft=hop_length*2, hop_length=hop_length)
#     mels = numpy.log(mels + 1e-9) # add small number to avoid log(0)

#     # min-max scale to fit inside 8-bit range
#     img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
#     img = numpy.flip(img, axis=0) # put low frequencies at the bottom in image
#     img = 255-img # invert. make black==more energy

#     # save as PNG
#     skimage.io.imsave(out, img)


# if __name__ == '__main__':
#     # settings
#     hop_length = 512 # number of samples per time-step in spectrogram
#     n_mels = 128 # number of bins in spectrogram. Height of image
#     time_steps = 384 # number of time-steps. Width of image

#     # load audio
#     path = './TrimSounds/20kgbar-1.wav'
#     y, sr = librosa.load(path, offset=1.0, duration=10.0, sr=22050)
#     out = 'out.png'

#     # extract a fixed length window
#     start_sample = 0 # starting at beginning
#     length_samples = time_steps*hop_length
#     window = y[start_sample:start_sample+length_samples]
    
#     # convert to PNG
#     spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
#     print('wrote file', out)

# ---------------------------------------

# import os
# import matplotlib.pyplot as plt

# #for loading and visualizing audio files
# import librosa
# import librosa.display

# #to play audio
# # import IPython.display as ipd

# audio_fpath = "./TrimSounds/"
# audio_clips = os.listdir(audio_fpath)
# print("No. of .wav files in audio folder = ",len(audio_clips))

# x, sr = librosa.load(audio_fpath+audio_clips[2], sr=44100)

# print(type(x), type(sr))
# print(x.shape, sr)

# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)

# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# plt.colorbar()

# ---------------------------------

# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.io import wavfile
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
# sample_rate, samples = wavfile.read('./TrimSounds/20kgbar-1.wav')
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
# plt.pcolormesh(times, frequencies, spectrogram, shading='flat')
# plt.imshow(spectrogram)
# plt.show()

# ---------------------

import os 
import librosa
import librosa.display
import IPython.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

thud_file = "./TrimSounds/20kgbar-1.wav"
thud, sr = librosa.load(thud_file)

FRAME_SIZE = 2048
HOP_SIZE = 512

S_thud = librosa.stft(thud, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
# print(S_thud.shape)
# print(type(S_thud[0][0]))
Y_thud = np.abs(S_thud) ** 2
# print(Y_thud.shape)
# print(type(Y_thud[0][0]))

# def plot_spectogram(Y, sr, hop_length, y_axis="linear"):
#     plt.figure(figsize=(25,10))
#     librosa.display.specshow(Y,
#                              sr=sr,
#                              hop_length=hop_length,
#                              x_axis="time",
#                              y_axis=y_axis)
#     plt.colorbar(format="%+2.f")

# plot_spectogram(Y_thud, sr, HOP_SIZE)

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
plot_spectrogram(Y_thud, sr, HOP_SIZE)