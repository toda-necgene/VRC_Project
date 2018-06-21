import librosa,librosa.display
import sklearn
import matplotlib.pyplot as plt

file_l="../テスト.wav"
file_r="../train/Model/datasets/test/label.wav"

x,xfs = librosa.load(file_l, sr=16000)
y,yfs = librosa.load(file_r, sr=16000)

mfccsx = librosa.feature.mfcc(x, sr=xfs)
mfccsy = librosa.feature.mfcc(y, sr=yfs)
mfccsx = sklearn.preprocessing.scale(mfccsx, axis=1)
mfccsy = sklearn.preprocessing.scale(mfccsy, axis=1)
plt.subplot(2,1,1)
librosa.display.specshow(mfccsx, sr=xfs, x_axis='time')
plt.colorbar()

plt.subplot(2,1,2)
librosa.display.specshow(mfccsy, sr=yfs, x_axis='time')
plt.colorbar()

plt.show()