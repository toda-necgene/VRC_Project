import pyaudio
import numpy as np
import glob
import wave
import random
from copy import deepcopy
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss

    return noisy
FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "../train/Model/datasets/source/"
WAVE_TMP="tmp.wav"
WAVE_OUTPUT_FILENAME = "../train/Model/datasets/train/01/"
file=(WAVE_INPUT_FILENAME+"/結合済み.wav")
index=0
dms=[]
wf = wave.open(file, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')[0:4480000].reshape(2,-1)
data_realA=data[0]
data_realB=data[1]
time=80000
times=data_realA.shape[0]//time
mod=80000//5
for tt in range(5):
    starttime=mod*tt
    for i in range(times):
        ##音声抽出
        star=random.randint(0,5)
        dur=random.randint(15,25)
        data_A_N = deepcopy(data_realA)[i*time+starttime:(i+1)*time+starttime]
        data_B_N=deepcopy(data_realB)[i*time+starttime:(i+1)*time+starttime]
        sss=random.randint(1,5-1)
        #保管
        p=pyaudio.PyAudio()
        datanum=np.append(data_A_N,data_B_N)
        ww = wave.open(WAVE_OUTPUT_FILENAME+str(tt+1)+"-"+str(i+1)+".wav", 'wb')
        ww.setnchannels(1)
        ww.setsampwidth(p.get_sample_size(FORMAT))
        ww.setframerate(RATE)
        ww.writeframes(datanum.tobytes())
        ww.close()
