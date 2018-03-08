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
files=glob.glob(WAVE_INPUT_FILENAME+"/*.wav")
index=0
dms=[]

wf=wave.open(files[0],'rb')
dds=wf.readframes(CHUNK)
while dds !=b'':
    dms.append(dds)
    dds=wf.readframes(CHUNK)
dms=b''.join(dms)
data=np.frombuffer(dms,'int16')[0:160000]
data2=np.reshape(data, [2,50,1600])
data_realA=data2[0]
data_realB=data2[1]
p=pyaudio.PyAudio()
datanum=np.append(data_realA,data_realB)
ww = wave.open("../train/Model/datasets/train/02/tt.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(datanum.tobytes())
ww.close()
wf=wave.open(files[1],'rb')
dds=wf.readframes(CHUNK)
dms=[]
while dds !=b'':
    dms.append(dds)
    dds=wf.readframes(CHUNK)
dms=b''.join(dms)
data=np.frombuffer(dms,'int16')[0:160000]
data2=np.reshape(data, [2,50,1600])
data_realA=data2[0]
data_realB=data2[1]
p=pyaudio.PyAudio()
datanum=np.append(data_realA,data_realB)
ww = wave.open("../train/Model/datasets/train/02/tt2.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(datanum.tobytes())
ww.close()
for file in files[2:-1]:
    dms=[]
    wf=wave.open(file,'rb')
    dds=wf.readframes(CHUNK)
    while dds !=b'':
        dms.append(dds)
        dds=wf.readframes(CHUNK)
    dms=b''.join(dms)
    data=np.frombuffer(dms,'int16')[0:160000]
    data2=np.reshape(data, [2,50,1600])
    data_realA=data2[0]
    data_realB=data2[1]
    p=pyaudio.PyAudio()
    for i in range(20):
        ##音声加工
        star=random.randint(0,5)
        dur=random.randint(15,25)
        data_A_N = deepcopy(data_realA)
        data_B_N=deepcopy(data_realB)
        sss=random.randint(1,len(files)-1)
        dms = []
        wf = wave.open(files[sss], 'rb')
        dds = wf.readframes(CHUNK)
        while dds != b'':
            dms.append(dds)
            dds = wf.readframes(CHUNK)
        dms = b''.join(dms)
        data = np.frombuffer(dms, 'int16')[0:160000]
        data2 = np.reshape(data, [2, 50, 1600])
        data_realA_2 = data2[0]
        data_realB_2 = data2[1]

        #data_B_N.flags.writable=True
        if i!=0:
            for s in range(star,dur):
                if (star+dur)<50:
                    data_A_N[s]=data_realA_2[s]
                    data_B_N[s] = data_realB_2[s]
        ##ダウンサンプリング
        fs_target = 8000
        ##保存
        datanum=np.append(data_A_N,data_B_N)
        ww = wave.open(WAVE_OUTPUT_FILENAME+str(index)+"-"+str(i+1)+"-"+str(0)+".wav", 'wb')
        ww.setnchannels(1)
        ww.setsampwidth(p.get_sample_size(FORMAT))
        ww.setframerate(RATE)
        ww.writeframes(datanum.tobytes())
        ww.close()
    index+=1