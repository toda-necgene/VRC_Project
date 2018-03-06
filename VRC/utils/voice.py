import pyaudio
import numpy as np
import glob
import wave
import random
from copy import deepcopy
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    var = 0.1
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
WAVE_INPUT_FILENAME = "./Model/datasets/source/"
WAVE_TMP="tmp.wav"
WAVE_OUTPUT_FILENAME = "./Model/datasets/test/"
files=glob.glob(WAVE_INPUT_FILENAME+"/*.wav")
index=0
for file in files:
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
    for i in range(10):
        ##音声加工
        star=0
        dur=50
        data_B_N=deepcopy(data_realB)
        #data_B_N.flags.writable=True
        if i!=0:
            for s in range(star,star+dur):
                data_B_N[s]*=0
        ##ダウンサンプリング
        fs_target = 8000
        ##保存
        datanum=np.append(data_realA,data_B_N)
        ww = wave.open(WAVE_OUTPUT_FILENAME+str(index)+"-"+str(i+1)+"-"+str(0)+".wav", 'wb')
        ww.setnchannels(1)
        ww.setsampwidth(p.get_sample_size(FORMAT))
        ww.setframerate(RATE)
        ww.writeframes(datanum.tobytes())
        ww.close()
    index+=1