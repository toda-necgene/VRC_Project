from model.model_cpu import Model
import numpy as np
import scipy,scipy.signal
import pyaudio as pa
import pyworld as pw
import atexit
import time
path_to_networks = './best_model'

NFFT=1024
SHIFT=512
TERM=4096
fs = 44100
channels = 1
sampling_target=16000
samplingrate=441*160

def encode(data):
    fs=16000
    _f0,t=pw.dio(data,fs)
    f0=pw.stonemask(data,_f0,t,fs)
    sp=pw.cheaptrick(data,f0,t,fs)
    ap=pw.d4c(data,f0,t,fs)
    return f0[::4].astype(np.float32),(sp/20.0)[::4].astype(np.float32),ap[::4].astype(np.float32)
def decode(f0,sp,ap):
    ap = np.tile(ap.reshape(-1, 1, 513), (1, 4, 1)).astype(np.float)
    ap = ap.reshape(-1, 513)
    f0 = np.tile(f0.reshape(-1, 1), (1, 4)).astype(np.float)
    f0 = f0.reshape(-1)
    sp=np.tile(sp.reshape(-1,1,513),(1,4,1)).astype(np.float)*20.0
    sp=sp.reshape(-1,513)
    return pw.synthesize(f0,sp,ap,16000)

net=Model("./setting.json")
net.load()
p_in = pa.PyAudio()
py_format = p_in.get_format_from_width(2)
use_device_index = 0
data=np.zeros(TERM)

#Process Of guess
def process(data):
    output=net.sess.run(net.test_outputaB,feed_dict={net.input_model_test:data})
    return output
inf=p_in.get_default_output_device_info()
up = int(TERM *fs/ sampling_target)+1
stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = up*2,
		input = True,
		output = True)
rdd=np.zeros(SHIFT)
rdd2=np.zeros(SHIFT)
tt=time.time()
def terminate():
    stream.stop_stream()
    stream.close()
    p_in.terminate()
    print("Stream Stop")
la=np.zeros([5])
atexit.register(terminate)
las=np.zeros([SHIFT])
noise_filter=np.zeros(SHIFT)
print("ノイズ取得中")
t=0.0
tt = time.time()
print("変換　開始")
while stream.is_active():
    ins=stream.read(up)
    inputs = np.frombuffer(ins,dtype=np.int16).astype(np.float32)/32767.0
    inputs=scipy.signal.resample(inputs,TERM)
    inputs=np.clip(inputs,-1.0,1.0)
    inp=np.append(las,inputs).reshape(TERM+SHIFT)
    las = inputs[-SHIFT:]
    roll_stride=SHIFT//2
    f0,sp,ap = encode(inp.copy())
    sp=sp.astype(np.float32)
    res=np.asarray(sp.reshape(1,15,513,1))
    resp = process(res.copy())
    resb = decode(f0*1.8,resp[0],ap)
    res = (np.clip(resb,-1.0,1.0).reshape(-1)*32767)
    res=scipy.signal.resample(res,up)
    vs=res.astype(np.int16).tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    print("CPS:%1.3f" % (np.mean(la)/(up/fs)))
    output = stream.write(vs)
    tt = time.time()
