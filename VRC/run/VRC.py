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
fs = 16000
channels = 1
gain=0.9
pitch=1.32

def encode(data):
    fs=16000
    _f0,t=pw.dio(data,fs)
    f0=pw.stonemask(data,_f0,t,fs)
    sp=pw.cheaptrick(data,f0,t,fs)
    ap=pw.d4c(data,f0,t,fs)
    return f0.astype(np.float64),np.clip((np.log(sp)+15)/20,-1.0,1.0).astype(np.float64),ap.astype(np.float64)
def decode(f0,sp,ap):
    ap = ap.reshape(-1, 513).astype(np.float)
    f0 = f0.reshape(-1).astype(np.float)
    sp = np.exp(sp.reshape(-1, 1, 513).astype(np.float) * 20 - 15)
    sp=sp.reshape(-1,513).astype(np.float)
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
# up = int(TERM *fs/ sampling_target)+1
up = TERM
stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = TERM*2,
		input = True,
		output = True)
rdd=np.zeros(SHIFT)
rdd2=np.zeros(SHIFT)
tt=time.time()
def normlize(f0,sp):
    me=np.mean(sp,axis=1)
    ra=np.tile((0.0007/(me+1e-8)).reshape(-1,1),(1,513))
    ra[me<5e-7]=1e-8
    sp=sp*(ra)*0.8+sp*0.2
    sp[f0 == 0] =1e-8

    return sp
def terminate():
    stream.stop_stream()
    stream.close()
    p_in.terminate()
    print("Stream Stop")
la=np.zeros([5])
atexit.register(terminate)
las=np.zeros([NFFT])
print("変換　開始")
while stream.is_active():
    ins=stream.read(up)
    tt = time.time()
    inputs = np.frombuffer(ins,dtype=np.int16).astype(np.float64)/32767.0*gain
    inputs=np.clip(inputs,-1.0,1.0)
    inp=np.append(las,inputs).reshape(TERM+NFFT)
    las = inputs[-NFFT:]
    f0,sp,ap = encode(inp.copy())
    sp=sp.astype(np.float32)
    sp=normlize(f0,sp)
    res=sp.reshape(1,65,513,1)
    resp = process(res.copy())
    resb = decode(f0*pitch,resp[0],ap)
    res = (np.clip(resb,-1.0,1.0).reshape(-1)*32767)
    vs=res.astype(np.int16)[SHIFT:-SHIFT].tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    print("CPS:%1.3f" % float(np.mean(la)/(up/fs)))
    output = stream.write(vs)
