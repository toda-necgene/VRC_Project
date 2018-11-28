from model.model_cpu import Model
import numpy as np
import pyaudio as pa
import pyworld as pw
import atexit
import time
args=dict()
a=np.load("./voice_profile.npy")

args["pitch_rate_mean_s"] = a[0]
args["pitch_rate_mean_t"] = a[1]
args["pitch_rate_var"] = a[2]

padding=1024
padding_shift=512
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
tt=time.time()
def terminate():
    stream.stop_stream()
    stream.close()
    p_in.terminate()
    print("Stream Stop")
la=np.zeros([5])
atexit.register(terminate)
las=np.zeros([padding])
print("変換　開始")
while stream.is_active():
    ins=stream.read(up)
    tt = time.time()
    inputs = np.frombuffer(ins,dtype=np.int16).astype(np.float64)/32767.0*gain
    inputs=np.clip(inputs,-1.0,1.0)
    inp=np.append(las,inputs).reshape(TERM+padding)
    las = inputs[-padding:]
    f0,sp,ap = encode(inp.copy())
    sp=sp.astype(np.float32)
    res=sp.reshape(1,65,513,1)
    resp = process(res.copy())
    resb = decode((f0-args["pitch_rate_mean_s"])*args["pitch_rate_var"]+args["pitch_rate_mean_t"],resp[0],ap)
    res = (np.clip(resb,-1.0,1.0).reshape(-1)*32767)
    vs=res.astype(np.int16)[padding_shift:-padding_shift].tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    print("CPS:%1.3f" % float(np.mean(la)/(up/fs)))
    output = stream.write(vs)
