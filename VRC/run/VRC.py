from model.model_cpu import Model
import numpy as np
import scipy,scipy.signal
import pyaudio as pa
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

def fft(data):
    time_ruler = data.shape[0] // SHIFT
    if data.shape[0] % SHIFT == 0:
        time_ruler -= 1
    window = np.hamming(NFFT)
    pos = 0
    wined = np.zeros([time_ruler, NFFT])
    for fft_index in range(time_ruler):
        frame = data[pos:pos + NFFT]
        wined[fft_index] = frame * window
        pos += SHIFT
    fft_r = scipy.fft(wined, n=NFFT, axis=1)
    re = fft_r.real.reshape(time_ruler, -1)
    im = fft_r.imag.reshape(time_ruler, -1)
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
    d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
    spec = np.concatenate((c, d), 2)
    return spec

def ifft(datanum_in, red):
    data = datanum_in
    a = np.clip(data[:, :, 0], a_min=-60, a_max=10)
    sss = np.exp(a)
    p = np.sqrt(sss)
    r = p * (np.cos(data[:, :, 1]))
    i = p * (np.sin(data[:, :, 1]))
    dds = np.concatenate((r.reshape(r.shape[0], r.shape[1], 1), i.reshape(i.shape[0], i.shape[1], 1)), 2)
    datanum = dds[:, :, 0] + 1j * dds[:, :, 1]
    fft_s = scipy.ifft(datanum, n=NFFT, axis=1)
    fft_data = fft_s.real
    v = fft_data[:, :NFFT // 2]
    reds = fft_data[-1, NFFT // 2:].copy()
    lats = np.roll(fft_data[:, NFFT // 2:], 1, axis=0)
    lats[0, :] = red
    spec = np.reshape(v + lats, (-1))
    return spec, reds


net=Model("./setting.json")
net.load()
p_in = pa.PyAudio()
py_format = p_in.get_format_from_width(2)
use_device_index = 0
data=np.zeros(TERM)

#Process Of guess
def process(data):
    output=net.sess.run(net.fake_aB_image_test,feed_dict={net.input_model_test:data})
    return output
inf=p_in.get_default_output_device_info()
stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = TERM*2,
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
las=np.zeros([SHIFT+TERM//2])
noise_filter=np.zeros(SHIFT)
print("ノイズ取得中")
t=0.0
up = int(TERM *fs/ sampling_target)
tt = time.time()
print("変換　開始")
while stream.is_active():
    ins=stream.read(up)
    inputs = np.frombuffer(ins,dtype=np.int16).astype(np.float32)/32767.0
    inputs=scipy.signal.resample(inputs,TERM)
    inputs=np.clip(inputs,-1.0,1.0)
    inp=np.append(las,inputs).reshape(TERM+SHIFT+TERM//2)
    las = inputs[-SHIFT-TERM//2:]
    roll_stride=SHIFT//2
    n = fft(inp.copy())[:,:SHIFT,:].astype(np.float32)
    # print(n.shape)
    res=np.asarray([n[:8],n[4:]])

    resp = process(res.copy())
    res2 = resp.copy()[:, :, ::-1, :]
    ress = np.append(resp, res2, axis=2)
    resb,rdd = ifft(ress[0],rdd)
    resb = (np.clip(resb,-1.0,1.0).reshape(-1)*32767)
    res2, rdd2 = ifft(ress[1], rdd2)
    res2 = (np.clip(res2, -1.0, 1.0).reshape(-1) * 32767)
    res=res2
    res[:TERM//2]=(res[:TERM//2]+resb[-TERM//2:])//2
    res=scipy.signal.resample(res,up)
    vs=res.astype(np.int16).tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    print("CPS:%1.3f" % (np.mean(la)/(up/fs)))
    output = stream.write(vs)
    tt = time.time()
