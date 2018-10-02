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
fft_freq=np.fft.fftfreq(NFFT,samplingrate)
def get_pitch(data):
    wined=data.reshape(-1,NFFT)*np.hamming(NFFT)
    fft_r = scipy.fft(wined, n=NFFT, axis=1)
    re = fft_r.real.reshape(-1,NFFT)
    im = fft_r.imag.reshape(-1,NFFT)
    c = np.power(re, 2) + np.power(im, 2).reshape(-1,NFFT)
    fft_s = scipy.ifft(c, n=NFFT, axis=1)
    sdf = fft_s.real.reshape(-1)
    m=0.0
    l1 = 0.0
    l2 = 0.0
    list_maximum=[]
    nsdf=np.zeros_like(sdf)
    for i in range(data.shape[0]):
        x = data[i]
        m += x * x
        inv = data.shape[0] - i - 1
        x = data[inv]
        m += x * x
        nsdf[inv] = 2.0 * sdf[inv].real / m
        h = nsdf[inv] - l1
        if np.sign(h)!=np.sign(l2) and nsdf[inv]>0:
            list_maximum.append(inv)
        l1=nsdf[inv]
        l2=h
    inv_mx=np.max(nsdf[list_maximum])
    pitch=0.0
    for i in range(len(list_maximum)):
        if inv_mx*0.8<nsdf[[list_maximum[i]]]:
            pitch=nsdf[[list_maximum[i]]]
            break
    return pitch

def fft(data):
    time_ruler = data.shape[0] // SHIFT
    if data.shape[0] % SHIFT == 0:
        time_ruler -= 1
    pos = 0
    wined = np.zeros([time_ruler, NFFT])
    win=np.hamming(NFFT)
    for fft_index in range(time_ruler):
        frame = data[pos:pos + NFFT]
        wined[fft_index] = frame*win
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
    v = fft_data[:, :SHIFT]
    reds = fft_data[-1, SHIFT:].copy()
    lats = np.roll(fft_data[:, SHIFT:].copy(), 1, axis=0)
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
    pitch_source = get_pitch(inputs)

    inputs=np.clip(inputs,-1.0,1.0)
    inp=np.append(las,inputs).reshape(TERM+SHIFT)
    las = inputs[-SHIFT:]
    roll_stride=SHIFT//2
    n = fft(inp.copy())[:,:SHIFT,:].astype(np.float32)
    res=n.reshape(1,-1,SHIFT,2)
    resp = process(res.copy())
    res2 = resp.copy()[:, :, ::-1, :]
    ress = np.append(resp, res2, axis=2)
    resb,rdd = ifft(ress[0],rdd)
    pitch_target = get_pitch(resb.copy())
    res = (np.clip(resb,-1.0,1.0).reshape(-1)*32767)
    res=scipy.signal.resample(res,up)
    vs=res.astype(np.int16).tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    print("CPS:%1.3f" % (np.mean(la)/(up/fs)))
    print(pitch_source,pitch_target)
    output = stream.write(vs)
    tt = time.time()
