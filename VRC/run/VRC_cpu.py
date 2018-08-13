from ..model.model_proto import Model
import numpy as np
import scipy
import pyaudio as pa
import atexit
import time
path_to_networks = './Network'
graph_filename = './graph'
NFFT=1024
SHIFT=512
TERM=4096
dilation_size=7
boost=1.0
otp_boost=1.0
bass_cut=20.0
noise_filter_rate=0.95
print(np.fft.fftfreq(NFFT,1.0/16000)[13])
def preem(data):
    dd=np.roll(data.copy(), -1)
    dd[-1]=0.0
    return data - 0.97*dd
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


net=Model("../setting.json")
net.load()
p_in = pa.PyAudio()
py_format = p_in.get_format_from_width(2)
fs = 16000
channels = 1
use_device_index = 0
data=np.zeros(TERM)

#Process Of guess
def process(data):
    output=net.sess.run(net.fake_aB_image_test,feed_dict={net.input_model_test:data})
    return output
inf=p_in.get_default_output_device_info()
print(inf)
stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = TERM,
		input = True,
		output = True)
rdd=np.zeros(SHIFT)
rdd2=np.zeros(SHIFT)
rdd3=np.zeros(SHIFT)
rdd4=np.zeros(SHIFT)
tt=time.time()
def terminate():
    stream.stop_stream()
    stream.close()
    p_in.terminate()
    print("Stream Stop")
la=np.zeros([5])
atexit.register(terminate)
las=np.zeros([SHIFT*dilation_size+SHIFT])
noise_filter=np.zeros(SHIFT)
print("ノイズ取得中")
t=0.0
while t<3:
    ins = stream.read(TERM)
    inp = np.frombuffer(ins, dtype=np.int16).reshape(TERM).astype(np.float32) / 32767.0 * boost
    s = fft(inp.copy())[:, :SHIFT, :].astype(np.float32)
    if np.mean(noise_filter)!=0.0:
        noise_filter=(np.mean(s[:,:,0],axis=0)+noise_filter)/2
    else:
        noise_filter = np.mean(s[:, :, 0], axis=0)
    t+=TERM/fs
print("変換　開始")
while stream.is_active():
    ins=stream.read(TERM)
    tt = time.time()
    inputs = np.frombuffer(ins,dtype=np.int16).reshape(TERM).astype(np.float32)/32767.0*boost
    inputs=np.clip(inputs,-1.0,1.0)
    # inputs=preem(inputs)
    inp=np.append(las,inputs).reshape(TERM+SHIFT*(dilation_size+1))
    las = inputs[-SHIFT*dilation_size-SHIFT:]
    roll_stride=SHIFT//2
    n = fft(inp.copy())[:,:SHIFT,:].astype(np.float32)
    n[:,:,0]-=noise_filter*noise_filter_rate
    inp2=np.roll(inp.copy(),roll_stride)
    inp2[:roll_stride]=0.0
    n2 = fft(inp2)[:, :SHIFT, :].astype(np.float32)
    n2[:, :, 0] -= noise_filter*noise_filter_rate
    inp3 = np.roll(inp.copy(), roll_stride*2)
    inp3[:roll_stride*2] = 0.0
    n3 = fft(inp3)[:, :SHIFT, :].astype(np.float32)
    inp4 = np.roll(inp.copy(), roll_stride*3)
    inp4[:roll_stride*3] = 0.0
    n4 = fft(inp4)[:, :SHIFT, :].astype(np.float32)

    res=np.asarray([n,n2,n3,n4])
    # res=np.asarray([n,n2])
    resp = process(res.copy())
    resp[:,:,:,:]-=res[:,:8,:,:]*0.2
    resp[:, :, 48:, 0] -= 2.1
    # resp[:, :, 100:, 0] -= 4.1
    # resp[:, :, :, 0]-=-noise_filter*0.2
    # res = res
    res2 = resp.copy()[:, :, ::-1, :]
    ress = np.append(resp, res2, axis=2)
    res,rdd = ifft(ress[0],rdd)
    res2,rdd2 = ifft(ress[1],rdd2)
    res2=np.roll(res2,-roll_stride)
    res2[-roll_stride:]=0.0
    res3,rdd3 = ifft(ress[2],rdd3)
    res3 = np.roll(res3, -roll_stride*2)
    res3[-roll_stride*2:] = 0.0
    res4,rdd4 = ifft(ress[3],rdd4)
    res4 = np.roll(res4, -roll_stride*3)
    res4[-roll_stride*3:] = 0.0
    respond=(res+res2+res3+res4)/4
    # respond = (res + res2) / 2
    res = (np.clip((respond)*otp_boost,-0.8,0.8).reshape(-1)*32767)
    vs=(res.astype(np.int16)).tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    # print("CPS:%1.3f" % (np.mean(la)/(TERM/fs)))
    output = stream.write(vs)
