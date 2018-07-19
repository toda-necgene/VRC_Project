from Converter import Model
import numpy as np
import scipy
import pyaudio as pa
import atexit
import time
NFFT=1024
SHIFT=512
TERM=4096
gain=1.0
sgain=1.5
noise_gate=0.5
DIL=7
ac=1
ab=0


def fft( data,noise_presets,me):

    time_ruler = data.shape[0] // SHIFT
    if data.shape[0] % SHIFT== 0:
        time_ruler -= 1
    window = np.hamming(NFFT)
    pos = 0
    wined = np.zeros([time_ruler, NFFT])
    for fft_index in range(time_ruler):
        frame = data[pos:pos + NFFT]
        wined[fft_index] = frame * window
        pos +=  SHIFT
    fft_r = scipy.fft(wined, n=NFFT, axis=1)
    re = fft_r.real.reshape(time_ruler, -1)
    im = fft_r.imag.reshape(time_ruler, -1)
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
    c[:,:,0] +=noise_presets*noise_gate
    c[:,:,0] += me*noise_gate
    d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
    spec = np.concatenate((c, d), 2)
    return spec


def ifft(data, redi):
    a = data
    p = np.sqrt(np.exp(np.clip(a[:, :, 0], -30.0, 10.0)))
    r = p * (np.cos(a[:, :, 1]))
    i = p * (np.sin(a[:, :, 1]))
    dds = np.concatenate((r.reshape(r.shape[0], r.shape[1], 1), i.reshape(i.shape[0], i.shape[1], 1)), 2)
    data = dds[:, :, 0] + 1j * dds[:, :, 1]
    fft_s = scipy.ifft(data, n=NFFT, axis=1)
    fft_data = fft_s.real
    v = fft_data[:, :NFFT // 2]
    reds = fft_data[-1, NFFT // 2:].copy()
    lats = np.roll(fft_data[:, NFFT // 2:], 1, axis=0)
    lats[0, :] = redi
    spec = np.reshape((v + lats)/2, (-1))
    return spec, reds

net=Model("../setting.json")
net.build_model()
net.load()
p_in = pa.PyAudio()
py_format = p_in.get_format_from_width(2)
fs = 16000
channels = 1
use_device_index = 0
data=np.zeros(TERM)

#Process Of guess
def process(data,data2):
    dd=np.asarray([data,data2]).reshape([2,data.shape[0],data.shape[1],data.shape[2]])
    output=net.sess.run(net.fake_aB_image_test,feed_dict={net.input_model_test:dd})
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
rddb=np.zeros(SHIFT)
tt=time.time()
def terminate():
    stream.stop_stream()
    stream.close()
    p_in.terminate()
    print("Stream Stop")
la=np.zeros([5])
atexit.register(terminate)
ipt = TERM+SHIFT+SHIFT*DIL
dd=SHIFT*DIL+SHIFT
las=np.zeros([dd])
noise_presets=np.tile(net.args["noise_set"],(net.input_size_test[1],1))
me=-np.mean(noise_presets)

while stream.is_active():
    inputs = np.frombuffer(stream.read(TERM),dtype=np.int16).reshape(TERM).astype(np.float32)/32767.0*sgain
    tt = time.time()
    inp=np.append(las,inputs)
    inp2=np.pad(inp.copy(),(SHIFT//2,0),"constant")[:-SHIFT//2]
    r = max(0, ipt - inp.shape[0])
    if r > 0:
        inp = np.pad(inp,  (r, 0), 'constant')
    las = inputs[-dd:]
    res = fft(inp.reshape(-1),noise_presets,me)[:,:SHIFT,:]
    resb= fft(inp2.reshape(-1),noise_presets,me)[:,:SHIFT,:]
    res = process(res,resb)
    res2 = res.copy()[:, :, ::-1, :]
    res2[:,:,:,1]*=-1
    res = np.append(res, res2, axis=2)
    resa,rdd = ifft(res[0],rdd)
    resb, rddb = ifft(res[1], rddb)
    resb=np.pad(resb[SHIFT//2:],(0,SHIFT//2),"constant")
    res=(resa+resb)*gain
    res = np.clip(res,-1.0,1.0).reshape(-1)*32767

    vs=(res[-TERM:].astype(np.int16)).tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    print("CPS:%1.3f" % (np.mean(la)))
    output = stream.write(vs)
