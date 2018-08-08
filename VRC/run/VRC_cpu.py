from Converter import Model
import numpy as np
import scipy
import pyaudio as pa
import atexit
import time
path_to_networks = './Network'
graph_filename = './graph'
NFFT=128
SHIFT=64
TERM=4096
ac=1
ab=0


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
    data = datanum_in[ac:, ab:, :]
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
    tt = time.time()
    output,_,_=net.sess.run(net.fake_B_image,feed_dict={net.input_model:data})
    # print("TT:"+str(time.time()-tt))
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
tt=time.time()
def terminate():
    stream.stop_stream()
    stream.close()
    p_in.terminate()
    print("Stream Stop")
la=np.zeros([5])
atexit.register(terminate)
las=np.zeros([SHIFT])
while stream.is_active():
    inputs = np.frombuffer(stream.read(TERM),dtype=np.int16).reshape(TERM).astype(np.float32)/32767.0
    tt = time.time()
    inp=np.append(las,inputs).reshape(1,TERM+SHIFT,1)
    las = inputs[-SHIFT:]
    res = np.zeros([1, TERM//SHIFT, SHIFT, 2])
    n = fft(inp.reshape(-1))[:,:SHIFT,:]
    scales = np.sqrt(np.var(n[ :, :, 0], axis=1) + 1e-8)
    means = np.mean(n[:, :, 0], axis=1)
    # scales=np.var(n[:, :, 0], axis=1)
    mms = 1 / scales
    scl = np.tile(np.reshape(means, (-1, 1)), (1, SHIFT))
    n[ :, :, 0] = np.einsum("ij,i->ij", n[ :, :, 0] - scl, mms)

    res[0] = n.astype(np.float32)
    res = process(res)
    filter=-5.0
    means[means < filter] = -32.0
    scales[means < filter] = 0.1

    res[0,:, :, 0] = np.einsum("ij,i->ij", res[0,:, :, 0] , scales)
    res[0,:, :, 0]+= scl

    res2 = res.copy()[:, :, ::-1, :]
    res = np.append(res, res2, axis=2)

    res,rdd = ifft(res[0],rdd)
    res = np.clip(res,-0.8,0.8).reshape(-1)*32767

    vs=(res[-TERM:].astype(np.int16)).tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    print("CPS:%1.3f" % (np.mean(la)))
    output = stream.write(vs)
