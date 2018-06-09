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
    ac = max([NFFT - spec.shape[0], 0])
    ab = max([NFFT - spec.shape[1], 0])
    spec = np.pad(spec, ((ac, 0), (ab, 0), (0, 0)), "constant",constant_values=-32)
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
    spec=np.append(spec,reds)
    return spec, reds

def filter_fft(inp,means,scales):
    a = inp.copy()
    scales2 = np.sqrt(np.var(a[:, :, 0], axis=1) + 1e-8)
    means2 = np.mean(a[:, :, 0], axis=1)
    ss = scales / (scales2 + 1e-32)
    means_mask=means.copy()
    means_mask[means_mask<-4.5]=-60.0
    sm = np.tile((means_mask-1.0 - means2).reshape(-1, 1), (1, NFFT))
    c = a[:, :, 0]
    c = np.einsum("ij,i->ij",c+ sm , ss)
    a[:, :, 0] = c
    return a

net=Model("../setting.json")
net.build_model()
net.load()
p_in = pa.PyAudio()
py_format = p_in.get_format_from_width(2)
fs = 16000
channels = 1
chunk = 8000
use_device_index = 0
data=np.zeros(chunk)

#Process Of guess
def process(data):
    tt = time.time()
    output=net.sess.run(net.fake_B_image,feed_dict={net.input_model:data})
    # print("TT:"+str(time.time()-tt))
    return output
inf=p_in.get_default_output_device_info()
print(inf)
stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = chunk,
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
las=np.zeros([192])
while stream.is_active():
    inputs = np.frombuffer(stream.read(chunk),dtype=np.int16).reshape(8000).astype(np.float32)/32767.0
    tt = time.time()
    inp=np.append(las,inputs).reshape(1,8192,1)
    las = inputs[-192:]
    res = np.zeros([1, 128, 128, 2])
    n = fft(inp.reshape(-1))
    # scales = np.sqrt(np.var(n[ :, :, 0], axis=1) + 1e-8)
    means = np.mean(n[:, :, 0], axis=1)
    scales=np.tile([0.2],(NFFT))
    mms = 1 / scales
    scl = np.tile(np.reshape(means, (-1, 1)), (1, NFFT))
    n[ :, :, 0] = np.einsum("ij,i->ij", n[ :, :, 0] - scl, mms)

    res[0] = n.astype(np.float32)
    res = process(res)
    res[0,:, :, 0] = np.einsum("ij,i->ij", res[0,:, :, 0] , scales)
    res[0,:, :, 0]+= scl

    scales2 = np.sqrt(np.var(res[0,:, :, 0], axis=1) + 1e-8)
    means2 = np.mean(res[0,:, :, 0], axis=1)
    res,rdd = ifft(res[0],rdd)
    res = res.reshape(-1)*32767

    vs=(res[-8000:].astype(np.int16)).tobytes()
    la=np.append(la,time.time() - tt)
    la=la[-5:]
    print("CPS:%1.3f" % (np.mean(la)))
    output = stream.write(vs)
