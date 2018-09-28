from mvnc import mvncapi as mvnc
import numpy as np
import pyaudio as pa
import atexit
import time
path_to_networks = './Network'
graph_filename = './graph'
NFFT=512
SHIFT=64
ac=1
ab=0

def terminate():
    input_fifo.destroy()
    output_fifo.destroy()
    graph.destroy()
    device.close()
    device.destroy()
    stream.stop_stream()
    stream.close()
    p_in.terminate()
atexit.register(terminate)
def fft(data):
    time_ruler = data.shape[0] // SHIFT
    if data.shape[0] % SHIFT == 0:
        time_ruler -= 1
    pos = 0
    wined = np.zeros([time_ruler, NFFT])
    for fft_index in range(time_ruler):
        frame = data[pos:pos + NFFT]
        wined[fft_index] = frame
        pos += SHIFT
    fft_r = np.fft.fft(wined, n=NFFT, axis=1)
    re = fft_r.real.reshape(time_ruler, -1)
    im = fft_r.imag.reshape(time_ruler, -1)
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
    d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
    spec = np.concatenate((c, d), 2)
    ac = max([NFFT - spec.shape[0], 0])
    ab = max([NFFT - spec.shape[1], 0])
    spec = np.pad(spec, ((ac, 0), (ab, 0), (0, 0)), "constant")
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
    fft_s = np.fft.ifft(datanum, n=NFFT, axis=1)
    fft_data = fft_s.real
    v = fft_data[:, :NFFT // 2]
    reds = fft_data[-1, NFFT // 2:].copy()
    lats = np.roll(fft_data[:, NFFT // 2:], 1, axis=0)
    lats[0, :] = red
    spec = np.reshape(v + lats, (-1))
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

devices = mvnc.enumerate_devices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.open()


with open( graph_filename, mode='rb') as f:
    graphfile = f.read()

graph = mvnc.Graph("graph_encode")
input_fifo,output_fifo=graph.allocate_with_fifos(device,graphfile,input_fifo_type=mvnc.FifoType.HOST_WO, input_fifo_data_type=mvnc.FifoDataType.FP32, input_fifo_num_elem=2,
        output_fifo_type=mvnc.FifoType.HOST_RO, output_fifo_data_type=mvnc.FifoDataType.FP32, output_fifo_num_elem=2)

p_in = pa.PyAudio()
py_format = p_in.get_format_from_width(2)
fs = 16000
channels = 1
chunk = 8192
use_device_index = 0
data=np.zeros(chunk)
#Process Of guess
def process(data):
    datan=np.transpose(data,[0,3,1,2]).astype(np.float32)
    tts = time.time()
    graph.queue_inference_with_fifo_elem(input_fifo,output_fifo,datan,None)
    output, userobj = output_fifo.read_elem()
    print("task takes:" + str(time.time() - tts))

    # output=np.transpose(output,[1,2,0])
    output=output.reshape(128,128,2)
    print(output.shape)
    return output

stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = chunk,
		input = True,
		output = True)
rdd=np.zeros(SHIFT)
tt=time.time()
while stream.is_active():
    print("task takes:"+str(time.time()-tt))
    tt=time.time()
    inputs = np.frombuffer(stream.read(chunk),dtype=np.int16).reshape(1,8192,1).astype(np.float32)/32767.0
    res = np.zeros([1, 128, 128, 2])
    n = fft(inputs.reshape(-1))

    scales = np.sqrt(np.var(n[ :, :, 0], axis=1) + 1e-8)
    means = np.mean(n[:, :, 0], axis=1)
    mms = 1 / scales
    scl = np.tile(np.reshape(means, (-1, 1)), (1, NFFT))
    n[ :, :, 0] = np.einsum("ij,i->ij", n[ :, :, 0] - scl, mms)
    res[0] = n.astype(np.float32)


    res = process(res)
    res=filter_fft(res,means,scales)
    res,rdd = ifft(res,rdd)
    res = res.reshape(-1)
    vs=(res.astype(np.int16)).tobytes()
    output = stream.write(vs)
