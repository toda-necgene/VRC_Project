from mvnc import mvncapi as mvnc
import numpy as np
import pyaudio as pa
import atexit
path_to_networks = './Network'
graph_filename = 'graph'
def fft(data):
    rate=16000
    NFFT=64
    time_song=float(data.shape[0])/rate
    time_unit=1/rate
    start=0
    stop=time_song
    step=(NFFT//2)*time_unit
    time_ruler=np.arange(start,stop,step)
    window=np.hamming(NFFT)
    spec=np.zeros([len(time_ruler),(NFFT),2])
    pos=0
    for fft_index in range(len(time_ruler)):
        frame=data[pos:pos+NFFT]/32767.0
        if len(frame)==NFFT:
            wined=frame*window
            fft=np.fft.fft(wined)
            fft_data=np.asarray([fft.real,fft.imag])
            fft_data=np.transpose(fft_data, (1,0))
            for i in range(len(spec[fft_index])):
                spec[fft_index][i]=fft_data[i]
            pos+=NFFT//2
    return spec
def ifft(data):
    data=data[:,:,0]+1j*data[:,:,1]
    time_ruler=data.shape[0]
    window=np.hamming(64)
    spec=np.zeros([])
    lats = np.zeros([32])
    pos=0
    for _ in range(time_ruler):
        frame=data[pos]
        fft=np.fft.ifft(frame)
        fft_data=fft.real
        fft_data/=window
        v = lats + fft_data[:32]
        lats = fft_data[32:]
        spec=np.append(spec,v)
        pos+=1
    return spec[1:]
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()


with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()

graph = device.AllocateGraph(graphfile)

p_in = pa.PyAudio()
py_format = p_in.get_format_from_width(2)
fs = 16000
channels = 1
chunk = 160000
use_device_index = 0
data=np.zeros(chunk)
#Process Of guess
def process(data):
    graph.LoadTensor(data.astype(np.float16), 'user object')
    output, userobj = graph.GetResult()
    return output

stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = chunk,
		input = True,
		output = True)

while stream.is_active():
    inputs = np.frombuffer(stream.read(chunk),dtype=np.float32).reshape(1,1,160000)
    res = np.zeros([1, 256, 64, 2])
    n = fft(inputs.reshape(-1))
    res[0] = (n)
    red = np.log(np.abs(res[:, :, :, 0] + 1j * res[:, :, :, 1]) ** 2 + 1e-16)
    res = process(red)
    res = ifft(res[0]) * 32767
    res = res.reshape(-1)
    vs=res.astype(np.int16).getbuffer()
    output = stream.write(vs)

def terminate():
    graph.DeallocateGraph()
    device.CloseDevice()
    stream.stop_stream()
    stream.close()
    p_in.terminate()
atexit.register(terminate)