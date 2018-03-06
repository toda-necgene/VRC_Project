from mvnc import mvncapi as mvnc
import numpy as np
import pyaudio as pa
import atexit
def encode(inp):
    mu=255.
    return np.sign(inp)*np.log(1.+mu*inp)/np.log(1.+mu)
def decode(inp):
    mu=255.
    return np.sign(inp)*(1./mu)*((1.+mu)**np.abs(inp)-1.)
path_to_networks = './Network'
graph_filename = 'graph'

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
    inp=inputs/32767.
    inp=encode(inp).astype(np.int32)
    inp=(inp+1.0)/2.0
    input_data=np.eye(256)[inp]
    vc = process(input_data)
    vr=vc.argmax(axis = 3)/128.-1.0
    vs=decode(vr)
    vs=vs.astype(np.int16).getbuffer()
    vs=vs*32767.
    output = stream.write(vs)

def terminate():
    graph.DeallocateGraph()
    device.CloseDevice()
    stream.stop_stream()
    stream.close()
    p_in.terminate()
atexit.register(terminate)