from mvnc import mvncapi as mvnc
import numpy as np
import pyaudio as pa
import atexit

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
    inputs = stream.read(chunk)
    input_data=np.asarray(inputs).reshape(1,1,160000)
    vc = process(input_data)
    output = stream.write(vc)
def terminate():
    graph.DeallocateGraph()
    device.CloseDevice()
    stream.stop_stream()
    stream.close()
    pa.terminate()
atexit.register(terminate)