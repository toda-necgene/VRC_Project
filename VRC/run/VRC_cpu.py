import numpy as np
import pyaudio as pa
import atexit
import time
from Converter import Model
def encode(inp):
    mu=255.
    return np.sign(inp)*np.log(1.+mu*inp)/np.log(1.+mu)
def decode(inp):
    mu=255.
    return np.sign(inp)*(1./mu)*((1.+mu)**np.abs(inp)-1.)

p_in = pa.PyAudio()
py_format = p_in.get_format_from_width(2)
fs = 16000
channels = 1
chunk = 8192
use_device_index = 0
data=np.zeros(chunk)

m = Model(False)
m.build_model()
chk = "./Network"
m.load(chk)
print(" [*] finished succesfully")
stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = chunk,
		input = True,
		output = True)
print(" [*] start converting")
tt=time.time()
while stream.is_active():
    inputs = np.frombuffer(stream.read(chunk),dtype=np.int16).reshape(1,8192,1)
    #本処理
    t2 = time.time()
    vc = m.convert(inputs)
    #後処理
    vs=vc.astype(np.int16)
    vs=vs.tobytes()
    output = stream.write(vs)
    t4=time.time()
    print("%2.4f" % (time.time()-tt))
    tt = time.time()
def terminate():
    stream.stop_stream()
    stream.close()
    p_in.terminate()
atexit.register(terminate)