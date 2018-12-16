from model.model_cpu import Model
from multiprocessing import Queue,Process
import numpy as np
import pyaudio as pa
import pyworld as pw
import atexit
import scipy.signal
# Process Of guess

padding = 1024
padding_shift = 512
TERM = 4096
fs = 48000
channels = 1
gain = 1.0
up = 4096*3

def encode(data):
    fs = 16000
    f0, t = pw.dio(data, fs)
    f0 = pw.stonemask(data, f0, t, fs)
    sp = pw.cheaptrick(data, f0, t, fs)
    ap = pw.d4c(data, f0, t, fs)
    return f0.astype(np.float64), np.clip((np.log(sp) + 15) / 20, -1.0, 1.0).astype(np.float64), ap.astype(np.float64)


def decode(f0, sp, ap):
    ap = ap.reshape(-1, 513).astype(np.float)
    f0 = f0.reshape(-1).astype(np.float)
    sp = np.exp(sp.reshape(-1, 513).astype(np.float) * 20 - 15)
    ww = pw.synthesize(f0, sp, ap, 16000)
    return ww

def process(queue_in,queue_out):
    a=np.load("./voice_profile.npy")
    net=Model("./setting.json")
    net.load()
    queue_out.put("ok")
    while True:
        if not queue_in.empty():
            ins = queue_in.get()
            inputs = np.frombuffer(ins, dtype=np.int16).astype(np.float64) / 32767.0
            inputs = np.clip(inputs, -1.0, 1.0)
            if np.max(inputs) > 0.2:
                inputs=scipy.signal.resample(inputs,TERM)
                f0, sp, ap = encode(inputs.copy())
                data = sp.reshape(1, -1, 513, 1)
                output = net.sess.run(net.test_outputaB, feed_dict={net.input_model_test: data})
                resb = decode((f0 - a[0]) * a[2] + a[1],
                              output[0], ap)
                res = (np.clip(resb, -1.0, 1.0).reshape(-1) * 32767)
                res = scipy.signal.resample(res, up)
                vs = res.astype(np.int16)
                vs = vs.tobytes()
                queue_out.put(vs)
if __name__ == '__main__':
    args=dict()
    p_in = pa.PyAudio()

    q_in = Queue()
    q_out = Queue()
    p = Process(target=process, args=(q_in, q_out))
    p.start()
    def terminate():
        stream.stop_stream()
        stream.close()
        p_in.terminate()
        print("Stream Stop")
    atexit.register(terminate)
    while True:
        if not q_out.empty():
            vs = q_out.get()
            if vs=="ok":
                break
    stream=p_in.open(format = pa.paInt16,
            channels = 1,
            rate = fs,
            frames_per_buffer = up,
            input = True,
            output = True,
            output_device_index=2)
    stream.start_stream()
    print("convertion start")
    while stream.is_active():
        ins=stream.read(up)
        q_in.put(ins)
        vs=np.zeros([up]).tobytes()
        if not q_out.empty():
            vs = q_out.get()
        stream.write(vs,up)
