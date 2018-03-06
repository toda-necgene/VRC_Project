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
chunk = 2048
use_device_index = 0
data=np.zeros(chunk)

m = Model(False)
m.build_model()
chk = "./Network"
m.load(chk)
print(" [*] finished succesfully")
cur_ins= None
cur_res_2 = np.zeros((m.in_put_size[0], m.in_put_size[1], m.in_put_size[2]), dtype=np.int16)
#Process Of guess
def process(in_put,cur_in,cur_res):
    cur_res_s=cur_res
    cur_res_ans = np.zeros((m.in_put_size[0], m.in_put_size[1], m.in_put_size[2]), dtype=np.int16)
    times = in_put.shape[1] // (m.out_put_size[2]) + 1
    if in_put.shape[1] % (m.out_put_size[1] * m.batch_size) == 0:
        times -= 1
    otp = np.array([], dtype=np.int16)
    otp2 = np.empty((1,256), dtype=np.int16)
    for t in range(times):
        red = np.zeros((1, m.in_put_size[1], m.in_put_size[2]))
        start_pos = m.out_put_size[1] * (t) + ((in_put.shape[1]) % m.out_put_size[1])
        resorce = np.reshape(in_put[0,  max(0,start_pos - m.in_put_size[1]-1):min(start_pos, in_put.shape[1]),:],
                             (-1, 256))
        r = max(0, m.in_put_size[1] - resorce.shape[0])
        if r > 0:
            if cur_in is not None:
                res_bef = cur_in[0,max(0,cur_in.shape[1]-r-1):-1,:]
                resorce = np.append(res_bef,resorce,axis=0)
            else:
                resorce = np.pad(resorce, ((r, 0), (0, 0)), 'constant')
        red[0] = resorce
        red = red.reshape((m.in_put_size[0], m.in_put_size[1], m.in_put_size[2],1))
        res = m.sess.run(m.fakeB_otp,
                            feed_dict={m.real_data: red})
        cur_res_s = np.append(cur_res_s,res,axis=1)
        cur_res_s = cur_res_s[:, max(0, cur_res_s.shape[1]-m.in_put_size[1]-1):-1, :]
        otp=np.reshape(cur_res_s,(m.in_put_size[0],m.in_put_size[1],m.in_put_size[2],1))
        res2 = m.sess.run(m.fakeB_otp_2,
                             feed_dict={m.real_data:otp })
        otp2 = np.append(otp2, res2[0, :,:],axis=0)
    output = otp2[otp2.shape[0] - in_put.shape[1] - 1:-1,:]
    return output,cur_res_s

stream=p_in.open(format = pa.paInt16,
		channels = 1,
		rate = fs,
		frames_per_buffer = chunk,
		input = True,
		output = True)
print(" [*] start converting")
tt=time.time()
while stream.is_active():
    inputs = np.frombuffer(stream.read(chunk),dtype=np.int16).reshape(1,2048)
    #前処理
    t1=time.time()
    inp=inputs/32767.
    inp=encode(inp)
    inp=(inp+1.0)/2.0*255
    inp=inp.astype(np.int16)
    input_data=np.eye(256)[inp]
    #本処理
    t2 = time.time()
    vc,cur_res_2 = process(input_data,cur_ins,cur_res_2)
    #後処理
    t3 = time.time()
    cur_ins=(input_data)
    vr=vc.argmax(axis = 1)/128.-1.0
    vs=decode(vr)
    vs = vs * 32767.
    vs=vs.astype(np.int16)
    vs=vs.tobytes()
    output = stream.write(vs)
    t4=time.time()
    print("%2.4f(%1.4f|%1.4f|%1.4f)/%1.4f" % (time.time()-tt,t2-t1,t3-t2,t4-t3,chunk/fs))
    tt = time.time()
def terminate():
    stream.stop_stream()
    stream.close()
    p_in.terminate()
atexit.register(terminate)