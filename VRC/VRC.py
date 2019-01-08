from multiprocessing import Queue,Process,freeze_support
from model import generator
import numpy as np
import os
import pyaudio as pa
import pyworld as pw
import atexit
import json
import tensorflow as tf
class Model:
    def __init__(self,path):
        self.args = dict()

        # default setting
        self.args["model_name"] = "VRC"
        self.args["version"] = "18.12.22"

        self.args["checkpoint_dir"] = "./trained_models"

        self.args["input_size"] = 4096
        self.args["pitch_rate_var"]=1.0
        self.args["pitch_rate_mean_s"]=0.0
        self.args["pitch_rate_mean_t"]=0.0

        # reading json file
        try:
            with open(path, "r") as f:
                dd = json.load(f)
                keys = dd.keys()
                for j in keys:
                    data = dd[j]
                    keys2 = data.keys()
                    for k in keys2:
                        if k in self.args:
                            if type(self.args[k]) == type(data[k]):
                                self.args[k] = data[k]
                            else:
                                print(
                                    " [W] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(
                                        type(self.args[k])) + "\"")
                        elif k[0] == "#":
                            pass
                        else:
                            print(" [W] Argument \"" + k + "\" is not exsits.")

        except json.JSONDecodeError as e:
            print(" [W] JSONDecodeError: ", e)
            print(" [W] Use default setting")
        except FileNotFoundError:
            print(" [W] Setting file is not found :", path)
            print(" [W] Use default setting")

        # initializing paramaters
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        # shapes of inputs
        self.input_size_test = [1, 52,513,1]
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))

        #inputs place holder
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_G-net_A")
        #creating generator

        with tf.variable_scope("generator_1"):
            self.test_outputaB = generator(self.input_model_test, reuse=None)

        #saver
        self.saver = tf.train.Saver()


    def load(self):
        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(" [*] Reading checkpoint...")
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(self.args["checkpoint_dir"], model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt :
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.epoch=self.saver
            return True
        else:
            return False


def encode(data):
    f0, t = pw.dio(data, 16000)
    f0 = pw.stonemask(data, f0, t, 16000)
    sp = pw.cheaptrick(data, f0, t, 16000)
    ap = pw.d4c(data, f0, t, 16000)
    return f0.astype(np.float64), np.clip((np.log(sp) + 15) / 20, -1.0, 1.0).astype(np.float64), ap.astype(np.float64)


def decode(f0, sp, ap):
    ap = ap.reshape(-1, 513).astype(np.float)
    f0 = f0.reshape(-1).astype(np.float)
    sp = np.exp(sp.reshape(-1, 513).astype(np.float) * 20 - 15)
    ww = pw.synthesize(f0, sp, ap, 16000)
    return ww

def process(queue_in, queue_out):
    net = Model("./setting.json")
    net.load()
    a = np.load("./voice_profile.npy")
    queue_out.put("ok")
    while True:
        if not queue_in.empty():
            ins = queue_in.get()
            inputs = np.frombuffer(ins, dtype=np.int16).astype(np.float64) / 32767.0
            inputs = np.clip(inputs, -1.0, 1.0)
            f0, sp, ap = encode(inputs.copy())
            data = sp.reshape(1, -1, 513, 1)
            output = net.sess.run(net.test_outputaB, feed_dict={net.input_model_test: data})
            resb = decode((f0 - a[0]) * a[2] + a[1],output[0], ap)
            res = (np.clip(resb, -1.0, 1.0).reshape(-1) * 32767)
            vs = res.astype(np.int16)
            vs = vs.tobytes()
            queue_out.put(vs)
if __name__ == '__main__':
    args=dict()
    padding=1024
    padding_shift=512
    TERM=4096
    fs = 16000
    channels = 1
    gain=1.0
    up = 4096
    q_in = Queue()
    q_out = Queue()

    p_in = pa.PyAudio()
    p = Process(target=process, args=(q_in, q_out))
    p.start()
    while True:
        if not q_out.empty():
            vs = q_out.get()
            if vs=="ok":
                break
    print("変換　開始")
    def terminate():
        stream.stop_stream()
        stream.close()
        p_in.terminate()
        print("Stream Stop")
        freeze_support()
    atexit.register(terminate)
    stream = p_in.open(format=pa.paInt16,
                       channels=1,
                       rate=fs,
                       frames_per_buffer=up,
                       input=True,
                       output=True)
    stream.start_stream()
    while stream.is_active():
        ins=stream.read(up)
        q_in.put(ins)
        vs=np.zeros([up],dtype=np.int16).tobytes()
        if not q_out.empty():
            vs = q_out.get()
        stream.write(vs)
