from multiprocessing import Queue,Process,freeze_support
from model import generator
import numpy as np
import os
import pyaudio as pa
import atexit
import tensorflow as tf

import util

class Model:
    def __init__(self, path, load=False):
        self.args = util.config_reader(path, {
            "model_name": "VRC",
            "version": "18.12.22",
            "checkpoint_dir" : "./trained_models",
            "input_size": 4096,
        })

        # initializing paramaters
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        # shapes of inputs
        self.input_size_test = [1, 52, 513, 1]
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))

        self.build_model()

        if load:
            self.load()

    def build_model(self):

        #inputs place holder
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_G-net_A")
        #creating generator

        with tf.variable_scope("generator_1"):
            self.test_output_aB = generator(self.input_model_test, reuse=None, training=False)

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
        if ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.epoch=self.saver
            return True
        else:
            return False

    def convert(self, data):
        data = self.sess.run(self.test_output_aB, feed_dict={self.input_model_test, data})
        return data

def process(queue_in, queue_out):
    net = Model("./setting.json", load=True)
    
    f0_translater = util.generate_f0_translater("./voice_profile.npy")
    queue_out.put("ok")
    while True:
        if not queue_in.empty():
            ins = queue_in.get()
            inputs = np.frombuffer(ins, dtype=np.int16).astype(np.float64) / 32767.0
            inputs = np.clip(inputs, -1.0, 1.0)
            f0, sp, ap = util.encode(inputs.copy())
            data = sp.reshape(1, -1, 513, 1)
            output = net.convert(data)
            resb = util.decode(f0_translater(f0),output[0], ap)
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
