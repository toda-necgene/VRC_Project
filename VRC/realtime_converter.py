from multiprocessing import Queue, Process, freeze_support
import numpy as np
import os
import pyaudio as pa
import atexit
import tensorflow as tf

from VRC.model import Model as w2w
from VRC.waver import Waver


class Model:
    def __init__(self, model, load=None):
        self.sess = tf.InteractiveSession(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions()))

        test_input_size = model.input_size.copy()
        test_input_size[0] = 1
        #inputs place holder
        self.input_model_test = tf.placeholder(tf.float32, test_input_size,
                                               "inputs_G-net_A")
        #creating generator

        with tf.variable_scope("generator_1"):
            self.test_output_aB = model.generator(
                self.input_model_test, reuse=None, training=False)

        #saver
        self.saver = tf.train.Saver()

        if load is not None:
            # initialize variables
            self.sess.run(tf.global_variables_initializer())
            print(" [*] Reading checkpoint...")

            checkpoint_dir = os.path.join(load, model.name)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # pylint: disable=E1101
                self.saver.restore(self.sess,
                                   os.path.join(checkpoint_dir, ckpt_name))
            else:
                raise Exception("Load error checkpoint")

    def convert(self, data):
        data = self.sess.run(
            self.test_output_aB, feed_dict={self.input_model_test, data})
        return data


def process(queue_in, queue_out):
    waver = Waver()
    f0_transfer = waver.generate_f0_transfer(*(np.load('./profile.npy')))
    net = Model(w2w(), load='trained\\VRC1.0.2')

    queue_out.put("ok")
    while True:
        if not queue_in.empty():
            ins = queue_in.get()
            inputs = np.frombuffer(
                ins, dtype=np.int16).astype(np.float64) / 32767.0
            inputs = np.clip(inputs, -1.0, 1.0)
            f0, _, ap, psp = waver.encode_block(inputs)
            output = waver.decode(f0_transfer(f0), ap, psp=net.convert(psp))
            queue_out.put((output * 32767).tobytes())


if __name__ == '__main__':
    args = dict()
    padding = 1024
    padding_shift = 512
    TERM = 4096
    fs = 16000
    channels = 1
    gain = 1.0
    up = 4096
    q_in = Queue()
    q_out = Queue()

    p_in = pa.PyAudio()
    p = Process(target=process, args=(q_in, q_out))
    p.start()
    while True:
        if not q_out.empty():
            vs = q_out.get()
            if vs == "ok":
                break

    print("変換　開始")

    def terminate():
        stream.stop_stream()
        stream.close()
        p_in.terminate()
        print("Stream Stop")
        freeze_support()

    atexit.register(terminate)

    stream = p_in.open(
        format=pa.paInt16,
        channels=1,
        rate=fs,
        frames_per_buffer=up,
        input=True,
        output=True)
    stream.start_stream()
    while stream.is_active():
        ins = stream.read(up)
        q_in.put(ins)
        vs = np.zeros([up], dtype=np.int16).tobytes()
        if not q_out.empty():
            vs = q_out.get()
        stream.write(vs)
