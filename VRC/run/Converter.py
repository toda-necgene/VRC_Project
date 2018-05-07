import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
class Model:
    def __init__(self,debug):
        self.batch_size=1
        self.depth=5
        self.input_ch=1
        self.input_size=[self.batch_size,8192,1]
        self.input_size_model=[self.batch_size,64,256,1]

        self.dataset_name="wave2wave_1.0.4"
        self.output_size=[self.batch_size,8192,1]
        self.CHANNELS=[4**i+1 for i in range(self.depth+1)]
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85)))
        if debug:
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    def build_model(self):

        self.input_model=tf.placeholder(tf.float32, [self.batch_size,256,64], "inputs_convert")
        with tf.variable_scope("generator_1"):
            self.fake_B_image=self.generator(tf.reshape(self.input_model,[self.batch_size,256,64,1]), reuse=False,name="gen")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()
    def convert(self,inputs):
        tt=time.time()
        res = np.zeros([1, 256, 64, 2])
        n = fft(inputs.reshape(-1))
        res[0] = (n)
        red = np.log(np.abs(res[:, :, :, 0] + 1j * res[:, :, :, 1]) ** 2 + 1e-16)
        res = self.sess.run(self.fake_B_image,feed_dict={self.input_model:red})
        res = ifft(res[0]) * 32767
        res = res.reshape(-1)
        return res
    def generator(self,current_outputs,reuse,name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        current=current_outputs

        connections=[ ]

        for i in range(self.depth):
            current=self.down_layer(current,self.CHANNELS[i+1])
            connections.append(current)
        for i in range(self.depth):
            current+=connections[self.depth-i-1]
            current=self.up_layer(current,self.CHANNELS[self.depth-i-1],i!=(self.depth-1))
        return current
    def up_layer(self,current,output_shape,bn):
        ten=tf.nn.leaky_relu(current)
        ten=tf.layers.conv2d_transpose(ten, output_shape,kernel_size=4 ,strides=(1,1), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
        if(bn):
            ten=tf.layers.batch_normalization(ten,axis=3,training=False,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        return ten
    def down_layer(self,current,output_shape):
        ten=tf.layers.batch_normalization(current,axis=3,training=False,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        ten=tf.layers.conv2d(ten, output_shape,kernel_size=4 ,strides=(1,1), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
        ten=tf.nn.leaky_relu(ten)
        return ten
    def save(self, checkpoint_dir, step):
        model_name = "wave2wave.model"
        model_dir = "%s_%s_%s layers" % (self.dataset_name, self.batch_size,self.depth)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s layers" % (self.dataset_name, self.batch_size,self.depth)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.epoch=self.saver
            return True
        else:
            return False
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
