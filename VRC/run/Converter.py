import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
class Model:

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
