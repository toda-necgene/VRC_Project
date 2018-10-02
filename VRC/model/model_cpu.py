import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import json
import shutil
from .model import discriminator,generator
class Model:
    def __init__(self,path):
        self.args = dict()

        # default setting
        self.args["model_name"] = "wave2wave"
        self.args["version"] = "1.0.0"

        self.args["checkpoint_dir"] = "./trained_models"
        self.args["best_checkpoint_dir"] = "./best_model"
        self.args["wave_otp_dir"] = "./havests"
        self.args["train_data_dir"] = "./datasets/train"
        self.args["test_data_dir"] = "./datasets/test"

        self.args["test"] = True
        self.args["tensorboard"] = True
        self.args["debug"] = False

        self.args["batch_size"] = 32
        self.args["input_size"] = 4096
        self.args["NFFT"] = 1024
        self.args["dilated_size"]=15
        self.args["g_lr_max"] = 2e-4
        self.args["g_lr_min"] = 2e-6
        self.args["d_lr_max"] = 2e-4
        self.args["d_lr_min"] = 2e-6
        self.args["g_b1"] = 0.5
        self.args["g_b2"] = 0.999
        self.args["d_b1"] = 0.5
        self.args["d_b2"] = 0.999
        self.args["weight_Cycle_Pow"] = 100.0
        self.args["weight_Cycle_Pha"] = 100.0
        self.args["weight_GAN"] = 1.0
        self.args["train_epoch"] = 1000
        self.args["start_epoch"] = 0
        self.args["save_interval"] = 10
        self.args["lr_decay_term"] = 20

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
        self.args["SHIFT"] = self.args["NFFT"] // 2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        # shapes of inputs
        ss = self.args["input_size"] // self.args["SHIFT"]
        self.input_size_model = [self.args["batch_size"], ss, self.args["NFFT"] // 2, 2]
        self.input_size_test = [None, ss, self.args["NFFT"] // 2, 2]

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))

        if bool(self.args["debug"]):
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
        if self.args["wave_otp_dir"] is not "False":
            self.args["wave_otp_dir"] = self.args["wave_otp_dir"] + self.args["name_save"] + "/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])

        self.build_model()
    def build_model(self):

        #inputs place holder
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_G-net_A")
        self.input_model_testa =self.input_model_test*0.1
        #creating generator
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1"):
                self.fake_aB_image_testa = generator(self.input_model_testa, reuse=None, train=False)
                self.fake_aB_image_test=10*self.fake_aB_image_testa

        #saver
        self.saver = tf.train.Saver()



    def convert(self,in_put):
        #function of test
        #To convert wave file
        back_load=self.args["SHIFT"]
        use_num = 4
        tt=time.time()
        ipt_size=self.args["input_size"]+self.args["SHIFT"]+self.args["SHIFT"]*self.args["dilated_size"]
        ipt=ipt_size+back_load
        times=in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        res3 = np.zeros([1,self.args["NFFT"],2], dtype=np.float32)
        rss=np.zeros([use_num,self.input_size_model[2]],dtype=np.float64)

        for t in range(times):
            # Preprocess

            # Padiing
            start_pos=ipt_size*t+(in_put.shape[0]%ipt_size)
            resorce=np.reshape(in_put[max(0,start_pos-ipt):start_pos],(-1))
            r=max(0,ipt-resorce.shape[0])
            if r>0:
                resorce=np.pad(resorce,(r,0),'constant')
            # FFT
            ters=back_load//use_num
            res=self.fft(resorce[-ipt_size:].copy()/32767.0)
            res=res[:,:self.args["SHIFT"],:].reshape(1,-1,self.args["SHIFT"],2)
            for r in range(1,use_num):
                pp=ters*r
                resorce2=resorce[pp:pp+ipt_size].copy()
                resorce2=self.fft(resorce2/32767)[:,:self.args["SHIFT"],:].reshape(1,-1,self.args["SHIFT"],2)
                res=np.append(res,resorce2,axis=0)
            # running network
            response=self.sess.run(self.fake_aB_image_test,feed_dict={ self.input_model_test:res})
            res2 = response.copy()[:, :, ::-1, :]
            response = np.append(response, res2, axis=2)
            response[:,:,self.args["SHIFT"]:,1]*=-1
            response=np.clip(response,-60.0,12.0)
            # Postprocess

            # IFFT
            rest=np.zeros(self.args["input_size"])
            for i in range(response.shape[0]):
                resa, rss[i] = self.ifft(response[i], rss[i])
                if i != 0:
                    resa = np.roll(resa, -ters*i, axis=0)
                    resa[-ters*i:] = 0
                rest+=(resa)[-self.args["input_size"]:]
            res3 = np.append(res3, response[0,:,:,:]).reshape(-1,self.args["NFFT"],2)

            res = np.clip(rest, -1.0, 1.0)*32767

            # chaching results
            res=res.reshape(-1).astype(np.int16)
            otp=np.append(otp,res)
        h=otp.shape[0]-in_put.shape[0]
        if h>0:
            otp=otp[h:]

        return otp.reshape(-1),time.time()-tt,res3[1:]

    def load(self):
        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(" [*] Reading checkpoint...")
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(self.args["checkpoint_dir"], model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.epoch=self.saver
            return True
        else:
            return False

    def fft(self,data):

        time_ruler = data.shape[0] // self.args["SHIFT"]
        if data.shape[0] % self.args["SHIFT"] == 0:
            time_ruler -= 1
        pos = 0
        win = np.hamming(self.args["NFFT"])
        wined = np.zeros([time_ruler, self.args["NFFT"]])
        for fft_index in range(time_ruler):
            frame = data[pos:pos + self.args["NFFT"]]
            wined[fft_index] = frame*win
            pos += self.args["SHIFT"]
        fft_r = np.fft.fft(wined, n=self.args["NFFT"], axis=1)
        re = fft_r.real.reshape(time_ruler, -1)
        im = fft_r.imag.reshape(time_ruler, -1)
        c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
        c=np.clip(c,-10,10)
        d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
        spec = np.concatenate((c, d), 2)
        return spec
    def ifft(self,data,redi):
        a=data
        a[:, :, 0]=np.clip(a[:, :, 0],a_min=-20,a_max=20)
        sss=np.exp(a[:,:,0])
        p = np.sqrt(sss)
        r = p * (np.cos(a[:, :, 1]))
        i = p * (np.sin(a[:, :, 1]))
        dds = np.concatenate((r.reshape(r.shape[0], r.shape[1], 1), i.reshape(i.shape[0], i.shape[1], 1)), 2)
        data=dds[:,:,0]+1j*dds[:,:,1]
        fft_s = np.fft.ifft(data,n=self.args["NFFT"], axis=1)
        fft_data = fft_s.real
        v = fft_data[:, :self.args["NFFT"]// 2].copy()
        reds = fft_data[-1, self.args["NFFT"] // 2:].copy()
        lats = np.roll(fft_data[:, self.args["NFFT"] // 2:].copy(), 1, axis=0 )
        lats[0, :]=redi
        spec = np.reshape(v + lats, (-1))/2
        return spec,reds



