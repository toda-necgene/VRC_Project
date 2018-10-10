import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import json
from .model import generator
import pyworld as pw
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
        self.args["dilated_size"]=0
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
        self.args["pitch_rate"]=1.0

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
        self.input_size_model = [self.args["batch_size"], 58,513,1]
        self.input_size_test = [None, 58,513,1]

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
        self.input_model_testa =self.input_model_test
        #creating generator
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1"):
                self.test_outputaB = generator(self.input_model_testa[:,:,:,:1]*0.1, reuse=None, train=False)

        #saver
        self.saver = tf.train.Saver()



    def convert(self,in_put):
        #function of test
        #To convert wave file
        back_load=self.args["SHIFT"]
        use_num = 2
        tt=time.time()
        ipt_size=self.args["input_size"]+self.args["SHIFT"]+self.args["SHIFT"]*self.args["dilated_size"]
        ipt=ipt_size+back_load
        times=in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        res3 = np.zeros([1,self.args["NFFT"],2], dtype=np.float32)

        for t in range(times):
            # Preprocess

            # Padiing
            start_pos=ipt_size*t+(in_put.shape[0]%ipt_size)
            resorce=np.reshape(in_put[max(0,start_pos-ipt):start_pos],(-1))
            r=max(0,ipt-resorce.shape[0])
            if r>0:
                resorce=np.pad(resorce,(r,0),'constant')
            # FFT
            ap2=list()
            ff = list()
            ters=back_load//use_num
            f0,res,ap=encode((resorce[-ipt_size:].copy()/32767.0).astype(np.double))
            ff.append(f0)
            ap2.append(ap)
            res=res.reshape(1,-1,514,1)
            for r in range(1,use_num):
                pp=ters*r
                resorce2=resorce[pp:pp+ipt_size].copy()
                f0,resorce2,ap2_o=encode((resorce2/32767).astype(np.double))
                resorce2=resorce2.reshape(1,-1,514,1)
                ff.append(f0)
                ap2.append(ap2_o)
                res=np.append(res,resorce2,axis=0)
            # running network
            response=self.sess.run(self.test_outputaB,feed_dict={ self.input_model_test:res})
            # Postprocess

            rest=np.zeros(self.args["input_size"])
            for i in range(response.shape[0]):
                f0=ff[i]*self.args["pitch_rate"]
                resa= decode(f0,response[i],ap2[i])
                if i != 0:
                    resa = np.roll(resa, -ters*i, axis=0)
                    resa[-ters*i:] = 0
                rest+=resa[-self.args["input_size"]:]
            res3 = np.append(res3, response[0,:,:,:])

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

def encode(data):
    fs=16000
    _f0,t=pw.dio(data,fs)
    f0=pw.stonemask(data,_f0,t,fs)
    sp=pw.cheaptrick(data,f0,t,fs)
    ap=pw.d4c(data,f0,t,fs)
    sp=np.log(sp)
    return f0,sp,ap
def decode(f0,sp,ap):
    return pw.synthesize(f0,np.exp(sp),ap,16000)



