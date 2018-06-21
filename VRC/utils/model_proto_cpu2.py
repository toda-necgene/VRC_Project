from glob import glob
import tensorflow as tf
import os
import time
from six.moves import xrange
import numpy as np
import wave
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import pyaudio
from datetime import datetime
import json
import shutil
import math
import matplotlib.pyplot as plt
class Model:
    def __init__(self,path):
        self.args=dict()
        self.args["checkpoint_dir"]="./trained_models"
        self.args["wave_otp_dir"] = "False"
        self.args["train_data_num"]=500
        self.args["batch_size"]=1
        self.args["depth"] =4
        self.args["d_depth"] = 4
        self.args["train_epoch"]=500
        self.args["stop_itr"] = -1
        self.args["start_epoch"]=0
        self.args["test"]=True
        self.args["log"] = True
        self.args["tensorboard"]=False
        self.args["hyperdash"]=False
        self.args["stop_argument"]=True
        self.args["stop_value"] = 0.5
        self.args["input_size"] = 8192
        self.args["weight_Cycle"]=1.0
        self.args["weight_GAN1"] = 1.0
        self.args["weight_GAN2"] = 1.0
        self.args["NFFT"]=128
        self.args["debug"] = False
        self.args["noise"] = False
        self.args["cupy"] = False
        self.args["D_channels"] =[2]
        self.args["G_channel"] = 32
        self.args["G_channels"] = [32]
        self.args["strides_g"] = [2,2]
        self.args["strides_g2"] = [1, 1]
        self.args["strides_d"] = [2,2]
        self.args["filter_g"] = [8,8]
        self.args["filter_g2"] = [2, 2]
        self.args["filter_d"] = [4,4]
        self.args["model_name"] = "wave2wave"
        self.args["version"] = "1.0.0"
        self.args["log_eps"] = 1e-8
        self.args["g_lr"]=2e-4
        self.args["g_b1"] = 0.5
        self.args["g_b2"] = 0.999
        self.args["d_b1"] = 0.5
        self.args["d_b2"] = 0.999
        self.args["train_d_scale"]=1.0
        self.args["train_interval"]=10
        self.args["save_interval"]=1
        self.args["test_dir"] = "./test"
        self.args["dropbox"]="False"
        self.args["architect"] = "flatnet"
        self.args["label_noise"]=0.0
        self.args["train_data_path"]="./train/Model/datasets/train/"
        if os.path.exists(path):
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
                                        " [!] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(
                                            type(self.args[k])) + "\"")
                            elif k[0] == "#":
                                pass
                            else:
                                print(" [!] Argument \"" + k + "\" is not exsits.")

            except json.JSONDecodeError as e:
                 print(' [x] JSONDecodeError: ', e)
        else:
            print( " [!] Setting file is not found")
        if len(self.args["D_channels"]) != (self.args['d_depth']):
            print(" [!] Channels length and depth+1 must be equal ." + str(len(self.args["D_channels"])) + "vs" + str(self.args['d_depth']))
            self.args["D_channels"] = [min([2 ** (i + 1) - 2, 254]) for i in range(self.args['d_depth'])]
        if len(self.args["G_channels"]) != (self.args['depth']*2):
            print(" [!] Channels length and depth*2 must be equal ." + str(len(self.args["G_channels"])) + "vs" + str(
                self.args['gepth']*2))
            self.args["D_channels"] = [min([2 ** (i + 1) - 2, 254]) for i in range(self.args['depth']*2)]
        self.args["SHIFT"] = self.args["NFFT"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        ss=self.args["input_size"]//self.args["SHIFT"]
        self.input_size_model=[self.args["batch_size"],ss,self.args["NFFT"]//2,2]
        sd=self.args["input_size"]//self.args["SHIFT"]
        for i in range(self.args["d_depth"]):
            sd=(sd-self.args["filter_d"][0])//self.args["strides_d"][0]+1
        self.input_size_label=[self.args["batch_size"],sd,24]
        print(self.input_size_label)
        print("model input size:"+str(self.input_size_model))
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))
        if bool(self.args["debug"]):
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
        if  self.args["wave_otp_dir"] is not "False" :
            self.args["wave_otp_dir"]=self.args["wave_otp_dir"]+ self.args["name_save"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])
            shutil.copy(path,self.args["wave_otp_dir"]+"setting.json")
            self.args["log_file"]=self.args["wave_otp_dir"]+"log.txt"
        self.checkpoint_dir = self.args["checkpoint_dir"]

    def build_model(self):

        #inputs place holder
        #入力
        self.input_modela=tf.placeholder(tf.float32, self.input_size_model, "input_A")
        self.input_modelb = tf.placeholder(tf.float32, self.input_size_model, "input_B")
        self.label_modela = tf.placeholder(tf.float32, self.input_size_label, "label_A")
        self.label_modelb = tf.placeholder(tf.float32, self.input_size_label, "label_B")

        self.training=tf.placeholder(tf.float32,[1],name="Training")
        self.noise = tf.placeholder(tf.float32, [self.args["batch_size"]], "inputs_Noise")

        a_true_noised=self.input_modela+tf.random_normal(self.input_modela.shape,0,self.noise[0])
        b_true_noised = self.input_modelb + tf.random_normal(self.input_modelb.shape, 0, self.noise[0])

        #creating discriminator inputs
        #D-netの入力の作成
        #creating discriminator
        #D-net（判別側)の作成
        self.d_judge_F_logits=[]
        with tf.variable_scope("discrims"):

            with tf.variable_scope("discrimB"):
                self.d_judge_BR = discriminator(b_true_noised[:,:,:,:], None, self.args["filter_d"],
                                                                       self.args["strides_d"], self.args["d_depth"],
                                                                       self.args["D_channels"],a="B")

            with tf.variable_scope("discrimA"):
                self.d_judge_AR = discriminator(a_true_noised[:,:,:,:], None, self.args["filter_d"],
                                                                        self.args["strides_d"], self.args["d_depth"],
                                                                        self.args["D_channels"],"A")
            # with tf.variable_scope("discrimB2"):
            #     self.d_judge_BR0 = discriminator(b_true_noised[:,:,:,:1], None, self.args["filter_d"],
            #                                                            self.args["strides_d"], self.args["d_depth"],
            #                                                            self.args["D_channels"],a="B2")
            #
            #     self.d_judge_BF0 = discriminator(self.fake_aB_image[:,:,:,:1], True, self.args["filter_d"],
            #                                                            self.args["strides_d"], self.args["d_depth"],
            #                                                            self.args["D_channels"],a="B2")
            # with tf.variable_scope("discrimA2"):
            #     self.d_judge_AR0 = discriminator(a_true_noised[:,:,:,:1], None, self.args["filter_d"],
            #                                                             self.args["strides_d"], self.args["d_depth"],
            #                                                             self.args["D_channels"],a="A2")
            #     self.d_judge_AF0 = discriminator(self.fake_bA_image[:,:,:,:1], True, self.args["filter_d"],
            #                                                             self.args["strides_d"], self.args["d_depth"],
            #                                                             self.args["D_channels"],a="A2")


        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrims")
        #objective-functions of discriminator
        #D-netの目的関数
        # self.d_loss_AR = tf.reduce_mean(tf.squared_difference(self.d_judge_AR_logits,tf.ones_like(self.label_modela)))
        # self.d_loss_AF = tf.reduce_mean(tf.squared_difference(self.d_judge_AF_logits,tf.zeros_like(self.label_modela)))
        # self.d_loss_BR = tf.reduce_mean(tf.squared_difference(self.d_judge_BR_logits,tf.ones_like(self.label_modelb)))
        # self.d_loss_BF = tf.reduce_mean(tf.squared_difference(self.d_judge_BF_logits,tf.zeros_like(self.label_modelb)))
        #
        # self.d_loss_AR0 = tf.reduce_mean(tf.squared_difference(self.d_judge_AR0, self.label_modela))
        # self.d_loss_AF0 = tf.reduce_mean(tf.squared_difference(self.d_judge_AF0, self.label_modela * 0))
        # self.d_loss_BR0 = tf.reduce_mean(tf.squared_difference(self.d_judge_BR0, self.label_modelb))
        # self.d_loss_BF0 = tf.reduce_mean(tf.squared_difference(self.d_judge_BF0, self.label_modelb * 0))


        #saver
        #保存の準備
        self.saver = tf.train.Saver()


    def save(self, checkpoint_dir, step):
        model_name = "wave2wave.model"
        model_dir =  self.args["name_save"]
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    def load(self):
        # initialize variables
        # 変数の初期化
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(" [*] Reading checkpoint...")
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.epoch=self.saver
            return True
        else:
            return False





def discriminator(inp,reuse,f,s,depth,chs,a):
    current=tf.reshape(inp, [inp.shape[0],inp.shape[1],inp.shape[2],2])
    for i in range(depth):
        stddevs=math.sqrt(2.0/(f[0]*f[1]*int(current.shape[3])))
        ten = tf.layers.conv2d(current, chs[i], kernel_size=f, strides=s, padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        ten=tf.layers.batch_normalization(ten,training=True)
        if i!=depth-1:
            current = tf.nn.leaky_relu(ten)
        else:
            current=ten
    if a is "A" and reuse is not None:
        print(" [*] bottom shape:"+str(current.shape))
    #出力サイズB*H*24
    return current

def shift(data_inps,pitch):
    data_inp=data_inps.reshape(-1)
    return scale(time_strech(data_inp,1/pitch),data_inp.shape[0])
    # return time_strech(data_inp,1/pitch)
    # return scale(data_inp,data_inp.shape[0]/2)

def scale(inputs,len_wave):
    x=np.linspace(0.0,inputs.shape[0]-1,len_wave)
    ref_x_n=(x+0.5).astype(int)
    spec=inputs[ref_x_n[...]]
    return spec.reshape(-1)
def time_strech(datanum,speed):
    term_s = int(16000 * 0.05)
    fade=term_s//2
    pulus=int(term_s*speed)
    data_s=datanum.reshape(-1)
    spec=np.zeros(1)
    ifs=np.zeros(fade)
    for i_s in np.arange(0.0,data_s.shape[0],pulus):
        st=int(i_s)
        fn=min(int(i_s+term_s+fade),data_s.shape[0])
        dd=data_s[st:fn]
        if i_s + pulus >= data_s.shape[0]:
            spec = np.append(spec, dd)
        else:
            ds_in = np.linspace(0, 1, fade)
            ds_out = np.linspace(1, 0, fade)
            stock = dd[:fade]
            dd[:fade] = dd[:fade] * ds_in
            if st != 0:
                dd[:fade] += ifs[:fade]
            else:
                dd[:fade] += stock * np.linspace(1, 0, fade)
            if fn!=data_s.shape[0]:
                ifs = dd[-fade:] * ds_out
            spec=np.append(spec,dd[:-fade])
    return spec[1:]
def nowtime():
    return datetime.now().strftime("%Y_%m_%d %H_%M_%S")

def upload(voice,to,comment=""):
    voiced=voice.astype(np.int16)
    p=pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    ww = wave.open(to+".wav", 'wb')
    ww.setnchannels(1)
    ww.setsampwidth(p.get_sample_size(FORMAT))
    ww.setframerate(16000)
    ww.writeframes(voiced.reshape(-1).tobytes())
    ww.close()
    p.terminate()

def isread(path):
    wf=wave.open(path,"rb")
    ans=np.zeros([1],dtype=np.int16)
    dds=wf.readframes(1024)
    while dds != b'':
        ans=np.append(ans,np.frombuffer(dds,"int16"))
        dds = wf.readframes(1024)
    wf.close()
    ans=ans[1:]
    i=160000-ans.shape[0]
    if i>0:
        ans=np.pad(ans,(0,i),"constant")
    else:
        ans=ans[:160000]
    return ans
def bn(ten,depth):
    # return tf.layers.batch_normalization(inp,axis=3, training=False, trainable=True,
    #                                     gamma_initializer=tf.ones_initializer())
    name = "batch_normalization"
    with tf.variable_scope(name+str(depth)):
        sc = ten.shape[3]
        gamma = tf.get_variable("gamma", sc, tf.float32, tf.zeros_initializer(), trainable=True)
        beta = tf.get_variable("beta", sc, tf.float32, tf.zeros_initializer(), trainable=True)
        m,v=tf.nn.moments(ten,[0,1,2])
    return tf.nn.batch_normalization(ten, m, v, beta, gamma, 1e-8)

def imread(path):
    return np.load(path)
def mask_scale(dd,f,t,power):
    dd[:,f:t,0]-=(power/100*-dd[:,f:t,0])

    return dd
def mask_const(dd,f,t,power):
    dd[:,f:t,0]-=power
    # dd[:,:,1]=dd[:,:,1]*1.12
    return dd
def filter_clip(dd,f=1.5):
    dxf=np.maximum(dd,-f)+f+np.minimum(dd,f)-f
    return -dxf*0.5

def filter_mean(dd):
    dxx1=np.roll(dd,1)
    dxx1[:1]=dd[:1]
    dxx2=np.roll(dd,2)
    dxx2[:2] = dd[:2]
    dxx3= np.roll(dd, 3)
    dxx3[:3] = dd[:3]
    dxx4 = np.roll(dd, 4)
    dxx4[:4] = dd[:4]
    return (dd+dxx1+dxx2+dxx3+dxx4)/5.0