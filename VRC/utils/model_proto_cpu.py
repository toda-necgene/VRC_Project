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
        self.args["depth"] =[4]
        self.args["d_depth"] = 4
        self.args["train_epoch"]=500
        self.args["stop_itr"] = -1
        self.args["start_epoch"]=0
        self.args["test"]=True
        self.args["log"] = True
        self.args["tensorboard"]=False
        self.args["hyperdash"]=False
        self.args["input_size"] = 8192
        self.args["weight_Cycle"]=1.0
        self.args["weight_GAN"] = 1.0
        self.args["NFFT"]=128
        self.args["debug"] = False
        self.args["noise"] = False
        self.args["cupy"] = False
        self.args["D_channels"] =[2]
        self.args["G_channels"] = [32]
        self.args["strides_d"] = [2,2]
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
        self.args["dilations"]=[1]
        self.args["dilation_size"]=7
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
        if len(self.args["D_channels"]) != (self.args['d_depth'] + 1):
            print(" [!] Channels length and depth+1 must be equal ." + str(len(self.args["D_channels"])) + "vs" + str(self.args['d_depth'] + 1))
            self.args["D_channels"] = [min([2 ** (i + 1) - 2, 254]) for i in range(self.args['d_depth'] + 1)]
        self.args["SHIFT"] = self.args["NFFT"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        ss=self.args["input_size"]//self.args["SHIFT"]+self.args["dilation_size"]
        self.input_size_model=[None,ss+self.args["dilation_size"]+1,self.args["NFFT"]//2,2]
        self.input_size_test = [None, ss, self.args["NFFT"] // 2, 2]
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

        else:
            self.dbx=None
        self.checkpoint_dir = self.args["checkpoint_dir"]
        self.build_model()
    def build_model(self):

        #inputs place holder
        #入力
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_G-net_A")

        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1"):
                self.fake_aB_image_test = generator(self.input_model_test, reuse=None,
                                                chs=self.args["G_channels"], depth=self.args["depth"],
                                                d=self.args["dilations"],
                                                train=False)


        #saver
        #保存の準備
        self.saver = tf.train.Saver()



    def convert(self,in_put):
        #function of test
        #To convert wave file
        #テスト用関数
        #wave file　変換用


        tt=time.time()
        ipt=self.args["input_size"]+self.args["SHIFT"]*self.args["dilation_size"]+self.args["SHIFT"]
        times=in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        res3 = np.zeros([1,self.args["NFFT"],2], dtype=np.float32)
        rss=np.zeros([self.input_size_model[2]],dtype=np.float64)
        for t in range(times):
            # Preprocess
            # 前処理

            # Padiing
            # サイズ合わせ
            start_pos=self.args["input_size"]*t+(in_put.shape[0]%self.args["input_size"])
            resorce=np.reshape(in_put[max(0,start_pos-ipt):start_pos],(-1))
            r=max(0,ipt-resorce.shape[0])
            if r>0:
                resorce=np.pad(resorce,(r,0),'constant')
            # FFT
            # 短時間高速離散フーリエ変換
            res=self.fft(resorce.reshape(-1)/32767.0)
            # running network
            # ネットワーク実行
            res=res[:,:self.args["SHIFT"],:].reshape([1,-1,self.args["SHIFT"],2])
            res=self.sess.run(self.fake_aB_image_test,feed_dict={ self.input_model_test:res})
            res2=res.copy()[:,:,::-1,:]
            res=np.append(res,res2,axis=2)
            res[:,:,self.args["SHIFT"]:,1]*=-1
            a=res[0].copy()
            res3 = np.append(res3, a).reshape(-1,self.args["NFFT"],2)


            # Postprocess
            # 後処理

            # IFFT
            # 短時間高速離散逆フーリエ変換
            res,rss=self.ifft(a,rss)
            # 変換後処理
            res=np.clip(res,-1.0,1.0)
            res=res*32767
            # chaching results
            # 結果の保存
            res=res.reshape(-1).astype(np.int16)
            otp=np.append(otp,res[-self.args["input_size"]:])
        h=otp.shape[0]-in_put.shape[0]
        if h>0:
            otp=otp[h:]

        return otp.reshape(-1),time.time()-tt,res3[1:]



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

    def fft(self,data):

        time_ruler = data.shape[0] // self.args["SHIFT"]
        if data.shape[0] % self.args["SHIFT"] == 0:
            time_ruler -= 1
        window = np.hamming(self.args["NFFT"])
        pos = 0
        wined = np.zeros([time_ruler, self.args["NFFT"]])
        for fft_index in range(time_ruler):
            frame = data[pos:pos + self.args["NFFT"]]
            wined[fft_index] = frame * window
            pos += self.args["SHIFT"]
        fft_r = np.fft.fft(wined, n=self.args["NFFT"], axis=1)
        re = fft_r.real.reshape(time_ruler, -1)
        im = fft_r.imag.reshape(time_ruler, -1)
        c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
        d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
        spec = np.concatenate((c, d), 2)
        return spec
    def ifft(self,data,redi):
        a=data
        a[:, :, 0]=np.clip(a[:, :, 0],a_min=-100000,a_max=88)
        sss=np.exp(a[:,:,0])
        p = np.sqrt(sss)
        r = p * (np.cos(a[:, :, 1]))
        i = p * (np.sin(a[:, :, 1]))
        dds = np.concatenate((r.reshape(r.shape[0], r.shape[1], 1), i.reshape(i.shape[0], i.shape[1], 1)), 2)
        data=dds[:,:,0]+1j*dds[:,:,1]
        fft_s = np.fft.ifft(data,n=self.args["NFFT"], axis=1)
        fft_data = fft_s.real
        # fft_data[:]/=window
        v = fft_data[:, :self.args["NFFT"]// 2]
        reds = fft_data[-1, self.args["NFFT"] // 2:].copy()
        lats = np.roll(fft_data[:, self.args["NFFT"] // 2:], 1, axis=0 )
        lats[0, :]=redi
        spec = np.reshape(v + lats, (-1))
        return spec,reds




def discriminator(inp,reuse,depth,chs,train=True):
    current=inp
    for i in range(depth):
        stddevs=math.sqrt(2.0/(16*int(current.shape[3])))
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[2,5], strides=[1,2], padding="VALID",kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        ten = tf.layers.batch_normalization(ten, axis=3, trainable=False, training=train, reuse=reuse,
                                            name="bn_disc" + str(i))
        # ten=tf.layers.dropout(ten,0.125,training=train)
        current = tf.nn.leaky_relu(ten)
    print(" [*] bottom shape:"+str(current.shape))
    h4=tf.reshape(current, [-1,current.shape[1]*current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
    return ten
def generator(current_outputs,reuse,depth,chs,d,train):
    return block_res(current_outputs, chs,1,depth,reuse,d,train)

def block_res(current,chs,rep_pos,depth,reuses,d,train=True):
    ten = current
    tenM=[]
    times=depth[0]
    res=depth[1]
    for i in range(times-1):
        stddevs = math.sqrt(2.0 / ( 4 * int(ten.shape[3])))
        tenA = tf.layers.conv2d(ten, chs[i], kernel_size=[1, 4], strides=[1, 4], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                               data_format="channels_last", reuse=reuses, name="convA"+str(i) + str(rep_pos),
                               dilation_rate=(1, 1))

        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bnA_en"+str(i) + str(rep_pos))

        ten=tf.nn.leaky_relu(tenA)
        tenM.append(ten)

    stddevs = math.sqrt(2.0 / (4 * int(ten.shape[3])))
    ten = tf.layers.conv2d(ten, chs[times-1], kernel_size=[1, 2], strides=[1, 2], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                           data_format="channels_last", reuse=reuses, name="conv"+str(times-1) + str(rep_pos),
                           dilation_rate=(1, 1))
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bn"+str(times-1) + str(rep_pos))
    ten = tf.nn.leaky_relu(ten)
    stddevs = math.sqrt(2.0 / (2 * int(ten.shape[3])))
    tms=times
    for i in range(len(d)):
        ten = tf.layers.conv2d(ten, chs[i + tms], kernel_size=[2, 1], strides=[1, 1], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                               data_format="channels_last", reuse=reuses, name="conv_p" + str(i),
                               dilation_rate=(d[i], 1))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bn_p" + str(i))

        ten = tf.nn.leaky_relu(ten)
    tms+=len(d)
    for i in range(res):
        stddevs = math.sqrt(2.0 / (7 * int(ten.shape[3])))

        tenA=ten

        # ten = tf.layers.conv2d(tenA, chs[tms+i], [1,7], [1,1], padding="SAME",
        #                                   kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
        #                                   data_format="channels_last", reuse=reuses, name="res_convA"+str(i) + str(rep_pos))
        # ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
        #                                      name="bnA"+str(tms+i) + str(rep_pos))
        # ten = tf.nn.leaky_relu(ten)
        ten = tf.layers.conv2d(tenA, chs[tms + i] * 4, [1, 7], [1, 4], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                               data_format="channels_last", reuse=reuses, name="res_conv" + str(i) + str(rep_pos))

        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bnA" + str(tms + i) + str(rep_pos))
        ten = tf.nn.leaky_relu(ten)
        ten = tf.layers.conv2d_transpose(ten, chs[tms + i], [1, 7], [1, 4], padding="SAME",
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                                         use_bias=False,
                                         data_format="channels_last", reuse=reuses,
                                         name="res_deconv" + str(i) + str(rep_pos))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bnB" + str(tms + i) + str(rep_pos))
        ten = tf.nn.leaky_relu(ten)

        ten=ten+tenA
    tms+=res
    ten = deconve_with_ps(ten, [1, 2], chs[tms], rep_pos, reuses=reuses, name="00")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                         name="bn"+str(times+res) + str(rep_pos))
    ten = tf.nn.leaky_relu(ten)
    for i in range(times-1):
        ten+=tenM[times-i-2][:,:8,:,:]
        ten = deconve_with_ps(ten, [1, 4], chs[tms+1+i], rep_pos, reuses=reuses, name=str(i+1))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bn"+str(i+1+times+res) + str(rep_pos))
        ten=tf.nn.leaky_relu(ten)

    tenA=ten
    tenA = tf.layers.conv2d(tenA, 16, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                            data_format="channels_last", reuse=reuses, name="res_last1A"  + str(rep_pos))
    tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bnAL" + str( 1 + times + res) + str(rep_pos))
    tenA=tf.nn.leaky_relu(tenA)
    tenA = tf.layers.conv2d(tenA, 1, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                            data_format="channels_last", reuse=reuses, name="res_last2A" + str(rep_pos))

    tenB=ten
    tenB = tf.layers.conv2d(tenB, 16, [1, 1], [1, 1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                           data_format="channels_last", reuse=reuses, name="res_last1B" + str(rep_pos))
    tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=reuses,
                                         name="bnBL" + str(1 + times + res) + str(rep_pos))
    tenB = tf.nn.leaky_relu(tenB)
    tenB = tf.layers.conv2d(tenB, 1, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                            data_format="channels_last", reuse=reuses, name="res_last2B" + str(rep_pos))
    ten=tf.concat([tenA,tenB],3)
    return ten
def deconve_with_ps(inp,r,otp_shape,depth,reuses=None,name=""):
    chs_r=r[0]*r[1]*otp_shape
    stddevs = math.sqrt(2.0 / int(inp.shape[3]))
    ten = tf.layers.conv2d(inp, chs_r, kernel_size=[1,1], strides=[1,1], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                           data_format="channels_last", reuse=reuses, name="deconv_ps1"+name + str(depth))
    b_size = -1
    in_h = ten.shape[1]
    in_w = ten.shape[2]
    ten = tf.reshape(ten, [b_size, r[0], r[1], in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [b_size, in_h * r[0], in_w * r[1], otp_shape])
    return ten[:,:,:,:]

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