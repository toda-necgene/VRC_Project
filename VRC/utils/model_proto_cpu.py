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
        self.args["dilation_size"] = 7
        self.args["debug"] = False
        self.args["noise"] = False
        self.args["cupy"] = False
        self.args["D_channels"] =[2]
        self.args["G_channel"] = 32
        self.args["G_channels"] = [32]
        self.args["strides_g"] = [[2,2]]
        self.args["strides_d"] = [2,2]
        self.args["filter_g"] = [[8,8]]
        self.args["filter_d"] = [4,4]
        self.args["dilations"] = [1,1,1,1]
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
        self.args["lr_decay_term"] = 100
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
                self.args['depth']*2))
            self.args["D_channels"] = [min([2 ** (i + 1) - 2, 254]) for i in range(self.args['depth']*2)]
        self.args["SHIFT"] = self.args["NFFT"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        ss=self.args["input_size"]//self.args["SHIFT"]
        self.input_size_model=[None,ss*2+self.args["dilation_size"] ,self.args["NFFT"]//2,2]
        self.input_size_test = [1, ss+self.args["dilation_size"], self.args["NFFT"] // 2, 2]
        self.output_size = [1, ss, self.args["NFFT"] // 2, 2]
        print("model input size:"+str(self.output_size))
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

    def build_model(self):

        #inputs place holder
        #入力

        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "input_T")
        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1",reuse=tf.AUTO_REUSE):

                self.fake_aB_image_test = generator(self.input_model_test, reuse=None,
                                                chs=self.args["G_channel"], depth=self.args["depth"],
                                                f=self.args["filter_g"],
                                                s=self.args["strides_g"],
                                                d=self.args["dilations"],
                                                type=self.args["architect"],
                                                train=False,name="1", chs2=self.args["G_channels"])

        #saver
        #保存の準備
        self.saver = tf.train.Saver()



    def convert(self,in_put):
        #function of test
        #To convert wave file
        #テスト用関数
        #wave file　変換用


        tt=time.time()
        ipt=self.args["input_size"]+self.args["SHIFT"]+self.args["SHIFT"]*self.args["dilation_size"]
        times=in_put.shape[1]//(self.args["input_size"])+1
        if in_put.shape[1]%((self.args["input_size"])*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        res3 = np.zeros([1,self.args["NFFT"],2], dtype=np.float32)
        rss=np.zeros([self.input_size_test[2]],dtype=np.float64)
        for t in range(times):
            # Preprocess
            # 前処理

            # Padiing
            # サイズ合わせ
            start_pos=self.args["input_size"]*t+(in_put.shape[1]%self.args["input_size"])
            resorce=np.reshape(in_put[0,max(0,start_pos-ipt):start_pos,0],(1,-1))
            r=max(0,ipt-resorce.shape[1])
            if r>0:
                resorce=np.pad(resorce,((0,0),(r,0)),'constant')
            red=resorce
            red=red.reshape((1,ipt))
            res = np.zeros([1,self.input_size_test[1],self.input_size_test[2],self.input_size_test[3]])
            # FFT
            # 短時間高速離散フーリエ変換
            n=self.fft(red[0].reshape(-1)/32767.0)
            res[0]=n[:,:self.args["SHIFT"]]
            # means = np.mean(res[0,:,:,0], axis=1)
            # means = np.tile(np.reshape(means, (-1, 1)), (1, self.args["SHIFT"]))
            # res[0, :, :, 0] = res[0, :, :, 0] - means
            # scales =np.reshape(np.sqrt(np.var(res[0,:,:,0], axis=1) + 1e-8), (-1))
            # mms = 1 / scales
            # res[0, :, :, 0] = np.einsum("ij,i->ij", res[0, :, :, 0], mms)
            # running network
            # ネットワーク実行
            res=res[:,:self.args["SHIFT"],:]
            res=self.sess.run(self.fake_aB_image_test,feed_dict={ self.input_model_test:res})
            res2=res.copy()[:,:,::-1,:]
            res=np.append(res,res2,axis=2)
            res[:,:,self.args["SHIFT"]:,1]*=-1

            a = res[0].copy()
            # c = a[:, :, 0]
            # scales_mask = scales.copy()[:self.output_size[1]]+1.7
            # means_mask = means.copy()[:self.output_size[1]]-3.5
            # c = np.einsum("ij,i->ij", c, scales_mask)
            # sm = np.tile(means_mask, (1, 2))
            # c = c + sm
            # c=np.clip(c,-60.0,3.0)
            # a[:, :, 0] = c
            res3 = np.append(res3, a).reshape(-1,self.args["NFFT"],2)


            # Postprocess
            # 後処理

            # IFFT
            # 短時間高速離散逆フーリエ変換
            res,rss=self.ifft(a,rss)
            # 変換後処理
            # bsd = filter_mean(res)
            # res = filter_clip(res, f=0.5)
            res=np.clip(res,-1.0,1.0)
            res=res*32767
            # chaching results
            # 結果の保存
            res=res.reshape(-1).astype(np.int16)
            otp=np.append(otp,res[-8192:])
        h=otp.shape[0]-in_put.shape[1]
        if h>0:
            otp=otp[h:]

        return otp.reshape(1,in_put.shape[1],in_put.shape[2]),time.time()-tt,res3[1:]


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




def discriminator(inp,reuse,f,s,depth,chs,a):
    current=tf.reshape(inp, [-1,inp.shape[1],inp.shape[2],2])
    for i in range(depth):
        stddevs=math.sqrt(2.0/(f[0]*f[1]*int(current.shape[3])))
        ten=current
        if i!=0:
            ten = tf.layers.batch_normalization(ten, name="bn_disk" + str(i), training=True, reuse=reuse)
        ten = tf.layers.conv2d(ten, chs[i], kernel_size=f, strides=s, padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        if i!=depth-1:
            current = tf.nn.leaky_relu(ten)
        else:
            current=ten
    if a is "A" and reuse is not None:
        print(" [*] bottom shape:"+str(current.shape))
    #出力サイズB*H*24
    return current
def generator(current_outputs,reuse,depth,chs,chs2,f,s,d,type,train,name):
    if type == "ps_decay_flatnet":
        return generator_flatnet_decay(current_outputs, reuse, depth, chs2, f, s,d, 1, train)
    elif type == "double_decay_flatnet":
        return generator_flatnet_decay(current_outputs, reuse, depth, chs2, f, s,d, 3, train)
    elif type == "decay_flatnet":
        return generator_flatnet_decay(current_outputs, reuse, depth, chs2, f, s,d, 0, train)
    elif type == "ps_unet":
        return generator_unet(current_outputs, reuse, depth, chs, f, s, 1)
    elif type == "hybrid_unet":
        return generator_unet(current_outputs, reuse, depth, chs, f, s, 2)
    else :
        return  generator_unet(current_outputs,reuse,depth,chs,f,s)
def generator_flatnet_decay(current_outputs,reuse,depth,chs,f,s,d,ps,train):
    current=current_outputs
    #main process
    for i in range(depth):
        connections = current
        if ps==1:
            ten = block_ps(current, chs[i*2+1],chs[i*2],f[i], i, reuse,i!=depth-1,train=train)
        elif ps == 3:
            ten = block_double(current, chs[i * 2 + 1], chs[i * 2], f[i],s[i], i, reuse, i != depth - 1,pixs=f[i], train=train)
        else :
            ten = block_dc(current,chs[i*2+1],chs[i*2], f[i], s[i], i, reuses=reuse, shake=i != depth - 1,train=train)
        if i!=depth-1:
            ims=ten.shape[3]//connections.shape[3]
            if ims!=0:
                connections=tf.tile(connections,[1,1,1,ims])
            elif connections.shape[3]>ten.shape[3]:
                connections = connections[:,:,:,:ten.shape[3]]
            ims2 = int(ten.shape[1])-int(connections.shape[1])
            if ims2<0:
                connections = connections[:, :ims2, :, :]
            current = ten + connections
        else:
            current=ten
    current=dilations(current,d,reuse,train,chs,depth)
    return current
def dilations(inp,d,reuse,train,chs,startd):
    ten=inp
    stddevs = math.sqrt(2.0 / (2 * 1 * int(ten.shape[3])))
    for i in range(len(d)):
        tenB = tf.nn.tanh(ten)
        ten = ten * tenB

        ten = tf.layers.conv2d(ten, chs[i + startd * 2], kernel_size=[2, 1], strides=[1, 1], padding="VALID",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                                data_format="channels_last", reuse=reuse, name="conv11" + str(startd * 2 + i),
                                dilation_rate=(d[i], 1))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="bn11" + str(startd*2+i))
    return ten
def block_dc(current,output_shape,chs,f,s,depth,reuses,shake,train):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))

    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=s, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv11"+str(depth))
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bn11" + str(depth))
    tt = tf.pad(ten, ((0, 0), (0, 0), (4, 0), (0, 0)), "reflect")
    ten = tt[:, :, :-4, :]
    ten = tf.nn.leaky_relu(ten,name="lrelu"+str(depth))

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten = tf.layers.conv2d_transpose(ten, output_shape, kernel_size=f, strides=s, padding="VALID",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                                     data_format="channels_last",reuse=reuses,name="deconv11"+str(depth))
    if shake:
        ten=tf.nn.leaky_relu(ten)
    return ten
def block_ps(current,output_shape,chs,f,depth,reuses,relu,train):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv21"+str(depth))

    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bn11" + str(depth))
    ten = tf.nn.leaky_relu(ten, name="lrelu" + str(depth))

    if relu:
        tt = tf.pad(ten, ((0, 0), (0, 0), (2, 0), (0, 0)), "reflect")
        ten = tt[:, :, :-2, :]+ten
    ten=deconve_with_ps(ten,f[0],output_shape,depth,reuses=reuses)
    if relu:
        ten=tf.nn.leaky_relu(ten)
    return ten
def block_hybrid(current,f,chs,depth,reuses,shake,train):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv21"+str(depth))
    # ten = tf.contrib.layers.instance_norm(ten,reuse=reuses,scope="g_net"+name+str(depth))
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bn11" + str(depth))
    ten = tf.nn.leaky_relu(ten,name="lrelu"+str(depth))

    pos=tf.constant(np.linspace(1.0,0.1,int(ten.shape[2])),dtype=tf.float32,shape=ten.shape)
    ten1 = ten
    ten2 = ten
    if shake:
        ten1 = tf.manip.roll(ten, 2, 2)
        ten2 = tf.manip.roll(ten, -2, 1)
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten1=deconve_with_ps(ten1,f[0],2,depth,reuses=reuses)
    ten2 =  tf.layers.conv2d_transpose(ten2, 2, kernel_size=f, strides=f, padding="VALID",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                                       data_format="channels_last", reuse=reuses, name="deconv11" + str(depth))

    if shake:
        ten1 = tf.nn.leaky_relu(ten1)
        ten2 = tf.nn.relu(ten2)
    ten=(ten1+ten2)*0.5
    return ten
def block_double(current,output_shape,chs,f,s,depth,reuses,shake,pixs=[2,2],train=True):
    ten=current
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=s, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                           data_format="channels_last", reuse=reuses, name="conv11" + str(depth),dilation_rate=(1,1))
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bn11" + str(depth))

    ten = tf.nn.leaky_relu(ten, name="lrelu" + str(depth))
    tenB=ten
    if shake:
        tt = tf.pad(tenB, ((0, 0), (0, 0), (1, 0), (0, 0)), "reflect")
        tenB = tt[:, :, :-1, :]
    tenB = deconve_with_ps(tenB, pixs, output_shape, depth, reuses=reuses,name="02")
    tenA = deconve_with_ps(ten, pixs, output_shape, depth, reuses=reuses,name="01")

    ten = (tenA + tenB) * 0.5
    if shake:
        tenC = tf.nn.tanh(ten)
        ten = ten * tenC

    return ten

def deconve_with_ps(inp,r,otp_shape,depth,f=[1,1],reuses=None,name=""):
    chs_r=r[0]*r[1]*otp_shape
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(inp.shape[3])))
    ten = tf.layers.conv2d(inp, chs_r, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                           data_format="channels_last", reuse=reuses, name="deconv_ps1"+name + str(depth))
    b_size = -1
    in_h = ten.shape[1]
    in_w = ten.shape[2]
    ten = tf.reshape(ten, [b_size, r[0], r[1], in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [b_size, in_h * r[0], in_w * r[1], otp_shape])
    return ten[:,:,:,:]

def generator_unet(current_outputs,reuse,depth,chs,f,s,ps=0):
    current=current_outputs
    connections=[ ]
    for i in range(depth):
        connections.append(current)
        current = down_layer(current, chs*(i+1) ,f,s,reuse,i)
    print("shape of structure:"+str([[int(c.shape[0]),int(c.shape[1]),int(c.shape[2]),int(c.shape[3])] for c in connections]))
    for i in range(depth):
        current=up_layer(current,chs*(depth-i-1) if (depth-i-1)!=0 else 2,f,s,i,i!=(depth-1),depth-i-1>2,reuse,ps=ps)
        if i!=depth-1:
            current += connections[depth - i -1]
    return tf.reshape(current,current_outputs.shape)

def up_layer(current,output_shape,f,s,depth,bn=True,do=False,reuse=None,ps=0):
    ten=tf.nn.leaky_relu(current)
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    if ps==0:
        ten=tf.layers.conv2d_transpose(ten, output_shape, kernel_size=f, strides=s, padding="VALID",
                         kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",
                         name="deconv" + str(depth), reuse=reuse)
    elif ps==1:
        ten=deconve_with_ps(ten,f[0],output_shape,depth,reuses=reuse)
    else:
        ten1 = tf.layers.conv2d_transpose(ten, output_shape, kernel_size=f, strides=s, padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",
                               name="deconv" + str(depth), reuse=reuse)
        ten2 = deconve_with_ps(ten, f[0], output_shape, depth, reuses=reuse)
        ten = ten1 + ten2
    if bn:
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2),name="bn_u"+str(depth),reuse=reuse)
    return ten
def down_layer(current,output_shape,f,s,reuse,depth):
    ten=current
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(current.shape[3])))
    if depth!=0:
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2),name="bn_d"+str(depth),reuse=reuse)

    ten=tf.layers.conv2d(ten, output_shape,kernel_size=f ,strides=s, padding="VALID",kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),data_format="channels_last",name="conv"+str(depth),reuse=reuse)
    ten=tf.nn.leaky_relu(ten)
    return ten

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