import tensorflow as tf
import os
import time
import numpy as np
import wave
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import pyaudio
from datetime import datetime
import json
import shutil
import math
BN_FLAG=False
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
        self.args["weight_Norm"]=1.0
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
        self.args["pitch_rate"] = 1.0
        self.args["pitch_res"]=563.0
        self.args["pitch_tar"]=563.0
        self.args["test_dir"] = "./test"
        self.args["architect"]="flatnet"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    dd = json.load(f)
                    keys = dd.keys()
                    for j in keys:
                        data=dd[j]
                        keys2 = data.keys()
                        for k in keys2:
                            if k in self.args:
                                if type(self.args[k]) == type(data[k]):
                                    self.args[k] = data[k]
                                else:
                                    print(" [!] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(
                                        type(self.args[k])) + "\"")
                            elif k[0] == "#":
                                pass
                            else:
                                print(" [!] Argument \"" + k + "\" is not exsits.")
            except json.JSONDecodeError as e:
                print(' [x] JSONDecodeError: ', e)
        else:
            print(" [!] Setting file is not found")
        if len(self.args["D_channels"]) != (self.args['d_depth'] + 1):
            print(" [!] Channels length and depth+1 must be equal ." + str(len(self.args["D_channels"])) + "vs" + str(self.args['d_depth'] + 1))
            self.args["D_channels"] = [min([2 ** (i + 1) - 2, 254]) for i in range(self.args['d_depth'] + 1)]
        if self.args["pitch_rate"]==1.0:
            self.args["pitch_rate"] = self.args["pitch_tar"]/self.args["pitch_res"]
            print(" [!] pitch_rate is not found . calculated value : "+str(self.args["pitch_rate"]))
        self.args["SHIFT"] = self.args["NFFT"]//2
        ss=int(self.args["input_size"])//int(self.args["SHIFT"])
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        self.input_size_model=[self.args["batch_size"],ss,self.args["NFFT"]//2,2]
        print("model input size:"+str(self.input_size_model))
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))
        if bool(self.args["debug"]):
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
        if  self.args["wave_otp_dir"] is not "False" :
            self.args["wave_otp_dir"]=self.args["wave_otp_dir"]+ self.args["name_save"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])
            shutil.copy(path,self.args["wave_otp_dir"]+"info.json")
            self.args["log_file"]=self.args["wave_otp_dir"]+"log.txt"
        self.checkpoint_dir = self.args["checkpoint_dir"]

    def build_model(self):


        #inputs place holder
        #入力
        self.input_model=tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net")
        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):
            with tf.variable_scope("generator_1"):
                self.fake_B_image =generator(tf.reshape(self.input_model,self.input_size_model), reuse=False,chs=self.args["G_channel"],chs2=self.args["G_channels"],depth=self.args["depth"],f=self.args["filter_g"],s=self.args["strides_g"],type=self.args["architect"],name="1")
            self.noise = tf.placeholder(tf.float32, [self.args["batch_size"]], "inputs_Noise")

        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")

        #saver
        #保存の準備
        self.saver = tf.train.Saver()



    def convert(self,in_put):
        #function of test
        #To convert wave file
        #テスト用関数
        #wave file　変換用

        tt=time.time()
        ipt=self.args["input_size"]+self.args["SHIFT"]
        times=in_put.shape[1]//(self.args["input_size"])+1
        if in_put.shape[1]%((self.args["input_size"])*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        otp2=np.asarray([],dtype=np.int16)
        otp3= np.asarray([[[]]], dtype=np.float32)
        rest = np.asarray([[[]]], dtype=np.float32)

        otp4 = np.asarray([[[]]], dtype=np.float32)
        rss=np.zeros([self.args["SHIFT"]])
        rss4=np.zeros([self.args["SHIFT"]])
        for t in range(times):
            # Preprocess
            # 前処理

            # Padiing
            # サイズ合わせ
            red=np.zeros((self.args["batch_size"]-1,ipt))
            start_pos=self.args["input_size"]*t+(in_put.shape[1]%self.args["input_size"])
            resorce=np.reshape(in_put[0,max(0,start_pos-ipt):start_pos,0],(1,-1)).astype(np.float32)
            r=max(0,ipt-resorce.shape[1])
            if r>0:
                resorce=np.pad(resorce,((0,0),(r,0)),'constant')
            red=np.append(resorce,red)
            red=red.reshape((self.args["batch_size"],ipt))
            res = np.zeros(self.input_size_model)
            # FFT
            # 短時間高速離散フーリエ変換
            MEAN_RANGE=2
            for i in range(self.args["batch_size"]):
                n=self.fft(red[i].reshape(-1))
                rest = np.append(rest, n[:, :, :].copy())
                res[i]=n[:,:self.args["SHIFT"],:]
            sts = res.shape[1]//MEAN_RANGE
            samples=res[0, :, :, 0].reshape(MEAN_RANGE,-1)
            means = np.mean(samples, axis=1)
            filter = -5.95
            means[means < filter] = -17.0
            means = np.tile(np.reshape(means, (-1, 1)), (sts, self.args["SHIFT"]))
            res[0, :, :, 0] =res[0, :, :, 0]- means
            samples = res[0, :, :, 0].reshape(MEAN_RANGE,-1)
            scales = np.tile(np.reshape(np.sqrt(np.var(samples, axis=1) + 1e-8), (-1)), (sts))
            mms = 1 / scales
            scales[means[:,0] < filter] = 0.001
            res[0, :, :, 0] = np.einsum("ij,i->ij", res[0, :, :, 0], mms)

            # filter = -4.75
            # means[means < filter] = -17.0
            # scl = np.tile(np.reshape(means, (-1, 1)), (1, self.args["SHIFT"]))
            # res[0, :, :, 0] = np.einsum("ij,i->ij", res[0, :, :, 0] , mms)+scl

            # running network
            # ネットワーク実行
            res2=self.sess.run(self.fake_B_image,feed_dict={ self.input_model:res})
            res3 = res2.copy()[:, :, ::-1, :]
            res2= np.append(res2,res3, axis=2)
            # res2[:,:,self.args["SHIFT"]:,1]*=-1

            # resas = np.append(resas, res[0])
            # Postprocess
            # 後処理
            a=res2[0].copy()
            c = a[:, :, 0]
            scales_mask = scales.copy()
            means_mask = means.copy()
            c = np.einsum("ij,i->ij", c, scales_mask)
            sm=np.tile(means_mask,(1,2))
            c=c+sm
            a[:,:,0]=c

            b=res2[0].copy()
            scales_mask = (scales.copy()+3)/2
            sm = (sm-9.52)/2
            c = b[:, :, 0]
            c = np.einsum("ij,i->ij", c, scales_mask)
            c = c + sm
            b[:,:,0] = c
            b[:,:,0] = np.clip(b[:,:,0], -60.0, 3.0)
            otp3 = np.append(otp3, a[:, :, :].copy())
            otp4 = np.append(otp4, a[:, :, :].copy())

            # otp3 = np.append(otp3, a[ 2:, :, :])

            # IFFT
            # 短時間高速離散逆フーリエ変換
            res2,rss=self.ifft(a,rss)
            res2=np.clip(res2,-1.0,1.0)
            res2=res2*32767
            res4, rss4 = self.ifft(b, rss4)
            res4 = np.clip(res4, -1.0, 1.0)
            res4 = res4 * 32767
            # chaching results
            # 結果の保存
            res2=res2.reshape(-1).astype(np.int16)
            res4 = res4.reshape(-1).astype(np.int16)
            # print(res2.shape)
            otp=np.append(otp,res2[-8192:])
            otp2 = np.append(otp2, res4[-8192:])
        h=otp.shape[0]-in_put.shape[1]
        if h>0:
            otp=otp[h:]
        h = otp2.shape[0] - in_put.shape[1]
        if h > 0:
            otp2 = otp2[h:]
        restx=rest.reshape([-1,self.args["NFFT"],2])
        otpx = otp3.reshape([-1, self.args["NFFT"], 2])
        return otp.reshape(1,in_put.shape[1],in_put.shape[2]),otp2,np.abs(restx[:,:self.args["SHIFT"],:]-otpx[:,:self.args["SHIFT"],:]),otp4

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
            print(" [x] %s is not found." % checkpoint_dir)
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
        v = fft_data[:, :self.args["NFFT"]// 2]
        reds = fft_data[-1, self.args["NFFT"] // 2:].copy()
        lats = np.roll(fft_data[:, self.args["NFFT"] // 2:], 1, axis=0 )
        lats[0, :]=redi
        spec = np.reshape(v + lats, (-1))
        return spec,reds


def generator(current_outputs,reuse,depth,chs,chs2,f,s,type,name):
    if type == "flatnet":
        return generator_flatnet(current_outputs,reuse,depth,chs,f,s,0,name)
    elif type == "ps_flatnet":
        return generator_flatnet(current_outputs, reuse, depth, chs, f, s, 1,name)
    elif type == "hybrid_flatnet":
        return generator_flatnet(current_outputs, reuse, depth, chs, f, s, 2,name)
    elif type == "double_flatnet":
        return generator_flatnet(current_outputs, reuse, depth, chs, f, s, 3, name)
    elif type == "ps_decay_flatnet":
        return generator_flatnet_decay(current_outputs, reuse, depth, chs2, f, s, 1, name)
    elif type == "decay_flatnet":
        return generator_flatnet_decay(current_outputs, reuse, depth, chs2, f, s, 0, name)
    elif type == "ps_unet":
        return generator_unet(current_outputs, reuse, depth, chs, f, s, 1)
    elif type == "hybrid_unet":
        return generator_unet(current_outputs, reuse, depth, chs, f, s, 2)
    else :
        return  generator_unet(current_outputs,reuse,depth,chs,f,s)
def generator_flatnet(current_outputs,reuse,depth,chs,f,s,ps,name):
    current=current_outputs
    output_shape=int(current.shape[3])
    #main process
    for i in range(depth):
        connections = current
        if ps==1:
            ten = block_ps(current, output_shape,chs, f, i, reuse,i!=depth-1,name)
        elif ps==2 :
            ten=block3(current,f,chs,i,reuse,i!=depth-1,name)
        elif ps == 3:
            ten = block_double(current, f,s, chs, i, reuse, i != depth - 1, pixs=4 )
        else :
            ten = block_dc(current, output_shape, chs, f, s, i, reuses=reuse, shake=i != depth - 1,name=name)
        if i!=depth-1 and ps!=3:
            current = ten + connections
        else:
            current=ten
    return current
def generator_flatnet_decay(current_outputs,reuse,depth,chs,f,s,ps,name):
    current=current_outputs
    output_shape=chs
    #main process
    for i in range(depth):
        connections = current
        if ps==1:
            ten = block_ps(current, chs[i*2+1],chs[i*2],f, i, reuse,i!=depth-1,name)
        else :
            ten = block_dc(current,chs[i*2+1],chs[i*2], f, s, i, reuses=reuse, shake=i != depth - 1,name=name)
        if i!=depth-1 and ps!=3:
            ims=ten.shape[3]//connections.shape[3]
            if ims!=0:
                connections=tf.tile(connections,[1,1,1,ims])
            elif connections.shape[3]>ten.shape[3]:
                connections = connections[:,:,:,:ten.shape[3]]
            current = ten + connections
        else:
            current=ten
    return current
def block_double(current,f,s,chs,depth,reuses,shake,pixs=2):
    currenta=current
    tenA=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(tenA.shape[3])))
    tenA = tf.layers.conv2d(tenA, chs, kernel_size=f, strides=s, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv11"+str(depth))
    tenA = tf.layers.batch_normalization(tenA, axis=3, training=True, trainable=True, reuse=reuses,
                                            name="bn11" + str(depth))

    tenA = tf.nn.leaky_relu(tenA,name="lrelu"+str(depth))

    tt=tf.pad(tenA,((0,0),(0,0),(4,0),(0,0)),"reflect")
    ten1=tt[:,:,:-4,:]
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(tenA.shape[3])))
    tenA = tf.layers.conv2d_transpose(ten1, 2, kernel_size=f, strides=s, padding="VALID",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                                      data_format="channels_last", reuse=reuses, name="deconv11" + str(depth))

    tenB=current
    ps_f=[pixs,pixs]
    tenB = tf.layers.conv2d(tenB, chs, kernel_size=ps_f, strides=ps_f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                           data_format="channels_last", reuse=reuses, name="conv21" + str(depth))
    tenB = tf.layers.batch_normalization(tenB, axis=3, training=True, trainable=True, reuse=reuses,
                                         name="bn21" + str(depth))

    tenB = tf.nn.leaky_relu(tenB, name="lrelu" + str(depth))
    tenB = deconve_with_ps(tenB, pixs, 2, depth, reuses=reuses)

    ten = (tenA + tenB) * 0.5
    if shake:
        ten=tf.nn.leaky_relu(ten)
        ten=ten+currenta

    return ten

def block_dc(current,output_shape,chs,f,s,depth,reuses,shake,name):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))

    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=s, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv11"+str(depth))
    ten = tf.layers.batch_normalization(ten, axis=3, training=False, trainable=False, reuse=reuses,
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
def block_ps(current,output_shape,chs,f,depth,reuses,relu,name):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv21"+str(depth))

    ten = tf.layers.batch_normalization(ten, axis=3, training=False, trainable=True, reuse=reuses,
                                        name="bn11" + str(depth))
    tt = tf.pad(ten, ((0, 0), (0, 0), (2, 0), (0, 0)), "reflect")
    ten = tt[:, :, :-2, :]
    ten = tf.nn.leaky_relu(ten,name="lrelu"+str(depth))
    ten=deconve_with_ps(ten,f[0],output_shape,depth,reuses=reuses)
    if relu:
        ten=tf.nn.leaky_relu(ten)
    return ten
def block3(current,f,chs,depth,reuses,shake,name):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv21"+str(depth))
    # ten = tf.contrib.layers.instance_norm(ten,reuse=reuses,scope="g_net"+name+str(depth))
    ten = tf.layers.batch_normalization(ten, axis=3, training=True, trainable=True, reuse=reuses,
                                        name="bn11" + str(depth))
    ten = tf.nn.leaky_relu(ten,name="lrelu"+str(depth))
    n=(depth%2)*2-1

    pos=tf.constant(np.linspace(1.0,0.1,int(ten.shape[2])),dtype=tf.float32,shape=ten.shape)
    ten1 = ten*pos
    ten2 = ten*pos
    if shake:
        ten1 = tf.manip.roll(ten, n*4, 2)
        # ten2 = tf.manip.roll(ten, -n * 4, 1)
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
def deconve_with_ps(inp,r,otp_shape,depth,f=[1,1],reuses=None):
    chs_r=(r**2)*otp_shape
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(inp.shape[3])))
    ten = tf.layers.conv2d(inp, chs_r, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                           data_format="channels_last", reuse=reuses, name="deconv_ps1" + str(depth))
    b_size = ten.shape[0]
    in_h = ten.shape[1]
    in_w = ten.shape[2]
    ten = tf.reshape(ten, [b_size, r, r, in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [b_size, in_h * r, in_w * r, otp_shape])
    return ten


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
    if ps==0:
        ten=tf.layers.conv2d_transpose(ten, output_shape, kernel_size=f, strides=s, padding="VALID",
                         kernel_initializer=tf.contrib.layers.xavier_initializer(), data_format="channels_last",
                         name="deconv" + str(depth), reuse=reuse)
    elif ps==1:
        ten=deconve_with_ps(ten,f[0],output_shape,depth,reuses=reuse)
    else:
        ten1 = tf.layers.conv2d_transpose(ten, output_shape, kernel_size=f, strides=s, padding="VALID",
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), data_format="channels_last",
                               name="deconv" + str(depth), reuse=reuse)
        ten2 = deconve_with_ps(ten, f[0], output_shape, depth, reuses=reuse)
        ten = ten1 + ten2
    if bn:
        ten=tf.layers.batch_normalization(ten,axis=3,training=False,gamma_initializer=tf.random_normal_initializer(1.0, 0.2),name="bn_u"+str(depth),reuse=reuse)
    return ten
def down_layer(current,output_shape,f,s,reuse,depth):
    ten=current
    if depth!=0:
        ten=tf.layers.batch_normalization(current,axis=3,training=False,gamma_initializer=tf.random_normal_initializer(1.0, 0.2),name="bn_d"+str(depth),reuse=reuse)
    ten=tf.layers.conv2d(ten, output_shape,kernel_size=f ,strides=s, padding="VALID",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last",name="conv"+str(depth),reuse=reuse)
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
    ww = wave.open(to+nowtime()+comment+".wav", 'wb')
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
    dd[:,f:t,0]-=(power/100*dd[:,f:t,0])

    return dd
def mask_const(dd,f,t,power):
    dd[:,f:t,0]-=power
    # dd[:,:,1]=dd[:,:,1]*1.12
    return dd