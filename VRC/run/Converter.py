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
        self.args["G_channel2"] = 32
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
                with open(path,"r") as f:
                    data=json.load(f)
                    keys=data.keys()
                    for k in keys:
                        if k in self.args:
                            if type(self.args[k])==type(data[k]):
                                self.args[k]=data[k]
                            else:
                                print(" [!] Argumet \""+k+"\" is incorrect data type. Please change to \""+str(type(self.args[k]))+"\"")
                        else:
                            print(" [!] Argument \"" + k + "\" is not exsits.")
            except json.JSONDecodeError as e:
                 print(' [x] JSONDecodeError: ', e)
        else:
            print( " [!] Setting file is not found")
        if len(self.args["D_channels"]) != (self.args['d_depth'] + 1):
            print(" [!] Channels length and depth+1 must be equal ." + str(len(self.args["D_channels"])) + "vs" + str(self.args['d_depth'] + 1))
            self.args["D_channels"] = [min([2 ** (i + 1) - 2, 254]) for i in range(self.args['d_depth'] + 1)]
        if self.args["pitch_rate"]==1.0:
            self.args["pitch_rate"] = self.args["pitch_tar"]/self.args["pitch_res"]
            print(" [!] pitch_rate is not found . calculated value : "+str(self.args["pitch_rate"]))
        self.args["SHIFT"] = self.args["NFFT"]//2
        ss=int(self.args["input_size"])*2//int(self.args["NFFT"])
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        self.input_size_model=[self.args["batch_size"],self.args["NFFT"],self.args["NFFT"],2]
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
        self.input_model=tf.placeholder(tf.float32, self.input_size_model, "inputs_G-netA")
        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):
            with tf.variable_scope("generator_2"):
                self.fake_B_image =generator(tf.reshape(self.input_model,self.input_size_model), reuse=False,chs=self.args["G_channel"],depth=self.args["depth"],f=self.args["filter_g"],s=self.args["strides_g"],type=self.args["architect"],rate=1.0)
        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_1")
        #saver
        #保存の準備
        self.saver = tf.train.Saver()



    def convert(self,in_put):
        #function of test
        #To convert wave file
        #テスト用関数
        #wave file　変換用

        tt=time.time()
        ipt=self.args["input_size"]+self.args["NFFT"]
        times=in_put.shape[1]//(self.args["input_size"])+1
        if in_put.shape[1]%((self.args["input_size"])*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        otp2=np.asarray([],dtype=np.int16)
        otp3= np.asarray([[[]]], dtype=np.float32)
        rss=np.zeros([self.input_size_model[2]//2])
        rss4=np.zeros([self.input_size_model[2]//2])
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

            # changing pitch
            # ピッチ変更
            for i in range(self.args["batch_size"]):
                red[i] = shift(red[i] / 32767.0, self.args["pitch_rate"]).reshape(red[i].shape)
            # FFT
            # 短時間高速離散フーリエ変換
            for i in range(self.args["batch_size"]):
                n=self.fft(red[i].reshape(-1))
                res[i]=n
            scales = np.sqrt(np.var(res[0, :, :, 0], axis=1) + 1e-8)
            means = np.mean(res[0, :, :, 0], axis=1)
            mms = 1/scales
            scl = np.tile(np.reshape(means, (-1, 1)), (1, self.args["NFFT"]))
            res[0, :, :, 0] = np.einsum("ij,i->ij",res[0, :, :, 0]- scl,mms)

            # running network
            # ネットワーク実行

            res2=self.sess.run(self.fake_B_image,feed_dict={ self.input_model:res})
            # otp3 = np.append(otp3, res2[0,2:, :, :])

            # resas = np.append(resas, res[0])
            # Postprocess
            # 後処理
            a=res2[0].copy()
            scales2=np.sqrt(np.var(a[:,:,0],axis=1)+1e-8)
            means2 = np.mean(a[:, :, 0], axis=1)
            ss=np.tile((scales/(scales2+1e-32)).reshape(-1,1),(1,self.args["NFFT"]))
            sm=np.tile((means-means2).reshape(-1,1),(1,self.args["NFFT"]))
            c=a[:,:,0]

            c=c*ss
            c = c + sm
            a[:,:,0]=c

            # print([np.mean(scales-np.sqrt(np.var(c, axis=1) + 1e-8)),np.mean(means-np.mean(c, axis=1))])
            b=a.copy()

            b = mask_scale(b, 250, 770, -10)
            b = mask_scale(b, 0,250, -20)
            b = mask_scale(b, 770, 1024, -50)
            b = mask_const(b, 250, 770, 5)

            # b = mask_scale(b, 0, 1024, 10)

            scales3 = np.sqrt(np.var(b[:, :, 0], axis=1) + 1e-32)
            means3 = np.mean(b[:, :, 0], axis=1)
            scales_mask=scales.copy()
            means_mask = means.copy()
            means_mask[means_mask < -2.5] = -30.0
            ss2 = scales_mask / (scales3 + 1e-32)
            sm2=np.tile((means_mask-means3).reshape(-1,1),(1,self.args["NFFT"]))
            b[:,:,0]=np.einsum("ij,i->ij",b[:,:,0]+sm2,ss2)

            b = np.clip(b, -60.0, 9.0)

            otp3 = np.append(otp3, a[ 1:, :, :])

            # otp3 = np.append(otp3, a[ 2:, :, :])

            # IFFT
            # 短時間高速離散逆フーリエ変換
            res4, rss4 = self.ifft(b, rss4)
            res4 = np.clip(res4, -1.0, 1.0)
            res4 = res4 * 32767
            # chaching results
            # 結果の保存
            res4 = res4.reshape(-1).astype(np.int16)
            # print(res2.shape)
            otp2 = np.append(otp2, res4[-8192:])
        h = otp2.shape[0] - in_put.shape[1]
        if h > 0:
            otp2 = otp2[h:]
        return otp2.reshape(1,in_put.shape[1],in_put.shape[2])

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
    def ifft(self,data,red):
        a=np.clip(data[:, :, 0],a_min=-100000,a_max=88)
        sss=np.exp(a)
        p = np.sqrt(sss)
        r = p * (np.cos(data[:, :, 1]))
        i = p * (np.sin(data[:, :, 1]))
        dds = np.concatenate((r.reshape(r.shape[0], r.shape[1], 1), i.reshape(i.shape[0], i.shape[1], 1)), 2)
        datanum=dds[:,:,0]+1j*dds[:,:,1]
        window=np.hamming(self.args["NFFT"])
        fft_s = np.fft.ifft(datanum,n=self.args["NFFT"], axis=1)
        fft_data = fft_s.real
        # fft_data[:]/=window
        v = fft_data[:, :self.args["NFFT"]// 2]
        reds = fft_data[-1, self.args["NFFT"] // 2:].copy()
        lats = np.roll(fft_data[:, self.args["NFFT"] // 2:], 1,axis=0)
        lats[0, :] =red
        spec = np.reshape(v + lats, (-1))
        return spec,reds
def generator(current_outputs,reuse,depth,chs,f,s,rate,type):
    if type == "flatnet":
        return generator_flatnet(current_outputs,reuse,depth,chs,f,s,0)
    elif type == "ps_flatnet":
        return generator_flatnet(current_outputs, reuse, depth, chs, f, s, 1)
    elif type == "hybrid_flatnet":
        return generator_flatnet(current_outputs, reuse, depth, chs, f, s, 2)
    elif type == "ps_unet":
        return generator_unet(current_outputs, reuse, depth, chs, f, s, True)
    else :
        return  generator_unet(current_outputs,reuse,depth,chs,f,s)
def generator_flatnet(current_outputs,reuse,depth,chs,f,s,ps):
    current=current_outputs
    output_shape=int(current.shape[3])
    #main process
    for i in range(depth):
        connections = current
        fs=[2**(i//4+1),2**(i//4+1)]

        if ps==1:
            ten = block2(current, output_shape, fs, i, reuse,i!=depth-1)
        elif ps==2 :
            ten=block3(current,chs,fs,i,reuse,i!=depth-1)
        else :
            ten = block(current, output_shape, chs, f, s, i, reuse, i != depth - 1)
        current = ten + connections
    return current
def block(current,output_shape,chs,f,s,depth,reuses,relu):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))

    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=s, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv11"+str(depth))
    ten = tf.layers.batch_normalization(ten, axis=3, training=False, trainable=False,
                                        gamma_initializer=tf.ones_initializer(), reuse=reuses, name="bn11" + str(depth))

    ten = tf.nn.leaky_relu(ten,name="lrelu"+str(depth))

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten = tf.layers.conv2d_transpose(ten, output_shape, kernel_size=f, strides=s, padding="VALID",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                                     data_format="channels_last",reuse=reuses,name="deconv11"+str(depth))
    if relu:
        ten=tf.nn.relu(ten)
        ten = tf.layers.batch_normalization(ten, axis=3, training=False, trainable=False,
                                            gamma_initializer=tf.ones_initializer(), reuse=reuses,
                                            name="bn12" + str(depth))
    return ten
def block2(current,output_shape,f,depth,reuses,relu):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    chs_r=f[0]*f[1]*output_shape
    ten = tf.layers.conv2d(ten, chs_r, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv21"+str(depth))

    ten = tf.layers.batch_normalization(ten, axis=3, training=False, trainable=False,
                                        gamma_initializer=tf.ones_initializer(), reuse=reuses, name="bn21" + str(depth))

    ten = tf.nn.leaky_relu(ten,name="lrelu"+str(depth))
    ten=deconve_with_ps(ten,f[0],output_shape,depth,reuses=reuses)
    ten = tf.layers.batch_normalization(ten, axis=3, training=False, trainable=False,
                                        gamma_initializer=tf.ones_initializer(), reuse=reuses, name="bn22" + str(depth))

    if relu:
        ten=tf.nn.relu(ten)
    return ten
def block3(current,chs,f,depth,reuses,relu):
    ten=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv21"+str(depth))

    ten = tf.layers.batch_normalization(ten, axis=3, training=True, trainable=True,
                                        gamma_initializer=tf.ones_initializer(), reuse=reuses, name="bn21" + str(depth))

    ten = tf.nn.leaky_relu(ten,name="lrelu"+str(depth))
    ten1=deconve_with_ps(ten[:,:,:,1:],f[0],1,depth,reuses=reuses)
    ten2 =  tf.layers.conv2d_transpose(ten[:,:,:,:1], 1, kernel_size=f, strides=f, padding="VALID",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                                     data_format="channels_last",reuse=reuses,name="deconv11"+str(depth))
    ten=tf.concat([ten2,ten1],axis=3)
    if relu:
        ten=tf.nn.relu(ten)
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

def generator_unet(current_outputs,reuse,depth,chs,f,s,ps=False):
    current=current_outputs
    connections=[ ]
    for i in range(depth):
        connections.append(current)
        current = down_layer(current, chs*(i+1) ,f,s,reuse,i)
    print("shape of structure:"+str([[int(c.shape[0]),int(c.shape[1]),int(c.shape[2]),int(c.shape[3])] for c in connections]))
    for i in range(depth):
        current=up_layer(current,chs*(depth-i-1) if (depth-i-1)!=0 else 2,f,s,i,i!=(depth-1),depth-i-1>2,reuse,ps)
        if i!=depth-1:
            current += connections[depth - i -1]
    return tf.reshape(current,current_outputs.shape)

def up_layer(current,output_shape,f,s,depth,bn=True,do=False,reuse=None,ps=False):
    ten=tf.nn.leaky_relu(current)
    if ps:
        ten = deconve_with_ps(ten, f[0], output_shape, depth, reuses=reuse)
    else:
        ten=tf.layers.conv2d_transpose(ten, output_shape,kernel_size=f ,strides=s, padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last",name="deconv"+str(depth),reuse=reuse)
    if bn:
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2),name="bn_u"+str(depth),reuse=reuse)
    return ten
def down_layer(current,output_shape,f,s,reuse,depth):
    ten=tf.layers.batch_normalization(current,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2),name="bn_d"+str(depth),reuse=reuse)
    ten=tf.layers.conv2d(ten, output_shape,kernel_size=f ,strides=s, padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last",name="conv"+str(depth),reuse=reuse)
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