import tensorflow as tf
import os
import numpy as np
import wave
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
        self.args["weight_Cycle1"]=1.0
        self.args["weight_Cycle2"] = 1.0
        self.args["weight_GAN1"] = 1.0
        self.args["weight_GAN2"] = 1.0
        self.args["NFFT"]=128
        self.args["dilation_size"] = 7
        self.args["debug"] = False
        self.args["noise"] = False
        self.args["cupy"] = False
        self.args["D_channels"] =[2]
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
        self.args["SHIFT"] = self.args["NFFT"]//2
        ss=int(self.args["input_size"])//int(self.args["SHIFT"])
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        self.input_size_model=[2,ss+self.args["dilation_size"],self.args["NFFT"]//2,2]
        print("model input size:"+str(self.input_size_model))
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))

        if  self.args["wave_otp_dir"] is not "False" :
            self.args["wave_otp_dir"]=self.args["wave_otp_dir"]+ self.args["name_save"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])
            shutil.copy(path,self.args["wave_otp_dir"]+"info.json")
            self.args["log_file"]=self.args["wave_otp_dir"]+"log.txt"

        self.checkpoint_dir = self.args["checkpoint_dir"]

    def build_model(self):


        #入力
        self.input_model = tf.placeholder(tf.float32, self.input_size_model, "input")
        self.input_model_t = tf.placeholder(tf.float32, self.input_size_model, "input_b")#creating generator
        with tf.variable_scope("seed_net"):
            self.seed_TB,_ = seed_net(self.input_model_t[:,-8:,:,:], None,self.args["d_depth"],self.args["D_channels"], " ",train=False)
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):
            self.fake_image = generator(self.input_model, self.seed_TB, reuse=None,
                                                chs=self.args["G_channels"], depth=self.args["depth"],
                                                f=self.args["filter_g"],
                                                s=self.args["strides_g"],
                                                d=self.args["dilations"],
                                                type=self.args["architect"],
                                                train=False)
        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")

        #saver
        #保存の準備
        self.saver = tf.train.Saver()



    def convert(self,in_put,target):
        #テスト用関数
        #wave file　変換用
        ipt=self.args["input_size"]+self.args["SHIFT"]+self.args["SHIFT"]*self.args["dilation_size"]
        targ = target[0:ipt] / 32767.0
        tar = self.fft(targ)[:, :self.args["SHIFT"]].reshape(1, self.input_size_model[1], self.input_size_model[2],
                                                             self.input_size_model[3])
        tar =np.repeat(tar,2,axis=0)
        times=in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        otp3= np.asarray([[[]]], dtype=np.float32)
        rss=np.zeros([self.args["SHIFT"]])
        for t in range(times):
            # 前処理

            # サイズ合わせ
            start_pos=self.args["input_size"]*t+(in_put.shape[0]%self.args["input_size"])
            red=in_put[max(0,start_pos-ipt):start_pos].astype(np.float32)
            r=max(0,ipt-red.shape[0])
            if r>0:
                red=np.pad(red,(r,0),'constant')
            red=red.reshape(ipt)
            res = np.zeros(self.input_size_model)
            # 短時間高速離散フーリエ変換
            n=self.fft(red)
            res[0]=n[:,:self.args["SHIFT"],:]
            red2=np.pad(red,(self.args["SHIFT"],0),"constant")[:-self.args["SHIFT"]]
            n = self.fft(red2)
            res[1] = n[:, :self.args["SHIFT"], :]

            # ネットワーク実行
            res2=self.sess.run(self.fake_image,feed_dict={ self.input_model:res,self.input_model_t:tar})
            res3 = res2.copy()[:, :, ::-1, :]
            res2= np.append(res2,res3, axis=2)
            res2[:,:,self.args["SHIFT"]:,1]*=-1

            # 後処理
            a=res2[0].copy()
            otp3 = np.append(otp3, a[:, :, :].copy())

            # 短時間高速離散逆フーリエ変換
            res2,rss=self.ifft(a,rss)
            res2=np.clip(res2,-1.0,1.0)
            res2=res2*32767

            # 結果の保存
            res2=res2.reshape(-1).astype(np.int16)
            otp=np.append(otp,res2)
        h=otp.shape[0]-in_put.shape[0]
        if h>0:
            otp=otp[h:]
        return otp,otp3

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
        v = fft_data[:, :self.args["NFFT"]// 2]
        reds = fft_data[-1, self.args["NFFT"] // 2:].copy()
        lats = np.roll(fft_data[:, self.args["NFFT"] // 2:], 1, axis=0 )
        lats[0, :]=redi
        spec = np.reshape(v + lats, (-1))
        return spec,reds


def seed_net(inp,reuse,depth,chs,a,train=True):
    stddevs = math.sqrt(2.0 / (int(inp.shape[1]) * int(inp.shape[3])))
    current = tf.layers.conv2d(inp, 16, kernel_size=[inp.shape[1],1], strides=[1,1], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                               data_format="channels_last", name="seed_t", reuse=reuse)
    current = tf.layers.batch_normalization(current, name="bn_seed00", training=train, reuse=reuse)
    current = tf.nn.leaky_relu(current)
    for i in range(depth):
        stddevs = math.sqrt(2.0 / 4.0*chs[i])
        current = tf.layers.batch_normalization(current, name="bn_seed" + str(i), training=train, reuse=reuse)
        current=tf.layers.conv2d(current, chs[i], kernel_size=[1,7], strides=[1,4], padding="VALID",use_bias=False,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                               data_format="channels_last", name="conv_seed" + str(i), reuse=reuse)
        current = tf.nn.leaky_relu(current)
    current=tf.reshape(current,[-1,1,chs[depth-1],1])
    sig=current
    if a is "A" :
        print(" [*] bottom shape:"+str(current.shape))
    return sig,current
def generator(current,seed,reuse,depth,chs,f,s,d,type,train):
    if type == "hybrid_decay_flatnet":
        return generator_flatnet_decay(current, seed,reuse, depth, chs, f, s, d, 2, train)
    elif type == "mix_decay_flatnet":
        return generator_flatnet_decay(current, seed,reuse, depth, chs, f, s, d, 4, train)
    else :
        return  generator_unet(current,reuse,depth,chs,f,s)
def generator_flatnet_decay(c,seed,reuse,depth,chs,f,s,d,ps,train):
    #main process
    current=c
    for i in range(depth):
        connections = current
        if ps == 2:
            ten = block_hybrid(current,seed, chs[i], chs[i * 2], f[i], s[i], i, reuse, i != depth - 1, pixs=f[i],train=train)
        else:
            ten = block_mix(current,seed,chs[i], i, reuse,train=train)
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
    ten = inp
    ten2 = inp
    stddevs = math.sqrt(2.0 / (2 * 1 * int(ten.shape[3])))
    for i in range(len(d)):
        ten = tf.layers.conv2d(ten, chs[i+startd], kernel_size=[2, 1], strides=[1, 1], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                           data_format="channels_last", reuse=reuse, name="conv_p" + str(startd+i), dilation_rate=(d[i], 1))

        if i!=len(d)-1:
            ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="bn_p" + str(startd+i))

            ten = tf.nn.leaky_relu(ten)

        ten2 = tf.layers.conv2d(ten2, chs[i + startd], kernel_size=[2, 1], strides=[1, 1], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                               data_format="channels_last", reuse=reuse, name="conv_f" + str(startd+ i),
                               dilation_rate=(d[i], 1))
        if i!=len(d)-1:
            ten2 = tf.layers.batch_normalization(ten2, axis=3, training=train, trainable=True, reuse=reuse,
                                            name="bn_f" + str(startd+ i))
            ten2 = tf.nn.leaky_relu(ten2)
    current=tf.concat([ten,ten2],axis=3)
    return current
def block_hybrid(current,seed,output_shape,chs,f,s,depth,reuses,shake,pixs=[2,2],train=True):
    ten = current
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    tenA = tf.layers.conv2d(ten, chs, kernel_size=f, strides=s, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                           data_format="channels_last", reuse=reuses, name="conv11" + str(depth), dilation_rate=(1, 1))
    tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bn11" + str(depth))

    tenA = tf.nn.leaky_relu(tenA)

    tenA = tf.layers.conv2d_transpose(tenA,output_shape,f,s,padding="VALID",kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),data_format="channels_last", reuse=reuses, name="deconv" + str(depth))
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(ten.shape[3])))
    tenB = tf.layers.conv2d(ten, chs, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                           data_format="channels_last", reuse=reuses, name="conv12" + str(depth))
    tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bn11" + str(depth))

    tenB = tf.nn.leaky_relu(tenB)

    tenB = deconve_with_ps(tenB, pixs, output_shape, depth, reuses=reuses, name="01")
    if shake:
        ten = tf.nn.leaky_relu(tenB + tenA)
    else:
        ten=tenB + tenA
    return ten
def block_mix(current,seed,chs,depth,reuses,train=True):
    ten = current
    tenM=[]
    times=4
    res=3

    for i in range(times-1):
        stddevs = math.sqrt(2.0 / ( 4 * int(ten.shape[3])))
        ten = tf.layers.conv2d(ten, chs//(2**(times-i-1)), kernel_size=[1, 2], strides=[1, 2], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                               data_format="channels_last", reuse=reuses, name="conv"+str(i) + str(depth),
                               dilation_rate=(1, 1))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bn"+str(i) + str(depth))
        ten = tf.nn.leaky_relu(ten)
        tenM.append(ten)
    stddevs = math.sqrt(2.0 / (4 * int(ten.shape[3])))
    ten = tf.layers.conv2d(ten, chs, kernel_size=[1, 2], strides=[1, 2], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                           data_format="channels_last", reuse=reuses, name="conv"+str(times) + str(depth),
                           dilation_rate=(1, 1))
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bn"+str(times) + str(depth))
    ten = tf.nn.leaky_relu(ten)

    tenS = tf.layers.conv2d(seed, chs, kernel_size=[1, 2], strides=[1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                            data_format="channels_last", reuse=reuses, name="convS1" + str(depth))
    tenS = tf.layers.batch_normalization(tenS, axis=3, training=train, trainable=True, reuse=reuses,
                                         name="bnS1" + str(depth))

    tenS = tf.nn.leaky_relu(tenS)

    ten=ten+tenS
    for i in range(res):
        stddevs = math.sqrt(2.0 / (7 * int(ten.shape[3])))
        tenA=ten
        ten = tf.layers.conv2d(ten, chs, [1,7], [1,1], padding="SAME",
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                                          data_format="channels_last", reuse=reuses, name="res_conv"+str(i) + str(depth))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                             name="bn"+str(times+i+1) + str(depth))
        ten = tf.nn.leaky_relu(ten)

        ten=ten+tenA

    ten = deconve_with_ps(ten, [1, 2], chs//2, depth, reuses=reuses, name="00")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                         name="bn"+str(times+res+1) + str(depth))
    ten = tf.nn.leaky_relu(ten)
    for i in range(times-1):
        ten+=tenM[times-i-2]
        ten = deconve_with_ps(ten, [1, 2], chs//(2**(i+2)), depth, reuses=reuses, name=str(i+1))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bn"+str(i+2+times+res) + str(depth))
        ten=tf.nn.leaky_relu(ten)
    return ten
def deconve_with_ps(inp,r,otp_shape,depth,f=[1,1],reuses=None,name=""):
    chs_r=r[0]*r[1]*otp_shape
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(inp.shape[3])))
    ten = tf.layers.conv2d(inp, chs_r, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
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