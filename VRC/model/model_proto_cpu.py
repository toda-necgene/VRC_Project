import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import json
import shutil
from .model_proto import generator
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
        self.args["repeatations"]=1
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
                                                d=self.args["dilations"],r=self.args["repeatations"], train=False)


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
        rss2 = np.zeros([self.input_size_model[2]], dtype=np.float64)
        rssC = np.zeros([self.input_size_model[2]], dtype=np.float64)
        rssD = np.zeros([self.input_size_model[2]], dtype=np.float64)

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
            res=self.fft(resorce.copy()/32767.0)
            resorce2=np.roll(resorce.copy(),self.args["SHIFT"]//2,axis=0)
            resorce2[:self.args["SHIFT"]//2]=0.0
            resorce3 = np.roll(resorce.copy(), self.args["SHIFT"] // 2+self.args["SHIFT"], axis=0)
            resorce3[:self.args["SHIFT"] // 2+self.args["SHIFT"]] = 0.0
            resorce4 = np.roll(resorce.copy(), self.args["SHIFT"], axis=0)
            resorce4[:self.args["SHIFT"] ] = 0.0

            res2 = self.fft(resorce2.reshape(-1) / 32767.0)
            res3C=self.fft(resorce3.reshape(-1) / 32767.0)
            res4D=self.fft(resorce4.reshape(-1) / 32767.0)
            # running network
            # ネットワーク実行
            res=np.asarray([res[:,:self.args["SHIFT"],:],res2[:,:self.args["SHIFT"],:],res3C[:,:self.args["SHIFT"],:],res4D[:,:self.args["SHIFT"],:]]).reshape([4,-1,self.args["SHIFT"],2])
            # res =res[:, :self.args["SHIFT"], :].reshape([1, -1, self.args["SHIFT"], 2])

            response=self.sess.run(self.fake_aB_image_test,feed_dict={ self.input_model_test:res})
            res2=response[0].copy()[:,::-1,:]
            res=np.append(response[0].copy(),res2,axis=1)
            res[:,self.args["SHIFT"]:,1]*=-1
            resB = response[1].copy()[ :, ::-1, :]
            resB = np.append(response[1].copy(), resB, axis=1)
            resB[:, self.args["SHIFT"]:, 1] *= -1
            resC = response[2].copy()[:, ::-1, :]
            resC = np.append(response[1].copy(), resC, axis=1)
            resC[:, self.args["SHIFT"]:, 1] *= -1
            resD = response[3].copy()[:, ::-1, :]
            resD = np.append(response[3].copy(), resD, axis=1)
            resD[:, self.args["SHIFT"]:, 1] *= -1
            a=np.clip(res.copy(),-60.0,12.0)
            b=np.clip(resB.copy(),-60.0,12.0)
            c=np.clip(resC.copy(),-60.0,12.0)
            d=np.clip(resD.copy(),-60.0,12.0)
            res3 = np.append(res3, a).reshape(-1,self.args["NFFT"],2)


            # Postprocess
            # 後処理

            # IFFT
            # 短時間高速離散逆フーリエ変換
            res,rss=self.ifft(a,rss)
            resB2, rss2 = self.ifft(b, rss2)
            resC3, rssC = self.ifft(c, rssC)
            resD4, rssD = self.ifft(d, rssD)

            # 変換後処理
            resB2=np.roll(resB2,-self.args["SHIFT"]//2,axis=0)
            resB2[-self.args["SHIFT"]//2:]=0.0
            resC3 = np.roll(resC3, -self.args["SHIFT"] // 2-self.args["SHIFT"], axis=0)
            resC3[-self.args["SHIFT"] // 2-self.args["SHIFT"]:] = 0.0
            resD4 = np.roll(resD4, -self.args["SHIFT"], axis=0)
            resD4[-self.args["SHIFT"]:] = 0.0

            res=(res+resB2+resC3+resD4)/4
            res = np.clip(res, -1.0, 1.0)*32767

            # chaching results
            # 結果の保存
            res=res.reshape(-1).astype(np.int16)
            otp=np.append(otp,res[-self.args["input_size"]:])
        h=otp.shape[0]-in_put.shape[0]
        if h>0:
            otp=otp[h:]

        return otp.reshape(-1),time.time()-tt,res3[1:]

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
        # window=np.hamming(self.args["NFFT"])
        # window=np.clip(window,0.5,1.0)
        # fft_data[:]/=window
        v = fft_data[:, :self.args["NFFT"]// 2].copy()
        reds = fft_data[-1, self.args["NFFT"] // 2:].copy()
        lats = np.roll(fft_data[:, self.args["NFFT"] // 2:].copy(), 1, axis=0 )
        lats[0, :]=redi
        spec = np.reshape(v + lats, (-1))
        return spec,reds



