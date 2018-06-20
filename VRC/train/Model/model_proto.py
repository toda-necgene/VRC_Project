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
from hyperdash import Experiment
from datetime import datetime
import json
import shutil
import cupy
import dropbox
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
                self.args['gepth']*2))
            self.args["D_channels"] = [min([2 ** (i + 1) - 2, 254]) for i in range(self.args['depth']*2)]
        self.args["SHIFT"] = self.args["NFFT"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        ss=self.args["input_size"]//self.args["SHIFT"]
        self.input_size_model=[None,ss,self.args["NFFT"]//2,2]
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
        if self.args["dropbox"]!="False":
            self.dbx=dropbox.Dropbox(self.args["dropbox"])
            self.dbx.users_get_current_account()

        else:
            self.dbx=None
        self.checkpoint_dir = self.args["checkpoint_dir"]

    def build_model(self):

        #inputs place holder
        #入力
        self.input_modela=tf.placeholder(tf.float32, self.input_size_model, "input_A")
        self.input_modelb = tf.placeholder(tf.float32, self.input_size_model, "input_B")
        self.label_modela = tf.placeholder(tf.float32, self.input_size_label, "label_A")
        self.label_modelb = tf.placeholder(tf.float32, self.input_size_label, "label_B")

        self.training=tf.placeholder(tf.float32,[1],name="Training")
        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1",reuse=tf.AUTO_REUSE):
                self.fake_aB_image = generator(self.input_modela, reuse=None,chs=self.args["G_channel"], depth=self.args["depth"], f=self.args["filter_g"],
                                              s=self.args["strides_g"], chs2=self.args["G_channels"],type=self.args["architect"],train=True,name="1")
                self.fake_aB_image_test = generator(self.input_modelb, reuse=True,
                                                chs=self.args["G_channel"], depth=self.args["depth"],
                                                f=self.args["filter_g"],
                                                s=self.args["strides_g"],
                                                type=self.args["architect"],
                                                train=False,name="1", chs2=self.args["G_channels"])

            with tf.variable_scope("generator_2",reuse=tf.AUTO_REUSE):
                self.fake_bA_image = generator(self.input_modelb, reuse=None,
                                              chs=self.args["G_channel"], depth=self.args["depth"], f=self.args["filter_g"],
                                              s=self.args["strides_g"], chs2=self.args["G_channels"],type=self.args["architect"],train=True,name="2")

            with tf.variable_scope("generator_2",reuse=tf.AUTO_REUSE):
                self.fake_Ba_image = generator(self.fake_aB_image, reuse=True,
                                              chs=self.args["G_channel"], depth=self.args["depth"], f=self.args["filter_g"],
                                              s=self.args["strides_g"], chs2=self.args["G_channels"],type=self.args["architect"],train=True,name="2")
            with tf.variable_scope("generator_1",reuse=tf.AUTO_REUSE):
                self.fake_Ab_image = generator(self.fake_bA_image, reuse=True,
                                               chs=self.args["G_channel"], depth=self.args["depth"],
                                               f=self.args["filter_g"],
                                               s=self.args["strides_g"], chs2=self.args["G_channels"],type=self.args["architect"],train=True,name="1")
        self.noise = tf.placeholder(tf.float32, [self.args["batch_size"]], "inputs_Noise")

        ss=self.input_modela.shape[1:]
        a_true_noised=self.input_modela+tf.random_normal(ss,0,self.noise[0])
        b_true_noised = self.input_modelb + tf.random_normal(ss, 0, self.noise[0])

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

                self.d_judge_BF = discriminator(self.fake_aB_image[:,:,:,:], True, self.args["filter_d"],
                                                                       self.args["strides_d"], self.args["d_depth"],
                                                                       self.args["D_channels"],a="B")
            with tf.variable_scope("discrimA"):
                self.d_judge_AR = discriminator(a_true_noised[:,:,:,:], None, self.args["filter_d"],
                                                                        self.args["strides_d"], self.args["d_depth"],
                                                                        self.args["D_channels"],"A")
                self.d_judge_AF = discriminator(self.fake_bA_image[:,:,:,:], True, self.args["filter_d"],
                                                                        self.args["strides_d"], self.args["d_depth"],
                                                                        self.args["D_channels"],a="A")
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

        self.d_loss_AR = tf.reduce_mean(tf.squared_difference(self.d_judge_AR, tf.ones_like(self.d_judge_AR)))
        self.d_loss_AF = tf.reduce_mean(tf.squared_difference(self.d_judge_AF, tf.zeros_like(self.d_judge_AF)))
        self.d_loss_BR = tf.reduce_mean(tf.squared_difference(self.d_judge_BR, tf.ones_like(self.d_judge_BR)))
        self.d_loss_BF = tf.reduce_mean(tf.squared_difference(self.d_judge_BF, tf.zeros_like(self.d_judge_BF)))

        # self.d_loss_AR0 = tf.reduce_mean(tf.squared_difference(self.d_judge_AR0, tf.ones_like(self.d_judge_AR0)))
        # self.d_loss_AF0 = tf.reduce_mean(tf.squared_difference(self.d_judge_AF0, tf.zeros_like(self.d_judge_AF0)))
        # self.d_loss_BR0 = tf.reduce_mean(tf.squared_difference(self.d_judge_BR0, tf.ones_like(self.d_judge_BR0)))
        # self.d_loss_BF0 = tf.reduce_mean(tf.squared_difference(self.d_judge_BF0, tf.zeros_like(self.d_judge_BF0)))

        # self.d_lossA=(self.d_loss_AR+self.d_loss_AF+self.d_loss_AR0+self.d_loss_AF0)/2.0
        # self.d_lossB= (self.d_loss_BR + self.d_loss_BF+self.d_loss_BR0 + self.d_loss_BF0)/2.0
        self.d_lossA = (self.d_loss_AR + self.d_loss_AF )
        self.d_lossB = (self.d_loss_BR + self.d_loss_BF )

        self.d_loss=self.d_lossA+self.d_lossB
            # objective-functions of generator
        # G-netの目的関数

        # L1 norm lossA
        saa=tf.abs(self.fake_Ba_image[:,:,:,0]-self.input_modela[:,:,:,0])
        sbb=tf.abs(self.fake_Ba_image[:,:,:,1]-self.input_modela[:,:,:,1])
        L1B=saa+sbb

        # Gan lossA
        # DSb=tf.reduce_mean(-tf.log(self.d_judge_BF+1e-32))
        # DSA1 = tf.reduce_mean(tf.squared_difference(self.d_judge_AF0, tf.ones_like(self.d_judge_AF0)))
        DSA2 = tf.reduce_mean(tf.squared_difference(self.d_judge_AF, tf.ones_like(self.d_judge_AF)))
        # DSB1 = tf.reduce_mean(tf.squared_difference(self.d_judge_BF0, tf.ones_like(self.d_judge_BF0)))
        DSB2 = tf.reduce_mean(tf.squared_difference(self.d_judge_BF, tf.ones_like(self.d_judge_BF)))

        # generator lossA
        # self.g_loss_aB = L1B * self.args["weight_Cycle"]+tf.reduce_mean(self.args["weight_GAN1"]*DSB1+self.args["weight_GAN2"]*DSB2)
        self.g_loss_aB = L1B * self.args["weight_Cycle"] + tf.reduce_mean(self.args["weight_GAN2"] * DSB2)

        # self.g_loss_aB =  tf.reduce_mean(self.args["weight_GAN"] * DSb)

        # L1 norm lossB
        sa=tf.abs(self.fake_Ab_image[:,:,:,0]-self.input_modelb[:,:,:,0] )
        sb=tf.abs(self.fake_Ab_image[:,:,:,1]-self.input_modelb[:,:,:,1] )
        L1bAAb = sa+sb
        # Gan loss
        # DSA = tf.reduce_mean(-tf.log(self.d_judge_AF+1e-32))

        # L1UBA =16.0/(tf.abs(self.fake_bA_image[:,:,:,0]-self.fake_aB_image[:,:,:,0])+1e-8)
        # L1UBA =tf.maximum(L1UBA,tf.ones_like(L1UBA))
        # generator loss
        # self.g_loss_bA = L1bAAb * self.args["weight_Cycle"] + tf.reduce_mean( self.args["weight_GAN1"]*DSA1+self.args["weight_GAN2"]*DSA2)
        self.g_loss_bA = L1bAAb * self.args["weight_Cycle"] + tf.reduce_mean( self.args["weight_GAN2"] * DSA2)

        # self.g_loss_bA =  tf.reduce_mean( self.args["weight_GAN"]*DSA)

        self.g_loss=self.g_loss_aB+self.g_loss_bA
        #BN_UPDATE
        self.update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #tensorboard functions
        #tensorboard 表示用関数
        self.g_loss_all= tf.summary.scalar("g_loss_cycle_A", tf.reduce_mean(L1bAAb),family="g_loss")
        # self.g_loss_gan = tf.summary.scalar("g_loss_gan_A_0", tf.reduce_mean(DSA1),family="g_loss")
        self.g_loss_gan2 = tf.summary.scalar("g_loss_gan_A_1", tf.reduce_mean(DSA2), family="g_loss")
        self.dscore = tf.summary.scalar("dscore_AF", tf.reduce_mean(self.d_judge_AF),family="d_score")
        self.dscore3 = tf.summary.scalar("dscore_AR", tf.reduce_mean(self.d_judge_AR), family="d_score")
        # self.g_loss_sum_1 = tf.summary.merge([self.g_loss_all, self.g_loss_gan, self.g_loss_gan2, self.dscore])
        self.g_loss_sum_1 = tf.summary.merge([self.g_loss_all, self.g_loss_gan2, self.dscore,self.dscore3])

        self.g_loss_all2 = tf.summary.scalar("g_loss_cycle_B", tf.reduce_mean(L1B),family="g_loss")
        # self.g_loss_gan3 = tf.summary.scalar("g_loss_gan_B_0", tf.reduce_mean(DSB1),family="g_loss")
        self.g_loss_gan4 = tf.summary.scalar("g_loss_gan_B_1", tf.reduce_mean(DSB2), family="g_loss")
        self.dscore2 = tf.summary.scalar("dscore_BF", tf.reduce_mean(self.d_judge_BF),family="d_score")
        self.dscore4 = tf.summary.scalar("dscore_BR", tf.reduce_mean(self.d_judge_BR), family="d_score")

        # self.g_loss_uba = tf.summary.scalar("g_loss_distAB", tf.reduce_mean(L1UBA), family="g_loss")
        # self.g_loss_sum_2 = tf.summary.merge([self.g_loss_all2, self.g_loss_gan3, self.g_loss_gan4, self.dscore2])
        self.g_loss_sum_2 = tf.summary.merge([self.g_loss_all2, self.g_loss_gan4, self.dscore2,self.dscore4])

        self.d_loss_sumA = tf.summary.scalar("d_lossA", tf.reduce_mean(self.d_lossA),family="d_loss")
        self.d_loss_sumB = tf.summary.scalar("d_lossB", tf.reduce_mean(self.d_lossB),family="d_loss")

        self.result=tf.placeholder(tf.float32, [1,1,160000], name="FB")
        self.result1 = tf.placeholder(tf.float32, [1,2560,128,2], name="FBI0")
        im1=tf.transpose(self.result1[:,:,:,:1],[0,2,1,3])
        im2 = tf.transpose(self.result1[:, :, :, 1:], [0, 2, 1, 3])
        self.fake_B_sum = tf.summary.audio("fake_B", tf.reshape(self.result,[1,160000,1]), 16000, 1)
        self.fake_B_sum2 = tf.summary.image("fake_B_image01", im1, 1)
        self.fake_B_sum3 = tf.summary.image("fake_B_image02", im2, 1)
        self.g_test_epo=tf.placeholder(tf.float32,name="g_test_epoch_end")
        self.g_test_epoch = tf.summary.merge([tf.summary.scalar("g_test_epoch_end", self.g_test_epo,family="test")])
        self.g_test_epo2 = tf.placeholder(tf.float32, name="g_test_epoch_end")
        self.g_test_epoch2 = tf.summary.merge([tf.summary.scalar("g_test_epoch_image", self.g_test_epo2, family="test")])

        self.tb_results=tf.summary.merge([self.fake_B_sum,self.fake_B_sum2,self.fake_B_sum3,self.g_test_epoch,self.g_test_epoch2])

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
        res3 = np.zeros([1,self.args["NFFT"],2], dtype=np.float32)
        rss=np.zeros([self.input_size_model[2]],dtype=np.float64)
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
            res = np.zeros([1,self.input_size_model[1],self.input_size_model[2],self.input_size_model[3]])
            # FFT
            # 短時間高速離散フーリエ変換
            n=self.fft(red[0].reshape(-1)/32767.0)
            res[0]=n[:,:self.args["SHIFT"]]
            means = np.mean(res[0,:,:,0], axis=1)
            means = np.tile(np.reshape(means, (-1, 1)), (1, self.args["SHIFT"]))
            res[0, :, :, 0] = res[0, :, :, 0] - means
            scales =np.reshape(np.sqrt(np.var(res[0,:,:,0], axis=1) + 1e-8), (-1))
            mms = 1 / scales
            res[0, :, :, 0] = np.einsum("ij,i->ij", res[0, :, :, 0], mms)
            # running network
            # ネットワーク実行
            res=res[:,:self.args["SHIFT"],:]
            res=self.sess.run(self.fake_aB_image_test,feed_dict={ self.input_modelb:res,self.training:np.asarray([1.0])})
            res2=res.copy()[:,:,::-1,:]
            res=np.append(res,res2,axis=2)
            res[:,:,self.args["SHIFT"]:,1]*=-1
            # resas = np.append(resas, res[0])
            a = res[0].copy()
            c = a[:, :, 0]
            scales_mask = scales.copy()
            means_mask = means.copy()
            c = np.einsum("ij,i->ij", c, scales_mask)
            sm = np.tile(means_mask, (1, 2))
            c = c + sm
            a[:, :, 0] = c
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



    def train(self):

        # setting paramaters
        # パラメータ
        tln=self.args["train_d_scale"]
        lr_g_opt=self.args["g_lr"]
        beta_g_opt=self.args["g_b1"]
        beta_2_g_opt=self.args["g_b2"]
        lr_d_opt=lr_g_opt*tln
        beta_d_opt=self.args["d_b1"]
        beta_2_d_opt=self.args["d_b2"]
        # naming output-directory
        # 出力ディレクトリ
        taken_times=np.empty([1])

        self.lod="[glr="+str(lr_g_opt)+",gb="+str(beta_g_opt)+",dlr="+str(lr_d_opt)+",db="+str(beta_d_opt)+"]"

        lr_g_opt3 = lr_g_opt*(0.1**(self.args["start_epoch"]//self.args["lr_decay_term"]))
        lr_d_opt3 = lr_g_opt*(0.1**(self.args["start_epoch"]//self.args["lr_decay_term"]))
        lr_g=tf.placeholder(tf.float32,None,name="g_lr")
        lr_d=tf.placeholder(tf.float32,None,name="d_lr")
        g_optim = tf.train.AdamOptimizer(lr_g, beta_g_opt, beta_2_g_opt).minimize(self.g_loss,
                                                                                       var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(lr_d, beta_d_opt, beta_2_d_opt).minimize(self.d_loss,
                                                                                       var_list=self.d_vars)

        time_of_epoch=np.zeros(1)

        # logging
        # ログ出力
        if self.args["tensorboard"]:
            self.writer = tf.summary.FileWriter("./logs/"+self.lod+self.args["name_save"], self.sess.graph)

        if self.args["wave_otp_dir"] != "False" and  os.path.exists(self.args["wave_otp_dir"]):
            with open(self.args["log_file"], "w") as f:
                if self.args["stop_argument"]:
                    f.write("epochs,d_score,%g_loss,test_varidation")
                else:
                    f.write("epochs,test_varidation")
                f.flush()
        # initialize training info
        # 学習の情報の初期化
        start_time = time.time()
        log_data_g = np.empty(0)
        log_data_d = np.empty(0)
        # loading net
        # 過去の学習データの読み込み
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print(" [*] reading data path :"+self.args["train_data_path"]+'/Source_data/')
        # loading training data directory
        # トレーニングデータの格納ディレクトリの読み込み
        data = glob(self.args["train_data_path"]+'/Source_data/*-wave.npy')
        datal = glob(self.args["train_data_path"] + '/Source_data/*-stri.npy')
        data2 = glob(self.args["train_data_path"] + '/Answer_data/*-wave.npy')
        data2l = glob(self.args["train_data_path"] + '/Answer_data/*-stri.npy')

        # loading test data
        # テストデータの読み込み
        test=isread('./Model/datasets/test/test.wav')[0:160000].astype(np.float32)
        label=isread('./Model/datasets/test/label.wav')[0:160000].astype(np.float32)
        label2 = imread('./Model/datasets/test/label2.npy')
        # times of one epoch
        # 回数計算
        train_data_num = min(len(data), self.args["train_data_num"])
        train_data_num2 = min(len(data2), self.args["train_data_num"])
        print(" [*] data found",len(data),len(data2))
        batch_idxs = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num2)]

        #　学習データをメモリに乗っける
        batch_files = data[:train_data_num]
        batch_files2 = data2[:train_data_num]
        batch_files_l = datal[:train_data_num]
        batch_files2_l = data2l[:train_data_num]

        batch_sounds_r = np.asarray([(imread(batch_file)) for batch_file in batch_files])
        batch_sounds_t = np.asarray([(imread(batch_file)) for batch_file in batch_files2])
        batch_label_r = np.asarray([(imread(batch_file)) for batch_file in batch_files_l])
        batch_label_t = np.asarray([(imread(batch_file)) for batch_file in batch_files2_l])

        # hyperdash
        if self.args["hyperdash"]:
            self.experiment=Experiment(self.args["name_save"]+"_G1")
            self.experiment.param("lr_g_opt", lr_g_opt)
            self.experiment.param("beta_g_opt", beta_g_opt)
            self.experiment.param("training_interval", self.args["train_interval"])
            self.experiment.param("learning_rate_scale", tln)
        ts=0.0

        for epoch in range(self.args["start_epoch"],self.args["train_epoch"]):
            # shuffling training data
            # トレーニングデータのシャッフル
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)
            test1=0.0
            ts = 0.0
            counter=0
            ti=batch_idxs
            if self.args["test"] and epoch%self.args["save_interval"]==0:
                print(" [*] Epoch %3d testing" % epoch)
                #testing
                #テスト
                out_puts,taken_time_test,im=self.convert(test.reshape(1,-1,1))
                im = im.reshape([-1, self.args["NFFT"], 2])
                otp_im=np.append(np.clip((im[:,:,0]+30)/40,0.0,1.0).reshape([1,-1,self.args["NFFT"],1]),np.clip((im[:,:,1]+3.15)/6.30,0.0,1.0).reshape([1,-1,self.args["NFFT"],1]),axis=3)
                out_put=out_puts.astype(np.float32)/32767.0
                # loss of tesing
                #テストの誤差
                test1=np.mean(np.abs(out_puts.reshape(1,-1,1)[0]-label.reshape(1,-1,1)[0]))
                test2 = np.mean(np.abs(im - label2[-im.shape[0]:]))

                #hyperdash
                if self.args["hyperdash"]:
                    self.experiment.metric("testG",test1)
                    self.experiment.metric("testGimage", test2)

                #writing epoch-result into tensorboard
                #tensorboardの書き込み
                if self.args["tensorboard"]:
                    rs=self.sess.run(self.tb_results,feed_dict={ self.result:out_put.reshape(1,1,-1),self.result1:otp_im,self.g_test_epo:test1,self.g_test_epo2:test2})
                    self.writer.add_summary(rs, epoch)

                #saving test result
                #テストの結果の保存
                if os.path.exists(self.args["wave_otp_dir"]):
                    plt.subplot(211)
                    ins=np.transpose(im[:,:,0],(1,0))
                    plt.imshow(ins,aspect="auto")
                    plt.clim(-30,10)
                    if epoch==self.args["start_epoch"]:
                        plt.colorbar()
                    plt.subplot(212)
                    ins = np.transpose(im[:, :, 1], (1, 0))
                    plt.imshow(ins, aspect="auto")
                    plt.clim(-3.141593, 3.141593)
                    if epoch == self.args["start_epoch"]:
                        plt.colorbar()
                    path=self.args["wave_otp_dir"]+nowtime()
                    plt.savefig(path+".png")
                    upload(out_puts,path)
                    if self.dbx is not None:
                        print(" [*] Files uploading")
                        with open(path+".png","rb") as ff:
                            self.dbx.files_upload(ff.read(),"/apps/tensorflow_watching_app/Latestimage.png",mode=dropbox.files.WriteMode.overwrite)
                        with open(path + ".wav", "rb") as ff:
                            self.dbx.files_upload(ff.read(), "/apps/tensorflow_watching_app/Latestwave.wav",mode=dropbox.files.WriteMode.overwrite)
                        print(" [*] Files uploaded!!")

                print(" [*] Epoch %3d tested in %3.3f" % (epoch, taken_time_test))

            print(" [*] Epoch %3d started" % epoch)

            for idx in xrange(0, batch_idxs):
                # loading trainig data
                # トレーニングデータの読み込み
                st=self.args["batch_size"]*idx
                # getting training data
                # トレーニングデータの取得
                res_t=np.asarray([batch_sounds_r[ind] for ind in index_list[st:st+self.args["batch_size"]]])
                tar=np.asarray([batch_sounds_t[ind] for ind in index_list2[st:st+self.args["batch_size"]]])
                res_l=np.asarray([batch_label_t[ind] for ind in index_list2[st:st+self.args["batch_size"]]])
                tar_l = np.asarray([batch_label_r[ind] for ind in index_list2[st:st + self.args["batch_size"]]])

                rate = 1.0 - 0.5 ** (epoch // 50 + 1)
                # G-netの学習
                nos = np.random.rand(self.args["batch_size"]) * self.args["label_noise"]
                # self.sess.run([g_optim,self.update_ops],feed_dict={ self.input_modela:res_t,self.label_modela:res_l,self.input_modelb:tar,self.label_modelb:tar_l, self.noise:nos,self.training:np.asarray([rate])})
                self.sess.run([g_optim,self.update_ops],feed_dict={ self.input_modela:res_t,self.input_modelb:tar, self.noise:nos,self.training:np.asarray([rate]),lr_g:lr_g_opt3})
                # Update D network (2times)
                nos = np.random.rand(self.args["batch_size"]) * self.args["label_noise"]
                #self.sess.run([d_optim],
                 #             feed_dict={self.input_modelb: tar, self.input_modela: res_t, self.label_modela: res_l,
                 #                        self.label_modelb: tar_l, self.noise: nos, self.training: np.asarray([rate])})
                self.sess.run([d_optim],
                              feed_dict={self.input_modelb: tar, self.input_modela: res_t,
                                        self.noise: nos, self.training: np.asarray([rate]),lr_d:lr_d_opt3})
                self.sess.run([d_optim],
                              feed_dict={self.input_modelb: tar, self.input_modela: res_t,
                                         self.noise: nos, self.training: np.asarray([rate]), lr_d: lr_d_opt3})

                # tensorboardの保存
                if self.args["tensorboard"] and (counter+ti*epoch)%self.args["train_interval"]==0:
                    nos = np.random.rand(self.args["batch_size"]) * 0.0
                    # hg,hd,hg2, hd2=self.sess.run([self.g_loss_sum_1,self.d_loss_sumA,self.g_loss_sum_2, self.d_loss_sumB],feed_dict={self.input_modela:res_t,self.label_modela:res_l, self.label_modelb:tar_l,self.input_modelb:tar ,self.noise:nos ,self.training:np.asarray([1.0]) })
                    hg, hd, hg2, hd2= self.sess.run(
                        [self.g_loss_sum_1, self.d_loss_sumA, self.g_loss_sum_2, self.d_loss_sumB],
                        feed_dict={self.input_modela: res_t,self.input_modelb: tar, self.noise: nos, self.training: np.asarray([1.0])})

                    self.writer.add_summary(hg, counter + ti * epoch)
                    self.writer.add_summary(hd, counter + ti * epoch)
                    self.writer.add_summary(hg2, counter+ti*epoch)
                    self.writer.add_summary(hd2, counter+ti*epoch)
                counter+=1

            #saving model
            #モデルの保存
            if epoch%self.args["save_interval"]==0:
                self.save(self.args["checkpoint_dir"], epoch)
            if self.args["log"] and self.args["wave_otp_dir"]!="False":
                with open(self.args["log_file"],"a") as f:
                    if self.args["stop_argument"]:
                        f.write("\n %6d,%5.5f,%5.5f,%10.5f" % (
                        epoch, float(np.mean(log_data_d)), float(np.mean(log_data_g)), float(test1)))
                    else:
                        f.write("%6d,%10.5f" % (epoch, float(test1)))
                    f.write("\n")
                    f.flush()
            if self.args["hyperdash"] and self.args["stop_argument"] :
                self.experiment.metric("ScoreD", np.mean(log_data_d))
                self.experiment.metric("lossG", np.mean(log_data_g))

            # initializing epoch info
            # エポック情報の初期化
            log_data_g = np.empty(0)
            log_data_d = np.empty(0)

            if self.args["stop_itr"] != -1:
                count=counter+ti*epoch
                # console outputs
                taken_time = time.time() - start_time
                start_time = time.time()
                ft = taken_time * (self.args["stop_itr"] - count - 1)
                print(" [*] Epoch %5d finished in %.2f (preprocess %.3f) ETA: %3d:%2d:%2.1f" % (
                epoch, taken_time, ts, ft // 3600, ft // 60 % 60, ft % 60))
                time_of_epoch = np.append(time_of_epoch, np.asarray([taken_time, ts]))
                if count>self.args["stop_itr"]:
                    break
            else :
                #console outputs
                count = counter + ti * epoch
                taken_time = time.time() - start_time
                taken_times=np.append(taken_time,taken_times)
                if taken_times.shape[0]>20:
                    taken_times=taken_times[0:20]
                tsts=np.mean(taken_times)
                start_time = time.time()
                ft=tsts*(self.args["train_epoch"]-epoch-1)
                print(" [*] Epoch %5d (iterations: %10d)finished in %.2f (preprocess %.3f) ETA: %3d:%2d:%2.1f" % (epoch,count,taken_time,ts,ft//3600,ft//60%60,ft%60))
                time_of_epoch=np.append(time_of_epoch,np.asarray([taken_time,ts]))
            if epoch%self.args["lr_decay_term"]==0:
                lr_d_opt3 = lr_d_opt * (0.1 ** (epoch // 100))
                lr_g_opt3 = lr_g_opt * (0.1 ** (epoch // 100))
        self.save(self.args["checkpoint_dir"], epoch)
        print(" [*] Finished!! in "+ str(np.sum(time_of_epoch[::2])))

        if self.args["log"] and self.args["wave_otp_dir"] != "False":
            with open(self.args["log_file"], "a") as f:
                f.write("\n time on 1 epoch:" +str(np.mean(time_of_epoch[::2]))+" preprocess :"+str(np.mean(time_of_epoch[1::2])))
                f.write("\n")
                f.flush()
        # hyperdash
        if self.args["hyperdash"]:
            self.experiment.end()
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
        if self.args["cupy"] :
            wineds = cupy.asarray(wined, dtype=cupy.float64)
            fft_rs = cupy.fft.fft(wineds, n=self.args["NFFT"], axis=1)
            fft_r = cupy.asnumpy(fft_rs)
        else:
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
        if self.args["cupy"]:
            eep = cupy.asarray(data, dtype=cupy.complex128)
            fft_se = cupy.fft.ifft(eep,n=self.args["NFFT"], axis=1)
            fft_s = cupy.asnumpy(fft_se)
        else:
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
def generator(current_outputs,reuse,depth,chs,chs2,f,s,type,train,name):
    if type == "flatnet":
        return generator_flatnet(current_outputs,reuse,depth,chs,f,s,0,train,name)
    elif type == "ps_flatnet":
        return generator_flatnet(current_outputs, reuse, depth, chs, f, s, 1,train,name)
    elif type == "hybrid_flatnet":
        return generator_flatnet(current_outputs, reuse, depth, chs, f, s, 2,train,name)
    elif type == "double_flatnet":
        return generator_flatnet(current_outputs, reuse, depth, chs, f, s, 3, train, name)
    elif type == "ps_decay_flatnet":
        return generator_flatnet_decay(current_outputs, reuse, depth, chs2, f, s, 1, train)
    elif type == "double_decay_flatnet":
        return generator_flatnet_decay(current_outputs, reuse, depth, chs2, f, s, 3, train)
    elif type == "decay_flatnet":
        return generator_flatnet_decay(current_outputs, reuse, depth, chs2, f, s, 0, train)
    elif type == "ps_unet":
        return generator_unet(current_outputs, reuse, depth, chs, f, s, 1)
    elif type == "hybrid_unet":
        return generator_unet(current_outputs, reuse, depth, chs, f, s, 2)
    else :
        return  generator_unet(current_outputs,reuse,depth,chs,f,s)
def generator_flatnet(current_outputs,reuse,depth,chs,f,s,ps,train,name):
    current=current_outputs
    output_shape=int(current.shape[3])
    #main process
    for i in range(depth):
        connections = current
        if ps==1:
            ten = block_ps(current, output_shape,chs, f, i, reuse,i!=depth-1,train)
        elif ps==2 :
            ten=block_hybrid(current,f,chs,i,reuse,i!=depth-1,train)
        elif ps == 3:
            ten = block_double(current, f,s, chs, i, reuse, i != depth - 1, pixs=4,train=train )
        else :
            ten = block_dc(current, output_shape, chs, f, s, i, reuses=reuse, shake=i != depth - 1,train=train)
        if i!=depth-1:
            current = ten + connections
        else:
            current=ten
    return current
def generator_flatnet_decay(current_outputs,reuse,depth,chs,f,s,ps,train):
    current=current_outputs
    #main process
    for i in range(depth):
        connections = current
        if ps==1:
            ten = block_ps(current, chs[i*2+1],chs[i*2],f, i, reuse,i!=depth-1,train=train)
        elif ps == 3:
                ten = block_double(current, chs[i * 2 + 1], chs[i * 2], f,s, i, reuse, i != depth - 1,pixs=16, train=train)
        else :
            ten = block_dc(current,chs[i*2+1],chs[i*2], f, s, i, reuses=reuse, shake=i != depth - 1,train=train)
        if i!=depth-1:
            ims=ten.shape[3]//connections.shape[3]
            if ims!=0:
                connections=tf.tile(connections,[1,1,1,ims])
            elif connections.shape[3]>ten.shape[3]:
                connections = connections[:,:,:,:ten.shape[3]]
            current = ten + connections
        else:
            current=ten
    return current

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
    tt = tf.pad(ten, ((0, 0), (0, 0), (2, 0), (0, 0)), "reflect")
    ten = tt[:, :, :-2, :]
    ten = tf.nn.leaky_relu(ten,name="lrelu"+str(depth))
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
def block_double(current,output_shape,chs,f,s,depth,reuses,shake,pixs=2,train=True):
    tenA=current

    stddevs = math.sqrt(2.0 / (f[0] * f[1] * chs))
    tenA = tf.layers.conv2d(tenA, chs, kernel_size=f, strides=s, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), data_format="channels_last",reuse=reuses,name="conv11"+str(depth))
    tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bn11" + str(depth))

    tenA = tf.nn.leaky_relu(tenA,name="lrelu"+str(depth))


    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(tenA.shape[3])))
    tenA = tf.layers.conv2d_transpose(tenA, output_shape, kernel_size=f, strides=s, padding="VALID",
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                                      data_format="channels_last", reuse=reuses, name="deconv11" + str(depth))

    tenB=current
    ps_f=[pixs,pixs]
    tenB = tf.layers.conv2d(tenB, chs, kernel_size=ps_f, strides=ps_f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                           data_format="channels_last", reuse=reuses, name="conv21" + str(depth))
    tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=reuses,
                                         name="bn21" + str(depth))
    inl=tf.random_normal_initializer(mean=0.5,stddev=1.0)

    # nas = tf.get_variable("pos_gate"+str(depth),shape=[1,1,tenB.shape[2],1],trainable=True,initializer=inl)
    # nas = tf.tile(nas,[tenB.shape[0],tenB.shape[1],1,tenB.shape[3]])
    # nas = tf.clip_by_value(nas,0.0,1.0)
    # tt = tf.pad(tenB*nas, ((0, 0), (0, 0), (2, 0), (0, 0)), "reflect")
    # tenB = tt[:, :, :-2, :]
    tenB = tf.nn.leaky_relu(tenB, name="lrelu" + str(depth))
    tenB = deconve_with_ps(tenB, pixs, output_shape, depth, reuses=reuses)

    ten = (tenA + tenB) * 0.5
    if shake:
        ten=tf.nn.leaky_relu(ten)

    return ten

def deconve_with_ps(inp,r,otp_shape,depth,f=[1,1],reuses=None):
    chs_r=(r**2)*otp_shape
    stddevs = math.sqrt(2.0 / (f[0] * f[1] * int(inp.shape[3])))
    ten = tf.layers.conv2d(inp, chs_r, kernel_size=f, strides=f, padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                           data_format="channels_last", reuse=reuses, name="deconv_ps1" + str(depth))
    b_size = -1
    in_h = ten.shape[1]
    in_w = ten.shape[2]
    ten = tf.reshape(ten, [b_size, r, r, in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [b_size, in_h * r, in_w * r, otp_shape])
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