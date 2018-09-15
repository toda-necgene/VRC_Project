from glob import glob
import tensorflow as tf
import os
import time
from six.moves import xrange
import numpy as np
import wave
import librosa,sklearn
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import pyaudio
from hyperdash import Experiment
from datetime import datetime
import json
import shutil
import cupy
import dropbox
from .model import discriminator,generator
import matplotlib.pyplot as plt
class Model:
    def __init__(self,path):
        self.args=dict()
        self.args["checkpoint_dir"]="./trained_models"
        self.args["best_checkpoint_dir"]="./trained_models2"
        self.args["wave_otp_dir"] = "False"
        self.args["test_data_dir"] ="None"
        self.args["batch_size"]=1
        self.args["depth"] =[4]
        self.args["d_depth"] = 4
        self.args["train_epoch"]=500
        self.args["start_epoch"]=0
        self.args["test"]=True
        self.args["tensorboard"]=False
        self.args["hyperdash"]=False
        self.args["input_size"] = 8192
        self.args["weight_Cycle_Pow"]=1.0
        self.args["weight_Cycle_Fre"]=1.0
        self.args["weight_GAN"] = 1.0
        self.args["NFFT"]=1024
        self.args["debug"] = False
        self.args["cupy"] = False
        self.args["D_channels"] =[2]
        self.args["G_channels"] = [32]
        self.args["model_name"] = "wave2wave"
        self.args["version"] = "1.0.0"
        self.args["g_lr_max"]=2e-4
        self.args["g_lr_min"] = 2e-8
        self.args["d_lr_max"] = 2e-4
        self.args["d_lr_min"] = 2e-8
        self.args["g_b1"] = 0.5
        self.args["g_b2"] = 0.999
        self.args["d_b1"] = 0.5
        self.args["d_b2"] = 0.999
        self.args["train_interval"]=10
        self.args["save_interval"]=1
        self.args["test_dir"] = "./test"
        self.args["dropbox"]="False"
        self.args["dilations"]=[1]
        self.args["dilation_size"]=7
        self.args["repeatations"]=1
        self.args["lr_decay_term"]=100
        self.args["train_data_dir"]="./train/Model/datasets/train/"
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
        self.args["SHIFT"] = self.args["NFFT"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        ss=self.args["input_size"]//self.args["SHIFT"]+self.args["dilation_size"]
        self.input_size_model=[None,ss+self.args["dilation_size"],self.args["NFFT"]//2,2]
        # self.input_size_test = [None, ss, self.args["NFFT"] // 2, 2]
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
        if self.args["dropbox"]!="False":
            self.dbx=dropbox.Dropbox(self.args["dropbox"])
            self.dbx.users_get_current_account()

        else:
            self.dbx=None
        self.checkpoint_dir = self.args["checkpoint_dir"]
        self.build_model()
    def build_model(self):

        #inputs place holder
        #入力
        self.input_modela=tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net_A")
        self.input_modelb = tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net_B")
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_G-net_A")

        self.input_modela1 = self.input_modela[:, -8:, :, :]*0.1
        self.input_modelb1 = self.input_modelb[:, -8:, :, :]*0.1
        self.input_model_tests=self.input_model_test[:,-8:,:,:]*0.1
        self.training=tf.placeholder(tf.float32,[1],name="Training")
        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1"):
                self.fake_aB_image12,ax1 = generator(self.input_modela1, reuse=None,
                                              chs=self.args["G_channels"], depth=self.args["depth"], d=self.args["dilations"],train=True,r=self.args["repeatations"])

                self.fake_aB_image_test ,_= generator(self.input_model_tests, reuse=True,
                                                chs=self.args["G_channels"], depth=self.args["depth"],
                                                d=self.args["dilations"],
                                                train=False,r=self.args["repeatations"])
                self.fake_aB_image_test*=10
            with tf.variable_scope("generator_2"):
                self.fake_bA_image12,bx1 = generator(self.input_modelb1, reuse=None,
                                              chs=self.args["G_channels"], depth=self.args["depth"],d=self.args["dilations"],train=True,r=self.args["repeatations"])
            self.fake_aB_image = self.fake_aB_image12
            self.fake_bA_image = self.fake_bA_image12

            with tf.variable_scope("generator_2",reuse=True):
                self.fake_Ba_image,_ = generator(self.fake_aB_image, reuse=True,
                                              chs=self.args["G_channels"], depth=self.args["depth"], d=self.args["dilations"],train=True,r=self.args["repeatations"])
            with tf.variable_scope("generator_1",reuse=True):
                self.fake_Ab_image,_ = generator(self.fake_bA_image, reuse=True,
                                               chs=self.args["G_channels"], depth=self.args["depth"],d=self.args["dilations"],train=True,r=self.args["repeatations"])

        ff=self.args["input_size"]//self.args["SHIFT"]
        a_true_noised=self.input_modela1[:,-ff:,:,:]
        b_true_noised = self.input_modelb1[:,-ff:,:,:]

        #creating discriminator inputs
        #D-netの入力の作成
        #creating discriminator
        #D-net（判別側)の作成
        self.d_judge_F_logits=[]
        with tf.variable_scope("discrims"):

            with tf.variable_scope("discrimB"):
                self.d_judge_BR= discriminator(b_true_noised, None, self.args["d_depth"],
                                                                       self.args["D_channels"])

                self.d_judge_BF = discriminator(self.fake_aB_image12, True, self.args["d_depth"],
                                                                       self.args["D_channels"])
                self.d_judge_BF2 = list()
                for bnbn in ax1:
                    self.d_judge_BF2.append(discriminator(bnbn, True, self.args["d_depth"],
                                                    self.args["D_channels"]))

            with tf.variable_scope("discrimA"):
                self.d_judge_AR = discriminator(a_true_noised, None,  self.args["d_depth"],
                                                                        self.args["D_channels"])
                self.d_judge_AF = discriminator(self.fake_bA_image12, True,  self.args["d_depth"],
                                                                        self.args["D_channels"])
                self.d_judge_AF2=list()
                for bnbn in bx1:
                    self.d_judge_AF2.append(discriminator(bnbn, True, self.args["d_depth"],
                                                    self.args["D_channels"]))


        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")

        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrims")
        #objective-functions of discriminator
        #D-netの目的関数
        self.d_loss_AR = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"],1]),predictions=self.d_judge_AR))
        self.d_loss_AF = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros([self.args["batch_size"],1]),predictions=self.d_judge_AF))
        self.d_loss_AF2=list()
        for n in self.d_judge_AF2:
            self.d_loss_AF2.append(tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros([self.args["batch_size"], 1]), predictions=n)))
        # self.d_loss_AF2 =tf.add_n(self.d_loss_AF2)
        self.d_loss_BR = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"],1]), predictions=self.d_judge_BR))
        self.d_loss_BF = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros([self.args["batch_size"],1]),predictions=self.d_judge_BF))
        self.d_loss_BF2=list()
        for n in self.d_judge_BF2:
            self.d_loss_BF2.append(tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros([self.args["batch_size"], 1]), predictions=n)))
        # self.d_loss_BF2 = tf.add_n(self.d_loss_BF2)
        self.d_lossA=(self.d_loss_AR+self.d_loss_AF)
        self.d_lossB= (self.d_loss_BR + self.d_loss_BF)
        # dl2norm=tf.add_n([tf.nn.l2_loss(w) for w in self.d_vars])*1e-4
        self.d_loss=self.d_lossA+self.d_lossB
        # objective-functions of generator
        # G-netの目的関数

        # L1 norm lossA
        saa=tf.reduce_mean(tf.abs(self.fake_Ba_image[:,:,:,0]-self.input_modela1[:,-ff:,:,0]))* self.args["weight_Cycle_Pow"]
        sbb=tf.reduce_mean(tf.abs(self.fake_Ba_image[:,:,:,1]-self.input_modela1[:,-ff:,:,1]))* self.args["weight_Cycle_Fre"]
        L1B=0.5*(saa+sbb)

        # Gan lossA
        DSb = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"],1]),predictions=self.d_judge_BF))
        DSb2= list()
        for n in self.d_judge_BF2:
            DSb2.append(tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"], 1]), predictions=n)))
        # DSb2=tf.add_n(DSb2)
        # generator lossA
        self.g_loss_aB = L1B +DSb* self.args["weight_GAN"]
        # L1 norm lossB
        sa=tf.reduce_mean(tf.abs(self.fake_Ab_image[:,:,:,0]-self.input_modelb1[:,-ff:,:,0]))* self.args["weight_Cycle_Pow"]
        sb=tf.reduce_mean(tf.abs(self.fake_Ab_image[:,:,:,1]-self.input_modelb1[:,-ff:,:,1] ))* self.args["weight_Cycle_Fre"]
        L1bAAb = 0.5*(sa+sb)
        # Gan loss
        DSA = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"],1]),predictions=self.d_judge_AF))
        DSA2=list()
        for n in self.d_judge_AF2:
            DSA2.append(tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"], 1]), predictions=n)))
        # DSA2=tf.add_n(DSA2)
        # L1UBA =16.0/(tf.abs(self.fake_bA_image[:,:,:,0]-self.fake_aB_image[:,:,:,0])+1e-8)
        # L1UBA =tf.maximum(L1UBA,tf.ones_like(L1UBA))
        # generator loss
        self.g_loss_bA = L1bAAb + DSA * self.args["weight_GAN"]
        self.g_loss=self.g_loss_aB+self.g_loss_bA
        #BN_UPDATE
        self.update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #tensorboard functions
        #tensorboard 表示用関数
        self.g_loss_all= tf.summary.scalar("g_loss_cycle_A", tf.reduce_mean(L1bAAb),family="g_loss")
        self.g_loss_gan = tf.summary.scalar("g_loss_gan_A", tf.reduce_mean(DSA),family="g_loss")
        self.dscore = tf.summary.scalar("dscore_A", tf.reduce_mean(self.d_judge_AF),family="d_score")
        self.g_loss_sum_1 = tf.summary.merge([self.g_loss_all, self.g_loss_gan, self.dscore])

        self.g_loss_all2 = tf.summary.scalar("g_loss_cycle_B", tf.reduce_mean(L1B),family="g_loss")
        self.g_loss_gan2 = tf.summary.scalar("g_loss_gan_B", tf.reduce_mean(DSb),family="g_loss")
        self.dscore2 = tf.summary.scalar("dscore_B", tf.reduce_mean(self.d_judge_BF),family="d_score")
        # self.g_loss_uba = tf.summary.scalar("g_loss_distAB", tf.reduce_mean(L1UBA), family="g_loss")
        self.g_loss_sum_2 = tf.summary.merge([self.g_loss_all2, self.g_loss_gan2, self.dscore2])

        self.d_loss_sumA = tf.summary.scalar("d_lossA", tf.reduce_mean(self.d_lossA),family="d_loss")
        self.d_loss_sumB = tf.summary.scalar("d_lossB", tf.reduce_mean(self.d_lossB),family="d_loss")

        self.result=tf.placeholder(tf.float32, [1,1,160000], name="FB")
        self.result1 = tf.placeholder(tf.float32, [1,None,self.args["NFFT"],2], name="FBI0")
        im1=tf.transpose(self.result1[:,:,:,:1],[0,2,1,3])
        im2 = tf.transpose(self.result1[:, :, :, 1:], [0, 2, 1, 3])
        self.fake_B_sum = tf.summary.audio("fake_B", tf.reshape(self.result,[1,160000,1]), 16000, 1)
        self.fake_B_sum2 = tf.summary.image("fake_B_image01", im1, 1)
        self.fake_B_sum3 = tf.summary.image("fake_B_image02", im2, 1)
        self.g_test_epo=tf.placeholder(tf.float32,name="g_test_epoch_end")
        self.g_test_epo2=tf.placeholder(tf.float32,name="g_test_epoch_end_mfcc")
        self.g_test_epoch = tf.summary.merge([tf.summary.scalar("g_test_epoch_end", self.g_test_epo,family="test"),tf.summary.scalar("g_test_mfcc_loss", self.g_test_epo2,family="test")])

        self.tb_results=tf.summary.merge([self.fake_B_sum,self.fake_B_sum2,self.fake_B_sum3,self.g_test_epoch])

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
            res3 = np.append(res3, a[-7:,:,:]).reshape(-1,self.args["NFFT"],2)


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
            otp=np.append(otp,res[-8192:])
        h=otp.shape[0]-in_put.shape[1]
        if h>0:
            otp=otp[h:]

        return otp.reshape(1,in_put.shape[1],in_put.shape[2]),time.time()-tt,res3[1:]



    def train(self):

        # setting paramaters
        # パラメータ
        lr_g_opt_max=self.args["g_lr_max"]
        lr_g_opt_min=self.args["g_lr_min"]
        beta_g_opt=self.args["g_b1"]
        beta_2_g_opt=self.args["g_b2"]
        lr_d_opt_max=self.args["d_lr_max"]
        lr_d_opt_min = self.args["d_lr_min"]
        beta_d_opt=self.args["d_b1"]
        beta_2_d_opt=self.args["d_b2"]
        T_cur=0
        T_pow=1.0
        ch=50000
        test_mfcc=999999
        best=999999
        T=self.args["lr_decay_term"]
        # naming output-directory
        # 出力ディレクトリ


        self.lod="[glr="+str(lr_g_opt_max)+",gb="+str(beta_g_opt)+",dlr="+str(lr_d_opt_max)+",db="+str(beta_d_opt)+"]"
        lr_g = tf.placeholder(tf.float32, None, name="g_lr")
        lr_d = tf.placeholder(tf.float32, None, name="d_lr")
        g_optim = tf.train.AdamOptimizer(lr_g, beta_g_opt, beta_2_g_opt).minimize(self.g_loss,
                                                                                  var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(lr_d, beta_d_opt, beta_2_d_opt).minimize(self.d_loss,
                                                                                  var_list=self.d_vars)

        tt_me=list()
        time_of_epoch=np.zeros(1)

        # logging
        # ログ出力
        if self.args["tensorboard"]:
            self.writer = tf.summary.FileWriter("./logs/"+self.lod+self.args["name_save"], self.sess.graph)

        if self.args["wave_otp_dir"] != "False" and  os.path.exists(self.args["wave_otp_dir"]):
            with open(self.args["log_file"], "w") as f:
                f.write("epochs,test_varidation")
                f.flush()
        # initialize training info
        ti=0
        # loading net
        # 過去の学習データの読み込み
        if self.load():
            print(" [*] Load SUCCESSED.")
        else:
            print(" [!] Load FAILED.")

        # loading training data directory
        # トレーニングデータの格納ディレクトリの読み込み
        data = glob(self.args["train_data_dir"]+'/Source_data/*')
        data2 = glob(self.args["train_data_dir"] + '/Answer_data/*')
        # loading test data
        # テストデータの読み込み
        test=isread(self.args["test_data_dir"]+'/test.wav')[0:160000].astype(np.float32)
        label=isread(self.args["test_data_dir"]+'/label.wav')[0:160000].astype(np.float32)
        # times of one epoch
        # 回数計算
        train_data_num = min(len(data),len(data2))

        batch_idxs = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num)]

        #　学習データをメモリに乗っける
        batch_files = data[:train_data_num]
        batch_files2 = data2[:train_data_num]
        print(" [*] loading dataset...")
        batch_sounds_r = np.asarray([(imread(batch_file)) for batch_file in batch_files])
        batch_sounds_t = np.asarray([(imread(batch_file)) for batch_file in batch_files2])
        print(" [*] %d data loaded!!",train_data_num)
        # hyperdash
        if self.args["hyperdash"]:
            self.experiment=Experiment(self.args["name_save"]+"_G1")
            self.experiment.param("lr_g_opt", lr_g_opt_max)
            self.experiment.param("beta_g_opt", beta_g_opt)
            self.experiment.param("training_interval", self.args["train_interval"])

        # 学習の情報の初期化
        radeon_x,radeon_fs=librosa.load(self.args["test_data_dir"]+'/label.wav',sr=16000)
        radeon = librosa.feature.mfcc(radeon_x, sr=radeon_fs)
        # radeon = sklearn.preprocessing.scale(radeon, axis=1)
        start_time = time.time()
        for epoch in range(self.args["start_epoch"],self.args["train_epoch"]):
            # 学習率の計算
            lr_d_opt3 = lr_d_opt_min+0.5*(lr_d_opt_max-lr_d_opt_min)*(1+np.cos(T_cur/T*np.pi))*T_pow
            # lr_d_opt3 =lr_d_opt_min+(lr_d_opt_max-lr_d_opt_min)*T_pow
            lr_g_opt3 = lr_g_opt_min+0.5*(lr_d_opt_max-lr_g_opt_min)*(1+np.cos(T_cur/T*np.pi))*T_pow
            # lr_g_opt3 = lr_g_opt_min+(lr_g_opt_max-lr_g_opt_min) * T_pow

            # トレーニングデータのシャッフル
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)
            ts = 0.0
            ipt = self.input_size_model[1]
            counter=0
            if self.args["test"] and epoch%self.args["save_interval"]==0:
                print(" [*] Epoch %3d testing" % epoch)
                #testing
                #テスト
                out_puts,taken_time_test,im=self.convert(test.reshape(1,-1,1))
                im = im.reshape([-1, self.args["NFFT"], 2])
                otp_im=np.append(np.clip((im[:,:,0]+10)/20,0.0,1.0).reshape([1,-1,self.args["NFFT"],1]),np.clip((im[:,:,1]+3.15)/6.30,0.0,1.0).reshape([1,-1,self.args["NFFT"],1]),axis=3)
                out_put=out_puts.astype(np.float32)/32767.0
                #テストの誤差
                test1=np.mean(np.abs(np.abs(out_puts.reshape(1,-1,1)[0])-np.abs(label.reshape(1,-1,1)[0])))
                raxis = librosa.feature.mfcc(out_put.reshape(-1)*1.0, sr=radeon_fs)
                # raxis = sklearn.preprocessing.scale(raxis, axis=1)
                rnx=min(raxis.shape[1],radeon.shape[1])
                test_mfcc=np.sum(np.abs(radeon[:,-rnx:]-raxis[:,-rnx:]))
                #hyperdash
                if self.args["hyperdash"]:
                    self.experiment.metric("testG",test1)
                    self.experiment.metric("testMFCC", test_mfcc)
                #writing epoch-result into tensorboard
                #tensorboardの書き込み
                if self.args["tensorboard"]:
                    hg, hd, hg2, hd2 = self.sess.run(
                        [self.g_loss_sum_1, self.d_loss_sumA, self.g_loss_sum_2, self.d_loss_sumB],
                        feed_dict={self.input_modela: batch_sounds_r[0:self.args["batch_size"]], self.input_modelb: batch_sounds_t[0:self.args["batch_size"]]})
                    self.writer.add_summary(hg, counter + ti * epoch)
                    self.writer.add_summary(hd, counter + ti * epoch)
                    self.writer.add_summary(hg2, counter + ti * epoch)
                    self.writer.add_summary(hd2, counter + ti * epoch)
                    rs=self.sess.run(self.tb_results,feed_dict={ self.result:out_put.reshape(1,1,-1),self.result1:otp_im,self.g_test_epo:test1,self.g_test_epo2:test_mfcc})
                    self.writer.add_summary(rs, epoch)

                #saving test result
                #テストの結果の保存
                if os.path.exists(self.args["wave_otp_dir"]):
                    plt.clf()
                    plt.subplot(211)
                    ins=np.transpose(im[:,:,0],(1,0))
                    plt.imshow(ins,aspect="auto")
                    plt.clim(-10,10)
                    plt.colorbar()
                    plt.subplot(212)
                    ins = np.transpose(im[:, :, 1], (1, 0))
                    plt.imshow(ins, aspect="auto")
                    plt.clim(-3.141593, 3.141593)
                    plt.colorbar()
                    path=self.args["wave_otp_dir"]+nowtime()+"_e"+str(epoch)
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
                batch_sounds1 = np.asarray([batch_sounds_r[ind] for ind in index_list[st:st+self.args["batch_size"]]])
                batch_sounds2= np.asarray([batch_sounds_t[ind] for ind in index_list2[st:st+self.args["batch_size"]]])
                # calculating one iteration repetation times
                # 1イテレーション実行回数計算
                times=int(batch_sounds1.shape[1])//self.input_size_model[1]+1
                if int(batch_sounds1.shape[1])%self.input_size_model[1]==0:
                    times-=1
                # shuffle start time
                # 開始タイミングのシャッフル
                time_set=[j for j in range(times)]
                np.random.shuffle(time_set)

                ti=(batch_idxs*times)
                for t in time_set:
                    tm = time.time()
                    # calculating starting position
                    # 開始位置の計算
                    start_pos=self.input_size_model[1]*(t+1)

                    # getting training data
                    # トレーニングデータの取得
                    res_t=batch_sounds1[:,max(0,start_pos-ipt):start_pos]
                    tar=batch_sounds2[:,max(0,start_pos-ipt):start_pos]
                    ts+=time.time()-tm
                    # Update G network
                    # G-netの学習
                    # self.sess.run([g_optim,self.update_ops],feed_dict={ self.input_modela:res_t,self.input_modelb:tar,lr_g:lr_g_opt3})
                    # Update D network (1time)
                    for _ in range(2):
                        self.sess.run([d_optim],
                                      feed_dict={self.input_modelb: tar, self.input_modela: res_t,lr_d:lr_d_opt3})
                    # G-netの学習
                    self.sess.run([g_optim, self.update_ops],
                                  feed_dict={self.input_modela: res_t, self.input_modelb: tar, lr_g: lr_g_opt3})
                    # saving tensorboard
                    # tensorboardの保存
                    counter+=1

            #saving model
            #モデルの保存
            self.save(self.args["checkpoint_dir"], epoch)
            if epoch%self.args["save_interval"]==0 and test_mfcc<best:
                self.save(self.args["best_checkpoint_dir"], epoch)
                best=test_mfcc
            #console outputs
            count = counter + ti * epoch
            taken_time = time.time() - start_time
            start_time = time.time()
            tt_me.append(taken_time)
            if len(tt_me)>20:
                tt_me=tt_me[0:20]
            ft=np.mean(tt_me)*(self.args["train_epoch"]-epoch-1)
            print(" [*] Epoch %5d (iterations: %10d)finished in %.2f (preprocess %.3f) ETA: %3d:%2d:%2.1f" % (epoch,count,taken_time,ts,ft//3600,ft//60%60,ft%60))
            time_of_epoch=np.append(time_of_epoch,np.asarray([taken_time,ts]))

            T_cur += 1
            if T==T_cur:
                # T=T//2
                T_cur=0
                T_pow*=0.9
            # elif epoch%self.args["save_interval"]==0 and test_mfcc<ch :
            #     ch -= 5000
            #     T_cur=0
            #     T_pow*=0.9
        print(" [*] Finished!! in "+ str(np.sum(time_of_epoch[::2])))

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

        time_ruler = data.shape[0] // self.args["SHIFT"]-1
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
        c = np.clip(c, -10, 10)
        d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
        spec = np.concatenate((c, d), 2)
        return spec
    def ifft(self,data,redi):
        a=data
        a[:, :, 0]=np.clip(a[:, :, 0],a_min=-10,a_max=88)
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