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

        self.input_modela1 = self.input_modela[:, :self.input_size_test[1], :, :]
        self.input_modela2 = self.input_modela[:, -self.input_size_test[1]:, :, :]
        self.input_modelb1 = self.input_modelb[:, :self.input_size_test[1], :, :]
        self.input_modelb2 = self.input_modelb[:, -self.input_size_test[1]:, :, :]
        self.training=tf.placeholder(tf.float32,[1],name="Training")
        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1"):
                self.fake_aB_image12 = generator(self.input_modela1, reuse=None,
                                              chs=self.args["G_channels"], depth=self.args["depth"], d=self.args["dilations"],train=True)
                self.fake_aB_image23 = generator(self.input_modela2, reuse=True,
                                                 chs=self.args["G_channels"], depth=self.args["depth"],
                                                 d=self.args["dilations"], train=True)

                self.fake_aB_image_test = generator(self.input_model_test, reuse=True,
                                                chs=self.args["G_channels"], depth=self.args["depth"],
                                                d=self.args["dilations"],
                                                train=False)
            with tf.variable_scope("generator_2"):
                self.fake_bA_image12 = generator(self.input_modelb1, reuse=None,
                                              chs=self.args["G_channels"], depth=self.args["depth"],d=self.args["dilations"],train=True)
                self.fake_bA_image23 = generator(self.input_modelb2, reuse=True,
                                               chs=self.args["G_channels"], depth=self.args["depth"],
                                               d=self.args["dilations"], train=True)
            self.fake_aB_image = tf.concat([self.fake_aB_image12, self.fake_aB_image23], axis=1)[:,1:,:,:]
            self.fake_bA_image = tf.concat([self.fake_bA_image12, self.fake_bA_image23], axis=1)[:,1:,:,:]

            with tf.variable_scope("generator_2",reuse=True):
                self.fake_Ba_image = generator(self.fake_aB_image, reuse=True,
                                              chs=self.args["G_channels"], depth=self.args["depth"], d=self.args["dilations"],train=True)
            with tf.variable_scope("generator_1",reuse=True):
                self.fake_Ab_image = generator(self.fake_bA_image, reuse=True,
                                               chs=self.args["G_channels"], depth=self.args["depth"],d=self.args["dilations"],train=True)

        ff=self.args["input_size"]//self.args["SHIFT"]
        a_true_noised=self.input_modela[:,-ff:,:,:]
        b_true_noised = self.input_modelb[:,-ff:,:,:]

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
            with tf.variable_scope("discrimA"):
                self.d_judge_AR = discriminator(a_true_noised, None,  self.args["d_depth"],
                                                                        self.args["D_channels"])
                self.d_judge_AF = discriminator(self.fake_bA_image12, True,  self.args["d_depth"],
                                                                        self.args["D_channels"])


        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars_aB=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")
        self.g_vars_bA = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generators")

        self.d_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrims")
        self.d_vars_2=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrims")
        #objective-functions of discriminator
        #D-netの目的関数
        self.d_loss_AR = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"],1]),predictions=self.d_judge_AR))
        self.d_loss_AF = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros([self.args["batch_size"],1]),predictions=self.d_judge_AF))
        self.d_loss_BR = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"],1]), predictions=self.d_judge_BR))
        self.d_loss_BF = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros([self.args["batch_size"],1]),predictions=self.d_judge_BF))

        self.d_lossA=(self.d_loss_AR+self.d_loss_AF)
        self.d_lossB= (self.d_loss_BR + self.d_loss_BF)
        self.d_loss=self.d_lossA+self.d_lossB
        # objective-functions of generator
        # G-netの目的関数

        # L1 norm lossA
        saa=tf.losses.mean_squared_error(labels=self.input_modela[:,-ff:,:,0],predictions =self.fake_Ba_image[:,:,:,0])*2.0
        sbb=tf.losses.mean_squared_error(labels=self.input_modela[:,-ff:,:,1] ,predictions = self.fake_Ba_image[:,:,:,1])*80.0
        L1B=saa+sbb

        # Gan lossA
        DSb=tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"],1]),predictions=self.d_judge_BF))
        # generator lossA
        self.g_loss_aB = L1B * self.args["weight_Cycle"]+DSb* self.args["weight_GAN"]
        # L1 norm lossB
        sa=tf.losses.mean_squared_error(labels=self.input_modelb[:,-ff:,:,0] ,predictions = self.fake_Ab_image[:,:,:,0])*2.0
        sb=tf.losses.mean_squared_error(labels=self.input_modelb[:,-ff:,:,1] , predictions =self.fake_Ab_image[:,:,:,1])*80.0
        L1bAAb = sa+sb
        # Gan loss
        DSA = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones([self.args["batch_size"],1]),predictions=self.d_judge_AF))
        # L1UBA =16.0/(tf.abs(self.fake_bA_image[:,:,:,0]-self.fake_aB_image[:,:,:,0])+1e-8)
        # L1UBA =tf.maximum(L1UBA,tf.ones_like(L1UBA))
        # generator loss
        self.g_loss_bA = L1bAAb * self.args["weight_Cycle"] + DSA * self.args["weight_GAN"]
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
        self.g_test_epoch = tf.summary.merge([tf.summary.scalar("g_test_epoch_end", self.g_test_epo,family="test")])

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


        self.lod="[glr="+str(lr_g_opt)+",gb="+str(beta_g_opt)+",dlr="+str(lr_d_opt)+",db="+str(beta_d_opt)+"]"
        g_optim =tf.train.AdamOptimizer(lr_g_opt,beta_g_opt,beta_2_g_opt).minimize(self.g_loss, var_list=self.g_vars_bA)
        d_optim = tf.train.AdamOptimizer(lr_d_opt,beta_d_opt,beta_2_d_opt).minimize(self.d_loss, var_list=self.d_vars_1)

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
        # 学習の情報の初期化
        start_time = time.time()
        ti=0
        # loading net
        # 過去の学習データの読み込み
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loading training data directory
        # トレーニングデータの格納ディレクトリの読み込み
        data = glob(self.args["train_data_path"]+'/Source_data/*')
        data2 = glob(self.args["train_data_path"] + '/Answer_data/*')
        # loading test data
        # テストデータの読み込み
        test=isread('./Model/datasets/test/test.wav')[0:160000].astype(np.float32)
        label=isread('./Model/datasets/test/label.wav')[0:160000].astype(np.float32)
        # times of one epoch
        # 回数計算
        train_data_num = min(len(data),len(data2))

        batch_idxs = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num)]

        #　学習データをメモリに乗っける
        batch_files = data[:train_data_num]
        batch_files2 = data2[:train_data_num]
        batch_sounds_r = np.asarray([(imread(batch_file)) for batch_file in batch_files])
        batch_sounds_t = np.asarray([(imread(batch_file)) for batch_file in batch_files2])

        # hyperdash
        if self.args["hyperdash"]:
            self.experiment=Experiment(self.args["name_save"]+"_G1")
            self.experiment.param("lr_g_opt", lr_g_opt)
            self.experiment.param("beta_g_opt", beta_g_opt)
            self.experiment.param("training_interval", self.args["train_interval"])
            self.experiment.param("learning_rate_scale", tln)


        for epoch in range(self.args["start_epoch"],self.args["train_epoch"]):
            # shuffling training data
            # トレーニングデータのシャッフル
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)
            test1=0.0
            ts = 0.0
            ipt = self.input_size_model[1]
            counter=0
            if self.args["test"] and epoch%self.args["save_interval"]==0:
                print(" [*] Epoch %3d testing" % epoch)
                #testing
                #テスト
                out_puts,taken_time_test,im=self.convert(test.reshape(1,-1,1))
                im = im.reshape([-1, self.args["NFFT"], 2])
                print(im.shape)
                otp_im=np.append(np.clip((im[:,:,0]+30)/40,0.0,1.0).reshape([1,-1,self.args["NFFT"],1]),np.clip((im[:,:,1]+3.15)/6.30,0.0,1.0).reshape([1,-1,self.args["NFFT"],1]),axis=3)
                out_put=out_puts.astype(np.float32)/32767.0
                # loss of tesing
                #テストの誤差
                test1=np.mean(np.abs(out_puts.reshape(1,-1,1)[0]-label.reshape(1,-1,1)[0]))

                #hyperdash
                if self.args["hyperdash"]:
                    self.experiment.metric("testG",test1)
                #writing epoch-result into tensorboard
                #tensorboardの書き込み
                if self.args["tensorboard"]:
                    rs=self.sess.run(self.tb_results,feed_dict={ self.result:out_put.reshape(1,1,-1),self.result1:otp_im,self.g_test_epo:test1})
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
                    self.sess.run([g_optim,self.update_ops],feed_dict={ self.input_modela:res_t,self.input_modelb:tar})
                    # Update D network (1times)
                    self.sess.run([d_optim],
                                  feed_dict={self.input_modelb: tar, self.input_modela: res_t})
                    # saving tensorboard
                    # tensorboardの保存
                    if self.args["tensorboard"] and (counter+ti*epoch)%self.args["train_interval"]==0:
                        hg,hd,hg2, hd2=self.sess.run([self.g_loss_sum_1,self.d_loss_sumA,self.g_loss_sum_2, self.d_loss_sumB],feed_dict={self.input_modela:res_t, self.input_modelb:tar  })
                        self.writer.add_summary(hg, counter + ti * epoch)
                        self.writer.add_summary(hd, counter + ti * epoch)
                        self.writer.add_summary(hg2, counter+ti*epoch)
                        self.writer.add_summary(hd2, counter+ti*epoch)
                    counter+=1

            #saving model
            #モデルの保存
            self.save(self.args["checkpoint_dir"], epoch)
            if self.args["log"] and self.args["wave_otp_dir"]!="False":
                with open(self.args["log_file"],"a") as f:
                    f.write("%6d,%10.5f" % (epoch, float(test1)))
                    f.write("\n")
                    f.flush()

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
                start_time = time.time()
                ft=taken_time*(self.args["train_epoch"]-epoch-1)
                print(" [*] Epoch %5d (iterations: %10d)finished in %.2f (preprocess %.3f) ETA: %3d:%2d:%2.1f" % (epoch,count,taken_time,ts,ft//3600,ft//60%60,ft%60))
                time_of_epoch=np.append(time_of_epoch,np.asarray([taken_time,ts]))


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
        #
        ten = tf.layers.conv2d(tenA, chs[tms + i]*4, [1, 7], [1, 4], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                               data_format="channels_last", reuse=reuses, name="res_conv" + str(i) + str(rep_pos))

        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                             name="bnA"+str(tms+i) + str(rep_pos))
        ten = tf.nn.leaky_relu(ten)
        ten = tf.layers.conv2d_transpose(ten, chs[tms + i], [1, 7], [1, 4], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                               data_format="channels_last", reuse=reuses, name="res_deconv" + str(i) + str(rep_pos))
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
    tenA = tf.layers.conv2d(tenA, 4, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                            data_format="channels_last", reuse=reuses, name="res_last1A"  + str(rep_pos))
    tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuses,
                                        name="bnAL" + str( 1 + times + res) + str(rep_pos))
    tenA=tf.nn.leaky_relu(tenA)
    tenA = tf.layers.conv2d(tenA, 1, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs), use_bias=False,
                            data_format="channels_last", reuse=reuses, name="res_last2A" + str(rep_pos))

    tenB=ten
    tenB = tf.layers.conv2d(tenB, 4, [1, 1], [1, 1], padding="SAME",
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