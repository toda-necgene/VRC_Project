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
import dropbox,dropbox.files
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
        self.args["strides_d"] = [2,2]
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
        self.args["SHIFT"] = self.args["NFFT"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        ss=self.args["input_size"]//self.args["SHIFT"]
        self.input_size_model=[None,ss+ss+self.args["dilation_size"] ,self.args["NFFT"]//2,2]
        self.input_size_test = [1, ss+self.args["dilation_size"], self.args["NFFT"] // 2, 2]
        self.output_size = [1, ss, self.args["NFFT"] // 2, 2]
        print("model train input size:" + str(self.input_size_model))
        print("model output size:"+str(self.output_size))
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
        """
        モデルの生成
        :return:
        """
        #入力
        self.input_modela= tf.placeholder(tf.float32, self.input_size_model, "input_A")
        self.input_modelb = tf.placeholder(tf.float32, self.input_size_model, "input_B")
        self.input_diff_ab = tf.placeholder(tf.float32, self.input_size_model[0], "diff_ab")
        self.input_def = tf.placeholder(tf.float32, self.input_size_model, "def_0")
        self.tar_def=tf.placeholder(tf.float32, [None,1,self.args["D_channels"][-1],1], "def_tar_0")
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "input_T")
        self.input_model_test_s = tf.placeholder(tf.float32, self.input_size_test, "input_T_b")

        self.input_modela1=self.input_modela[:,:self.input_size_test[1],:,:]
        self.input_modela2 = self.input_modela[:, -self.input_size_test[1]:, :, :]
        self.input_modelb1=self.input_modelb[:,:self.input_size_test[1],:,:]
        self.input_modelb2 = self.input_modelb[:, -self.input_size_test[1]:, :, :]
        #シードの作成

        with tf.variable_scope("seed_net"):
            self.seed_A,self.reality_A=seed_net(self.input_modela1[:,:8,:,:],None,self.args["d_depth"],self.args["D_channels"],"A")
            self.seed_def, self.reality_def = seed_net(self.input_def[:, :8, :, :], True, self.args["d_depth"],
                                      self.args["D_channels"], "A")
            self.seed_B,self.reality_B=seed_net(self.input_modelb1[:,:8,:,:],True,self.args["d_depth"],self.args["D_channels"]," ")
            self.seed_TB,self.score_f = seed_net(self.input_model_test_s[:,-8:,:,:], True,self.args["d_depth"],self.args["D_channels"], " ",train=False)
        #G-net（生成側）の作成
        with tf.variable_scope("generators"):
            self.fake_aB_12_image = generator(self.input_modela1,self.seed_B, reuse=None,chs=self.args["G_channels"], depth=self.args["depth"],d=self.args["dilations"],net_type=self.args["architect"],train=True)
            self.fake_aB_23_image=generator(self.input_modela2,self.seed_B, reuse=True, chs=self.args["G_channels"],
                                                  depth=self.args["depth"],d=self.args["dilations"],
                                                  net_type=self.args["architect"], train=True)
            self.fake_aB_image_test = generator(self.input_model_test,self.seed_TB, reuse=True,
                                            chs=self.args["G_channels"], depth=self.args["depth"],
                                            d=self.args["dilations"],
                                            net_type=self.args["architect"],
                                            train=False)
            output_aB=tf.concat([self.fake_aB_12_image,self.fake_aB_23_image[:,:-1,:,:]],axis=1)
            self.fake_bA_12_image = generator(self.input_modelb1,self.seed_A, reuse=True,
                                          chs=self.args["G_channels"], depth=self.args["depth"], d=self.args["dilations"],net_type=self.args["architect"],train=True)
            self.fake_bA_23_image = generator(self.input_modelb2,self.seed_A, reuse=True,
                                           chs=self.args["G_channels"], depth=self.args["depth"],d=self.args["dilations"],
                                           net_type=self.args["architect"], train=True)
            output_bA = tf.concat([self.fake_bA_12_image, self.fake_bA_23_image[:,:-1,:,:]], axis=1)
            self.fake_Ba_image = generator(output_aB,self.seed_A, reuse=True,
                                          chs=self.args["G_channels"], depth=self.args["depth"],d=self.args["dilations"],net_type=self.args["architect"],train=True)
            self.fake_Ab_image = generator(output_bA,self.seed_B, reuse=True,
                                           chs=self.args["G_channels"], depth=self.args["depth"],d=self.args["dilations"],net_type=self.args["architect"],train=True)

        # 偽物のシード作成
        self.d_judge_F_logits=[]
        with tf.variable_scope("seed_net",reuse=True):
            self.seed_FB,self.reality_FB=seed_net(self.fake_aB_12_image,True,self.args["d_depth"],self.args["D_channels"]," ")
            self.seed_FA,self.reality_FA=seed_net(self.fake_bA_12_image,True,self.args["d_depth"],self.args["D_channels"]," ")

        # それぞれの変数取得
        self.g_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"seed_net")
        # 誤差関数を設定
        self.loss_functions()
        #バッチ正規化の更新
        self.update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #保存の準備
        self.saver = tf.train.Saver()

    def loss_functions(self):
        """
        **概要**
        それぞれの誤差関数（目的関数）を設定している。
        式は以下のように定義されている
        各項の説明
		"""

        # クラスタリングの誤差
        cluster_loss = tf.pow(self.input_diff_ab/5.0 - tf.sqrt(tf.reduce_sum(tf.pow(self.seed_A - self.seed_B, 2))),
                              2) * 0.5
        # GANにおける本物と偽物の評価の誤差
        reality_loss_FB = tf.losses.mean_squared_error(labels=tf.zeros_like(self.reality_FB),predictions=self.reality_FB)
        reality_loss_FA = tf.losses.mean_squared_error(labels=tf.zeros_like(self.reality_FA),predictions =self.reality_FA)
        reality_loss_A = tf.losses.mean_squared_error(labels=tf.ones_like(self.reality_A),predictions=self.reality_A)
        reality_loss_B = tf.losses.mean_squared_error(labels=tf.ones_like(self.reality_B),predictions=self.reality_B)
        reality_loss = (reality_loss_FA + reality_loss_FB + reality_loss_A + reality_loss_B)
        # クラスタリングにおける分布の制限誤差
        # circle_loss = 0.5 * (tf.pow(1.0 - tf.sqrt(tf.reduce_sum(tf.pow(self.seed_A, 2))), 2) + tf.pow(
        #     1.0 - tf.sqrt(tf.reduce_sum(tf.pow(self.seed_B, 2))+1e-32), 2))
        # クラスタリングの基準点の誤差
        reference_loss = tf.losses.mean_squared_error(labels=self.tar_def, predictions=self.seed_def)

        # S-netの目的関数
        self.d_loss = cluster_loss + reality_loss  + reference_loss # + circle_loss
        # G-netの目的関数
        # Cycle_loss
        self.g_loss_cycle_A=tf.reduce_mean(tf.abs(self.input_modelb[:, :8, :, :1] - self.fake_Ab_image[:, :8, :, :1]) * self.args["weight_Cycle1"])
        self.g_loss_cycle_A2 = tf.reduce_mean(tf.abs(self.input_modelb[:, :8, :, 1:] - self.fake_Ab_image[:, :8, :, 1:]) *self.args["weight_Cycle2"])
        self.g_loss_cycle_B =tf.reduce_mean(tf.abs(self.input_modela[:, :8, :, :1] - self.fake_Ba_image[:, :8, :, :1]) * self.args["weight_Cycle1"])
        self.g_loss_cycle_B2 = tf.reduce_mean(tf.abs(self.input_modela[:, :8, :, 1:] - self.fake_Ba_image[:, :8, :, 1:]) *
            self.args["weight_Cycle2"])
        # GAN_labelロス
        self.g_loss_GAN_A = tf.losses.mean_squared_error(labels=tf.ones_like(self.reality_FA),predictions=self.reality_FA)
        self.g_loss_GAN_B = tf.losses.mean_squared_error(labels=tf.ones_like(self.reality_FB),predictions=self.reality_FB)
        self.g_loss_GAN_A2 =  tf.losses.mean_squared_error(labels=self.seed_A, predictions=self.seed_FA)
        self.g_loss_GAN_B2 =  tf.losses.mean_squared_error(labels=self.seed_B, predictions=self.seed_FB)

        self.g_loss = (self.g_loss_cycle_A2 + self.g_loss_cycle_B2)+(self.g_loss_cycle_A + self.g_loss_cycle_B) + (self.g_loss_GAN_A + self.g_loss_GAN_B) * \
                      self.args["weight_GAN1"]+(self.g_loss_GAN_A2 + self.g_loss_GAN_B2) * \
                      self.args["weight_GAN2"]
        # tensorboard 表示用関数
        summary=list()
        summary.append(tf.summary.scalar("cluster", tf.reduce_mean(cluster_loss), family="d_loss"))
        summary.append(tf.summary.scalar("real", tf.reduce_mean(reality_loss_A+reality_loss_B), family="d_loss"))
        summary.append(tf.summary.scalar("fake", tf.reduce_mean(reality_loss_FA + reality_loss_FB), family="d_loss"))
        #summary.append(tf.summary.scalar("circle_10", tf.reduce_mean(circle_loss), family="d_loss"))
        summary.append(tf.summary.scalar("reference", tf.reduce_mean(reference_loss), family="d_loss"))
        summary.append(tf.summary.scalar("cycle_A_pow", tf.reduce_mean(self.g_loss_cycle_A), family="g_loss"))
        summary.append(tf.summary.scalar("cycle_A_fre", tf.reduce_mean(self.g_loss_cycle_A2), family="g_loss"))
        summary.append(tf.summary.scalar("cycle_B_pow", tf.reduce_mean(self.g_loss_cycle_B), family="g_loss"))
        summary.append(tf.summary.scalar("cycle_B_fre", tf.reduce_mean(self.g_loss_cycle_B2), family="g_loss"))
        summary.append(tf.summary.scalar("GAN_A", tf.reduce_mean(self.g_loss_GAN_A), family="g_loss"))
        summary.append(tf.summary.scalar("GAN_B", tf.reduce_mean(self.g_loss_GAN_B), family="g_loss"))
        summary.append(tf.summary.scalar("A_cluster", tf.reduce_mean(self.g_loss_GAN_A2), family="g_loss"))
        summary.append(tf.summary.scalar("B_cluster", tf.reduce_mean(self.g_loss_GAN_B2), family="g_loss"))

        self.loss_sum = tf.summary.merge(summary)

        self.result = tf.placeholder(tf.float32, [1, 1, 160000], name="FB")
        self.result1 = tf.placeholder(tf.float32, [1, 320, 1024, 2], name="FBI0")
        im1 = tf.transpose(self.result1[:, :, :, :1], [0, 2, 1, 3])
        im2 = tf.transpose(self.result1[:, :, :, 1:], [0, 2, 1, 3])
        self.fake_B_sum = tf.summary.audio("fake_B", tf.reshape(self.result, [1, 160000, 1]), 16000, 1)
        self.fake_B_sum2 = tf.summary.image("fake_B_image01", im1, 1)
        self.fake_B_sum3 = tf.summary.image("fake_B_image02", im2, 1)
        self.g_test_epo = tf.placeholder(tf.float32, name="g_test_epoch_end")
        self.g_test_epoch = tf.summary.merge([tf.summary.scalar("g_test_epoch_end", self.g_test_epo, family="test")])

        self.tb_results = tf.summary.merge([self.fake_B_sum, self.fake_B_sum2, self.fake_B_sum3, self.g_test_epoch])

    def convert(self,in_put,target):
        #テスト用関数
        tt=time.time()
        ipt=self.args["input_size"]+self.args["SHIFT"]+self.args["SHIFT"]*self.args["dilation_size"]
        targ = target[0:ipt]/32767.0
        tar=self.fft(targ)[:,:self.args["SHIFT"]].reshape(1,self.input_size_test[1],self.input_size_test[2],self.input_size_test[3])
        times=in_put.shape[1]//(self.args["input_size"])+1
        if in_put.shape[1]%((self.args["input_size"])*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        res3 = np.zeros([1,self.args["NFFT"],2], dtype=np.float32)
        rss=np.zeros([self.input_size_test[2]],dtype=np.float64)
        res4=None
        for t in range(times):
            # 前処理

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
            res,b,c=self.sess.run([self.fake_aB_image_test,self.seed_TB,self.score_f],feed_dict={ self.input_model_test:res ,self.input_model_test_s:tar})
            res2=res.copy()[:,:,::-1,:]
            res=np.append(res,res2,axis=2)
            res[:,:,self.args["SHIFT"]:,1]*=-1
            a = res[0].copy()
            res3 = np.append(res3, a).reshape(-1,self.args["NFFT"],2)
            res4 = b
            # 後処理
            # 短時間高速離散逆フーリエ変換
            res,rss=self.ifft(a,rss)
            # 変換後処理
            res=np.clip(res,-1.0,1.0)
            res=res*32767
            # 結果の保存
            res=res.reshape(-1).astype(np.int16)
            otp=np.append(otp,res[-8192:])
        h=otp.shape[0]-in_put.shape[1]
        if h>0:
            otp=otp[h:]
        return otp.reshape(1,in_put.shape[1],in_put.shape[2]),time.time()-tt,res3[1:],res4



    def train(self):

        # パラメータ
        tln=self.args["train_d_scale"]
        lr_g_opt=self.args["g_lr"]
        beta_g_opt=self.args["g_b1"]
        beta_2_g_opt=self.args["g_b2"]
        lr_d_opt=lr_g_opt*tln
        beta_d_opt=self.args["d_b1"]
        beta_2_d_opt=self.args["d_b2"]
        # 出力ディレクトリ
        taken_times=[]

        self.lod="[glr="+str(lr_g_opt)+",gb="+str(beta_g_opt)+",dlr="+str(lr_d_opt)+",db="+str(beta_d_opt)+"]"

        lr_g_opt3 = lr_g_opt*(0.1**(self.args["start_epoch"]//self.args["lr_decay_term"]))
        lr_d_opt3 = lr_d_opt*(0.1**(self.args["start_epoch"]//self.args["lr_decay_term"]))
        lr_g=tf.placeholder(tf.float32,None,name="g_lr")
        lr_d=tf.placeholder(tf.float32,None,name="d_lr")
        g_optim = tf.train.AdamOptimizer(lr_g, beta_g_opt, beta_2_g_opt).minimize(self.g_loss,
                                                                                       var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(lr_d, beta_d_opt, beta_2_d_opt).minimize(self.d_loss,
                                                                                       var_list=self.d_vars)

        time_of_epoch=np.zeros(1)

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
        # 学習の情報の初期化
        start_time = time.time()
        log_data_g = np.empty(0)
        log_data_d = np.empty(0)
        defa = np.zeros(self.args["D_channels"][-1]).reshape([1,1,-1,1])
        defa[0,0,0,0]=1.0
        # 過去の学習データの読み込み
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print(" [*] reading data path :"+self.args["train_data_path"]+'/Source_data/')
        # トレーニングデータの格納ディレクトリの読み込み
        data = glob(self.args["train_data_path"]+'/Source_data/*-wave.npy')

        # テストデータの読み込み
        test=isread('./Model/datasets/test/test.wav')[0:160000].astype(np.float32)
        label=isread('./Model/datasets/test/label.wav')[0:160000].astype(np.float32)
        label2 = isread('./Model/datasets/test/label2.wav')[0:160000].astype(np.float32)

        # 回数計算
        train_data_num = min(len(data), self.args["train_data_num"])
        print(" [*] data found",len(data))
        batch_idxs = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]
        index_list2=[h for h in range(train_data_num)]

        #　学習データをメモリに乗っける
        batch_files = data[:train_data_num]

        batch_sounds_r = np.asarray([(imread(batch_file)) for batch_file in batch_files])
        diff_list=np.loadtxt(self.args["train_data_path"]+'/diff.csv',delimiter=",")
        # hyperdash
        if self.args["hyperdash"]:
            self.experiment=Experiment(self.args["name_save"]+"_G1")
            self.experiment.param("lr_g_opt", lr_g_opt)
            self.experiment.param("beta_g_opt", beta_g_opt)
            self.experiment.param("training_interval", self.args["train_interval"])
            self.experiment.param("learning_rate_scale", tln)

        for epoch in range(self.args["start_epoch"],self.args["train_epoch"]):
            # トレーニングデータのシャッフル
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)
            test1=0.0
            ts = 0.0
            counter=0
            ti=batch_idxs
            if self.args["test"] and epoch%self.args["save_interval"]==0:
                print(" [*] Epoch %3d testing" % epoch)
                #テスト
                out_puts,taken_time_test,im,im2=self.convert(test.reshape(1,-1,1),label.reshape(-1))
                out_puts2, _, _, im3 = self.convert(test.reshape(1, -1, 1), label2.reshape(-1))

                im = im.reshape([-1, self.args["NFFT"], 2])
                im2 = im2.reshape([-1, 4, 4])
                im3 = im3.reshape([-1, 4, 4])

                otp_im=np.append(np.clip((im[:,:,0]+30)/40,0.0,1.0).reshape([1,-1,self.args["NFFT"],1]),np.clip((im[:,:,1]+3.15)/6.30,0.0,1.0).reshape([1,-1,self.args["NFFT"],1]),axis=3)
                out_put=out_puts.astype(np.float32)/32767.0
                #テストの誤差
                test1=np.mean(np.abs(out_puts.reshape(1,-1,1)[0]-label.reshape(1,-1,1)[0]))

                #hyperdash
                if self.args["hyperdash"]:
                    self.experiment.metric("testG",test1)

                #tensorboardの書き込み
                if self.args["tensorboard"]:
                    rs=self.sess.run(self.tb_results,feed_dict={ self.result:out_put.reshape(1,1,-1),self.result1:otp_im,self.g_test_epo:test1})
                    self.writer.add_summary(rs, epoch)

                #テストの結果の保存
                if os.path.exists(self.args["wave_otp_dir"]):
                    plt.subplot(211)
                    ins=np.transpose(im[:,:,0],(1,0))
                    plt.imshow(ins,aspect="auto")
                    plt.clim(-30,10)
                    plt.colorbar()
                    plt.subplot(212)
                    ins = np.transpose(im[:, :, 1], (1, 0))
                    plt.imshow(ins, aspect="auto")
                    plt.clim(-3.141593, 3.141593)
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

                    plt.clf()
                    plt.subplot(211)
                    ins = np.transpose(im2[0], (1, 0))
                    plt.imshow(ins, aspect="auto")
                    plt.colorbar()
                    plt.subplot(212)
                    ins = np.transpose(im3[0], (1, 0))
                    plt.imshow(ins, aspect="auto")
                    plt.colorbar()
                    path = self.args["wave_otp_dir"] + nowtime()+"_seedA"
                    plt.savefig(path + ".png")
                    plt.clf()
                    upload(out_puts2, path,comment="another_label")
                print(" [*] Epoch %3d tested in %3.3f" % (epoch, taken_time_test))

            print(" [*] Epoch %3d started" % epoch)

            for idx in xrange(0, batch_idxs):
                # トレーニングデータの読み込み
                st=self.args["batch_size"]*idx
                # トレーニングデータの取得
                ism=[-1,self.input_size_model[1],self.input_size_model[2],self.input_size_model[3]]
                post=batch_sounds_r[0,:-1].reshape([1,self.input_size_model[1],self.input_size_model[2],self.input_size_model[3]])
                res_t=np.asarray([batch_sounds_r[ind] for ind in index_list[st:st+self.args["batch_size"]]])[:,:-1].reshape(ism)
                tar=np.asarray([batch_sounds_r[ind] for ind in index_list2[st:st+self.args["batch_size"]]])[:,:-1].reshape(ism)
                res2=np.asarray([batch_sounds_r[ind] for ind in index_list[st:st+self.args["batch_size"]]])[:,-1]
                tar2 = np.asarray([batch_sounds_r[ind] for ind in index_list2[st:st + self.args["batch_size"]]])[:,-1]
                diff=np.asarray([diff_list[int(res2[ts])][int(tar2[ts])]for ts in range(self.args["batch_size"])])
                # S-netの学習 (2times)
                self.sess.run([d_optim],feed_dict={self.input_modelb: tar, self.input_modela: res_t,self.input_diff_ab:diff,self.input_def: post,self.tar_def:defa, lr_d: lr_d_opt3})
                self.sess.run([d_optim],
                              feed_dict={self.input_modelb: tar, self.input_modela: res_t, self.input_diff_ab: diff,
                                         self.input_def: post, self.tar_def: defa, lr_d: lr_d_opt3})
                # G-netの学習
                self.sess.run([g_optim, self.update_ops],
                              feed_dict={self.input_modela: res_t, self.input_modelb: tar, self.input_def: post,
                                         self.tar_def: defa, lr_g: lr_g_opt3})
                # tensorboardの保存
                if self.args["tensorboard"] and (counter+ti*epoch)%self.args["train_interval"]==0:
                    res_t = np.asarray([batch_sounds_r[ind] for ind in range(10,20)])[:,
                            :-1].reshape(ism)
                    tar = np.asarray([batch_sounds_r[ind] for ind in range(10)])[:,
                          :-1].reshape(ism)
                    res2 = np.asarray([batch_sounds_r[ind] for ind in range(10,20)])[:,
                           -1]
                    tar2 = np.asarray([batch_sounds_r[ind] for ind in range(10)])[:,
                           -1]
                    diff = np.asarray(
                        [diff_list[int(res2[ts])][int(tar2[ts])] for ts in range(10)])

                    h= self.sess.run(self.loss_sum,feed_dict={self.input_def: post, self.tar_def: defa,self.input_modela: res_t,self.input_modelb: tar,self.input_diff_ab:diff})
                    self.writer.add_summary(h, counter + ti * epoch)
                counter+=1

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
                taken_times.append(taken_time)
                if len(taken_times)>20:
                    taken_times=taken_times[-20:]
                tfa=np.asarray(taken_times)
                tsts=np.mean(tfa)
                start_time = time.time()
                ft=tsts*(self.args["train_epoch"]-epoch-1)
                print(" [*] Epoch %5d (iterations: %10d)finished in %.2f (preprocess %.3f) ETA: %3d:%2d:%2.1f" % (epoch,count,taken_time,ts,ft//3600,ft//60%60,ft%60))
                time_of_epoch=np.append(time_of_epoch,np.asarray([taken_time,ts]))
            if epoch%self.args["lr_decay_term"]==0:
                lr_d_opt3 = lr_d_opt * (0.1 ** (epoch // 100))
                lr_g_opt3 = lr_g_opt * (0.1 ** (epoch // 100))
        self.save(self.args["checkpoint_dir"],self.args["train_epoch"])
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
        a[:, :, 0]=np.clip(a[:, :, 0],a_min=-100000,a_max=20)
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




def seed_net(inp,reuse,depth,chs,a,train=True):
    stddevs = math.sqrt(2.0 / (int(inp.shape[1]) * 16))
    current = tf.layers.conv2d(inp, 16, kernel_size=[inp.shape[1],1], strides=[1,1], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                               data_format="channels_last", name="seed_t", reuse=reuse)
    current = tf.layers.batch_normalization(current, name="bn_seed00", training=train, reuse=reuse)
    current = tf.nn.leaky_relu(current)
    for i in range(depth):
        stddevs = math.sqrt(2.0 / 3.0*chs[i])
        current = tf.layers.batch_normalization(current, name="bn_seed" + str(i), training=train, reuse=reuse)
        current=tf.layers.conv2d(current, chs[i], kernel_size=[1,3], strides=[1,2], padding="VALID",use_bias=False,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                               data_format="channels_last", name="conv_seed" + str(i), reuse=reuse)
        current = tf.nn.leaky_relu(current)
    stddevs = 0.00001
    seed=tf.layers.conv2d(current,chs[-1], kernel_size=[1,1], strides=[1,1], padding="VALID",use_bias=False,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),
                               data_format="channels_last", name="conv_seed_alpha_0", reuse=reuse)
    seed = tf.reshape(seed, [-1, 1, chs[-1], 1])
    seed=seed/tf.sqrt(tf.reduce_sum(tf.pow(seed, 2))+1e-4)
    real=tf.reshape(current,[-1,chs[-1]])
    real = tf.layers.dense(real, units=1,kernel_initializer=tf.truncated_normal_initializer(stddev=0.00001),use_bias=False, name="conv_seed_beta_0", reuse=reuse)

    if a is "A" :
        print(" [*] bottom shape:"+str(seed.shape))
    return seed,real
def generator(current,seed,reuse,depth,chs,d,net_type,train):
    if net_type == "resnet":
        return generator_flatnet_decay(current, seed,reuse, depth, chs, d, 2, train)
    else :
        return generator_flatnet_decay(current, seed,reuse, depth, chs, d, 4, train)
def generator_flatnet_decay(c,seed,reuse,depth,chs,d,ps,train):
    #main process
    current=c
    for i in range(depth):
        connections = current
        if ps == 2:
            ten = block_res(current,seed,chs[i], i, reuse,train=train)
        else:
            ten = block_flat(current,seed,chs[i], i, reuse,train=train)
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
    for i in range(len(d)):
        stddevs = math.sqrt(2.0 / (2 * int(ten.shape[3])))
        ten = tf.layers.conv2d(ten, chs[i+startd], kernel_size=[2, 1], strides=[1, 1], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                           data_format="channels_last", reuse=reuse, name="conv_p" + str(startd+i), dilation_rate=(d[i], 1))

        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                    name="bn_p" + str(startd+i))

        ten = tf.nn.leaky_relu(ten)

        ten2 = tf.layers.conv2d(ten2, chs[i + startd], kernel_size=[2, 1], strides=[1, 1], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                               data_format="channels_last", reuse=reuse, name="conv_f" + str(startd+ i),
                               dilation_rate=(d[i], 1))
        ten2 = tf.layers.batch_normalization(ten2, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="bn_f" + str(startd+ i))
        ten2 = tf.nn.leaky_relu(ten2)
    ten2=tf.nn.tanh(ten2)*3.2
    current=tf.concat([ten,ten2],axis=3)
    return current
def block_res(current,seed,chs,depth,reuses,train=True):
    ten = current
    tenM=[]
    times=3
    res=5

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
    tenS=tf.reshape(seed,[-1,int(ten.shape[2])])
    tenS = tf.layers.dense(tenS,units= int(ten.shape[2]),name="FCS1" + str(depth),reuse=reuses)
    tenS=tf.reshape(tenS,[-1,1,int(ten.shape[2]),1])
    tenS = tf.layers.batch_normalization(tenS, axis=3, training=train, trainable=True, reuse=reuses,
                                         name="bnS1" + str(depth))

    ten=ten*tenS
    for i in range(res):
        stddevs = math.sqrt(2.0 / (7 * int(ten.shape[3])))
        tenA=ten
        ten = tf.layers.conv2d(ten, chs, [1,7], [1,1], padding="SAME",
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                                          data_format="channels_last", reuse=reuses, name="res_convA"+str(i) + str(depth))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                             name="bnA"+str(times+i+1) + str(depth))
        ten = tf.nn.leaky_relu(ten)
        ten=tf.layers.conv2d(ten, chs, [1,7], [1,1], padding="SAME",
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                                          data_format="channels_last", reuse=reuses, name="res_convB"+str(i) + str(depth))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bnB" + str(times + i + 1) + str(depth))
        ten=ten+tenA
        ten = tf.nn.leaky_relu(ten)
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

def block_flat(current,seed,chs,depth,reuses,train=True):
    ten = current
    tenM=[]
    times=3
    res=7

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

    tenS=tf.reshape(seed,[-1,int(ten.shape[2])])
    tenS = tf.layers.dense(tenS,units= int(ten.shape[2]),name="FCS1" + str(depth),reuse=reuses)
    tenS=tf.reshape(tenS,[-1,1,int(ten.shape[2]),1])
    tenS = tf.layers.batch_normalization(tenS, axis=3, training=train, trainable=True, reuse=reuses,
                                         name="bnS1" + str(depth))

    tenS = tf.tanh(tenS)
    ten=ten*tenS

    for i in range(res):
        stddevs = math.sqrt(2.0 / (7 * int(ten.shape[3])))
        tenA=ten
        ten = tf.layers.conv2d(ten, chs, [1,7], [1,2], padding="VALID",
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                                          data_format="channels_last", reuse=reuses, name="res_conv"+str(i) + str(depth))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                             name="bnA"+str(times+i+2) + str(depth))
        ten = tf.nn.leaky_relu(ten)
        stddevs = math.sqrt(2.0 / (8 * int(ten.shape[3])))
        ten=tf.layers.conv2d_transpose(ten, chs, [1,8], [1,2], padding="VALID",
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=stddevs),use_bias=False,
                                          data_format="channels_last", reuse=reuses, name="res_deconv"+str(i) + str(depth))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bnB" + str(times + i + 2) + str(depth))
        ten = tf.nn.leaky_relu(ten)
        ten=ten+tenA
    ten = deconve_with_ps(ten, [1, 2], chs//2, depth, reuses=reuses, name="00")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                         name="bn"+str(times+res+2) + str(depth))
    ten = tf.nn.leaky_relu(ten)
    for i in range(times-1):
        ten+=tenM[times-i-2]
        ten = deconve_with_ps(ten, [1, 2], chs//(2**(i+2)), depth, reuses=reuses, name=str(i+1))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bn"+str(i+3+times+res) + str(depth))
        ten=tf.nn.leaky_relu(ten)
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
    ww = wave.open(to+comment+".wav", 'wb')
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