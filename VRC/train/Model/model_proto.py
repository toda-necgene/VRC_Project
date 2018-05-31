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
        self.args["dropbox"]="False"
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
        ss=int(self.args["input_size"])*2//int(self.args["NFFT"])+1
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        self.input_size_model=[self.args["batch_size"],ss,self.args["NFFT"],2]
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
        self.input_model=tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net")
        self.input_model_label=tf.placeholder(tf.float32, self.input_size_model, "inputs_GD-net_target_label")
        self.training=tf.placeholder(tf.float32,[1],name="Training")
        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generator_1"):
            self.ifs=generator(tf.reshape(self.input_model,self.input_size_model), reuse=False,chs=self.args["G_channel"],depth=self.args["depth"],f=self.args["filter_g"],s=self.args["strides_g"],rate=self.training)
            self.fake_B_image=self.ifs[-1]
        self.noise = tf.placeholder(tf.float32, [self.args["batch_size"]], "inputs_Noise")

        b_true_noised=self.input_model_label+tf.random_normal(self.input_model_label.shape,0,self.noise[0])
        #creating discriminator inputs
        #D-netの入力の作成
        self.res3=[]
        for c in self.ifs:
           self.res3.append(tf.concat([self.input_model,c], axis=1))
        self.res1=tf.concat([self.input_model,self.fake_B_image], axis=1)

        self.res2=tf.concat([self.input_model,b_true_noised], axis=1)
        #creating discriminator
        #D-net（判別側)の作成
        self.d_judge_F_logits=[]
        with tf.variable_scope("discrim",reuse=tf.AUTO_REUSE):
            self.d_judge_F1,self.d_judge_F1_logits=discriminator(self.res1,False,self.args["filter_d"],self.args["strides_d"],self.args["d_depth"],self.args["D_channels"])
            for g  in self.res3:
                self.d_judge_F_logits.append(discriminator(g, True, self.args["filter_d"],
                                                                        self.args["strides_d"], self.args["d_depth"],
                                                                        self.args["D_channels"]))

            self.d_judge_R,self.d_judge_R_logits=discriminator(self.res2,True,self.args["filter_d"],self.args["strides_d"],self.args["d_depth"],self.args["D_channels"])

        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_1")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrim")


        c=1.0-self.noise
        a=tf.reshape(c,self.d_judge_F1_logits.shape)
        #objective-functions of discriminator
        #D-netの目的関数
        self.d_loss_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=a,logits=self.d_judge_R_logits))
        self.d_loss_F = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(self.d_judge_F1_logits.shape),logits=self.d_judge_F1_logits))

        self.d_loss_Fs=[]
        for f in self.d_judge_F_logits:
            self.d_loss_Fs.append(tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros(self.d_judge_F1_logits.shape),
                                                        logits=f[1])))

        self.d_loss=(self.d_loss_F+self.d_loss_R+tf.reduce_mean(self.d_loss_Fs))/3.0

        # objective-functions of generator
        # G-netの目的関数

        # L1 norm loss
        drml = []
        for f in self.ifs:
            a=tf.clip_by_value(tf.abs(self.input_model_label[:,:,:,0] - f[:,:,:,0])/60,0.0,1.0)
            b=tf.clip_by_value(tf.abs(self.input_model_label[:,:,:,1] - f[:,:,:,1])/3.141593/20,0.0,1.0)
            drml.append(a+b)
        L1 = tf.reduce_mean(drml)
        # Gan loss
        drm=[]
        for f in self.d_judge_F_logits:
            drm.append(-tf.log(f[0] + 1e-8))
        DSs=tf.reduce_mean(drm)
        # generator loss
        self.g_loss_1 = L1 * self.args["weight_Norm"]  + DSs

        #BN_UPDATE
        self.update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #tensorboard functions
        #tensorboard 表示用関数
        self.g_loss_all= tf.summary.scalar("g_loss_All", tf.reduce_mean(self.g_loss_1))
        self.g_loss_gan = tf.summary.scalar("g_loss_gan", tf.reduce_mean(DSs))
        self.dscore = tf.summary.scalar("dscore", tf.reduce_mean(self.d_judge_F1))
        self.g_loss_sum_1= tf.summary.merge([self.g_loss_all,self.g_loss_gan,self.dscore])
        self.d_loss_sum = tf.summary.scalar("d_loss", tf.reduce_mean(self.d_loss))
        self.result=tf.placeholder(tf.float32, [1,1,160000], name="FB")
        self.fake_B_sum = tf.summary.audio("fake_B", tf.reshape(self.result,[1,160000,1]), 16000, 1)
        self.g_test_epo=tf.placeholder(tf.float32,name="g_test_epoch_end")
        self.g_test_epoch = tf.summary.merge([tf.summary.scalar("g_test_epoch_end", self.g_test_epo)])
        self.tb_results=tf.summary.merge([self.fake_B_sum,self.g_test_epoch])

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
        res2 = np.asarray([[[]]], dtype=np.float32)
        rss=np.zeros([self.input_size_model[2]//2],dtype=np.float64)
        for t in range(times):
            # Preprocess
            # 前処理

            # Padiing
            # サイズ合わせ
            red=np.zeros((self.args["batch_size"]-1,ipt))
            start_pos=self.args["input_size"]*t+(in_put.shape[1]%self.args["input_size"])
            resorce=np.reshape(in_put[0,max(0,start_pos-ipt):start_pos,0],(1,-1))
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
            scales = np.sqrt(np.var(res[0, :, :, 0], axis=1) + 1e-64)
            means = np.mean(res[0, :, :, 0], axis=1)
            mms=np.tile(np.reshape(scales,(-1,1)),(1,self.args["NFFT"]))
            scl = np.tile(np.reshape(means, (-1, 1)), (1, self.args["NFFT"]))
            res[0,:,:,0]=res[0,:,:,0]*mms-scl
            # running network
            # ネットワーク実行
            res=self.sess.run(self.fake_B_image,feed_dict={ self.input_model:res,self.training:np.asarray([1.0])})
            # resas = np.append(resas, res[0])
            a=res[0].copy()
            scales2 = np.sqrt(np.var(a[:, :, 0], axis=1) + 1e-64)
            means2 = np.mean(a[:, :, 0], axis=1)
            ss = np.tile((scales / scales2).reshape(-1, 1), (1, self.args["NFFT"]))
            sm = np.tile((means - means2).reshape(-1, 1), (1, self.args["NFFT"]))
            c = a[:, :, 0]

            c = c + sm
            c = c * ss
            a[:, :, 0] = c
            # a = mask_scale(a, 250, 770, 10)
            a = mask_const(a, 250, 770, 8)
            # a = mask_scale(a, 250, 770, -10)
            res2 = np.append(res2, a[ 2:, :, :])


            # Postprocess
            # 後処理

            # IFFT
            # 短時間高速離散逆フーリエ変換
            res,rss=self.ifft(a,rss)
            # 変換後処理
            bsd = filter_mean(res)
            res = filter_clip(bsd, f=0.5)
            res=res*32767/2
            # chaching results
            # 結果の保存
            res=res.reshape(-1).astype(np.int16)
            otp=np.append(otp,res[-8192:])
        h=otp.shape[0]-in_put.shape[1]
        if h>0:
            otp=otp[h:]

        return otp.reshape(1,in_put.shape[1],in_put.shape[2]),time.time()-tt,res2



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
        g_optim_1 =tf.train.AdamOptimizer(lr_g_opt,beta_g_opt,beta_2_g_opt).minimize(self.g_loss_1, var_list=self.g_vars_1)
        d_optim = tf.train.AdamOptimizer(lr_d_opt,beta_d_opt,beta_2_d_opt).minimize(self.d_loss_F, var_list=self.d_vars)
        d_optim_R = tf.train.AdamOptimizer(lr_d_opt, beta_d_opt,beta_2_d_opt).minimize(self.d_loss_R, var_list=self.d_vars)
        d_optim2 = tf.train.AdamOptimizer(lr_d_opt, beta_d_opt, beta_2_d_opt).minimize(self.d_loss,
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
        DS=1.0
        log_data_g = np.empty(0)
        log_data_d = np.empty(0)
        ti=0
        # loading net
        # 過去の学習データの読み込み
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loading training data directory
        # トレーニングデータの格納ディレクトリの読み込み
        data = glob('./Model/datasets/train/Source_data/*')

        # loading test data
        # テストデータの読み込み
        test=isread('./Model/datasets/test/test.wav')[0:160000]
        label=isread('./Model/datasets/test/label.wav')[0:160000]

        # times of one epoch
        # 回数計算
        train_data_num = min(len(data), self.args["train_data_num"])
        batch_idxs = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]


        #　学習データをメモリに乗っける
        batch_files2 = data[:train_data_num]
        batch_files = ["./Model/datasets/train/Answer_data/" + os.path.basename(bn) for bn in data[0:train_data_num]]
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
            test1=0.0
            ts = 0.0
            ipt = self.input_size_model[1]
            counter=0
            if self.args["test"]:
                print(" [*] Epoch %3d testing" % epoch)
                #testing
                #テスト
                out_puts,taken_time_test,im=self.convert(test.reshape(1,-1,1))
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
                    rs=self.sess.run(self.tb_results,feed_dict={ self.result:out_put.reshape(1,1,-1),self.g_test_epo:test1})
                    self.writer.add_summary(rs, epoch)

                #saving test result
                #テストの結果の保存
                if os.path.exists(self.args["wave_otp_dir"]):
                    im=im.reshape([-1,self.args["NFFT"],2])
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
                batch_sounds2= np.asarray([batch_sounds_t[ind] for ind in index_list[st:st+self.args["batch_size"]]])
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
                    r=self.input_size_model[1]-res_t.shape[1]
                    if r>0:
                        res_t=np.pad(res_t,((0,0),(r,0),(0,0),(0,0)),"constant")
                        tar = np.pad(tar,((0,0),(r,0),(0,0),(0,0)),"constant")

                    # Update G network
                    # G-netの学習
                    rate=1.0-0.5**(epoch//50+1)
                    self.sess.run([g_optim_1,self.update_ops],feed_dict={ self.input_model:res_t, self.input_model_label:tar,self.training:np.asarray([rate])})

                    # Update D network (2times)
                    # D-netの学習(2回)
                    if self.args["stop_argument"]:
                        if DS>self.args["stop_value"] :
                            nos=np.random.rand(self.args["batch_size"])*0.5
                            self.sess.run([d_optim],feed_dict={self.input_model:res_t, self.input_model_label:tar ,self.noise:nos ,self.training:np.asarray([rate])})
                        nos = np.random.rand(self.args["batch_size"]) * 0.5
                        self.sess.run([d_optim_R],feed_dict={self.input_model: res_t, self.input_model_label: tar, self.noise: nos,self.training:np.asarray([rate])})
                    else :
                        nos = np.random.rand(self.args["batch_size"]) * 0.5
                        self.sess.run([d_optim2],
                                      feed_dict={self.input_model: res_t, self.input_model_label: tar, self.noise: nos,self.training:np.asarray([rate])})

                        # saving tensorboard
                    # tensorboardの保存
                    if self.args["tensorboard"] and (counter+ti*epoch)%self.args["train_interval"]==0:
                        nos = np.random.rand(self.args["batch_size"]) * 0.5
                        hg,hd=self.sess.run([self.g_loss_sum_1,self.d_loss_sum],feed_dict={self.input_model:res_t, self.input_model_label:tar ,self.noise:nos ,self.training:np.asarray([1.0]) })
                        self.writer.add_summary(hg, counter+ti*epoch)
                        self.writer.add_summary(hd, counter+ti*epoch)
                    if self.args["stop_argument"] and (counter+ti*epoch) % self.args["train_interval"] == 0:
                        nos = np.random.rand(self.args["batch_size"]) * 0.5
                        hg,hd = self.sess.run([self.g_loss_1,self.d_judge_F1], feed_dict={self.input_model: res_t, self.input_model_label: tar,self.noise:nos ,self.training:np.asarray([1.0])})
                        log_data_g=np.append(log_data_g,np.mean(hg))
                        log_data_d=np.append(log_data_d,np.mean(hd))
                        DS=np.mean(hd)
                    counter+=1

            #saving model
            #モデルの保存
            self.save(self.args["checkpoint_dir"], epoch)


            if self.args["log"] and self.args["wave_otp_dir"]!="False":
                with open(self.args["log_file"],"a") as f:
                    if self.args["stop_argument"]:
                        f.write("\n %6d,%5.5f,%5.5f,%10.5f" % (
                        epoch, float(np.mean(log_data_d)), float(np.mean(log_data_g)), float(test1)))
                    else:
                        f.write("%6d,%10.5f" % (epoch, float(test1)))
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
                start_time = time.time()
                ft=taken_time*(self.args["train_epoch"]-epoch-1)
                print(" [*] Epoch %5d (iterations: %10d)finished in %.2f (preprocess %.3f) ETA: %3d:%2d:%2.1f" % (epoch,count,taken_time,ts,ft//3600,ft//60%60,ft%60))
                time_of_epoch=np.append(time_of_epoch,np.asarray([taken_time,ts]))


        print(" [*] Finished!! in "+ str(np.sum(time_of_epoch[::2])))

        if self.args["log"] and self.args["wave_otp_dir"] != "False":
            with open(self.args["log_file"], "a") as f:
                f.write("\n \n time on 1 epoch:" +str(np.mean(time_of_epoch[::2]))+" preprocess :"+str(np.mean(time_of_epoch[1::2])))
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
        a=np.clip(data[:, :, 0],a_min=-100000,a_max=88)
        sss=np.exp(a)
        p = np.sqrt(sss)
        r = p * (np.cos(data[:, :, 1]))
        i = p * (np.sin(data[:, :, 1]))
        dds = np.concatenate((r.reshape(r.shape[0], r.shape[1], 1), i.reshape(i.shape[0], i.shape[1], 1)), 2)
        data=dds[:,:,0]+1j*dds[:,:,1]
        window=np.hamming(self.args["NFFT"])
        if self.args["cupy"]:
            eep = cupy.asarray(data, dtype=cupy.complex128)
            fft_se = cupy.fft.ifft(eep,n=self.args["NFFT"], axis=1)
            fft_s = cupy.asnumpy(fft_se)
        else:
            fft_s = np.fft.ifft(data,n=self.args["NFFT"], axis=1)

        fft_data = fft_s.real
        fft_data[:]/=window
        v = fft_data[:, :self.args["NFFT"]// 2]
        reds = fft_data[-1, self.args["NFFT"] // 2:].copy()
        lats = np.roll(fft_data[:, self.args["NFFT"] // 2:], 1, axis=0 )
        lats[0, :]=redi
        spec = np.reshape(v + lats, (-1))
        return spec,reds


    def test(self):
        self.load()
        with open(self.args["log_file"], "w") as f:
            f.write("name       :low-score    :meanscore")
        target=imread(self.args["test_dir"]+"/target.wav").reshape(1,-1,1)
        list_data=glob(self.args["test_dir"]+"/*")
        ipt = self.args["SHIFT"] + self.args["input_size"]
        times = target.shape[0] // (self.args["input_size"]) + 1
        if target.shape[0] % (self.args["input_size"] * self.args["batch_size"]) == 0:
            times -= 1

        for i in list_data:
            otp = np.array([], dtype=np.float32)
            da=imread(i).reshape(1,-1,1)
            for t in range(times):
                # Preprocess
                # 前処理
                # Padiing
                # サイズ合わせ
                red = np.zeros((self.args["batch_size"] - 1, ipt))
                start_pos = self.args["input_size"] * (t + 1)
                resorce = np.reshape(da[0, max(0, start_pos - ipt):start_pos, 0], (1, -1))
                r = max(0, ipt - resorce.shape[1])
                if r > 0:
                    resorce = np.pad(resorce, ((0, 0), (r, 0)), 'reflect')
                red = np.append(resorce, red)
                red = red.reshape((self.args["batch_size"], ipt))
                targ = np.reshape(target[0, max(0, start_pos - ipt):start_pos, 0], (1, -1))
                r = max(0, ipt - targ.shape[1])
                if r > 0:
                    targ = np.pad(targ, ((0, 0), (r, 0)), 'reflect')

                res = np.zeros(self.input_size_model)
                tar = np.zeros(self.input_size_model)
                # FFT
                # 短時間高速離散フーリエ変換
                for imr in range(self.args["batch_size"]):
                    n = self.fft(resorce[imr].reshape(-1)/ 32767.0)
                    tar[imr] = (self.fft(targ[imr].reshape(-1))/ 32767.0)
                    res[imr] = n
                nos=np.zeros([1])
                score=self.sess.run(self.d_judge_R,feed_dict={self.input_model:tar, self.input_model_label:res ,self.noise:nos,self.training:np.asarray([1.0])})
                otp=np.append(otp,score)
            with open(self.args["log_file"], "a") as f:
                f.write("\n %10s : %5.5f  :%5.5f" % (i,float(np.min(otp)),float(np.mean(otp))))
                f.flush()


def discriminator(inp,reuse,f,s,depth,chs):
    current=tf.cast(inp, tf.float32)
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=f, strides=s, padding="VALID",
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        current = tf.nn.leaky_relu(ten)
    h4=tf.reshape(current, [current.shape[0],-1])
    ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
    return tf.nn.sigmoid(ten),ten
def generator(current_outputs,reuse,depth,chs,f,s,rate):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    else:
        assert tf.get_variable_scope().reuse == False
    current=current_outputs
    ctr=[]
    output_shape=int(current.shape[3])
    #main process
    for i in range(depth):
        connections = current
        ten=block(current,output_shape,chs,f,s,i)
        # if i>1:
            # ten=tf.nn.dropout(ten,rate[0]-((1-rate[0])*0.5*(i/depth)))
        current = ten + connections
        ctr.append(current)
    return ctr
def block(current,output_shape,chs,f,s,depth):
    ten=current
    ten = tf.layers.batch_normalization(ten, axis=3, training=True,trainable=True,
                                        gamma_initializer=tf.ones_initializer())
    
    ten = tf.layers.conv2d(ten, chs, kernel_size=f, strides=s, padding="VALID",
                           kernel_initializer=tf.contrib.layers.xavier_initializer(), data_format="channels_last")
    ten = tf.nn.leaky_relu(ten)

    ten = tf.layers.batch_normalization(ten, axis=3, training=True, trainable=True,
                                        gamma_initializer=tf.ones_initializer())

    ten = tf.layers.conv2d_transpose(ten, output_shape, kernel_size=f, strides=s, padding="VALID",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     data_format="channels_last")

    # ten = tf.layers.batch_normalization(ten, axis=3, training=True, trainable=True,
    #                                     gamma_initializer=tf.ones_initializer())

    #Add_filter_Layer
    # with tf.variable_scope("add_layer_Layer_"+str(depth)):
    #     sc=ten.shape[1:]
    #     fig=tf.reshape(tf.get_variable("add_filter",sc,tf.float32,tf.ones_initializer(),trainable=True),[1,sc[0],sc[1],sc[2]])
    #     figs=tf.tile(fig,(ten.shape[0],1,1,1))
    #     ten = ten + figs

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