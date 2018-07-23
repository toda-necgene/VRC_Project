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
import random
from datetime import datetime
import json
import shutil
class Model:
    def __init__(self,path):
        self.args=dict()
        self.args["checkpoint_dir"]="./trained_models"
        self.args["wave_otp_dir"] = "False"
        self.args["train_data_num"]=500
        self.args["batch_size"]=1
        self.args["depth"] =4
        self.args["train_epoch"]=500
        self.args["test"]=True
        self.args["log"] = True
        self.args["tensorboard"]=False
        self.args["hyperdash"]=False
        self.args["stop_argument"]=True
        self.args["stop_value"] = 0.5
        self.args["input_size"] = 8192
        self.args["NFFT"]=128
        self.args["debug"] = False
        self.args["noise"] = False
        self.args["channels"] =[2]
        self.args["strides_g"] = [2,2]
        self.args["strides_d"] = [2,2]
        self.args["filter_g"] = [8,8]
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
        if len(self.args["channels"]) != (self.args['depth'] + 1):
            print(" [!] Channels length and depth+1 must be equal ." + str(len(self.args["channels"])) + "vs" + str(self.args['depth'] + 1))
            self.args["channels"] = [min([4 ** (i + 1) - 2, 254]) for i in range(self.args['depth'] + 1)]
        if self.args["pitch_rate"]==1.0:
            self.args["pitch_rate"] = self.args["pitch_tar"]/self.args["pitch_res"]
            print(" [!] pitch_rate is not found . calculated value : "+str(self.args["pitch_rate"]))
        self.args["SHIFT"] = self.args["NFFT"]//2
        ss=int(self.args["input_size"])*2//int(self.args["NFFT"])
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        self.input_size_model=[self.args["batch_size"],ss,self.args["NFFT"],2]
        print("model input size:"+str(self.input_size_model))
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85)))
        if bool(self.args["debug"]):
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
        self.args["log_file"] = self.args["wave_otp_dir"] + "log.txt"

        if  self.args["wave_otp_dir"] is not "False" and not os.path.exists(self.args["wave_otp_dir"]):
            self.args["wave_otp_dir"]=self.args["wave_otp_dir"]+ self.args["name_save"]+"/"
            os.makedirs(self.args["wave_otp_dir"])
            shutil.copy(path,self.args["wave_otp_dir"]+"info.json")
            self.args["log_file"]=self.args["wave_otp_dir"]+"log.txt"

    def build_model(self):


        #inputs place holder
        #入力
        self.input_model=tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net")
        self.input_model_label=tf.placeholder(tf.float32, self.input_size_model, "inputs_GD-net_target_label")

        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generator_1"):
            self.fake_B_image=generator(self.input_model, reuse=False,chs=self.args["channels"],depth=self.args["depth"],f=self.args["filter_g"],s=self.args["strides_g"])

        self.noise = tf.placeholder(tf.float32, [self.args["batch_size"]], "inputs_Noise")

        b_true_noised=self.input_model_label+tf.random_normal(self.input_model_label.shape,0,self.noise)
        #creating discriminator inputs
        #D-netの入力の作成
        self.res1=tf.concat([self.input_model,self.fake_B_image], axis=1)
        self.res2=tf.concat([self.input_model,b_true_noised], axis=1)
        #creating discriminator
        #D-net（判別側)の作成
        with tf.variable_scope("discrim",reuse=tf.AUTO_REUSE):
            self.d_judge_F1=discriminator(self.res1,False,self.args["filter_d"],self.args["strides_d"],self.args["depth"])
            self.d_judge_R=discriminator(self.res2,True,self.args["filter_d"],self.args["strides_d"],self.args["depth"])

        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_1")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrim")

        #objective-functions of generator
        #G-netの目的関数

        #L1 norm loss
        L1=tf.reduce_mean(tf.pow(self.input_model_label-self.fake_B_image,2)*0.5)
        #Gan loss
        DS=tf.reduce_mean(tf.pow(self.d_judge_F1-1,2)*0.5)
        #generator loss
        self.g_loss_1=L1*10+DS

        a=1-self.noise
        b=-1
        #objective-functions of discriminator
        #D-netの目的関数
        self.d_loss_R = tf.reduce_mean(tf.pow(self.d_judge_R-a,2)*0.5)
        self.d_loss_F = tf.reduce_mean(tf.pow(self.d_judge_F1-b,2)*0.5)
        self.d_loss=tf.reduce_mean(self.d_loss_R+self.d_loss_F)

        #tensorboard functions
        #tensorboard 表示用関数
        self.g_loss_all= tf.summary.scalar("g_loss_All", tf.reduce_mean(self.g_loss_1))
        self.g_loss_gan = tf.summary.scalar("g_loss_gan", tf.reduce_mean(DS))
        self.dscore = tf.summary.scalar("dscore", tf.reduce_mean(self.d_judge_F1))
        self.g_loss_sum_1= tf.summary.merge([self.g_loss_all,self.g_loss_gan,self.dscore])
        self.d_loss_sum = tf.summary.merge([tf.summary.scalar("d_loss", tf.reduce_mean(self.d_loss)),tf.summary.scalar("d_loss_F", tf.reduce_mean(self.d_loss_F))])
        self.result=tf.placeholder(tf.float32, [1,1,160000], name="FB")
        self.fake_B_sum = tf.summary.audio("fake_B", tf.reshape(self.result,[1,160000,1]), 16000, 1)
        self.g_test_epo=tf.placeholder(tf.float32,name="g_test_epoch_end")
        self.g_test_epoch = tf.summary.scalar("g_test_epoch_end", self.g_test_epo)
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
        ipt=self.args["SHIFT"]+self.args["input_size"]
        times=in_put.shape[1]//(self.args["input_size"])+1
        if in_put.shape[1]%(self.args["input_size"]*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        otp2 = np.array([], dtype=np.int16)

        for t in range(times):
            # Preprocess
            # 前処理

            # Padiing
            # サイズ合わせ
            red=np.zeros((self.args["batch_size"]-1,self.args["input_size"]))
            start_pos=self.args["input_size"]*(t+1)
            resorce=np.reshape(in_put[0,max(0,start_pos-ipt):start_pos,0],(1,-1))
            r=max(0,ipt-resorce.shape[1])
            if r>0:
                resorce=np.pad(resorce,((0,0),(r,0)),'reflect')
            red=np.append(resorce,red)
            red=red.reshape((self.args["batch_size"],ipt))
            res = np.zeros(self.input_size_model)

            # changing pitch
            # ピッチ変更
            for i in range(self.args["batch_size"]):
                red[i] = shift(red[i] / 32767.0, self.args["pitch_rate"]).reshape(resorce[i].shape)
                r = self.args["SHIFT"] - res[i].shape[0] % self.args["SHIFT"]
                if r != self.args["SHIFT"]:
                    red[i] = np.pad(red[i], (0, r), "reflect")

            # FFT
            # 短時間高速離散フーリエ変換
            for i in range(self.args["batch_size"]):
                n=self.fft(red[i].reshape(-1))
                res[i]=n

            # running network
            # ネットワーク実行
            res=self.sess.run(self.fake_B_image,feed_dict={ self.input_model:res})


            # Postprocess
            # 後処理

            # IFFT
            # 短時間高速離散逆フーリエ変換
            res=self.ifft(res[0])*32767/2
            # chaching results
            # 結果の保存
            res=res.reshape(-1)
            otp=np.append(otp,res)
        h=otp.shape[0]-in_put.shape[1]-1
        if h>0:
            otp=otp[h:-1]
        return otp.reshape(1,in_put.shape[1],in_put.shape[2]),time.time()-tt



    def train(self):
        self.checkpoint_dir=self.args["checkpoint_dir"]
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
        d_optim = tf.train.AdamOptimizer(lr_d_opt,beta_d_opt,beta_2_d_opt).minimize(self.d_loss, var_list=self.d_vars)
        d_optim_R = tf.train.AdamOptimizer(lr_d_opt, beta_d_opt,beta_2_d_opt).minimize(self.d_loss_R, var_list=self.d_vars)

        # initialize variables
        # 変数の初期化
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # logging
        # ログ出力
        if self.args["tensorboard"]:
            self.writer = tf.summary.FileWriter("./logs/"+self.lod+self.args["name_save"], self.sess.graph)
        if self.args["wave_otp_dir"] != "False" and not os.path.exists(self.args["wave_otp_dir"]):
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
        # loading net
        # 過去の学習データの読み込み
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loading training data directory
        # トレーニングデータの格納ディレクトリの読み込み
        data = glob('./Model/datasets/train/01/*')

        # loading test data
        # テストデータの読み込み
        test=imread('./Model/datasets/test/test.wav')[0:160000]
        label=imread('./Model/datasets/test/label.wav')[0:160000]

        # times of one epoch
        # 回数計算
        batch_idxs = min(len(data), self.args["train_data_num"]) // self.args["batch_size"]

        # hyperdash
        if self.args["hyperdash"]:
            self.experiment=Experiment(self.args["name_save"]+"_G1")
            self.experiment.param("lr_g_opt", lr_g_opt)
            self.experiment.param("beta_g_opt", beta_g_opt)
            self.experiment.param("training_interval", self.args["train_interval"])
            self.experiment.param("learning_rate_scale", tln)

        for epoch in range(0,self.args["train_epoch"]):
            # shuffling training data
            # トレーニングデータのシャッフル
            np.random.shuffle(data)



            ipt = self.args["SHIFT"] + self.args["input_size"]
            counter=0
            print(" [*] Epoch %3d testing" % epoch)
            if self.args["test"]:
                #testing
                #テスト
                out_puts,taken_time=self.convert(test.reshape(1,-1,1))
                out_put=(out_puts.astype(np.float32)/32767.0)

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
                    upload(out_puts,self.args["wave_otp_dir"])

            print(" [*] Epoch %3d started" % epoch)

            for idx in xrange(0, batch_idxs):
                # loading trainig data
                # トレーニングデータの読み込み
                batch_files = data[idx*self.args["batch_size"]:(idx+1)*self.args["batch_size"]]
                batch = np.asarray([(imread(batch_file)) for batch_file in batch_files])
                batch_sounds = np.array(batch).astype(np.int16).reshape(self.args["batch_size"],2,80000)

                # calculating one iteration repetation times
                # 1イテレーション実行回数計算
                times=80000//self.args["input_size"]
                if int(80000)%self.args["input_size"]==0:
                    times-=1
                # shuffle start time
                # 開始タイミングのシャッフル
                time_set=[j for j in range(times)]
                random.shuffle(time_set)



                ti=(batch_idxs*times//self.args["train_interval"])+1

                for t in time_set:

                    # calculating starting position
                    # 開始位置の計算
                    start_pos=self.args["input_size"]*(t+1)

                    # getting training data
                    # トレーニングデータの取得
                    target=np.reshape(batch_sounds[:,0,max(0,start_pos-ipt):start_pos],(self.args["batch_size"],-1))/32767.0
                    resorce=np.reshape(batch_sounds[:,1,max(0,start_pos-ipt):start_pos],(self.args["batch_size"],-1))/32767.0


                    # preprocessing of input
                    # 入力の前処理

                    # padding
                    # サイズ合わせ
                    r=max(0,ipt-resorce.shape[1])
                    if r>0:
                        resorce=np.pad(resorce,((0,0),(r,0)),'reflect')
                    r=max(0,ipt-target.shape[1])
                    if r>0:
                        target=np.pad(target,((0,0),(r,0)),'reflect')


                    # changing pitch
                    # ピッチ変更
                    res = np.zeros(resorce.shape)
                    for i in range(self.args["batch_size"]):
                        res[i] = shift(resorce[i],self.args["pitch_rate"]).reshape(resorce[i].shape)
                        r = self.args["SHIFT"] - res[i].shape[0] % self.args["SHIFT"]
                        if r != self.args["SHIFT"]:
                            res[i] = np.pad(res[i], (0, r), "reflect")

                    # FFT
                    # 短時間高速離散フーリエ変換
                    res_i=np.zeros(self.input_size_model)
                    tar=np.zeros(self.input_size_model)
                    for i in range(self.args["batch_size"]):
                        res_i[i]=(self.fft(res[i]))
                        tar[i]=(self.fft(target[i]))
                    res_t=res_i.reshape(self.input_size_model)

                    # Update G network
                    # G-netの学習
                    self.sess.run([g_optim_1],feed_dict={ self.input_model:res_t, self.input_model_label:tar })
                    # Update D network (2times)
                    # D-netの学習(2回)
                    if self.args["stop_argument"] and DS>self.args["stop_value"]:
                        nos=np.random.rand(self.args["batch_size"])*0.5
                        self.sess.run([d_optim],feed_dict={self.input_model:res_t, self.input_model_label:tar ,self.noise:nos })
                    nos = np.random.rand(self.args["batch_size"]) * 0.5
                    self.sess.run([d_optim_R],feed_dict={self.input_model:res_t, self.input_model_label:tar ,self.noise:nos })

                    # saving tensorboard
                    # tensorboardの保存
                    if self.args["tensorboard"] and counter%self.args["train_interval"]==0:
                        nos = np.random.rand(self.args["batch_size"]) * 0.5
                        hg,hd=self.sess.run([self.g_loss_sum_1,self.d_loss_sum],feed_dict={self.input_model:res_t, self.input_model_label:tar ,self.noise:nos  })
                        self.writer.add_summary(hg, counter//self.args["train_interval"]+ti*epoch)
                        self.writer.add_summary(hd, counter//self.args["train_interval"]+ti*epoch)
                    if self.args["stop_argument"] and counter % self.args["train_interval"] == 0:
                        nos = np.random.rand(self.args["batch_size"]) * 0.5
                        hg,hd = self.sess.run([self.g_loss_1,self.d_judge_F1], feed_dict={self.input_model: res_t, self.input_model_label: tar,self.noise:nos })
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
                        f.write("%6d,%5.5f,%5.5f,%10.5f" % (epoch,float(np.mean(log_data_d)),float(np.mean(log_data_g)),float(test1)))
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

            #console outputs
            taken_time = time.time() - start_time
            print(" [*] Epoch %5d finished in %.2f" % (epoch,taken_time))
            start_time=time.time()

        #hyperdash
        if self.args["hyperdash"]:
            self.experiment.end()
        print(" [*] Finished!! on "+nowtime())

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
        time_ruler=data.shape[0]//self.args["SHIFT"]-1
        window=np.hamming(self.args["NFFT"])
        spec=np.zeros([time_ruler,self.args["NFFT"],2])
        pos=0
        for fft_index in range(time_ruler):
            frame=data[pos:pos+self.args["NFFT"]]
            wined=frame*window
            fft_result=np.fft.fft(wined)
            c = np.log(np.power(fft_result.real, 2) + np.power(fft_result.imag, 2) + 1e-24)
            d = np.arctan2(fft_result.imag, fft_result.real)
            fft_data=np.asarray([c,d])
            fft_data=np.transpose(fft_data, (1,0))
            for i in range(len(spec[fft_index])):
                spec[fft_index][i]=fft_data[i]
            pos+=self.args["SHIFT"]
        return spec
    def ifft(self,data):
        p = np.sqrt(np.exp(data[:, :, 0]))
        r = p * (np.cos(data[:, :, 1]))
        i = p * (np.sin(data[:, :, 1]))
        dds = np.concatenate((r.reshape(r.shape[0], r.shape[1], 1), i.reshape(i.shape[0], i.shape[1], 1)), 2)
        data=dds[:,:,0]+1j*dds[:,:,1]
        time_ruler=data.shape[0]
        window=np.hamming(self.args["NFFT"])
        spec=np.zeros([])
        lats = np.zeros(self.args["SHIFT"])
        pos=0
        for _ in range(time_ruler):
            frame=data[pos]
            fft_result=np.fft.ifft(frame)
            fft_data=fft_result.real
            fft_data/=window
            v = lats + fft_data[:self.args["NFFT"]//2]
            lats = fft_data[self.args["NFFT"]//2:]
            spec=np.append(spec,v)
            pos+=1
        return spec[1:]

#model architectures

def discriminator(inp,reuse,f,s,depth):
    current=tf.cast(inp, tf.float32)
    for i in range(depth):
        current = down_layer(current, 2 ** (i + 1), f, s)
        current = down_layer(current, 2 ** (i + 1), f, s)
    h4=tf.reshape(current, [1,-1])
    print("Dence shape:" + str(h4.shape))
    ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
    return ten

def generator(current_outputs,reuse,depth,chs,f,s):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    else:
        assert tf.get_variable_scope().reuse == False
    current=current_outputs
    connections=[ ]
    for i in range(depth):
        connections.append(current)
        current = down_layer(current, chs[i+1],f,s)
    print("shape of structure:"+str([[int(c.shape[0]),int(c.shape[1]),int(c.shape[2]),int(c.shape[3])] for c in connections]))
    for i in range(depth):
        current=up_layer(current,chs[depth-i-1],f,s,i!=(depth-1),depth-i-1>2)
        if i!=depth-1:
            current += connections[depth - i -1]
    return current

def up_layer(current,output_shape,f,s,bn=True,do=False):
    ten=tf.nn.leaky_relu(current)
    ten=tf.layers.conv2d_transpose(ten, output_shape,kernel_size=f ,strides=s, padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
    if bn:
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
    if do:
        ten=tf.nn.dropout(ten, 0.5)
    return ten
def down_layer(current,output_shape,f,s):
    ten=tf.layers.batch_normalization(current,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
    ten=tf.layers.conv2d(ten, output_shape,kernel_size=f ,strides=s, padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
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
    pulus=int(term_s*speed)
    data_s=datanum.reshape(-1)
    spec=np.zeros(1)
    ifs=np.zeros(pulus//2)
    for i_s in np.arange(0.0,data_s.shape[0],pulus):
        dd=data_s[int(i_s):int(i_s+term_s)]
        fade = min(int(pulus/2),dd.shape[0])
        ds_in= np.linspace(0,1,fade)
        ds_out =  np.linspace(1,0,fade)
        stock=dd[:fade]
        dd[:fade] = dd[:fade] * ds_in
        if  i_s!=0:
            dd[:fade]+=ifs[:fade]
        else :
            dd[:fade] += stock * np.linspace(1, 0, fade)
        ifs=dd[-fade:]*ds_out
        if i_s+pulus>=data_s.shape[0]:
            spec = np.append(spec, dd)
        else:
            spec=np.append(spec,dd[:-fade])
    return spec[1:]

def nowtime():
    return datetime.now().strftime("%Y_%m_%d %H_%M_%S")

def upload(voice,to):
    voiced=voice.astype(np.int16)
    p=pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    ww = wave.open(to+nowtime()+".wav", 'wb')
    ww.setnchannels(1)
    ww.setsampwidth(p.get_sample_size(FORMAT))
    ww.setframerate(16000)
    ww.writeframes(voiced.reshape(-1).tobytes())
    ww.close()
    p.terminate()

def imread(path):
    wf = wave.open(path, 'rb')
    ans=np.empty([],dtype=np.int16)

    bb=wf.readframes(1024)
    while bb != b'':
        ans=np.append(ans,np.frombuffer(bb,"int16"))
        bb=wf.readframes(1024)
    wf.close()
    i=160000-ans.shape[0]
    if i>0:
        ans=np.pad(ans, (0,i), "constant")
    else:
        ans=ans[0:160000]
    return ans