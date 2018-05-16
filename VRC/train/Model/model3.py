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
        self.args["model_name"] = "wave2wave"
        self.args["version"] = "1.0.0"
        self.args["g_lr"]=2e-4
        self.args["g_b1"] = 0.5
        self.args["g_b2"] = 0.999
        self.args["d_b1"] = 0.5
        self.args["d_b2"] = 0.999
        self.args["train_d_scale"]=1.0
        self.args["train_interval"]=10
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

        ss=int(self.args["input_size"])*2//int(self.args["NFFT"])
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        self.input_size_model=[self.args["batch_size"],ss,self.args["NFFT"],2]
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85)))
        if bool(self.args["debug"]):
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)



    def build_model(self):

        #DとGの学習のしやすさの比率は[ G:D = 1:12000 ]

        #inputs place holder
        #入力
        self.input_model=tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net")
        self.input_model_label=tf.placeholder(tf.float32, self.input_size_model, "inputs_GD-net_target_label")

        #前処理
        self.input_model_p=tf.pow(self.input_model[:,:,:,0],2)+tf.pow(self.input_model[:,:,:,1],2)
        self.input_model_2=tf.reshape(tf.log(self.input_model_p+1e-10),[self.input_size_model[0],self.input_size_model[1],self.input_size_model[2],1])

        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generator_1"):
            self.fake_B_image=generator(self.input_model_2, reuse=False,chs=self.args["channels"],depth=self.args["depth"])

        self.noise = tf.placeholder(tf.float32, [self.args["batch_size"]], "inputs_Noise")

        b_true=self.input_model_label+tf.random_normal(self.input_model_label.shape,0,self.noise)
        #creating discriminator inputs
        #D-netの入力の作成
        self.res1=tf.concat([self.input_model,self.fake_B_image], axis=1)
        self.res2=tf.concat([self.input_model,b_true], axis=1)
        #creating discriminator
        #D-net（判別側)の作成
        with tf.variable_scope("discrim",reuse=tf.AUTO_REUSE):
            self.d_judge_F1=discriminator(self.res1,False)
            self.d_judge_R=discriminator(self.res2,True)

        #getting individual variabloes
        #それぞれの変数取得
        self.g_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_1")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrim")

        #objective-functions of generator
        #G-netの目的関数

        #L1 norm loss
        L1=tf.reduce_mean(tf.abs(self.input_model_label-self.fake_B_image))
        #Gan loss
        DS=tf.reduce_mean(tf.pow(self.d_judge_F1-1,2)*0.5)
        #generator loss
        self.g_loss_1=L1*2+DS

        a=1-self.noise
        b=0
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
        times=in_put.shape[1]//(self.args["input_size"])+1
        if in_put.shape[1]%(self.args["input_size"]*self.args["batch_size"])==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        for t in range(times):
            # Preprocess
            # 前処理

            # Padiing
            # サイズ合わせ
            red=np.zeros((self.args["batch_size"]-1,self.args["input_size"]))
            start_pos=self.args["input_size"]*t+((in_put.shape[1])%self.args["input_size"])
            resorce=np.reshape(in_put[0,max(0,start_pos-self.args["input_size"]):start_pos,0],(1,-1))
            r=max(0,self.args["input_size"]-resorce.shape[1])
            if r>0:
                resorce=np.pad(resorce,((0,0),(r,0)),'constant')
            red=np.append(resorce,red)
            red=red.reshape((self.args["batch_size"],self.args["input_size"]))
            res=np.zeros(self.input_size_model)

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
            res=self.ifft(res[0])*32767

            # chaching results
            # 結果の保存
            res=res.reshape(-1)
            otp=np.append(otp,res)

        h=otp.shape[0]-in_put.shape[1]-1
        if h!=-1:
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

        # initialize training info
        # 学習の情報の初期化
        start_time = time.time()
        DS=1.0
        # loading net
        # 過去の学習データの読み込み
        if self.load(self.checkpoint_dir):
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

            # initializing epoch info
            # エポック情報の初期化
            log_data_g=np.empty(0)
            log_data_d =np.empty(0)

            counter=0

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
                    start_pos=self.args["input_size"]*t+(80000%self.args["input_size"])

                    # getting training data
                    # トレーニングデータの取得
                    target=np.reshape(batch_sounds[:,0,max(0,start_pos-self.args["input_size"]):start_pos],(self.args["batch_size"],-1))
                    resorce=np.reshape(batch_sounds[:,1,max(0,start_pos-self.args["input_size"]):start_pos],(self.args["batch_size"],-1))


                    # preprocessing of input
                    # 入力の前処理

                    # padding
                    # サイズ合わせ
                    r=max(0,self.args["input_size"]-resorce.shape[1])
                    if r>0:
                        resorce=np.pad(resorce,((0,0),(r,0)),'constant')
                    r=max(0,self.args["input_size"]-target.shape[1])
                    if r>0:
                        target=np.pad(target,((0,0),(r,0)),'constant')

                    # FFT
                    # 短時間高速離散フーリエ変換
                    res=np.zeros(self.input_size_model)
                    tar=np.zeros(self.input_size_model)
                    for i in range(self.args["batch_size"]):
                        res[i]=(self.fft(resorce[i]))
                        tar[i]=(self.fft(target[i]))
                    res_t=res.reshape(self.input_size_model)

                    # Update G network
                    # G-netの学習
                    self.sess.run([g_optim_1],feed_dict={ self.input_model:res_t, self.input_model_label:tar })
                    # Update D network (2times)
                    # D-netの学習(2回)
                    if DS>self.args["stop_value"]:
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
                    if self.args["log"] and counter % self.args["train_interval"] == 0:
                        nos = np.random.rand(self.args["batch_size"]) * 0.5
                        hg,hd = self.sess.run([self.g_loss_1,self.d_judge_F1], feed_dict={self.input_model: res_t, self.input_model_label: tar,self.noise:nos })
                        log_data_g=np.append(log_data_g,np.mean(hg))
                        log_data_d=np.append(log_data_d,np.mean(hd))
                        DS=np.mean(hd)
                    counter+=1

            #saving model
            #モデルの保存
            self.save(self.args["checkpoint_dir"], epoch)

            if self.args["test"]:
                #testing
                #テスト
                out_puts,taken_time=self.convert(test.reshape(1,-1,1))
                out_put=(out_puts.astype(np.float32)/32767.0)

                # loss of tesing
                #テストの誤差
                test1=np.mean(np.abs(out_puts-label.reshape(1,-1,1)))

                #hyperdash
                if self.args["hyperdash"]:
                    self.experiment.metric("testG",test1)
                    if self.args["log"]:
                        self.experiment.metric("ScoreD", np.mean(log_data_d))
                        self.experiment.metric("lossG", np.mean(log_data_g))

                #writing epoch-result into tensorboard
                #tensorboardの書き込み
                if self.args["tensorboard"]:
                    rs=self.sess.run(self.tb_results,feed_dict={ self.result:out_put.reshape(1,1,-1),self.g_test_epo:test1})
                    self.writer.add_summary(rs, epoch)

                #saving test result
                #テストの結果の保存
                if os.path.exists(self.args["wave_otp_dir"]):
                    upload(out_puts,self.args["wave_otp_dir"])
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
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.epoch=self.saver
            return True
        else:
            return False
    def fft(self,data):
        rate=16000
        time_song=float(data.shape[0])/rate
        time_unit=1/rate
        start=0
        stop=time_song
        step=(self.args["NFFT"]//2)*time_unit
        time_ruler=np.arange(start,stop,step)
        window=np.hamming(self.args["NFFT"])
        spec=np.zeros([len(time_ruler),self.args["NFFT"],2])
        pos=0
        for fft_index in range(len(time_ruler)):
            frame=data[pos:pos+self.args["NFFT"]]/32767.0
            if len(frame)==self.args["NFFT"]:
                wined=frame*window
                fft_result=np.fft.fft(wined)
                fft_data=np.asarray([fft_result.real,fft_result.imag])
                fft_data=np.transpose(fft_data, (1,0))
                for i in range(len(spec[fft_index])):
                    spec[fft_index][i]=fft_data[i]
                pos+=self.args["NFFT"]//2
        return spec
    def ifft(self,data):
        data=data[:,:,0]+1j*data[:,:,1]
        time_ruler=data.shape[0]
        window=np.hamming(self.args["NFFT"])
        spec=np.zeros([])
        lats = np.zeros([self.args["NFFT"]//2])
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

def discriminator(inp,reuse):
    inputs=tf.cast(inp, tf.float32)
    h1 = tf.nn.leaky_relu(tf.layers.conv2d(inputs, 4,[4,4], strides=[2,2], padding="VALID",data_format="channels_last",name="dis_01",reuse=reuse))
    h1_2 = tf.nn.leaky_relu(tf.layers.conv2d(h1, 8, [1, 1], strides=[1, 1], padding="VALID", data_format="channels_last", name="dis_01_2",reuse=reuse))
    h2 = tf.nn.leaky_relu(tf.layers.conv2d(tf.layers.batch_normalization(h1_2,training=False), 16,[4,4], strides=[2,2], padding="VALID",data_format="channels_last",name="dis_02",reuse=reuse))
    h2_2 = tf.nn.leaky_relu(tf.layers.conv2d(tf.layers.batch_normalization(h2, training=False),32, [4, 4], strides=[1, 1],padding="VALID", data_format="channels_last", name="dis_02_2", reuse=reuse))
    h3 = tf.nn.leaky_relu(tf.layers.conv2d(tf.layers.batch_normalization(h2_2,training=False), 64,[4,4], strides=[2,2], padding="VALID",data_format="channels_last",name="dis_03",reuse=reuse))
    h3_2 = tf.nn.leaky_relu(tf.layers.conv2d(tf.layers.batch_normalization(h3, training=False), 128, [4, 4], strides=[1, 1], padding="VALID",data_format="channels_last", name="dis_03_2", reuse=reuse))
    h4 = tf.nn.leaky_relu(tf.layers.conv2d(tf.layers.batch_normalization(h3_2,training=False), 64,[16,6], strides=1, padding="VALID",data_format="channels_last",name="dis_04",reuse=reuse))
    h4_2 = tf.nn.leaky_relu(tf.layers.conv2d(tf.layers.batch_normalization(h4, training=False), 32, [10, 4], strides=[1, 1], padding="VALID",data_format="channels_last", name="dis_04_2", reuse=reuse))
    h4=tf.reshape(h4_2, [1,-1])
    ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
    return ten

def generator(current_outputs,reuse,depth,chs):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    else:
        assert tf.get_variable_scope().reuse == False
    current=current_outputs
    connections=[ ]
    for i in range(depth):
        connections.append(current)
        current = down_layer(current, chs[i+1])
    for i in range(depth):
        current=up_layer(current,chs[depth-i-1],i!=(depth-1),depth-i-1>2)
        if i!=depth-1:
            current += connections[depth - i - 1]
    return tf.nn.tanh(current)

def up_layer(current,output_shape,bn=True,do=False):
    ten=tf.nn.leaky_relu(current)
    ten=tf.layers.conv2d_transpose(ten, output_shape,kernel_size=4 ,strides=(2,2), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
    if bn:
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
    if do:
        ten=tf.nn.dropout(ten, 0.5)
    return ten
def down_layer(current,output_shape):
    ten=tf.layers.batch_normalization(current,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
    ten=tf.layers.conv2d(ten, output_shape,kernel_size=4 ,strides=(2,2), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
    ten=tf.nn.leaky_relu(ten)
    return ten

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