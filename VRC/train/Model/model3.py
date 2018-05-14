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
class Model:
    def __init__(self,debug):
        self.batch_size=1
        self.depth=6
        self.train_epoch=500
        self.input_ch=1
        self.NFFT=64
        self.tensorboard = True
        self.hyperdash =True
        self.wave_output="z://waves/"
        self.input_size=[self.batch_size,8192,1]
        self.input_size_model=[self.batch_size,256,64,2]
        self.dataset_name="wave2wave_1.4.0"
        self.output_size=[self.batch_size,8192,1]
        self.CHANNELS=[min([2**(i+1),128]) for i in range(self.depth+1)]
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85)))
        if debug:
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)



    def build_model(self):

        #inputs place holder
        #入力
        self.input_model=tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net")
        self.input_model_label=tf.placeholder(tf.float32, self.input_size_model, "inputs_GD-net_target_label")

        #creating generator
        #G-net（生成側）の作成
        with tf.variable_scope("generator_1"):
            self.fake_B_image=generator(tf.reshape(self.input_model,[self.batch_size,256,64,1]), reuse=False,chs=self.CHANNELS,depth=self.depth)

        #creating discriminator inputs
        #D-netの入力の作成
        self.res1=tf.concat([self.input_model,self.fake_B_image], axis=1)
        self.res2=tf.concat([self.input_model,self.input_model_label], axis=1)
        #creating discriminator
        #D-net（判別側)の作成
        with tf.variable_scope("discrim",reuse=tf.AUTO_REUSE):
            self.d_judge_F1,self.d_judge_F1_logits=discriminator(self.res1,False)
            self.d_judge_R,self.d_judge_R_logits=discriminator(self.res2,True)

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
        self.g_loss_1=L1+DS

        #objective-functions of discriminator
        #D-netの目的関数
        self.d_loss_R = tf.reduce_mean(tf.pow(self.d_judge_R-1,2)*0.5)
        self.d_loss_F = tf.reduce_mean(tf.pow(self.d_judge_F1,2)*0.5)
        self.d_loss=tf.reduce_mean(self.d_loss_R+self.d_loss_F)

        #tensorboard functions
        #tensorboard 表示用関数
        self.g_loss_sum_1 = tf.summary.scalar("g_loss", tf.reduce_mean(self.g_loss_1))
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
        times=in_put.shape[1]//(self.output_size[1])+1
        if in_put.shape[1]%(self.output_size[1]*self.batch_size)==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        for t in range(times):
            # Preprocess
            # 前処理

            # Padiing
            # サイズ合わせ
            red=np.zeros((self.input_size[0]-1,self.input_size[1],self.input_size[2]))
            start_pos=self.output_size[1]*t+((in_put.shape[1])%self.output_size[1])
            resorce=np.reshape(in_put[0,max(0,start_pos-self.input_size[1]):start_pos,0],(1,-1))
            r=max(0,self.input_size[1]-resorce.shape[1])
            if r>0:
                resorce=np.pad(resorce,((0,0),(r,0)),'constant')
            red=np.append(resorce,red)
            red=red.reshape((self.input_size[0],self.input_size[1],self.input_size[2]))
            res=np.zeros([self.batch_size,256,64,2])

            # FFT
            # 短時間高速離散フーリエ変換
            for i in range(self.batch_size):
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



    def train(self,args):
        self.checkpoint_dir=args.checkpoint_dir
        # setting paramaters
        # パラメータ
        lr_g_opt=2e-4
        beta_g_opt=0.5
        beta_2_g_opt=0.999
        lr_d_opt=2e-4
        beta_d_opt=0.1

        # naming output-directory
        # 出力ディレクトリ
        self.lod="[glr="+str(lr_g_opt)+",gb="+str(beta_g_opt)+",dlr="+str(lr_d_opt)+",db="+str(beta_d_opt)+"]"
        g_optim_1 =tf.train.AdamOptimizer(lr_g_opt,beta_g_opt,beta_2_g_opt).minimize(self.g_loss_1, var_list=self.g_vars_1)
        d_optim = tf.train.AdamOptimizer(lr_d_opt,beta_d_opt).minimize(self.d_loss, var_list=self.d_vars)

        # initialize variables
        # 変数の初期化
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # logging
        # ログ出力
        if self.tensorboard:
            self.writer = tf.summary.FileWriter("./logs/"+self.lod+self.dataset_name, self.sess.graph)

        # initialize training info
        # 学習の情報の初期化
        start_time = time.time()

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
        batch_idxs = min(len(data), args.train_size) // self.batch_size

        # hyperdash
        if self.hyperdash:
            self.experiment=Experiment(self.dataset_name+"_G1")
            self.experiment.param("lr_g_opt", lr_g_opt)
            self.experiment.param("beta_g_opt", beta_g_opt)
            self.experiment.param("lr_d_opt", lr_d_opt)
            self.experiment.param("beta_d_opt", beta_d_opt)
            self.experiment.param("depth", self.depth)

        for epoch in range(0,self.train_epoch):
            # shuffling training data
            # トレーニングデータのシャッフル
            np.random.shuffle(data)

            # initializing epoch info
            # エポック情報の初期化
            counter=0

            print(" [*] Epoch %3d started" % epoch)

            for idx in xrange(0, batch_idxs):
                # loading trainig data
                # トレーニングデータの読み込み
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = np.asarray([(imread(batch_file)) for batch_file in batch_files])
                batch_sounds = np.array(batch).astype(np.int16).reshape(self.batch_size,2,80000)

                # calculating one iteration repetation times
                # 1イテレーション実行回数計算
                times=80000//self.output_size[1]
                if int(80000)%self.output_size[1]==0:
                    times-=1
                # shuffle start time
                # 開始タイミングのシャッフル
                time_set=[j for j in range(times)]
                random.shuffle(time_set)


                # tensorboard interval
                # tensorboard保存のインターバル
                tb_interval=10
                ti=(batch_idxs*times//tb_interval)

                for t in time_set:

                    # calculating starting position
                    # 開始位置の計算
                    start_pos=self.output_size[1]*t+(80000%self.output_size[1])

                    # getting training data
                    # トレーニングデータの取得
                    target=np.reshape(batch_sounds[:,0,max(0,start_pos-self.output_size[1]):start_pos],(self.batch_size,-1))
                    resorce=np.reshape(batch_sounds[:,1,max(0,start_pos-self.input_size[1]):start_pos],(self.batch_size,-1))


                    # preprocessing of input
                    # 入力の前処理

                    # padding
                    # サイズ合わせ
                    r=max(0,self.input_size[1]-resorce.shape[1])
                    if r>0:
                        resorce=np.pad(resorce,((0,0),(r,0)),'constant')
                    r=max(0,self.output_size[1]-target.shape[1])
                    if r>0:
                        target=np.pad(target,((0,0),(r,0)),'constant')

                    # FFT
                    # 短時間高速離散フーリエ変換
                    res=np.zeros([self.batch_size,256,64,2])
                    tar=np.zeros([self.batch_size,256,64,2])
                    for i in range(self.batch_size):
                        res[i]=(self.fft(resorce[i]))
                        tar[i]=(self.fft(target[i]))
                    res_t=res.reshape([self.batch_size,256,64,2])

                    # Update G network
                    # G-netの学習
                    self.sess.run([g_optim_1],feed_dict={ self.input_model:res_t, self.input_model_label:tar })
                    # Update D network (2times)
                    # D-netの学習(2回)
                    self.sess.run([d_optim],feed_dict={self.input_model:res_t, self.input_model_label:tar  })
                    self.sess.run([d_optim],feed_dict={self.input_model:res_t, self.input_model_label:tar })

                    # saving tensorboard
                    # tensorboardの保存
                    if self.tensorboard and counter%tb_interval==0:
                        hg=self.sess.run(self.g_loss_sum_1,feed_dict={self.input_model:res_t, self.input_model_label:tar  })
                        hd = self.sess.run(self.d_loss_sum,feed_dict={self.input_model: res_t, self.input_model_label: tar})
                        self.writer.add_summary(hg, counter//tb_interval+ti*epoch)
                        self.writer.add_summary(hd, counter//tb_interval+ti*epoch)

                    counter+=1

            #saving model
            #モデルの保存
            self.save(args.checkpoint_dir, epoch)


            #testing
            #テスト
            out_puts,taken_time=self.convert(test.reshape(1,-1,1))
            out_put=(out_puts.astype(np.float32)/32767.0)

            # loss of tesing
            #テストの誤差
            test1=np.mean(np.abs(out_puts-label.reshape(1,-1,1)))

            #hyperdash
            if self.hyperdash:
                self.experiment.metric("testG",test1)

            #writing epoch-result into tensorboard
            #tensorboardの書き込み
            if self.tensorboard:
                rs=self.sess.run(self.tb_results,feed_dict={ self.result:out_put.reshape(1,1,-1),self.g_test_epo:test1})
                self.writer.add_summary(rs, epoch)

            #saving test result
            #テストの結果の保存
            if self.wave_output is not "FALSE":
                upload(out_puts,self.wave_output)

            #console outputs
            taken_time = time.time() - start_time
            print(" [*] Epoch %5d finished in %.2f" % (epoch,taken_time))
            start_time=time.time()

        #hyperdash
        if self.hyperdash:
            self.experiment.end()
        print(" [*] Finished!! on "+nowtime())

    def save(self, checkpoint_dir, step):
        model_name = "wave2wave.model"
        model_dir = "%s_%s_%s layers" % (self.dataset_name, self.batch_size,self.depth)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s layers" % (self.dataset_name, self.batch_size,self.depth)
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
        step=(self.NFFT//2)*time_unit
        time_ruler=np.arange(start,stop,step)
        window=np.hamming(self.NFFT)
        spec=np.zeros([len(time_ruler),self.NFFT,2])
        pos=0
        for fft_index in range(len(time_ruler)):
            frame=data[pos:pos+self.NFFT]/32767.0
            if len(frame)==self.NFFT:
                wined=frame*window
                fft_result=np.fft.fft(wined)
                fft_data=np.asarray([fft_result.real,fft_result.imag])
                fft_data=np.transpose(fft_data, (1,0))
                for i in range(len(spec[fft_index])):
                    spec[fft_index][i]=fft_data[i]
                pos+=self.NFFT//2
        return spec
    def ifft(self,data):
        data=data[:,:,0]+1j*data[:,:,1]
        time_ruler=data.shape[0]
        window=np.hamming(self.NFFT)
        spec=np.zeros([])
        lats = np.zeros([self.NFFT//2])
        pos=0
        for _ in range(time_ruler):
            frame=data[pos]
            fft_result=np.fft.ifft(frame)
            fft_data=fft_result.real
            fft_data/=window
            v = lats + fft_data[:self.NFFT//2]
            lats = fft_data[self.NFFT//2:]
            spec=np.append(spec,v)
            pos+=1
        return spec[1:]

#model architectures

def discriminator(inp,reuse):
    inputs=tf.cast(inp, tf.float32)
    h1 = tf.nn.leaky_relu(tf.layers.conv2d(inputs, 4,2, strides=2, padding="VALID",data_format="channels_last",name="dis_01",reuse=reuse))
    h2 = tf.nn.leaky_relu(tf.layers.conv2d(h1, 8,4, strides=4, padding="VALID",data_format="channels_last",name="dis_02",reuse=reuse))
    h3 = tf.nn.leaky_relu(tf.layers.conv2d(h2, 16,8, strides=8, padding="VALID",data_format="channels_last",name="dis_03",reuse=reuse))
    h4 = tf.nn.leaky_relu(tf.layers.conv2d(h3, 4,16, strides=1, padding="VALID",data_format="channels_last",name="dis_04",reuse=reuse))
    h4=tf.reshape(h4, [1,-1])
    ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
    ot=tf.nn.sigmoid(ten)
    return ot,ten

def generator(current_outputs,reuse,depth,chs):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    else:
        assert tf.get_variable_scope().reuse == False
    current=current_outputs
    connections=[ ]
    for i in range(depth):
        current=down_layer(current,chs[i+1])
        connections.append(current)
    for i in range(depth):
        current+=connections[depth-i-1]
        current=up_layer(current,chs[depth-i-1],i!=(depth-1),depth-i-1>2)
    return current

def up_layer(current,output_shape,bn=True,do=False):
    ten=tf.nn.leaky_relu(current)
    ten=tf.layers.conv2d_transpose(ten, output_shape,kernel_size=4 ,strides=(1,1), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
    if bn:
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
    if do:
        ten=tf.nn.dropout(ten, 0.5)
    return ten
def down_layer(current,output_shape):
    ten=tf.layers.batch_normalization(current,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
    ten=tf.layers.conv2d(ten, output_shape,kernel_size=4 ,strides=(1,1), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
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