from glob import glob
import tensorflow as tf
import os
import time
import re
from six.moves import xrange
import numpy as np
import wave
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import pyaudio
from hyperdash import Experiment
import random
from datetime import datetime
import math
class Model:
    def __init__(self,debug):
        self.batch_size=1
        self.depth=6
        self.input_ch=1
        self.input_size=[self.batch_size,8192,1]
        self.input_size_model=[self.batch_size,64,256,1]

        self.dataset_name="wave2wave_1.1.0"
        self.output_size=[self.batch_size,8192,1]
        self.CHANNELS=[min([4**i+1,64]) for i in range(self.depth+1)]
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85)))
        if debug:
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    def build_model(self):

        self.input_model=tf.placeholder(tf.float32, [self.batch_size,256,64], "inputs_convert")
        self.input_model_label=tf.placeholder(tf.float32, [self.batch_size,256,64,2], "inputs_answer")
        self.input_wave_fake=tf.placeholder(tf.int16, [1, 80000,1], "inputs_discriminator_fake")
        self.input_wave_real=tf.placeholder(tf.int16, [1, 80000,1], "inputs_discriminator_real")
        self.input_wave_sour=tf.placeholder(tf.int16, [1, 80000,1], "inputs_discriminator_source")
        self.d_score=tf.placeholder(tf.float32, name="inputs_dsa")
        with tf.variable_scope("generator_1"):
            self.fake_B_image=self.generator(tf.reshape(self.input_model,[self.batch_size,256,64,1]), reuse=False,name="gen")

        self.res1=tf.concat([self.input_wave_sour,self.input_wave_fake], axis=1)
        self.res2=tf.concat([self.input_wave_sour,self.input_wave_real], axis=1)
        with tf.variable_scope("discrim",reuse=tf.AUTO_REUSE):
            self.d_judge_F1=self.discriminator(self.res1,False)
            self.d_judge_R=self.discriminator(self.res2,True)
            self.d_scale=1.0-tf.abs(self.d_judge_F1-self.d_judge_R)
        self.g_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_1")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrim")
        L1=tf.reduce_mean(tf.abs(self.input_model_label-self.fake_B_image))
        DS=tf.losses.mean_squared_error(labels=tf.ones(1),predictions=self.d_score)
        self.g_loss_1=tf.reduce_mean((L1)*(DS+1))
        self.d_loss_R = tf.losses.mean_squared_error(labels=tf.ones(self.d_judge_R.shape),predictions=self.d_judge_R)
        self.d_loss_F = tf.losses.mean_squared_error(labels=tf.zeros(self.d_judge_F1.shape),predictions=self.d_judge_F1)
        self.d_loss=tf.reduce_mean([self.d_loss_R,self.d_loss_F])
        self.g_loss_sum_1 = tf.summary.scalar("g_loss", tf.reduce_mean(self.g_loss_1))
        self.d_loss_sum = tf.summary.merge([tf.summary.scalar("d_loss", tf.reduce_mean(self.d_loss)),tf.summary.scalar("d_loss_F", tf.reduce_mean(self.d_loss_F))])
        self.exps=tf.placeholder(tf.float32, [1,1,160000], name="FB")
        self.fake_B_sum = tf.summary.audio("fake_B", tf.reshape(self.exps,[1,160000,1]), 16000, 1)
        self.g_test_epo_1=tf.placeholder(tf.float32,name="g_l_epo")
        self.g_test_epoch_1 = tf.summary.scalar("g_test_epoch", self.g_test_epo_1)
        self.d_score_epo=tf.placeholder(tf.float32,name="g_l_epo")
        self.d_score_epoch= tf.summary.scalar("d_score_epoch", self.d_score_epo)
        self.rrs=tf.summary.merge([self.fake_B_sum,self.g_test_epoch_1,self.d_score_epoch])
        self.saver = tf.train.Saver()
    def convert(self,in_put):
        tt=time.time()
        times=in_put.shape[1]//(self.output_size[1])+1
        if in_put.shape[1]%(self.output_size[1]*self.batch_size)==0:
            times-=1
        otp=np.array([],dtype=np.int16)
        for t in range(times):
            red=np.zeros((self.input_size[0]-1,self.input_size[1],self.input_size[2]))
            start_pos=self.output_size[1]*(t)+((in_put.shape[1])%self.output_size[1])
            resorce=np.reshape(in_put[0,max(0,start_pos-self.input_size[1]):start_pos,0],(1,-1))
            r=max(0,self.input_size[1]-resorce.shape[1])
            if r>0:
                resorce=np.pad(resorce,((0,0),(r,0)),'constant')
            red=np.append(resorce,red)
            red=red.reshape((self.input_size[0],self.input_size[1],self.input_size[2]))
            res=np.zeros([self.batch_size,256,64,2])
            for i in range(self.batch_size):
                n=fft(red[i].reshape(-1))
                res[i]=(n)
            red=np.log(np.abs(res[:,:,:,0]+1j*res[:,:,:,1])**2+1e-16)
            res=self.sess.run(self.fake_B_image,feed_dict={ self.input_model:red})
            res=ifft(res[0])*32767
            res=res.reshape(-1)
            otp=np.append(otp,res)
        h=otp.shape[0]-in_put.shape[1]-1
        if h!=-1:
            otp=otp[h:-1]
        return otp.reshape(1,in_put.shape[1],in_put.shape[2]),time.time()-tt
    def train_f(self,args):
        self.checkpoint_dir=args.checkpoint_dir
        lr_g_opt=2e-4
        beta_g_opt=0.5
        lr_d_opt=2e-6
        beta_d_opt=0.5
        self.lod="[glr="+str(lr_g_opt)+",gb="+str(beta_g_opt)+",dlr="+str(lr_d_opt)+",db="+str(beta_d_opt)+"]"
        g_optim_1 =tf.train.AdamOptimizer(lr_g_opt,beta_g_opt).minimize(self.g_loss_1, var_list=self.g_vars_1)
        d_optim = tf.train.RMSPropOptimizer(lr_d_opt,beta_d_opt).minimize(self.d_loss, var_list=self.d_vars)

        init_op = tf.global_variables_initializer()
        self.exp= np.zeros((self.batch_size,1,80000),dtype=np.int16)
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs/"+self.lod+self.dataset_name, self.sess.graph)
        counter = 1
        start_time = time.time()
        cv1=None
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        data = glob('./Model/datasets/train/01/*')
        data2 = glob('./Model/datasets/train/02/*')
        batch_idxs = min(len(data), args.train_size) // self.batch_size
        self.experiment=Experiment(self.dataset_name+"_G1")
        self.experiment.param("lr_g_opt", lr_g_opt)
        self.experiment.param("beta_g_opt", beta_g_opt)
        self.experiment.param("lr_d_opt", lr_d_opt)
        self.experiment.param("beta_d_opt", beta_d_opt)
        self.experiment.param("depth", self.depth)

        wds=1.0
        for epoch in range(0,500):

            np.random.shuffle(data)
            np.random.shuffle(data2)

            test=load_data('./Model/datasets/test/test.wav')[0:160000]
            label=load_data('./Model/datasets/test/label.wav')[0:160000]
            dps=0
            counter=0
            print("Epoch %3d start" % (epoch))
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = np.asarray([(load_data(batch_file)) for batch_file in batch_files])
                batch_images = np.array(batch).astype(np.int16).reshape(self.batch_size,2,80000)

                test_train=load_data(data2[idx%2]).reshape(1,2,80000)

                times=80000//self.output_size[1]
#                 times_added=0
                if int(80000)%self.output_size[1]==0:
                    times-=1
                ti=(batch_idxs*times//10)
#                 g_score=0
                #update_Punish_scale
                resorce_te=test_train[0,1,:].reshape(1,-1,1)
                target_te=test_train[0,0,:].reshape(1,-1,1)
                target_te=nos(target_te)
                cv1,_=self.convert(resorce_te)
                #Calcurate new d_score
                score1,dp=self.sess.run([self.d_judge_F1,self.d_scale],feed_dict={ self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te })
                self.sess.run([d_optim],feed_dict={self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te  })
                self.sess.run([d_optim],feed_dict={self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te  })
                hd,_=self.sess.run([self.d_loss_sum,d_optim],feed_dict={self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te  })
                self.writer.add_summary(hd, idx+batch_idxs*epoch)
                uds=np.mean((score1))*np.mean((dp))
                if math.isnan(uds):
                    uds=0.0
                ds=(uds+wds)/2
                d_score = (np.mean(ds))
                dps+=(d_score)
                times_set=[j for j in range(times)]
                random.shuffle(times_set)
                for t in times_set:
                    start_pos=self.output_size[1]*t+(80000%self.output_size[1])
                    target=np.reshape(batch_images[:,0,max(0,start_pos-self.output_size[1]):start_pos],(self.batch_size,-1))
                    resorce=np.reshape(batch_images[:,1,max(0,start_pos-self.input_size[1]):start_pos],(self.batch_size,-1))
                    r=max(0,self.input_size[1]-resorce.shape[1])

                    if r>0:
                        resorce=np.pad(resorce,((0,0),(r,0)),'constant')
                    r=max(0,self.output_size[1]-target.shape[1])
                    if r>0:
                        target=np.pad(target,((0,0),(r,0)),'constant')
                    res=np.zeros([self.batch_size,256,64,2])
                    tar=np.zeros([self.batch_size,256,64,2])
                    for i in range(self.batch_size):
                        res[i]=(fft(resorce[i]))
                        tar[i]=(fft(target[i]))
                    res=res.reshape([self.batch_size,256,64,2])
                    res=np.log(np.abs(res[:,:,:,0]+1j*res[:,:,:,1])**2+1e-16)
                    tar=tar.reshape([self.batch_size,256,64,2])
                    # Update G1 network
                    _,hg=self.sess.run([g_optim_1,self.g_loss_sum_1],feed_dict={ self.input_model:res, self.input_model_label:tar ,self.d_score:ds})
                    if counter%20==0:
                        self.writer.add_summary(hg, counter//10+ti*epoch)
                    counter+=1
                resorce_te= batch_images[0,1,:].reshape(1,-1,1)
                target_te= batch_images[0,0,:].reshape(1,-1,1)
                cv1,_=self.convert(resorce_te)
                score2,dp=self.sess.run([self.d_judge_F1,self.d_scale],feed_dict={ self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te })
                wds=np.mean((score2))*np.mean((dp))
                if math.isnan(wds):
                    wds=0.0
                self.sess.run([d_optim],feed_dict={self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te  })
                self.sess.run([d_optim],feed_dict={self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te  })
                hd,_=self.sess.run([self.d_loss_sum,d_optim],feed_dict={self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te  })
            self.save(args.checkpoint_dir, epoch+1)
            self.experiment.metric("errD",dps/batch_idxs)
            out_puts,taken_time=self.convert(test.reshape(1,-1,1))
            out_put=(out_puts.astype(np.float32)/32767.0)
            test1=np.mean(np.abs(out_puts-label.reshape(1,-1,1)))
            self.experiment.metric("testG",test1)
            ff="Z://data/"+self.dataset_name+".txt"
            f=open(ff,'a')
            f.write("TimeStamped:"+nowtime())
            f.write(",%2d,%4.1f,%f,%f\n" % (epoch+1,time.time() - start_time,(dps/batch_idxs),(test1)))
            f.close()
            rs=self.sess.run(self.rrs,feed_dict={ self.exps:out_put.reshape(1,1,-1),self.g_test_epo_1:(test1),self.d_score_epo:(dps/batch_idxs)})
            self.writer.add_summary(rs, epoch+1)
            print("test taken: %f secs" % (taken_time))
            upload(out_puts,"_u")
            upload(cv1,"_w")
            start_time = time.time()
        self.experiment.end()

    def discriminator(self,inp,reuse):
        inputs=tf.cast(inp, tf.float32)
        h1 = tf.nn.leaky_relu(tf.layers.conv1d(inputs, 4,4, strides=2, padding="VALID",data_format="channels_last",name="dis_01",reuse=reuse))
        h2 = tf.nn.leaky_relu(tf.layers.conv1d(tf.layers.batch_normalization(h1,training=True), 8,4, strides=2, padding="VALID",data_format="channels_last",name="dis_02",reuse=reuse))
        h3 = tf.nn.leaky_relu(tf.layers.conv1d(tf.layers.batch_normalization(h2,training=True), 16,4, strides=2, padding="VALID",data_format="channels_last",name="dis_03",reuse=reuse))
        h4 = tf.nn.leaky_relu(tf.layers.conv1d(tf.layers.batch_normalization(h3,training=True), 32,4, strides=2, padding="VALID",data_format="channels_last",name="dis_04",reuse=reuse))
        h4=tf.reshape(h4, [1,-1])
        ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
        return ten
    def generator(self,current_outputs,reuse,name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        current=current_outputs

        connections=[ ]

        for i in range(self.depth):
            current=self.down_layer(current,self.CHANNELS[i+1])
            connections.append(current)
        for i in range(self.depth):
            current+=connections[self.depth-i-1]
            current=self.up_layer(current,self.CHANNELS[self.depth-i-1],i!=(self.depth-1),self.depth-i-1>2)
        return current
    def up_layer(self,current,output_shape,bn,do):
        ten=tf.nn.leaky_relu(current)
        ten=tf.layers.conv2d_transpose(ten, output_shape,kernel_size=4 ,strides=(1,1), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
        if(bn):
            ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        if(do):
            ten=tf.nn.dropout(ten, 0.5)
        return ten
    def down_layer(self,current,output_shape):
        ten=tf.layers.batch_normalization(current,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        ten=tf.layers.conv2d(ten, output_shape,kernel_size=4 ,strides=(1,1), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
        ten=tf.nn.leaky_relu(ten)
        return ten
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
def fft(data):
    rate=16000
    NFFT=64
    time_song=float(data.shape[0])/rate
    time_unit=1/rate
    start=0
    stop=time_song
    step=(NFFT//2)*time_unit
    time_ruler=np.arange(start,stop,step)
    window=np.hamming(NFFT)
    spec=np.zeros([len(time_ruler),(NFFT),2])
    pos=0
    for fft_index in range(len(time_ruler)):
        frame=data[pos:pos+NFFT]/32767.0
        if len(frame)==NFFT:
            wined=frame*window
            fft=np.fft.fft(wined)
            fft_data=np.asarray([fft.real,fft.imag])
            fft_data=np.transpose(fft_data, (1,0))
            for i in range(len(spec[fft_index])):
                spec[fft_index][i]=fft_data[i]
            pos+=NFFT//2
    return spec
def ifft(data):
    data=data[:,:,0]+1j*data[:,:,1]
    time_ruler=data.shape[0]
    window=np.hamming(64)
    spec=np.zeros([])
    lats = np.zeros([32])
    pos=0
    for _ in range(time_ruler):
        frame=data[pos]
        fft=np.fft.ifft(frame)
        fft_data=fft.real
        fft_data/=window
        v = lats + fft_data[:32]
        lats = fft_data[32:]
        spec=np.append(spec,v)
        pos+=1
    return spec[1:]
def nowtime():
    return datetime.now().strftime("%Y_%m_%d %H_%M_%S")
def upload(voice,strs):
    voiced=voice.astype(np.int16)
    p=pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    ww = wave.open("Z://waves/"+nowtime()+strs+".wav", 'wb')
    ww.setnchannels(1)
    ww.setsampwidth(p.get_sample_size(FORMAT))
    ww.setframerate(16000)
    ww.writeframes(voiced.reshape(-1).tobytes())
    ww.close()
    p.terminate()
def load_data(image_path):
    images = imread(image_path)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return images
def nos(ar):
    s=ar.shape[1]
    if random.randint(0,2)==0:
        ar+=(np.random.rand(ar.shape[0],ar.shape[1],ar.shape[2])*4-2).astype(np.int16)
    i=random.randint(0,80)
    ar=np.pad(ar, ((0,0),(i,0),(0,0)), "constant")[:,0:s,:]
    return ar
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