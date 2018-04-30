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
        self.batch_size=8
        self.depth=4
        self.input_ch=1
        self.input_size=[self.batch_size,8192,1]
        self.input_size_model=[self.batch_size,16, 513,1]

        self.dataset_name="wave2wave_1.0.0"
        self.output_size=[self.batch_size,8192,1]
        self.CHANNELS=[4**i+1 for i in range(self.depth+1)]
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85)))
        if debug:
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    def build_model(self):

        self.input_model=tf.placeholder(tf.float32, [self.batch_size,512, 32,2], "inputs")
        self.input_model_label=tf.placeholder(tf.float32, [self.batch_size,512, 32,2], "inputs")
        self.input_wave_fake=tf.placeholder(tf.int16, [1, 80000,1], "inputs")
        self.input_wave_real=tf.placeholder(tf.int16, [1, 80000,1], "inputs")
        self.input_wave_sour=tf.placeholder(tf.int16, [1, 80000,1], "inputs")
        self.d_score=tf.placeholder(tf.float32, name="inputs_dsa")
        with tf.variable_scope("generator_1"):
            self.fake_B_image=self.generator(self.input_model, reuse=False,name="gen")

        self.res1=tf.concat([self.input_wave_sour,self.input_wave_fake], axis=1)
        self.res2=tf.concat([self.input_wave_sour,self.input_wave_real], axis=1)
        with tf.variable_scope("discrim",reuse=tf.AUTO_REUSE):
            self.var=dict()
            w=tf.get_variable('w1', [2,1,4],initializer=tf.contrib.layers.xavier_initializer())
            self.var['w-1'] = w
            w=tf.get_variable('w2', [2,4,8],initializer=tf.contrib.layers.xavier_initializer())
            self.var['w-2'] = w
            w=tf.get_variable('w3', [2,8,16],initializer=tf.contrib.layers.xavier_initializer())
            self.var['w-3'] = w
            w=tf.get_variable('w4', [2,16,32],initializer=tf.contrib.layers.xavier_initializer())
            self.var['w-4'] = w
            self.d_judge_F1,self.d_judge_F1_logits=self.discriminator(self.res1,self.var,False)
            self.d_judge_R,self.d_judge_R_logits=self.discriminator(self.res2,self.var,True)

        self.g_vars_1=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_1")
        self.g_vars_2=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_2")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrim")
        L1=tf.losses.mean_squared_error(labels=self.input_model_label, predictions=self.fake_B_image)
        DS=-tf.log(self.d_score)
        self.g_loss_1=L1*(100+DS)
        self.d_loss_R = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.random_uniform(self.d_judge_R.shape,0.9,1.0),  logits=self.d_judge_R_logits)
        self.d_loss_F = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.random_uniform(self.d_judge_F1.shape,0.1,0.3), logits=self.d_judge_F1_logits)
        self.d_loss=tf.reduce_mean([self.d_loss_R,self.d_loss_F])
        self.g_loss_sum_1 = tf.summary.scalar("g_loss", tf.reduce_mean(self.g_loss_1))
        self.d_loss_sum = tf.summary.merge([tf.summary.scalar("d_loss_R", tf.reduce_mean(self.d_loss_R)),tf.summary.scalar("d_loss_F", tf.reduce_mean(self.d_loss_F))])
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
            start_pos=self.output_size[2]*(t)+((in_put.shape[1])%self.output_size[1])
            resorce=np.reshape(in_put[0,0,max(0,start_pos-self.input_size[1]):min(start_pos,in_put.shape[1])],(1,1,-1))
            r=max(0,self.input_size[1]-resorce.shape[1])
            if r>0:
                resorce=np.pad(resorce,((0,0),(0,0),(r,0)),'constant')
            red=np.append(resorce,red)
            red=red.reshape((self.input_size[0],self.input_size[1],self.input_size[2]))
            res=np.zeros([self.batch_size,512, 32,2])
            for i in range(self.batch_size):
                    n=fft(red[i].reshape(-1))
                    res[i]=(n)
            red=res.reshape([self.batch_size,512, 32,2])
            red=red.reshape(self.batch_size,512, 32,2)
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
        lr_g_opt=1e-1
        beta_g_opt=0.5
        lr_d_opt=1e-1
        beta_d_opt=0.1
        self.lod="[glr="+str(lr_g_opt)+",gb="+str(beta_g_opt)+"]"
        g_optim_1 =tf.train.AdamOptimizer(lr_g_opt,beta_g_opt).minimize(self.g_loss_1, var_list=self.g_vars_1)
        d_optim = tf.train.AdamOptimizer(lr_d_opt,beta_d_opt).minimize(self.d_loss, var_list=self.d_vars)

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
                ti=(batch_idxs*times)
#                 g_score=0
                #update_Punish_scale
                resorce_te=test_train[0,1,:].reshape(1,-1,1)
                target_te=test_train[0,0,:].reshape(1,-1,1)
                target_te=nos(target_te)
                cv1,_=self.convert(resorce_te)
                #Calcurate new d_score
                score1=self.sess.run([self.d_judge_F1],feed_dict={ self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te })
                # Update D network
                hd,_=self.sess.run([self.d_loss_sum,d_optim],feed_dict={self.input_wave_sour:resorce_te ,self.input_wave_fake:cv1 ,self.input_wave_real:target_te  })
                self.writer.add_summary(hd, idx+batch_idxs*epoch)
                ds=np.mean((score1))
                if math.isnan(ds):
                    ds=0.0
#                 print(ds)
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
                    res=np.zeros([self.batch_size,512, 32,2])
                    tar=np.zeros([self.batch_size,512, 32,2])
                    for i in range(resorce.shape[0]):
                        res[i]=(fft(resorce[i]))
                        tar[i]=(fft(resorce[i]))
                    res=res.reshape([self.batch_size,512, 32,2])
                    tar=tar.reshape([self.batch_size,512, 32,2])
                    # Update G1 network
                    _,hg=self.sess.run([g_optim_1,self.g_loss_sum_1],feed_dict={ self.input_model:res, self.input_model_label:tar ,self.d_score:ds})
                    if counter%20==0:
                        self.writer.add_summary(hg, counter+ti*epoch)
                    counter+=1
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
            upload(out_puts)
            start_time = time.time()
        self.experiment.end()

    def discriminator(self,inp,var,reuse):
        inputs=tf.cast(inp, tf.float32)
        w=var['w-1']
        h1 = tf.nn.leaky_relu(tf.nn.conv1d(inputs, w, stride=2, padding="VALID",data_format="NWC",name="dis_01"))
        w=var['w-2']
        h2 = tf.nn.leaky_relu(tf.nn.conv1d((h1), w, stride=2, padding="VALID",data_format="NWC",name="dis_02"))
        w=var['w-3']
        h3 = tf.nn.leaky_relu(tf.nn.conv1d((h2), w, stride=2, padding="VALID",data_format="NWC",name="dis_03"))
        w=var['w-4']
        h4 = tf.nn.leaky_relu(tf.nn.conv1d((h3), w, stride=2, padding="VALID",data_format="NWC",name="dis_04"))

        ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
        ot=tf.nn.sigmoid(ten)
        return ot,ten

    def waver(self,inp):
        ten=tf.layers.batch_normalization(inp,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        ten=tf.layers.conv2d(ten,256, kernel_size=2, strides=(1,1), padding="VALID")
        ten=tf.nn.leaky_relu(ten)
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        ten=tf.layers.conv2d(ten,128, kernel_size=2, strides=(1,1), padding="VALID")
        ten=tf.nn.leaky_relu(ten)
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        ten=tf.layers.conv2d(ten,32, kernel_size=2, strides=(1,1), padding="VALID")
        ten=tf.nn.leaky_relu(ten)
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        ten=tf.layers.conv2d(ten,1, kernel_size=2, strides=(1,1), padding="VALID")
        ten=tf.nn.leaky_relu(ten)
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
        ten=tf.reshape(ten, [self.batch_size,-1])
        ten=tf.layers.dense(ten,8192)
        return tf.nn.tanh(ten)
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
            current=self.up_layer(current,self.CHANNELS[self.depth-i-1])

        return current
    def up_layer(self,current,output_shape):
        ten=tf.nn.leaky_relu(current)
        ten=tf.layers.conv2d_transpose(ten, output_shape,kernel_size=4 ,strides=(1,1), padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer(),data_format="channels_last")
        ten=tf.layers.batch_normalization(ten,axis=3,training=True,gamma_initializer=tf.random_normal_initializer(1.0, 0.2))
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
    NFFT=32
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
        frame=data[pos:pos+NFFT]
        if len(frame)==NFFT:
            wined=frame*window
            fft=np.fft.fft(wined)
            fft_data=np.asarray([fft.real,fft.imag])
            fft_data=np.reshape(fft_data, (32,2))
            for i in range(len(spec[fft_index])):
                spec[fft_index][-i-1]=fft_data[i]
            pos+=NFFT//2
    return spec
def ifft(data):
    data=data[:,:,0]+1j*data[:,:,1]
    time_ruler=data.shape[0]
    window=np.hamming(32)
    spec=np.zeros([])
    pos=0
    for _ in range(time_ruler):
        frame=data[pos]/32767.0
        fft=np.fft.ifft(frame)
        fft_data=fft.real
        fft_data/=window
        spec=np.append(spec,fft_data)
        pos+=1

    return spec[1:]/5
def nowtime():
    return datetime.now().strftime("%Y_%m_%d %H_%M_%S")
def upload(voice):
    voiced=voice.astype(np.int16)
    p=pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    ww = wave.open("Z://waves/"+nowtime()+".wav", 'wb')
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
    s=ar.shape[2]
#     if random.randint(0,5)==0:
#         ar+=(np.random.rand(ar.shape[0],ar.shape[1],ar.shape[2])*4-2).astype(np.int16)
    i=random.randint(0,80)
    ar=np.pad(ar, ((0,0),(0,0),(i,0)), "constant")[:,:,0:s]
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