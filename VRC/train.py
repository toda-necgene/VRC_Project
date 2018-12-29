import os
import time
import json
from datetime import datetime

import wave
import pyaudio
import pyworld.pyworld as pw

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from model import discriminator,generator

from voice_to_dataset_cycle import create_dataset

class Model:
    def __init__(self,path_setting):
        self.start_time = time.time()
        self.tt_list = list()
        self.test=None
        self.label=None
        self.label_spec=None
        self.label_spec_v=None
        self.sounds_r=None
        self.sounds_t=None
        self.loop_num=0

        # setting default parameters

        self.args=dict()

        # name options
        self.args["model_name"] = "VRC"
        self.args["version"] = "1.0.0"
        # saving options
        self.args["checkpoint_dir"]="./trained_models"
        self.args["wave_otp_dir"] = "./harvests"
        #training-data options
        self.args["use_old_dataset"]=False
        self.args["train_data_dir"]="./dataset/train"
        self.args["test_data_dir"] ="./dataset/test"
        # learning details output options
        self.args["test"]=True
        self.args["tensor-board"]=False
        self.args["real_sample_compare"]=False
        # learning options
        self.args["batch_size"] = 1
        self.args["weight_Cycle"] = 100.0
        self.args["train_iteration"] = 600000
        self.args["start_epoch"] = 0
        # architecture option
        self.args["input_size"] = 4096


        # loading json setting file (more codes ./setting.json. manual is exist in ./setting-example.json)
        try:
            with open(path_setting, "r") as f:
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
                                    " [W] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(
                                        type(self.args[k])) + "\"")
                        elif k[0] == "#":
                            pass
                        else:
                            print(" [W] Argument \"" + k + "\" is not exsits.")

        except json.JSONDecodeError as e:
            print(" [W] JSONDecodeError: ", e)
            print(" [W] Use default setting")


        # loading f0 parameters
        a = np.load("./voice_profile.npy")
        self.args["pitch_rate_mean_s"] = a[0]
        self.args["pitch_rate_mean_t"] = a[1]
        self.args["pitch_rate_var"] = a[2]

        # shapes properties
        self.input_size_model=[self.args["batch_size"],52,513,1]
        self.input_size_test = [1, 52,513,1]
        self.output_size_model = [self.args["batch_size"], 65,513,1]

        # initializing harvest directory
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        if  self.args["wave_otp_dir"] is not "False" :
            self.args["wave_otp_dir"]=self.args["wave_otp_dir"]+ self.args["name_save"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])


        # creating data-set

        if not self.args["use_old_dataset"]:
            create_dataset()
        #inputs place holders

        self.input_model_A=tf.placeholder(tf.float32, self.input_size_model, "inputs_g_A")
        self.input_model_B = tf.placeholder(tf.float32, self.input_size_model, "inputs_g_B")
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_g_test")

        #creating generator (if you want to view more codes then ./model.py)
        with tf.variable_scope("generator_1"):
            fake_aB_image = generator(self.input_model_A, reuse=None,training=True)
            self.fake_aB_image_test= generator(self.input_model_test, reuse=True,training=False)
        with tf.variable_scope("generator_2"):
            fake_bA_image = generator(self.input_model_B, reuse=None,training=True)
        with tf.variable_scope("generator_2",reuse=True):
            fake_Ba_image = generator(fake_aB_image, reuse=True,training=True)
        with tf.variable_scope("generator_1",reuse=True):
            fake_Ab_image = generator(fake_bA_image, reuse=True,training=True)


        #creating discriminator (if you want to view more codes then ./model.py)
        with tf.variable_scope("discriminator_1"):
            d_judgeAR = discriminator(self.input_model_A, None)
            d_judgeAF = discriminator(fake_bA_image, True)
        with tf.variable_scope("discriminator_2"):
            d_judgeBR = discriminator(self.input_model_B,None)
            d_judgeBF = discriminator(fake_aB_image, True)

        #getting individual variables of architectures
        self.g_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_1")+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator_2")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discriminator_1")+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discriminator_2")
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #objective-functions of discriminator
        label1 = tf.ones_like(d_judgeAR)
        label0 = tf.zeros_like(d_judgeAR)

        # Least squared loss
        d_loss_AR = tf.squared_difference(label1, d_judgeAR)
        d_loss_AF = tf.squared_difference(label0, d_judgeAF)
        d_loss_BR = tf.squared_difference(label1, d_judgeBR)
        d_loss_BF = tf.squared_difference(label0, d_judgeBF)
        self.d_loss=d_loss_AR+d_loss_AF+d_loss_BR+d_loss_BF

        # objective-functions of generator

        # Cycle loss (L2 norm is better than L1 norm for keeping words)
        g_loss_cyc_A = tf.losses.absolute_difference(self.input_model_A,fake_Ba_image)
        g_loss_cyc_B = tf.losses.absolute_difference(self.input_model_B,fake_Ab_image)

        # Gan loss (using a difference of like WGAN )
        g_loss_gan_A = tf.squared_difference(label1 , d_judgeAF)
        g_loss_gan_B = tf.squared_difference(label1 , d_judgeBF)


        self.g_loss =tf.losses.compute_weighted_loss( g_loss_cyc_A  + g_loss_cyc_B,self.args["weight_Cycle"]) + g_loss_gan_B+ g_loss_gan_A

        #tensorboard functions
        if self.args["tensor-board"]:
            g_loss_cyc_A_display= tf.summary.scalar("g_loss_cycle_AtoA", tf.reduce_mean(g_loss_cyc_A),family="g_loss")
            g_loss_gan_A_display = tf.summary.scalar("g_loss_gan_BtoA", tf.reduce_mean(g_loss_gan_A),family="g_loss")
            g_loss_sum_A_display = tf.summary.merge([g_loss_cyc_A_display, g_loss_gan_A_display])

            g_loss_cyc_B_display = tf.summary.scalar("g_loss_cycle_BtoB", tf.reduce_mean(g_loss_cyc_B),family="g_loss")
            g_loss_gan_B_display = tf.summary.scalar("g_loss_gan_AtoB", tf.reduce_mean(g_loss_gan_B),family="g_loss")
            g_loss_sum_B_display = tf.summary.merge([g_loss_cyc_B_display,g_loss_gan_B_display])
            d_loss_sum_A_display = tf.summary.scalar("d_loss", tf.reduce_mean(self.d_loss),family="d_loss")

            self.loss_display=tf.summary.merge([g_loss_sum_A_display,g_loss_sum_B_display,d_loss_sum_A_display])
            self.result_score= tf.placeholder(tf.float32, name="FakeFFTScore")
            self.result_image_display= tf.placeholder(tf.float32, [1,None,512], name="FakeSpectrum")
            image_pow_display=tf.reshape(tf.transpose(self.result_image_display[:,:,:],[0,2,1]),[1,512,-1,1])
            fake_B_image_display = tf.summary.image("Fake_spectrum_AtoB", image_pow_display, 1)
            fake_B_FFT_score_display= tf.summary.scalar("g_error_AtoB", tf.reduce_mean(self.result_score),family="g_test")
            self.g_test_display=tf.summary.merge([fake_B_image_display,fake_B_FFT_score_display])

        # initializing running object
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # logging
        if self.args["tensor-board"]:
            self.writer = tf.summary.FileWriter("./logs/" + self.args["name_save"], self.sess.graph)
    def convert(self,in_put):
        #function of test
        #to convert wave file

        # calculating times to execute network
        conversion_start_time=time.time()
        input_size_one_term=self.args["input_size"]
        executing_times=in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"])==0:
            executing_times-=1

        otp=np.array([],dtype=np.int16)

        for t in range(executing_times):
            # pre-process
            # padding
            start_pos=max(0,self.args["input_size"]*t+(in_put.shape[0]%self.args["input_size"])-input_size_one_term)
            end_pos = self.args["input_size"]*t+(in_put.shape[0]%self.args["input_size"])
            resource_wave=in_put[start_pos:end_pos]
            r=max(0,input_size_one_term-resource_wave.shape[0])
            if r>0:
                resource_wave=np.pad(resource_wave,(r,0),'constant')
            # wave to WORLD
            f0,sp,ap=encode((resource_wave/32767).astype(np.float))
            sp=sp.reshape(self.input_size_test)
            #main process
            # running network
            result=self.sess.run(self.fake_aB_image_test,feed_dict={ self.input_model_test:sp})

            # post-process
            # f0 transforming
            f0=(f0-self.args["pitch_rate_mean_s"])*self.args["pitch_rate_var"]+self.args["pitch_rate_mean_t"]
            # WORLD to wave
            result_wave=decode(f0,result[0].copy().reshape(-1,513).astype(np.float),ap)*32767
            result_wave_fixed=np.clip(result_wave,-32767.0,32767.0)[:self.args["input_size"]]
            result_wave_int16=result_wave_fixed.reshape(-1).astype(np.int16)

            #adding result
            otp=np.append(otp,result_wave_int16)

        h=otp.shape[0]-in_put.shape[0]
        if h>0:
            otp=otp[h:]

        return otp,time.time()-conversion_start_time



    def train(self):

        # naming output-directory
        lr=tf.placeholder(tf.float32)
        with tf.control_dependencies(self.update_ops):
            g_optimizer = tf.train.AdamOptimizer(lr, 0.5, 0.999).minimize(self.g_loss,var_list=self.g_vars)
        d_optimizer = tf.train.AdamOptimizer(lr, 0.5, 0.999).minimize(self.d_loss,var_list=self.d_vars)

        # loading net
        if self.load():
            print(" [I] Load success.")
        else:
            print(" [I] Load failed.")

        # loading training data directory
        # loading test data
        if self.args["test"]:
            self.test  = wave_read(self.args["test_data_dir"]+'/test.wav')
            if self.args["real_sample_compare"]:
                self.label = wave_read(self.args["test_data_dir"] + '/label.wav')
                im = fft(self.label[800:156000]/32767)
                self.label_spec = np.mean(im, axis=0)
                self.label_spec_v = np.std(im, axis=0)

        # preparing training-data
        batch_files = self.args["train_data_dir"]+'/A.npy'
        batch_files2 = self.args["train_data_dir"]+'/B.npy'

        print(" [I] loading data-set ...")
        self.sounds_r = np.load(batch_files)
        self.sounds_t = np.load(batch_files2)
        # times of one epoch
        train_data_num = min(self.sounds_r.shape[0],self.sounds_t.shape[0])
        self.loop_num = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num)]
        print(" [I] %d data loaded!!" % train_data_num)
        self.sounds_r=self.sounds_r.reshape([self.sounds_r.shape[0],self.sounds_r.shape[1],self.sounds_r.shape[2],1])
        self.sounds_t=self.sounds_t.reshape([self.sounds_t.shape[0],self.sounds_t.shape[1],self.sounds_t.shape[2],1])

        # initializing training information
        start_time_all=time.time()
        train_epoch=self.args["train_iteration"]//self.loop_num+1
        one_itr_num=self.loop_num*self.args["batch_size"]
        iterations=0
        max_lr = 2e-4
        min_lr = 2e-5
        T_c=0
        T=50000
        # main-training
        for epoch in range(train_epoch):
            # shuffling train_data_index
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)

            if self.args["test"] :
                self.test_and_save(epoch,iterations,one_itr_num)
            for idx in range(0, self.loop_num):
                # getting mini-batch
                st=self.args["batch_size"]*idx
                batch_sounds_resource = np.asarray([self.sounds_r[ind] for ind in index_list[st:st+self.args["batch_size"]]])
                batch_sounds_target= np.asarray([self.sounds_t[ind] for ind in index_list2[st:st+self.args["batch_size"]]])
                opt=np.cos(T_c/T*np.pi*0.5)*(max_lr-min_lr)+min_lr
                # update D network
                self.sess.run(d_optimizer, feed_dict={self.input_model_A: batch_sounds_resource,self.input_model_B: batch_sounds_target,lr:opt})
                # update G network
                self.sess.run(g_optimizer,feed_dict={self.input_model_A: batch_sounds_resource, self.input_model_B: batch_sounds_target,lr:opt})
                T_c+=1
                if T==T_c:
                    T_c=0
                    max_lr=min_lr
                    min_lr*=0.1
                iterations+=1
        self.test_and_save(train_epoch,iterations,one_itr_num)
        taken_time_all=time.time()-start_time_all
        hour_display=taken_time_all//3600
        minute_display=taken_time_all//60%60
        second_display=int(taken_time_all%60)
        print(" [I] ALL train process finished successfully!! in %06d : %02d : %02d" % (hour_display, minute_display, second_display))

    def test_and_save(self,epoch,itr,one_itr_num):

        # last testing
        out_puts, _ = self.convert(self.test)

        # fixing harvests types
        out_put = out_puts.copy().astype(np.float32) / 32767.0

        # calculating power spectrum
        im = fft(out_put[800:156000])
        spec = np.mean(im, axis=0)
        spec_v = np.std(im, axis=0)
        diff=spec-self.label_spec
        diff2=spec_v-self.label_spec_v
        score=np.mean(diff*diff+diff2*diff2)
        otp_im = im.copy().reshape(1,-1,512)

        # writing epoch-result into tensor-board
        if self.args["tensor-board"]:
            tb_result = self.sess.run(self.loss_display,
                                      feed_dict={self.input_model_A: self.sounds_r[0:self.args["batch_size"]],
                                                 self.input_model_B: self.sounds_t[0:self.args["batch_size"]]})
            self.writer.add_summary(tb_result, itr)
            rs = self.sess.run(self.g_test_display, feed_dict={self.result_image_display: otp_im,self.result_score:score})
            self.writer.add_summary(rs, itr)

        # saving test harvests
        if os.path.exists(self.args["wave_otp_dir"]):

            # saving fake spectrum
            plt.clf()
            plt.subplot(3,1,1)
            ins = np.transpose(im, (1, 0))
            plt.imshow(ins, vmin=-15, vmax=5,aspect="auto")
            plt.subplot(3,1,2)
            plt.plot(out_put[800:156000])
            plt.ylim(-1,1)
            if self.args["real_sample_compare"]:
                plt.subplot(3, 1, 3)
                plt.plot(diff * diff + diff2 * diff2)
                plt.ylim(0, 100)
            name_save = "%s%04d.png" % (self.args["wave_otp_dir"],epoch)
            plt.savefig(name_save)
            name_save = "./latest.png"
            plt.savefig(name_save)
            
            #saving fake waves
            path_save = self.args["wave_otp_dir"] + datetime.now().strftime("%m-%d_%H-%M-%S") + "_" + str(epoch)
            voiced = out_puts.astype(np.int16)[800:156000]
            p = pyaudio.PyAudio()
            FORMAT = pyaudio.paInt16
            ww = wave.open(path_save + ".wav", 'wb')
            ww.setnchannels(1)
            ww.setsampwidth(p.get_sample_size(FORMAT))
            ww.setframerate(16000)
            ww.writeframes(voiced.reshape(-1).tobytes())
            ww.close()
            p.terminate()
        self.save(self.args["checkpoint_dir"],epoch,self.saver)
        taken_time = time.time() - self.start_time
        self.start_time = time.time()
        # output console
        if itr!=0:
            # eta
            self.tt_list.append(taken_time/one_itr_num)
            if len(self.tt_list) > 5:
                self.tt_list = self.tt_list[1:-1]
            eta = np.mean(self.tt_list) * (self.args["train_iteration"] - itr)
            print(" [I] Iteration %06d / %06d finished. ETA: %02d:%02d:%02d takes %2.3f secs" % (itr, self.args["train_iteration"], eta // 3600, eta // 60 % 60, int(eta % 60), taken_time))

        if self.args["real_sample_compare"]:
            print(" [I] Epoch %04d tested. score=%.5f" % (epoch,float(score)))
        else:
            print(" [I] Epoch %04d tested." % epoch)

    def save(self, checkpoint_dir, step,saver):
        model_name = "wave2wave.model"
        model_dir =  self.args["name_save"]
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    def load(self):
        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(" [I] Reading checkpoint...")
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(self.args["checkpoint_dir"], model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt is not None and ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

def encode(data):
    fs=16000
    _f0,t=pw.dio(data,fs)
    f0=pw.stonemask(data,_f0,t,fs)
    sp=pw.cheaptrick(data,f0,t,fs)
    ap=pw.d4c(data,f0,t,fs)
    return f0.astype(np.float64),np.clip((np.log(sp)+15)/20,-1.0,1.0).astype(np.float64),ap.astype(np.float64)

def decode(f0,sp,ap):
    ap = ap.reshape(-1, 513).astype(np.float)
    f0 = f0.reshape(-1).astype(np.float)
    sp = np.exp(sp.reshape(-1, 1, 513).astype(np.float) * 20 - 15)
    sp=sp.reshape(-1,513).astype(np.float)
    return pw.synthesize(f0,sp,ap,16000)


def wave_read(path_file):
    wf=wave.open(path_file,"rb")
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

def fft(data):

    time_ruler = data.shape[0] // 512
    if data.shape[0] % 512 == 0:
        time_ruler -= 1
    window = np.hamming(1024)
    pos = 0
    wined = np.zeros([time_ruler, 1024])
    for fft_index in range(time_ruler):
        frame = data[pos:pos + 1024]
        r=1024-frame.shape[0]
        if r>0:
            frame=np.pad(frame,(0,r),"constant")
        wined[fft_index] = frame * window
        pos += 512
    fft_r = np.fft.fft(wined, n=1024, axis=1)
    re = fft_r.real.reshape(time_ruler, -1)
    im = fft_r.imag.reshape(time_ruler, -1)
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1)[:,512:]
    return np.clip(c,-15.0,5.0)


if __name__ == '__main__':
    path="./setting.json"
    net = Model(path)
    net.train()