from glob import glob
import os
import time
import json
from datetime import datetime

import wave
import pyaudio
import pyworld as pw

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

from .model import discriminator,generator

class Model:
    def __init__(self,path):


        self.args=dict()

        # default setting paramater

        self.args["model_name"] = "VRC"
        self.args["version"] = "1.0.0"

        self.args["checkpoint_dir"]="./trained_models"
        self.args["wave_otp_dir"] = "./havests"
        self.args["train_data_dir"]="./datasets/train"
        self.args["test_data_dir"] ="./datasets/test"

        self.args["test"]=True
        self.args["tensorboard"]=False
        self.args["debug"] = False

        self.args["batch_size"] = 1
        self.args["input_size"] = 4096
        self.args["padding"]=1024

        self.args["weight_Cycle"]=150.0
        self.args["weight_GAN"] = 1.0
        self.args["train_iteration"]=60000
        self.args["start_epoch"]=0

        # reading json file

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
                                    " [W] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(
                                        type(self.args[k])) + "\"")
                        elif k[0] == "#":
                            pass
                        else:
                            print(" [W] Argument \"" + k + "\" is not exsits.")

        except json.JSONDecodeError as e:
            print(" [W] JSONDecodeError: ", e)
            print(" [W] Use default setting")

        a = np.load("./voice_profile.npy")
        self.args["pitch_rate_mean_s"] = a[0]
        self.args["pitch_rate_mean_t"] = a[1]
        self.args["pitch_rate_var"] = a[2]
        # initializing hidden paramaters

        self.args["padding_shift"] = self.args["padding"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        # shapes setting
        self.input_size_model=[self.args["batch_size"],52,513,1]
        self.input_size_test = [1, 52,513,1]
        self.output_size_model = [self.args["batch_size"], 65,513,1]


        if bool(self.args["debug"]):
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)

        if  self.args["wave_otp_dir"] is not "False" :
            self.args["wave_otp_dir"]=self.args["wave_otp_dir"]+ self.args["name_save"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])

        self.sess = tf.Session()
        self.build_model()
    def build_model(self):

        #inputs place holder
        self.input_model_A=tf.placeholder(tf.float32, self.input_size_model, "inputs_g_A")
        self.input_model_B = tf.placeholder(tf.float32, self.input_size_model, "inputs_g_B")
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_g_test")

        #creating generator
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1"):
                fake_aB_image = generator(self.input_model_A, reuse=None, train=True)
                self.fake_aB_image_test= generator(self.input_model_test, reuse=True, train=False)
            with tf.variable_scope("generator_2"):
                fake_bA_image = generator(self.input_model_B, reuse=None, train=True)
                fake_Ba_image = generator(fake_aB_image, reuse=True,train=True)
            with tf.variable_scope("generator_1",reuse=True):
                fake_Ab_image = generator(fake_bA_image, reuse=True,train=True)


        #creating discriminator
        with tf.variable_scope("discriminators"):
            with tf.variable_scope("discriminators_B"):
                d_judge_BF = discriminator(fake_aB_image, None)
                d_judge_BS = tf.stop_gradient(tf.clip_by_value(discriminator(self.input_model_B, True),0.0,1.0) * 1 - tf.clip_by_value(d_judge_BF,0.0,1.0))
                self.R_rate_B =1 - d_judge_BS * 0.5
                F_rate=d_judge_BS*0.5
                D_input_R_B = self.R_rate_B * self.input_model_B + F_rate * fake_aB_image
                d_judge_BR= discriminator(D_input_R_B, True)
            with tf.variable_scope("discriminators_A"):
                d_judge_AF = discriminator(fake_bA_image, None)
                d_judge_AS = tf.stop_gradient(tf.clip_by_value(discriminator(self.input_model_A, True),0.0,1.0) * 1 - tf.clip_by_value(d_judge_AF,0.0,1.0))
                self.R_rate_A = 1 - d_judge_AS * 0.5
                F_rate = d_judge_AS * 0.5
                D_input_R_A = self.R_rate_A * self.input_model_A + F_rate * fake_bA_image
                d_judge_AR = discriminator(D_input_R_A, True)

        #getting individual variabloes
        self.g_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discriminators")
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #objective-functions of discriminator
        d_loss_AR = tf.losses.mean_squared_error(labels=self.R_rate_A, predictions=d_judge_AR)
        d_loss_AF = tf.losses.mean_squared_error(labels=tf.zeros_like(d_judge_AF), predictions=d_judge_AF)
        d_loss_BR = tf.losses.mean_squared_error(labels=self.R_rate_B, predictions=d_judge_BR)
        d_loss_BF = tf.losses.mean_squared_error(labels=tf.zeros_like(d_judge_BF), predictions=d_judge_BF)

        d_loss_A=d_loss_AR + d_loss_AF
        d_loss_B=d_loss_BR + d_loss_BF
        self.d_loss=d_loss_A + d_loss_B


        # objective-functions of generator

        # Cycle lossA
        g_loss_cyc_A = tf.losses.mean_squared_error(predictions=fake_Ba_image,labels=self.input_model_A)* self.args["weight_Cycle"]

        # Gan lossB
        g_loss_gan_B = tf.losses.mean_squared_error(labels=tf.ones_like(d_judge_BF), predictions=d_judge_BF) * self.args["weight_GAN"]
        # generator lossA
        g_loss_aB = g_loss_cyc_A +g_loss_gan_B


        # Cyc lossB
        g_loss_cyc_B = tf.losses.mean_squared_error(predictions=fake_Ab_image, labels=self.input_model_B) * self.args["weight_Cycle"]

        # Gan lossA
        g_loss_gan_A = tf.losses.mean_squared_error(labels=tf.ones_like(d_judge_AF), predictions=d_judge_AF) * self.args["weight_GAN"]
        # generator lossB
        g_loss_bA = g_loss_cyc_B + g_loss_gan_A

        # generator loss
        self.g_loss = g_loss_aB+g_loss_bA

        #tensorboard functions
        g_loss_cyc_A_display= tf.summary.scalar("g_loss_cycle_AtoA", tf.reduce_mean(g_loss_cyc_A),family="g_loss")
        g_loss_gan_A_display = tf.summary.scalar("g_loss_gan_BtoA", tf.reduce_mean(g_loss_gan_A),family="g_loss")
        g_loss_sum_A_display = tf.summary.merge([g_loss_cyc_A_display, g_loss_gan_A_display])

        g_loss_cyc_B_display = tf.summary.scalar("g_loss_cycle_BtoB", tf.reduce_mean(g_loss_cyc_B),family="g_loss")
        g_loss_gan_B_display = tf.summary.scalar("g_loss_gan_AtoB", tf.reduce_mean(g_loss_gan_B),family="g_loss")
        g_loss_sum_B_display = tf.summary.merge([g_loss_cyc_B_display,g_loss_gan_B_display])

        d_loss_sum_A_display = tf.summary.scalar("d_loss_A", tf.reduce_mean(d_loss_A),family="d_loss")
        d_loss_sum_B_display = tf.summary.scalar("d_loss_B", tf.reduce_mean(d_loss_B),family="d_loss")


        self.loss_display=tf.summary.merge([g_loss_sum_A_display,g_loss_sum_B_display,d_loss_sum_A_display,d_loss_sum_B_display])

        self.result_image_display= tf.placeholder(tf.float32, [1,None,512], name="FakeSpectrum")
        image_pow_display=tf.reshape(tf.transpose(self.result_image_display[:,:,:],[0,2,1]),[1,512,-1,1])
        fake_B_image_display = tf.summary.image("Fake_spectrum_AtoB", image_pow_display, 1)
        self.g_test_display=tf.summary.merge([fake_B_image_display])

        #saver
        self.saver = tf.train.Saver()

    def convert(self,in_put):
        #function of test
        #To convert wave file

        conversion_start_time=time.time()
        input_size_one_term=self.args["input_size"]
        executing_times=in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"])==0:
            executing_times-=1
        otp=np.array([],dtype=np.int16)

        for t in range(executing_times):
            # Preprocess

            # Padiing
            start_pos=self.args["input_size"]*t+(in_put.shape[0]%self.args["input_size"])
            resorce=in_put[max(0,start_pos-input_size_one_term):start_pos]
            r=max(0,input_size_one_term-resorce.shape[0])
            if r>0:
                resorce=np.pad(resorce,(r,0),'constant')
            # FFT
            f0,resource,ap=encode((resorce/32767).astype(np.float))
            resource=resource.reshape(self.input_size_test)
            #main process

            # running network
            result=self.sess.run(self.fake_aB_image_test,feed_dict={ self.input_model_test:resource})

            # Postprocess

            # IFFT
            f0=(f0-self.args["pitch_rate_mean_s"])*self.args["pitch_rate_var"]+self.args["pitch_rate_mean_t"]
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
        opt_lr=tf.placeholder(tf.float64)
        with tf.control_dependencies(self.update_ops):
            g_optim = tf.train.AdamOptimizer(opt_lr, 0.5, 0.999).minimize(self.g_loss,var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(opt_lr, 0.5, 0.999).minimize(self.d_loss,var_list=self.d_vars)

        # logging
        if self.args["tensorboard"]:
            self.writer = tf.summary.FileWriter("./logs/"+self.args["name_save"], self.sess.graph)

        # loading net
        if self.load():
            print(" [I] Load SUCCESSED.")
        else:
            print(" [I] Load FAILED.")

        # loading training data directory
        data = glob(self.args["train_data_dir"]+'/A/*')
        data2 = glob(self.args["train_data_dir"] + '/B/*')
        # loading test data
        self.test=isread(self.args["test_data_dir"]+'/test.wav')
        # prepareing training-data
        batch_files = data[0]
        batch_files2 = data2[0]

        print(" [I] loading dataset...")
        self.sounds_r = np.load(batch_files)
        self.sounds_t = np.load(batch_files2)
        # times of one epoch
        train_data_num = min(self.sounds_r.shape[0],self.sounds_t.shape[0])
        self.batch_idxs = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num)]
        print(" [I] %d data loaded!!" % train_data_num)
        self.sounds_r=self.sounds_r.reshape([self.sounds_r.shape[0],self.sounds_r.shape[1],self.sounds_r.shape[2],1])
        self.sounds_t=self.sounds_t.reshape([self.sounds_t.shape[0],self.sounds_t.shape[1],self.sounds_t.shape[2],1])

        # initializing training infomation
        start_time_all=time.time()
        tt_list=list()
        start_time = time.time()
        train_epoch=self.args["train_iteration"]//self.batch_idxs+1
        iterations=0
        # main-training
        for epoch in range(train_epoch):
            # shuffling train_data_index
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)

            if self.args["test"] and epoch % 1 == 0:
                self.test_and_save(epoch)
            for idx in range(0, self.batch_idxs):
                # getting batch
                if iterations==self.args["train_iteration"]:
                    break
                st=self.args["batch_size"]*idx
                batch_sounds_resource = np.asarray([self.sounds_r[ind] for ind in index_list[st:st+self.args["batch_size"]]])
                batch_sounds_target= np.asarray([self.sounds_t[ind] for ind in index_list2[st:st+self.args["batch_size"]]])

                lr=1e-3*(0.1**(iterations//100000))
                # update D network
                self.sess.run([d_optim],feed_dict={self.input_model_A: batch_sounds_resource, self.input_model_B: batch_sounds_target,opt_lr:lr})
                # update G network
                self.sess.run([g_optim],feed_dict={self.input_model_A: batch_sounds_resource, self.input_model_B: batch_sounds_target,opt_lr:lr})
                iterations+=1
            # calculating ETA
            if iterations == self.args["train_iteration"]:
                break
            taken_time = time.time() - start_time
            start_time = time.time()
            tt_list.append(taken_time)
            if len(tt_list)>10:
                tt_list=tt_list[1:-1]
            eta=np.mean(tt_list)*(train_epoch-epoch-1)
            # console outputs
            print(" [I] Iteration %04d / %04d finished. ETA: %02d:%02d:%02d takes %2.3f secs" % (iterations,self.args["train_iteration"],eta//3600,eta//60%60,int(eta%60),taken_time))


        self.test_and_save(train_epoch)
        taken_time_all=time.time()-start_time_all
        hour_display=taken_time_all//3600
        minute_display=taken_time_all//60%60
        second_display=int(taken_time_all%60)
        print(" [I] ALL train process finished successfully!! in %04d : %02d : %02d" % (hour_display, minute_display, second_display))

    def test_and_save(self,epoch):

        # last testing
        out_puts, taken_time_test = self.convert(self.test)
        # fixing havests types
        out_put = out_puts.copy().astype(np.float32) / 32767.0

        # calcurating power spectrum
        time_ruler = out_put.shape[0] // 512
        if out_put.shape[0] % 512 == 0:
            time_ruler -= 1
        pos = 0
        windowed = np.zeros([time_ruler, 1024])
        window = np.hamming(1024)
        for fft_index in range(time_ruler):
            frame = out_put[pos:pos + 1024]
            #padding input
            r = 1024 - frame.shape[0]
            if r > 0:
                frame = np.pad(frame, (0, r), "constant")
            windowed[fft_index] = frame * window
            pos += 512
        fft_result = np.fft.fft(windowed, n=1024, axis=1)
        im = np.log(np.power(fft_result.real, 2) + np.power(fft_result.imag, 2) + 1e-24).reshape(time_ruler, -1, 1)[:,:512]
        otp_im = im.copy().reshape(1,-1,512)

        # writing epoch-result into tensorboard
        if self.args["tensorboard"]:
            tb_result = self.sess.run(self.loss_display,
                                      feed_dict={self.input_model_A: self.sounds_r[0:self.args["batch_size"]],
                                                 self.input_model_B: self.sounds_t[0:self.args["batch_size"]]})
            self.writer.add_summary(tb_result, epoch)
            rs = self.sess.run(self.g_test_display, feed_dict={self.result_image_display: otp_im})
            self.writer.add_summary(rs, epoch)

        # saving test harvests
        if os.path.exists(self.args["wave_otp_dir"]):
            # saving fake spectrum
            plt.clf()
            ins = np.transpose(im[:,:,0], (1, 0))
            plt.imshow(ins, vmin=-25, vmax=5,aspect="auto")
            plt.colorbar()
            path = "%s%04d.png" % (self.args["wave_otp_dir"],epoch)
            plt.savefig(path)
            #saving fake waves
            path = self.args["wave_otp_dir"] + datetime.now().strftime("%m-%d_%H-%M-%S") + "_" + str(epoch)
            voiced = out_puts.astype(np.int16)
            p = pyaudio.PyAudio()
            FORMAT = pyaudio.paInt16
            ww = wave.open(path + ".wav", 'wb')
            ww.setnchannels(1)
            ww.setsampwidth(p.get_sample_size(FORMAT))
            ww.setframerate(16000)
            ww.writeframes(voiced.reshape(-1).tobytes())
            ww.close()
            p.terminate()

        self.save(self.args["checkpoint_dir"],epoch,self.saver)
        print(" [I] Epoch %04d tested. " % epoch)

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
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.epoch=self.saver
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