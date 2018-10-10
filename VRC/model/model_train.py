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
from datetime import datetime
import json
from .model import discriminator,generator
import matplotlib.pyplot as plt
import pyworld as pw

class Model:
    def __init__(self,path):


        self.args=dict()

        # default setting
        self.args["model_name"] = "wave2wave"
        self.args["version"] = "1.0.0"

        self.args["checkpoint_dir"]="./trained_models"
        self.args["best_checkpoint_dir"]="./best_model"
        self.args["wave_otp_dir"] = "./havests"
        self.args["train_data_dir"]="./datasets/train"
        self.args["test_data_dir"] ="./datasets/test"

        self.args["test"]=True
        self.args["tensorboard"]=True
        self.args["debug"] = False

        self.args["batch_size"] = 8
        self.args["input_size"] = 4096
        self.args["dilated_size"] = 0
        self.args["NFFT"]=1024
        self.args["KEPFILTERE"]=64

        self.args["g_lr_max"]=2e-4
        self.args["g_lr_min"] = 2e-6
        self.args["d_lr_max"] = 2e-4
        self.args["d_lr_min"] = 2e-6
        self.args["g_b1"] = 0.5
        self.args["g_b2"] = 0.999
        self.args["d_b1"] = 0.5
        self.args["d_b2"] = 0.999
        self.args["weight_Cycle_Pow"]=100.0
        self.args["weight_Cycle_f0"]=0.1
        self.args["weight_GAN"] = 1.0
        self.args["train_epoch"]=1000
        self.args["pre_train_epoch"] = 50
        self.args["P_train_epoch"]=100
        self.args["start_epoch"]=0
        self.args["save_interval"]=10
        self.args["lr_decay_term"]=20
        self.args["pitch_rate"]=1.0
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

        # initializing paramaters
        self.args["SHIFT"] = self.args["NFFT"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        # shapes of inputs
        self.input_size_model=[self.args["batch_size"],58,513,1]
        self.input_size_test = [1, 58,513,1]

        self.output_size_model = [self.args["batch_size"], 58,513,1]

        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))

        if bool(self.args["debug"]):
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
        if  self.args["wave_otp_dir"] is not "False" :
            self.args["wave_otp_dir"]=self.args["wave_otp_dir"]+ self.args["name_save"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])

        self.build_model()
    def build_model(self):

        #inputs place holder
        self.input_model_A=tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net_A")
        self.input_model_B = tf.placeholder(tf.float32, self.input_size_model, "inputs_G-net_B")

        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_G-net_test")

        input_model_A_fixed = self.input_model_A
        input_model_B_fixed = self.input_model_B
        input_model_test_fixed=self.input_model_test
        #creating generator
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1"):
                fake_aB_image12 = generator(input_model_A_fixed[:,:self.input_size_test[1],:,:], reuse=None, train=True)
                self.fake_aB_image_test_fixed= generator(input_model_test_fixed, reuse=True, train=False)
            with tf.variable_scope("generator_2"):
                fake_bA_image12 = generator(input_model_B_fixed[:,:self.input_size_test[1],:,:], reuse=None, train=True)
            with tf.variable_scope("generator_2",reuse=True):
                fake_Ba_image = generator(back_drop(fake_aB_image12[:,-self.input_size_test[1]:,:,:],0.75), reuse=True,train=True)
            with tf.variable_scope("generator_1",reuse=True):
                fake_Ab_image = generator(back_drop(fake_bA_image12[:,-self.input_size_test[1]:,:,:],0.75), reuse=True,train=True)

        #creating discriminator
        with tf.variable_scope("discrims"):

            with tf.variable_scope("discrimB"):
                d_judge_BR= discriminator(input_model_B_fixed[:,-self.input_size_test[1]:,:,:1], None)
                d_judge_BF = discriminator(fake_aB_image12, True)
                d_judge_AR = discriminator(input_model_A_fixed[:,-self.input_size_test[1]:,:,:1], True)
                d_judge_AF = discriminator(fake_bA_image12, True)

        #getting individual variabloes
        self.g_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrims")
        self.p_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "phase_decoder")
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #objective-functions of discriminator
        time_length=int(d_judge_AR.shape[1])
        label_one=tf.tile(tf.reshape(tf.one_hot(0,3),[1,1,3]),[self.args["batch_size"],time_length,1])
        label_two=tf.tile(tf.reshape(tf.one_hot(1,3),[1,1,3]),[self.args["batch_size"],time_length,1])
        label_three=tf.tile(tf.reshape(tf.one_hot(2,3),[1,1,3]),[self.args["batch_size"],time_length,1])
        d_loss_AR = tf.reduce_mean(tf.losses.mean_squared_error(labels=label_one,predictions=d_judge_AR))
        d_loss_AF = tf.reduce_mean(tf.losses.mean_squared_error(labels=label_three,predictions=d_judge_AF))
        d_loss_BR = tf.reduce_mean(tf.losses.mean_squared_error(labels=label_two,predictions=d_judge_BR))
        d_loss_BF = tf.reduce_mean(tf.losses.mean_squared_error(labels=label_three,predictions=d_judge_BF))

        d_loss_A=d_loss_AR + d_loss_AF
        d_loss_B=d_loss_BR + d_loss_BF
        self.d_loss=d_loss_A + d_loss_B


        # objective-functions of generator

        # Cycle lossA
        g_loss_cyc_A=tf.reduce_mean(tf.losses.mean_squared_error(predictions=fake_Ba_image[:,:,:-1,0],labels=input_model_A_fixed[:,-self.output_size_model[1]:,:-1,0]))* self.args["weight_Cycle_Pow"]
        g_loss_cyc_A += tf.reduce_mean(tf.losses.mean_squared_error(predictions=fake_Ba_image[:, :, -1:, 0],labels=input_model_A_fixed[:,-self.output_size_model[1]:, -1:, 0]))*self.args["weight_Cycle_f0"]

        # Gan lossB
        g_loss_gan_B = tf.losses.mean_squared_error(labels=label_two,predictions=d_judge_BF)* self.args["weight_GAN"]

        # generator lossA
        self.g_loss_aB = g_loss_cyc_A +g_loss_gan_B


        # Cyc lossB
        g_loss_cyc_B=tf.losses.mean_squared_error(predictions=fake_Ab_image[:,:,:-1,0],labels=input_model_B_fixed[:,-self.output_size_model[1]:,:-1,0])* self.args["weight_Cycle_Pow"]
        g_loss_cyc_B += tf.losses.mean_squared_error(predictions=fake_Ab_image[:, :, -1:, 0],labels=input_model_B_fixed[:, -self.output_size_model[1]:, -1:, 0]) *self.args["weight_Cycle_f0"]

        # Gan lossA
        g_loss_gan_A = tf.losses.mean_squared_error(labels=label_one,predictions=d_judge_AF)* self.args["weight_GAN"]

        # generator lossB
        self.g_loss_bA = g_loss_cyc_B + g_loss_gan_A

        # generator loss
        self.g_loss=self.g_loss_aB+self.g_loss_bA
        self.g_loss_pre = g_loss_cyc_B + g_loss_cyc_A
        #tensorboard functions
        g_loss_cyc_A_display= tf.summary.scalar("g_loss_cycle_A", tf.reduce_mean(g_loss_cyc_A),family="g_loss")
        g_loss_gan_A_display = tf.summary.scalar("g_loss_gan_A", tf.reduce_mean(g_loss_gan_A),family="g_loss")
        g_loss_sum_A_display = tf.summary.merge([g_loss_cyc_A_display, g_loss_gan_A_display])

        g_loss_cyc_B_display = tf.summary.scalar("g_loss_cycle_B", tf.reduce_mean(g_loss_cyc_B),family="g_loss")
        g_loss_gan_B_display = tf.summary.scalar("g_loss_gan_B", tf.reduce_mean(g_loss_gan_B),family="g_loss")
        g_loss_sum_B_display = tf.summary.merge([g_loss_cyc_B_display,g_loss_gan_B_display])

        d_loss_sum_A_display = tf.summary.scalar("d_lossA", tf.reduce_mean(d_loss_A),family="d_loss")
        d_loss_sum_B_display = tf.summary.scalar("d_lossB", tf.reduce_mean(d_loss_B),family="d_loss")


        self.loss_display=tf.summary.merge([g_loss_sum_A_display,g_loss_sum_B_display,d_loss_sum_A_display,d_loss_sum_B_display])

        self.result_audio_display= tf.placeholder(tf.float32, [1,1,160000], name="FB")
        self.result_image_display= tf.placeholder(tf.float32, [1,None,513], name="FBI0")
        image_pow_display=tf.reshape(tf.transpose(self.result_image_display[:,:,:],[0,2,1]),[1,513,-1,1])
        fake_B_audio_display = tf.summary.audio("fake_B", tf.reshape(self.result_audio_display,[1,160000,1]), 16000, 1)
        fake_B_image_display = tf.summary.image("fake_B_image_power", image_pow_display, 1)
        self.g_test_display=tf.summary.merge([fake_B_audio_display,fake_B_image_display])

        #saver
        self.saver = tf.train.Saver()
        self.saver_best = tf.train.Saver(max_to_keep=1)

    def convert(self,in_put):
        #function of test
        #To convert wave file

        conversion_start_time=time.time()
        input_size_one_term=self.args["input_size"]+self.args["SHIFT"]+self.args["dilated_size"]*self.args["SHIFT"]
        executing_times=in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"])==0:
            executing_times-=1
        otp=np.array([],dtype=np.int16)

        res_image = np.zeros([1,self.input_size_model[2]-1,1], dtype=np.float32)
        for t in range(executing_times):
            # Preprocess

            # Padiing
            start_pos=self.args["input_size"]*t+(in_put.shape[0]%self.args["input_size"])
            resorce=in_put[max(0,start_pos-input_size_one_term):start_pos]
            r=max(0,input_size_one_term-resorce.shape[0])
            if r>0:
                resorce=np.pad(resorce,(r,0),'constant')
            # FFT
            f0,resource,ap=encode((resorce/32767.0).astype(np.double))
            resource=resource.reshape(self.input_size_test)
            #main process

            # running network
            result=self.sess.run(self.fake_aB_image_test_fixed,feed_dict={ self.input_model_test:resource})

            res_image = np.append(res_image, result[0,:,:-1,:].copy(), axis=0)
            # Postprocess

            # IFFT
            result_wave=decode(f0*self.args["pitch_rate"],result[0].copy(),ap)

            result_wave_fixed=np.clip(result_wave,-1.0,1.0)[-self.args["input_size"]:]
            result_wave_int16=result_wave_fixed*32767

            # converting result
            result_wave_int16=result_wave_int16.reshape(-1).astype(np.int16)

            #adding result
            otp=np.append(otp,result_wave_int16)

        h=otp.shape[0]-in_put.shape[0]
        if h>0:
            otp=otp[h:]

        return otp,res_image[1:],time.time()-conversion_start_time



    def train(self):

        # setting paramaters
        lr_g_opt_max=self.args["g_lr_max"]
        beta_g_opt=self.args["g_b1"]
        beta_2_g_opt=self.args["g_b2"]
        lr_d_opt_max=self.args["d_lr_max"]
        beta_d_opt=self.args["d_b1"]
        beta_2_d_opt=self.args["d_b2"]
        T_cur=0
        self.best=999999
        T=self.args["lr_decay_term"]

        # naming output-directory
        lr_g = tf.placeholder(tf.float32, None, name="g_lr")
        lr_d = tf.placeholder(tf.float32, None, name="d_lr")
        g_optim_pre = tf.train.AdamOptimizer(1e-4, 0.9).minimize(self.g_loss_pre,var_list=self.g_vars)

        g_optim = tf.train.AdamOptimizer(lr_g, beta_g_opt, beta_2_g_opt).minimize(self.g_loss,
                                                                                  var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(lr_d, beta_d_opt, beta_2_d_opt).minimize(self.d_loss,
                                                                                  var_list=self.d_vars)

        tt_list=list()

        # logging
        if self.args["tensorboard"]:
            self.writer = tf.summary.FileWriter("./logs/A"+self.args["name_save"], self.sess.graph)

        # loading net
        if self.load():
            print(" [I] Load SUCCESSED.")
        else:
            print(" [I] Load FAILED.")

        # loading training data directory
        data = glob(self.args["train_data_dir"]+'/Source_data/*')
        data2 = glob(self.args["train_data_dir"] + '/Answer_data/*')
        # loading test data
        self.test=isread(self.args["test_data_dir"]+'/test.wav')[0:160000].astype(np.float32)

        # times of one epoch
        train_data_num = min(len(data),len(data2))
        self.batch_idxs = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num)]

        # prepareing training-data
        batch_files = data[:train_data_num]
        batch_files2 = data2[:train_data_num]

        print(" [I] loading dataset...")
        self.sounds_r = np.asarray([(imread(batch_file,self.input_size_model[1:])) for batch_file in batch_files])
        self.sounds_t = np.asarray([(imread(batch_file,self.input_size_model[1:])) for batch_file in batch_files2])
        print(" [I] %d data loaded!!" % train_data_num)

        # initializing training infomation
        start_time = time.time()
        start_time_all=time.time()
        for epoch in range(self.args["pre_train_epoch"]):
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)
            for idx in xrange(0, self.batch_idxs):
                st = self.args["batch_size"] * idx
                batch_sounds_resource = np.asarray(
                    [self.sounds_r[ind] for ind in index_list[st:st + self.args["batch_size"]]])
                batch_sounds_target = np.asarray(
                    [self.sounds_t[ind] for ind in index_list2[st:st + self.args["batch_size"]]])
                # update G network
                self.sess.run([g_optim_pre, self.update_ops],feed_dict={self.input_model_A: batch_sounds_resource, self.input_model_B: batch_sounds_target})
            print(" [I] Epoch %04d / %04d finished." % (epoch, self.args["pre_train_epoch"]))
        for epoch in range(self.args["train_epoch"]):

            # calculating learning-rate
            lr_d_culced=lr_d_opt_max
            lr_g_culced = lr_g_opt_max

            # shuffling train_data_index
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)

            prepareing_time_total = 0.0

            if self.args["test"] and epoch%self.args["save_interval"]==0:
               self.test_and_save(epoch)

            for idx in xrange(0, self.batch_idxs):
                start_preparing = time.time()
                # getting batch
                st=self.args["batch_size"]*idx
                batch_sounds_resource = np.asarray([self.sounds_r[ind] for ind in index_list[st:st+self.args["batch_size"]]])
                batch_sounds_target= np.asarray([self.sounds_t[ind] for ind in index_list2[st:st+self.args["batch_size"]]])
                # getting training data
                prepareing_time_total+=time.time()-start_preparing

                # update D network (2time)
                for _ in range(2):
                    self.sess.run([d_optim],
                                  feed_dict={self.input_model_A: batch_sounds_resource, self.input_model_B: batch_sounds_target,lr_d:lr_d_culced})
                # update G network
                self.sess.run([g_optim,  self.update_ops],
                              feed_dict={self.input_model_A: batch_sounds_resource, self.input_model_B: batch_sounds_target, lr_g: lr_g_culced})

            # calculating ETA
            taken_time = time.time() - start_time
            start_time = time.time()
            tt_list.append(taken_time)
            if len(tt_list)>20:
                tt_list=tt_list[0:20]
            eta=np.mean(tt_list)*(self.args["train_epoch"]-epoch-1)

            # console outputs
            print(" [I] Epoch %04d / %04d finished. ETA: %02d:%02d:%02d takes %3.2f secs(preprocess %2.3f secs)" % (epoch,self.args["train_epoch"],eta//3600,eta//60%60,int(eta%60),taken_time,prepareing_time_total))

            T_cur += 1

            # update learning_rate
            if T==T_cur:
                T=T*2
                T_cur=0

        self.test_and_save(self.args["train_epoch"])
        tnt=time.time()-start_time_all
        hour_f=tnt//3600
        minute_f=tnt//60%60
        second_f=int(tnt%60)
        if self.args["tensorboard"]:
            self.writer = tf.summary.FileWriter("./logs/B"+self.args["name_save"], self.sess.graph)

        print(" [I] ALL train process finished successfully!! in %04d : %02d : %02d" % (hour_f, minute_f, second_f))

    def test_and_save(self,epoch):

        # last testing
        out_puts, im, taken_time_test = self.convert(self.test)

        # fixing havests types
        out_put = out_puts.copy().astype(np.float32) / 32767.0
        otp_im = im.copy().reshape(1,-1,513)
        im = fft(out_puts)

        # writing epoch-result into tensorboard
        if self.args["tensorboard"]:
            tb_result = self.sess.run(self.loss_display,
                                      feed_dict={self.input_model_A: self.sounds_r[0:self.args["batch_size"]],
                                                 self.input_model_B: self.sounds_t[0:self.args["batch_size"]]})
            self.writer.add_summary(tb_result, self.batch_idxs * epoch)
            rs = self.sess.run(self.g_test_display, feed_dict={self.result_audio_display: out_put.reshape(1, 1, -1),
                                                               self.result_image_display: otp_im})
            self.writer.add_summary(rs, epoch)

        # saving test havests
        if os.path.exists(self.args["wave_otp_dir"]):
            plt.clf()
            ins = np.transpose(im[:,:,0], (1, 0))
            plt.imshow(ins, aspect="auto")
            plt.colorbar()
            path = self.args["wave_otp_dir"] + nowtime() + "_e" + str(epoch)
            plt.savefig(path + ".png")
            upload(out_puts, path)
        self.save(self.args["checkpoint_dir"],epoch,self.saver)
        print(" [I] Epoch %04d tested. time: %2.3f " % (epoch,taken_time_test))


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
def fft(data):
    NFFT=1024
    SHIFT=512
    time_ruler = data.shape[0] // SHIFT
    if data.shape[0] % SHIFT == 0:
        time_ruler -= 1
    pos = 0
    wined = np.zeros([time_ruler, NFFT])
    window=np.hamming(NFFT)
    for fft_index in range(time_ruler):
        frame = data[pos:pos + NFFT]
        r=NFFT-frame.shape[0]
        if r>0:
            frame=np.pad(frame,(0,r),"constant")
        wined[fft_index] = frame*window
        pos += SHIFT
    fft_r = np.fft.fft(wined, n=NFFT, axis=1)
    re = fft_r.real.reshape(time_ruler, -1)
    im = fft_r.imag.reshape(time_ruler, -1)
    spec = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
    return spec
def encode(data):
    fs=16000
    _f0,t=pw.dio(data,fs)
    f0=pw.stonemask(data,_f0,t,fs)
    sp=pw.cheaptrick(data,f0,t,fs)
    ap=pw.d4c(data,f0,t,fs)
    sp=np.log(sp)
    return f0,sp,ap
def decode(f0,sp,ap):
    return pw.synthesize(f0,np.exp(sp),ap,16000)

def back_drop(ten,rate):
    s = ten.get_shape()
    prop = tf.random_uniform(s, 0.0, 1.0) + rate
    prop = tf.floor(prop)
    tenA = ten * prop
    tenB = ten * (1 - prop)
    return tenA+tf.stop_gradient(tenB)
def nowtime():
    return datetime.now().strftime("%Y_%m_%d %H_%M_%S")

def upload(voice,to):
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
def imread(path,size):
    ns=np.load(path)
    return ns.reshape(size)