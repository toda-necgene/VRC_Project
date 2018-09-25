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
        self.args["NFFT"]=1024

        self.args["g_lr_max"]=2e-4
        self.args["g_lr_min"] = 2e-6
        self.args["d_lr_max"] = 2e-4
        self.args["d_lr_min"] = 2e-6
        self.args["g_b1"] = 0.5
        self.args["g_b2"] = 0.999
        self.args["d_b1"] = 0.5
        self.args["d_b2"] = 0.999
        self.args["weight_Cycle_Pow"]=100.0
        self.args["weight_Cycle_Pha"]=100.0
        self.args["weight_GAN"] = 1.0
        self.args["train_epoch"]=1000
        self.args["start_epoch"]=0
        self.args["save_interval"]=10
        self.args["lr_decay_term"]=20

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
        except FileNotFoundError:
            print(" [W] Setting file is not found :", path)
            print(" [W] Use default setting")

        # initializing paramaters
        self.args["SHIFT"] = self.args["NFFT"]//2
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        # shapes of inputs
        ss=self.args["input_size"]//self.args["SHIFT"]
        self.input_size_model=[self.args["batch_size"],ss,self.args["NFFT"]//2,2]
        self.input_size_test = [1, ss, self.args["NFFT"] // 2, 2]

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

        input_model_A_fixed = self.input_model_A*0.1
        input_model_B_fixed = self.input_model_B*0.1
        input_model_test_fixed=self.input_model_test*0.1

        #creating generator
        with tf.variable_scope("generators"):

            with tf.variable_scope("generator_1"):
                fake_aB_image= generator(input_model_A_fixed, reuse=None, train=True)

                fake_aB_image_test_fixed= generator(input_model_test_fixed, reuse=True, train=False)
                self.fake_aB_image_test=fake_aB_image_test_fixed*10
            with tf.variable_scope("generator_2"):
                fake_bA_image = generator(input_model_B_fixed, reuse=None, train=True)

            with tf.variable_scope("generator_2",reuse=True):
                fake_Ba_image = generator(back_drop(fake_aB_image,0.75), reuse=True,train=True)
            with tf.variable_scope("generator_1",reuse=True):
                fake_Ab_image = generator(back_drop(fake_bA_image,0.75), reuse=True,train=True)

        #creating discriminator
        with tf.variable_scope("discrims"):

            with tf.variable_scope("discrimB"):
                d_judge_BR= discriminator(input_model_B_fixed, None)
                d_judge_BF = discriminator(fake_aB_image, True)

            with tf.variable_scope("discrimA"):
                d_judge_AR = discriminator(input_model_A_fixed, None)
                d_judge_AF = discriminator(fake_bA_image, True)

        #getting individual variabloes
        self.g_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generators")
        self.d_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discrims")
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #objective-functions of discriminator
        d_loss_AR = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like( d_judge_AR),predictions=d_judge_AR))
        d_loss_AF = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros_like(d_judge_AF),predictions=d_judge_AF))
        d_loss_BR = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like( d_judge_BR),predictions=d_judge_BR))
        d_loss_BF = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.zeros_like(d_judge_BF),predictions=d_judge_BF))

        d_loss_A=d_loss_AR + d_loss_AF
        d_loss_B=d_loss_BR + d_loss_BF
        self.d_loss=d_loss_A + d_loss_B


        # objective-functions of generator

        # Cycle lossA
        g_loss_cyc_A_pow=tf.reduce_mean(tf.losses.mean_squared_error(predictions=fake_Ba_image[:,:,:,0],labels=input_model_A_fixed[:,:,:,0]))* self.args["weight_Cycle_Pow"]
        g_loss_cyc_A_pha=tf.reduce_mean(tf.losses.mean_squared_error(predictions=fake_Ba_image[:,:,:,1],labels=input_model_A_fixed[:,:,:,1]))* self.args["weight_Cycle_Pha"]
        g_loss_cyc_A=0.5*(g_loss_cyc_A_pow+g_loss_cyc_A_pha)

        # Gan lossB
        g_loss_gan_B = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(d_judge_BF),predictions=d_judge_BF))* self.args["weight_GAN"]

        # generator lossA
        self.g_loss_aB = g_loss_cyc_A +g_loss_gan_B


        # Cyc lossB
        g_loss_cyc_B_pow=tf.reduce_mean(tf.losses.mean_squared_error(predictions=fake_Ab_image[:,:,:,0],labels=input_model_B_fixed[:,:,:,0]))* self.args["weight_Cycle_Pow"]
        g_loss_cyc_B_pha=tf.reduce_mean(tf.losses.mean_squared_error(predictions=fake_Ab_image[:,:,:,1],labels=input_model_B_fixed[:,:,:,1] ))* self.args["weight_Cycle_Pha"]
        g_loss_cyc_B = 0.5*(g_loss_cyc_B_pow+g_loss_cyc_B_pha)

        # Gan lossA
        g_loss_gan_A = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.ones_like(d_judge_AF),predictions=d_judge_AF))* self.args["weight_GAN"]

        # generator lossB
        self.g_loss_bA = g_loss_cyc_B + g_loss_gan_A

        # generator loss
        self.g_loss=self.g_loss_aB+self.g_loss_bA


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
        self.result_image_display= tf.placeholder(tf.float32, [1,None,self.args["NFFT"],2], name="FBI0")
        image_pow_display=tf.transpose(self.result_image_display[:,:,:,:1],[0,2,1,3])
        image_pha_display = tf.transpose(self.result_image_display[:, :, :, 1:], [0, 2, 1, 3])
        fake_B_audio_display = tf.summary.audio("fake_B", tf.reshape(self.result_audio_display,[1,160000,1]), 16000, 1)
        fake_B_image_display = tf.summary.merge([tf.summary.image("fake_B_image_power", image_pow_display, 1),tf.summary.image("fake_B_image_phase", image_pha_display, 1)])
        self.g_test_pow_dif=tf.placeholder(tf.float32,name="g_test_epoch_end")
        g_test_value_display = tf.summary.scalar("g_test_power_distance", self.g_test_pow_dif,family="test")
        self.g_test_display=tf.summary.merge([fake_B_audio_display,fake_B_image_display,g_test_value_display])

        #saver
        #保存の準備
        self.saver = tf.train.Saver()
        self.saver_best = tf.train.Saver(max_to_keep=1)

    def convert(self,in_put):
        #function of test
        #To convert wave file

        conversion_start_time=time.time()
        input_size_one_term=self.args["input_size"]+self.args["SHIFT"]
        executing_times=in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"])==0:
            executing_times-=1
        otp=np.array([],dtype=np.int16)

        res_image = np.zeros([1,self.args["NFFT"],2], dtype=np.float32)
        remain_wave=np.zeros([self.input_size_model[2]],dtype=np.float64)
        for t in range(executing_times):
            # Preprocess

            # Padiing
            start_pos=self.args["input_size"]*(1+t)+(in_put.shape[0]%self.args["input_size"])
            resorce=in_put[max(0,start_pos-input_size_one_term):start_pos]
            r=max(0,input_size_one_term-resorce.shape[0])
            if r>0:
                resorce=np.pad(resorce,(r,0),'constant')
            # FFT
            resource=self.fft(resorce/32767.0)
            resource = resource[:, :self.args["SHIFT"], :].reshape([1, -1, self.args["SHIFT"], 2])

            #main process

            # running network
            result=self.sess.run(self.fake_aB_image_test,feed_dict={ self.input_model_test:resource})


            # Postprocess

            #fixing spectrogrum
            result_reverse = result.copy()[:, :, ::-1, :]
            result = np.append(result, result_reverse, axis=2)
            result[:, :, self.args["SHIFT"]:, 1] *= -1
            res_image = np.append(res_image, result[0].copy(),axis=0)

            # IFFT
            result_wave,remain_wave=self.ifft(result[0].copy(),remain_wave)

            result_wave_fixed=np.clip(result_wave,-1.0,1.0)
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
        lr_g_opt_min=self.args["g_lr_min"]
        beta_g_opt=self.args["g_b1"]
        beta_2_g_opt=self.args["g_b2"]
        lr_d_opt_max=self.args["d_lr_max"]
        lr_d_opt_min = self.args["d_lr_min"]
        beta_d_opt=self.args["d_b1"]
        beta_2_d_opt=self.args["d_b2"]
        T_cur=0
        T_pow=1.0
        self.best=999999
        T=self.args["lr_decay_term"]

        # naming output-directory
        lr_g = tf.placeholder(tf.float32, None, name="g_lr")
        lr_d = tf.placeholder(tf.float32, None, name="d_lr")
        g_optim = tf.train.AdamOptimizer(lr_g, beta_g_opt, beta_2_g_opt).minimize(self.g_loss,
                                                                                  var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(lr_d, beta_d_opt, beta_2_d_opt).minimize(self.d_loss,
                                                                                  var_list=self.d_vars)

        tt_list=list()

        # logging
        if self.args["tensorboard"]:
            self.writer = tf.summary.FileWriter("./logs/"+self.args["name_save"], self.sess.graph)

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
        label=isread(self.args["test_data_dir"]+'/label.wav')[0:160000].astype(np.float32)

        # prepareing test-target-spectrum
        input_size_one_term = self.args["input_size"] + self.args["SHIFT"]
        out_spectrum= np.zeros([1, self.args["NFFT"], 2], dtype=np.float32)
        fft_executing_times = label.shape[0] // (self.args["input_size"]) + 1
        if label.shape[0] % ((self.args["input_size"]) * self.args["batch_size"]) == 0:
            fft_executing_times -= 1

        # making test-target-spectrum

        for t in range(fft_executing_times):

            # Padiing
            start_pos = self.args["input_size"] * t + (label.shape[0] % self.args["input_size"])
            resorce = label[ max(0, start_pos - input_size_one_term):start_pos]
            r = max(0, input_size_one_term - resorce.shape[0])
            if r > 0:
                resorce = np.pad(resorce, (r, 0), 'constant')

            # FFT
            result_fft = self.fft(resorce / 32767.0)
            out_spectrum=np.append(out_spectrum,result_fft,axis=0)

        self.label_spectrum=out_spectrum[1:]

        # times of one epoch
        train_data_num = min(len(data),len(data2))
        self.batch_idxs = train_data_num // self.args["batch_size"]
        index_list=[h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num)]

        #　prepareing training-data
        batch_files = data[:train_data_num]
        batch_files2 = data2[:train_data_num]

        print(" [I] loading dataset...")
        self.sounds_r = np.asarray([(imread(batch_file)) for batch_file in batch_files])
        self.sounds_t = np.asarray([(imread(batch_file)) for batch_file in batch_files2])
        print(" [I] %d data loaded!!",train_data_num)

        # initializing training infomation
        start_time = time.time()
        start_time_all=time.time()

        for epoch in range(self.args["train_epoch"]):

            # calculating learning-rate
            lr_d_culced = lr_d_opt_min + 0.5*(lr_d_opt_max - lr_d_opt_min)*np.cos(np.pi*0.5*(T_cur/T)) * T_pow
            lr_g_culced = lr_g_opt_min + 0.5*(lr_g_opt_max - lr_g_opt_min)*np.cos(np.pi*0.5*(T_cur/T)) * T_pow
            # lr_d_culced=lr_d_opt_max
            # lr_g_culced = lr_g_opt_max

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
                self.sess.run([g_optim, self.update_ops],
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
                T_pow*=0.5

        self.test_and_save(self.args["train_epoch"])
        tnt=time.time()-start_time_all
        hour_f=tnt//3600
        minute_f=tnt//60%60
        second_f=int(tnt%60)
        print(" [I] All finished successfully!! in %04d : %02d : %02d"%(hour_f,minute_f,second_f))

    def test_and_save(self,epoch):

        # last testing
        out_puts, im, taken_time_test = self.convert(self.test)

        # fixing havests types
        out_put = out_puts.copy().astype(np.float32) / 32767.0
        otp_im = im.copy().reshape(1,-1,self.args["NFFT"],2)
        otp_im[:,:, :, 0] = np.clip((otp_im[:, :, :, 0] + 10.0) / 20.0, 0.0, 1.0)
        otp_im[:,:, :, 1] = np.clip((otp_im[:, :, :, 1] + 3.15) / 6.30, 0.0, 1.0)
        r = min(self.label_spectrum.shape[0], im.shape[0])
        test_score = np.mean(np.abs(self.label_spectrum[:r, :, 0] - im[:r, :, 0]))

        # writing epoch-result into tensorboard
        if self.args["tensorboard"]:
            tb_result = self.sess.run(self.loss_display,
                                      feed_dict={self.input_model_A: self.sounds_r[0:self.args["batch_size"]],
                                                 self.input_model_B: self.sounds_t[0:self.args["batch_size"]]})
            self.writer.add_summary(tb_result, self.batch_idxs * epoch)
            rs = self.sess.run(self.g_test_display, feed_dict={self.result_audio_display: out_put.reshape(1, 1, -1),
                                                               self.result_image_display: otp_im,
                                                               self.g_test_pow_dif: test_score})
            self.writer.add_summary(rs, epoch)

        # saving test havests
        if os.path.exists(self.args["wave_otp_dir"]):
            plt.clf()
            plt.subplot(211)
            ins = np.transpose(im[:, :, 0], (1, 0))
            plt.imshow(ins, aspect="auto")
            plt.clim(-10, 10)
            plt.colorbar()
            plt.subplot(212)
            ins = np.transpose(im[:, :, 1], (1, 0))
            plt.imshow(ins, aspect="auto")
            plt.clim(-3.141593, 3.141593)
            plt.colorbar()
            path = self.args["wave_otp_dir"] + nowtime() + "_e" + str(epoch)
            plt.savefig(path + ".png")
            upload(out_puts, path)

        print(" [I] Epoch %04d tested. score: %2.3f " % (epoch, float(test_score)))

        self.save(self.args["checkpoint_dir"], epoch, self.saver)
        if test_score < self.best:
            self.save(self.args["best_checkpoint_dir"], epoch, self.saver_best)


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
        # 変数の初期化
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

    def fft(self,data):

        time_ruler = data.shape[0] // self.args["SHIFT"]-1
        window = np.hamming(self.args["NFFT"])
        pos = 0
        wined = np.zeros([time_ruler, self.args["NFFT"]])
        for fft_index in range(time_ruler):
            frame = data[pos:pos + self.args["NFFT"]]
            wined[fft_index] = frame * window
            pos += self.args["SHIFT"]
        fft_r = np.fft.fft(wined, n=self.args["NFFT"], axis=1)
        re = fft_r.real.reshape(time_ruler, -1)
        im = fft_r.imag.reshape(time_ruler, -1)
        c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
        c = np.clip(c, -10, 10)
        d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
        spec = np.concatenate((c, d), 2)
        return spec
    def ifft(self,data,redi):
        a=data
        a[:, :, 0]=np.clip(a[:, :, 0],a_min=-10,a_max=88)
        sss=np.exp(a[:,:,0])
        p = np.sqrt(sss)
        r = p * (np.cos(a[:, :, 1]))
        i = p * (np.sin(a[:, :, 1]))
        dds = np.concatenate((r.reshape(r.shape[0], r.shape[1], 1), i.reshape(i.shape[0], i.shape[1], 1)), 2)
        data=dds[:,:,0]+1j*dds[:,:,1]
        fft_s = np.fft.ifft(data,n=self.args["NFFT"], axis=1)
        fft_data = fft_s.real
        # fft_data[:]/=window
        v = fft_data[:, :self.args["NFFT"]// 2]
        reds = fft_data[-1, self.args["NFFT"] // 2:].copy()
        lats = np.roll(fft_data[:, self.args["NFFT"] // 2:], 1, axis=0 )
        lats[0, :]=redi
        spec = np.reshape(v + lats, (-1))
        return spec,reds
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
def imread(path):
    return np.load(path)