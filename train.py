"""
製作者:TODA


実行すれば学習ができる。
設定はsetting.jsonを利用する。
"""
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
from model import discriminator, generator
from voice_to_dataset_cycle import create_dataset
class Model:
    """
    学習用モデルの定義です。
    """
    def __init__(self, path_setting):
        self.start_time = time.time()
        self.tt_list = list()
        self.test = None
        self.label = None
        self.label_spec = None
        self.label_spec_v = None
        self.sounds_r = None
        self.sounds_t = None
        self.loop_num = 0

        # setting default parameters

        self.args = dict()

        # name options
        self.args["model_name"] = "VRC"
        self.args["version"] = "1.0.0"
        # saving options
        self.args["checkpoint_dir"] = "./trained_models"
        self.args["wave_otp_dir"] = "./harvests"
        #training-data options
        self.args["use_old_dataset"] = False
        self.args["train_data_dir"] = "./dataset/train"
        self.args["test_data_dir"] = "./dataset/test"
        # learning details output options
        self.args["test"] = True
        self.args["tensor-board"] = False
        self.args["real_sample_compare"] = False
        # learning options
        self.args["batch_size"] = 1
        self.args["weight_cycle"] = 100.0
        self.args["train_iteration"] = 600000
        self.args["start_epoch"] = 0
        self.args["learning_rate"] = 8e-7
        self.args["test_interval"] = 1
        # architecture option
        self.args["input_size"] = 4096


        # loading json setting file
        # (more codes ./setting.json. manual is exist in ./setting-example.json)
        with open(path_setting, "r") as setting_raw_txt:
            try:
                json_loaded = json.load(setting_raw_txt)
                keys = json_loaded.keys()
                for j in keys:
                    data = json_loaded[j]
                    keys2 = data.keys()
                    for k in keys2:
                        if k in self.args:
                            if isinstance(self.args[k], type(data[k])):
                                self.args[k] = data[k]
                            else:
                                print(" [W] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(type(self.args[k])) + "\"")
                        elif k[0] == "#":
                            pass
                        else:
                            print(" [W] Argument \"" + k + "\" is not exsits.")
            except json.JSONDecodeError as er_message:
                print(" [W] JSONDecodeError: ", er_message)
                print(" [W] Use default setting")
        # shapes properties
        self.input_size_model = [self.args["batch_size"], 52, 1, 513]
        self.input_size_test = [1, 52, 1, 513]
        self.output_size_model = [self.args["batch_size"], 65, 1, 513]

        # initializing harvest directory
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        if  self.args["wave_otp_dir"] is not "False":
            self.args["wave_otp_dir"] = self.args["wave_otp_dir"] + self.args["name_save"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])


        # creating data-set

        if not self.args["use_old_dataset"]:
            create_dataset()

        # loading f0 parameters
        voice_profile = np.load("./voice_profile.npy")
        self.args["pitch_rate_mean_s"] = voice_profile[0]
        self.args["pitch_rate_mean_t"] = voice_profile[1]
        self.args["pitch_rate_var"] = voice_profile[2]

        #inputs place holders

        self.input_model_a = tf.placeholder(tf.float32, self.input_size_model, "inputs_g_a")
        self.input_model_b = tf.placeholder(tf.float32, self.input_size_model, "inputs_g_b")
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_g_test")
        self.time = tf.placeholder(tf.float32, None, "inputs_time_train")

        #creating generator (if you want to view more codes then ./model.py)
        with tf.variable_scope("generator_1"):
            fake_ab_image = generator(self.input_model_a, reuse=None)
            self.fake_ab_image_test = generator(self.input_model_test, reuse=True)
        with tf.variable_scope("generator_2"):
            fake_ba_image = generator(self.input_model_b, reuse=None)
            fake_bab_image = generator(fake_ab_image, reuse=True)
        with tf.variable_scope("generator_1", reuse=True):
            fake_aba_image = generator(fake_ba_image, reuse=True)


        #creating discriminator (if you want to view more codes then ./model.py)
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            inp = tf.concat([self.input_model_a, self.input_model_b, fake_ba_image, fake_ab_image], axis=0)
            d_judge = discriminator(inp, None)
            d_judge_to_g = discriminator(inp[self.args["batch_size"]*2:], reuse=True)

        #getting individual variables of architectures
        self.g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator_1")+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator_2")
        self.d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminator")
        _l = int(d_judge.shape[1])
        label_n = tf.tile(tf.reshape(tf.one_hot(0, 3), [1, 1, 3]), [self.args["batch_size"], _l, 1])
        label_a = tf.tile(tf.reshape(tf.one_hot(1, 3), [1, 1, 3]), [self.args["batch_size"], _l, 1])
        label_b = tf.tile(tf.reshape(tf.one_hot(2, 3), [1, 1, 3]), [self.args["batch_size"], _l, 1])
        label = tf.concat([label_a, label_b, label_n, label_n], axis=0)

        # crassifer loss(using a Least_Squared_Loss)
        self.d_loss = tf.squared_difference(label, d_judge)
        # objective-functions of generator
        # Cycle loss (L1 norm is better than L2 norm)
        g_loss_cyc_a = tf.losses.absolute_difference(labels=self.input_model_a, predictions=fake_bab_image)
        g_loss_cyc_b = tf.losses.absolute_difference(labels=self.input_model_b, predictions=fake_aba_image)
        # Gan loss (using a Least_Squared_Loss)
        g_loss_gan = tf.squared_difference(label[:self.args["batch_size"]*2], d_judge_to_g)
        self.g_loss = tf.losses.compute_weighted_loss(g_loss_cyc_a + g_loss_cyc_b, tf.cos(self.time*np.pi/2)*(self.args["weight_cycle"]-1.0)+1.0) + g_loss_gan
        #tensorboard functions
        if self.args["tensor-board"]:
            g_loss_cyc_a_display = tf.summary.scalar("g_loss_cycle_aba", tf.reduce_mean(g_loss_cyc_a), family="loss")
            g_loss_cyc_b_display = tf.summary.scalar("g_loss_cycle_bab", tf.reduce_mean(g_loss_cyc_b), family="loss")
            g_loss_gan_display = tf.summary.scalar("g_loss_gan", tf.reduce_mean(g_loss_gan), family="loss")
            g_loss_sum_display = tf.summary.merge([g_loss_cyc_a_display, g_loss_gan_display, g_loss_cyc_b_display])
            d_loss_sum_display = tf.summary.scalar("d_loss", tf.reduce_mean(self.d_loss), family="loss")
            self.loss_display = tf.summary.merge([g_loss_sum_display, d_loss_sum_display])
            self.result_score = tf.placeholder(tf.float32, name="fake_fft_score")
            self.result_image_display = tf.placeholder(tf.float32, [1, None, 512], name="fake_spectrum")
            image_pow_display = tf.reshape(tf.transpose(self.result_image_display[:, :, :], [0, 2, 1]), [1, 512, -1, 1])
            fake_b_image_display = tf.summary.image("fake_spectrum_ab", image_pow_display, 1)
            fake_b_fft_score_display = tf.summary.scalar("g_error_ab", tf.reduce_mean(self.result_score), family="g_test")
            self.g_test_display = tf.summary.merge([fake_b_image_display, fake_b_fft_score_display])

        # initializing running object
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # logging
        if self.args["tensor-board"]:
            self.writer = tf.summary.FileWriter("./logs/" + self.args["name_save"], self.sess.graph)
    def convert(self, in_put):
        """
        リアルタイムではない変換を行う。
        主にテスト用
        Parameters
        ----------
        in_put: np.array
        入力データ
        SamplingRate: 16000
        ValueRange  : [-1.0,1.0]
        dtype       : float64
        """
        #function of test
        #to convert wave file

        # calculating times to execute network
        conversion_start_time = time.time()
        input_size_one_term = self.args["input_size"]
        executing_times = in_put.shape[0]//(self.args["input_size"])+1
        if in_put.shape[0]%((self.args["input_size"])*self.args["batch_size"]) == 0:
            executing_times -= 1

        otp = np.array([], dtype=np.int16)

        for i in range(executing_times):
            # pre-process
            # padding
            start_pos = max(0, self.args["input_size"]*i+(in_put.shape[0]%self.args["input_size"])-input_size_one_term)
            end_pos = self.args["input_size"]*i+(in_put.shape[0]%self.args["input_size"])
            resource_wave = in_put[start_pos:end_pos]
            padding_num = max(0, input_size_one_term-resource_wave.shape[0])
            if padding_num > 0:
                resource_wave = np.pad(resource_wave, (padding_num, 0), 'constant')
            # wave to WORLD
            f0_estimation, sp_env, aperiodicity = encode((resource_wave/32767).astype(np.float))
            sp_env = sp_env.reshape(self.input_size_test)
            #main process
            # running network
            result = self.sess.run(self.fake_ab_image_test, feed_dict={self.input_model_test:sp_env})

            # post-process
            # f0 transforming
            f0_estimation = (f0_estimation-self.args["pitch_rate_mean_s"])*self.args["pitch_rate_var"]+self.args["pitch_rate_mean_t"]
            # WORLD to wave
            result_wave = decode(f0_estimation, result[0].copy().reshape(-1, 513).astype(np.float), aperiodicity)*32767
            result_wave_fixed = np.clip(result_wave, -32767.0, 32767.0)[:self.args["input_size"]]
            result_wave_int16 = result_wave_fixed.reshape(-1).astype(np.int16)

            #adding result
            otp = np.append(otp, result_wave_int16)

        head_cutnum = otp.shape[0]-in_put.shape[0]
        if head_cutnum > 0:
            otp = otp[head_cutnum:]

        return otp, time.time()-conversion_start_time



    def train(self):
        """
        学習を実行する。

        """
        # naming output-directory
        g_optimizer = tf.train.AdamOptimizer(self.args["learning_rate"], 0.5, 0.999).minimize(self.g_loss, var_list=self.g_vars)
        d_optimizer = tf.train.AdamOptimizer(self.args["learning_rate"], 0.5, 0.999).minimize(self.d_loss, var_list=self.d_vars)

        # loading net
        if self.load():
            print(" [I] Load success.")
        else:
            print(" [I] Load failed.")

        # loading training data directory
        # loading test data
        if self.args["test"]:
            self.test = wave_read(self.args["test_data_dir"]+'/test.wav')
            if self.args["real_sample_compare"]:
                self.label = wave_read(self.args["test_data_dir"] + '/label.wav')
                power_spec = fft(self.label[800:156000]/32767)
                self.label_spec = np.mean(power_spec, axis=0)
                self.label_spec_v = np.std(power_spec, axis=0)

        # preparing training-data
        batch_files = self.args["train_data_dir"]+'/A.npy'
        batch_files2 = self.args["train_data_dir"]+'/B.npy'

        print(" [I] loading data-set ...")
        self.sounds_r = np.load(batch_files)
        self.sounds_t = np.load(batch_files2)
        # times of one epoch
        train_data_num = min(self.sounds_r.shape[0], self.sounds_t.shape[0])
        self.loop_num = train_data_num // self.args["batch_size"]
        index_list = [h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num)]
        print(" [I] %d data loaded!!" % train_data_num)
        self.sounds_r = self.sounds_r.reshape([self.sounds_r.shape[0], self.sounds_r.shape[1], 1, self.sounds_r.shape[2]])
        self.sounds_t = self.sounds_t.reshape([self.sounds_t.shape[0], self.sounds_t.shape[1], 1, self.sounds_t.shape[2]])

        # initializing training information
        start_time_all = time.time()
        train_epoch = self.args["train_iteration"]//self.loop_num+1
        one_itr_num = self.loop_num
        iterations = 0
        # main-training
        for epoch in range(train_epoch):
            # shuffling train_data_index
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)
            time_per = iterations/self.args["train_iteration"]
            if self.args["test"] and epoch % self.args["test_interval"] == 0:
                self.test_and_save(epoch, iterations, one_itr_num, time_per)
            for idx in range(0, self.loop_num):
                # getting mini-batch
                time_per = iterations/self.args["train_iteration"]
                start_position = self.args["batch_size"]*idx
                batch_sounds_resource = self.sounds_r[index_list[start_position:start_position+self.args["batch_size"]]]
                batch_sounds_target = self.sounds_t[index_list2[start_position:start_position+self.args["batch_size"]]]
                # update D network
                self.sess.run(d_optimizer, feed_dict={self.input_model_a: batch_sounds_resource, self.input_model_b: batch_sounds_target, self.time:time_per})
                self.sess.run(d_optimizer, feed_dict={self.input_model_a: batch_sounds_resource, self.input_model_b: batch_sounds_target, self.time:time_per})
                # update G network
                self.sess.run(g_optimizer, feed_dict={self.input_model_a: batch_sounds_resource, self.input_model_b: batch_sounds_target, self.time:time_per})
                iterations += 1
                if self.args["train_iteration"] == iterations:
                    break
        self.test_and_save(train_epoch, iterations, one_itr_num, time_per)
        taken_time_all = time.time()-start_time_all
        hour_display = taken_time_all//3600
        minute_display = taken_time_all//60%60
        second_display = int(taken_time_all%60)
        print(" [I] ALL train process finished successfully!! in %06d : %02d : %02d" % (hour_display, minute_display, second_display))

    def test_and_save(self, epoch, itr, one_itr_num, time_per):
        """
        テストとセーブを行う。書き込み等も同時に行う。
        Parameters
        ----------
        epoch: int
            完了エポック数
        itr: int
            完了イテレーション数
        one_itr_num: int
            1エポック当たりのイテレーション数
        time_per: float
            学習の進行度
            Value_Range(0.0,1.0)
        """
        # testing
        out_puts, _ = self.convert(self.test)

        # fixing harvests types
        out_put = out_puts.copy().astype(np.float32) / 32767.0

        # calculating power spectrum
        image_power_spec = fft(out_put[800:156000])
        if self.args["real_sample_compare"]:
            spec_m = np.mean(image_power_spec, axis=0)
            spec_v = np.std(image_power_spec, axis=0)
            diff = spec_m-self.label_spec
            diff2 = spec_v-self.label_spec_v
            score = np.mean(diff*diff+diff2*diff2)
        otp_im = image_power_spec.copy().reshape(1, -1, 512)

        # writing epoch-result into tensor-board
        if self.args["tensor-board"]:
            tb_result = self.sess.run(self.loss_display,
                                      feed_dict={self.input_model_a: self.sounds_r[0:self.args["batch_size"]],
                                                 self.input_model_b: self.sounds_t[0:self.args["batch_size"]],
                                                 self.time:time_per})
            self.writer.add_summary(tb_result, itr)
            if self.args["real_sample_compare"]:
                result_test = self.sess.run(self.g_test_display, feed_dict={self.result_image_display: otp_im, self.result_score:score})
                self.writer.add_summary(result_test, itr)

        # saving test harvests
        if os.path.exists(self.args["wave_otp_dir"]):

            # saving fake spectrum
            plt.clf()
            plt.subplot(3, 1, 1)
            insert_image = np.transpose(image_power_spec, (1, 0))
            plt.imshow(insert_image, vmin=-15, vmax=5, aspect="auto")
            plt.subplot(3, 1, 2)
            plt.plot(out_put[800:156000])
            plt.ylim(-1, 1)
            if self.args["real_sample_compare"]:
                plt.subplot(3, 1, 3)
                plt.plot(diff * diff + diff2 * diff2)
                plt.ylim(0, 100)
            name_save = "%s%04d.png" % (self.args["wave_otp_dir"], epoch)
            plt.savefig(name_save)
            name_save = "./latest.png"
            plt.savefig(name_save)
            #saving fake waves
            pa_ins = pyaudio.PyAudio()
            path_save = self.args["wave_otp_dir"] + datetime.now().strftime("%m-%d_%H-%M-%S") + "_" + str(epoch)
            voiced = out_puts.astype(np.int16)[800:156000]
            wave_data = wave.open(path_save + ".wav", 'wb')
            wave_data.setnchannels(1)
            wave_data.setsampwidth(pa_ins.get_sample_size(pyaudio.paInt16))
            wave_data.setframerate(16000)
            wave_data.writeframes(voiced.reshape(-1).tobytes())
            wave_data.close()
            pa_ins.terminate()
        self.save(self.args["checkpoint_dir"], epoch, self.saver)
        taken_time = time.time() - self.start_time
        self.start_time = time.time()
        # output console
        if itr != 0:
            # eta
            self.tt_list.append(taken_time/one_itr_num/self.args["test_interval"])
            if len(self.tt_list) > 5:
                self.tt_list = self.tt_list[1:-1]
            eta = np.mean(self.tt_list) * (self.args["train_iteration"] - itr)
            print(" [I] Iteration %06d / %06d finished. ETA: %02d:%02d:%02d takes %2.3f secs" % (itr, self.args["train_iteration"], eta // 3600, eta // 60 % 60, int(eta % 60), taken_time))
        if self.args["real_sample_compare"]:
            print(" [I] Epoch %04d tested. score=%.5f" % (epoch, float(score)))
        else:
            print(" [I] Epoch %04d tested." % epoch)

    def save(self, checkpoint_dir, step, saver):
        """
        モデルのセーブ
        Parameters
        ----------
        checkpoint_dir : string
            ファイルの場所指定
        step: int
            学習の完了ステップ数（エポック）
        saver: tf.Saver
            保存用のオブジェクト
        """
        model_name = "wave2wave.model"
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
    def load(self):
        """
        モデルのロード
        変数の初期化も同時に行う
        Returns
        -------
        Flagment: bool
            うまくいけばTrue
            ファイルがなかったらFalse
        """
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
    """
    #音声をWorldに変換します

    Parameters
    ----------
    data : ndarray
        入力データ
        SamplingRate: 16000
        ValueRange  : [-1.0,1.0]
        dtype       : float64
    Returns
    -------
    World: list(3items)
        出力
        _f0 : f0 estimation
        Shape(N)
        dtype       : float64
        _sp : spectram envelobe
        Shape(N,513)
        dtype       : float64
        _ap : aperiodicity
        Shape(N,513)
        dtype       : float64
    """
    sampleing_rate = 16000
    _f0, _t = pw.dio(data, sampleing_rate)
    _f0 = pw.stonemask(data, _f0, _t, sampleing_rate)
    _sp = pw.cheaptrick(data, _f0, _t, sampleing_rate)
    _ap = pw.d4c(data, _f0, _t, sampleing_rate)
    return _f0.astype(np.float64), np.clip((np.log(_sp)+15)/20, -1.0, 1.0).astype(np.float64), _ap.astype(np.float64)

def decode(_f0, _sp, _ap):
    """
    #Worldを音声に変換します
    Parameters
    ----------
    _f0 : np.ndarray
        _f0 estimation
        Shape(N)
        dtype       : float64
    _sp : np.ndarray
        spectram envelobe
        Shape(N,513)
        dtype       : float64
    _ap : np.ndarray
        aperiodicity
        Shape(N,513)
        dtype       : float64
    Returns
    -------
    World: list(3items)
        #出力
        SamplingRate: 16000
        ValueRange  : [-1.0,1.0]
        dtype       : float64
    """
    _sp = np.exp(_sp * 20 - 15).astype(np.float)
    return pw.synthesize(_f0, _sp, _ap, 16000)


def wave_read(path_to_file):
    """
    #音声を読み込みます
     Parameters
    ----------
    path_to_file : string
        #ファイルまでのパス
    Returns
    -------
    ans: ndarray
        #音声
        ValueRange  : [-32767,32767]
        dtype       : int16
    """
    wave_data = wave.open(path_to_file, "rb")
    ans_data = np.zeros([1], dtype=np.int16)
    dds = wave_data.readframes(1024)
    while dds != b'':
        ans_data = np.append(ans_data, np.frombuffer(dds, "int16"))
        dds = wave_data.readframes(1024)
    wave_data.close()
    ans_data = ans_data[1:]
    return ans_data

def fft(data):
    """
    # stftを計算

     Parameters
    ----------
    data : np.ndarray
        音声データ
        ValueRange  : [-1.0,1.0]
        dtype       : float64
    Returns
    -------
    spec_po: np.ndarray
        パワースペクトラム
        power-spectram
        Shape : (n,512)
    """
    time_ruler = data.shape[0] // 512
    if data.shape[0] % 512 == 0:
        time_ruler -= 1
    window = np.hamming(1024)
    pos = 0
    wined = np.zeros([time_ruler, 1024])
    for fft_index in range(time_ruler):
        frame = data[pos:pos + 1024]
        padding_size = 1024-frame.shape[0]
        if padding_size > 0:
            frame = np.pad(frame, (0, padding_size), "constant")
        wined[fft_index] = frame * window
        pos += 512
    fft_r = np.fft.fft(wined, n=1024, axis=1)
    spec_re = fft_r.real.reshape(time_ruler, -1)
    spec_im = fft_r.imag.reshape(time_ruler, -1)
    spec_po = np.log(np.power(spec_re, 2) + np.power(spec_im, 2) + 1e-24).reshape(time_ruler, -1)[:, 512:]
    return np.clip(spec_po, -15.0, 5.0)


if __name__ == '__main__':
    NET = Model("./setting.json")
    NET.train()
