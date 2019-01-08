import os
import time
import json
from datetime import datetime

import wave

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from model import discriminator, generator

from voice_to_datasets_cycle import Voice2Dataset as V2D

import util

class Model:
    def __init__(self, path):
        self.args = util.config_reader(
            path,
            {
                #
                "model_name": "VRC",
                "version": "1.0.2",
                #
                "checkpoint_dir": "./trained_models",
                "wave_otp_dir": "./harvests",
                "train_data_dir": "./dataset/train",
                "test_data_dir": "./dataset/test",
                #
                "real_data_compare": False,
                "test": True,
                "tensorboard": False,
                #
                "batch_size": 1,
                "input_size": 4096,
                "weight_Cycle": 100.0,
                "train_iteration": 100000,
                "start_epoch": 0,
                #
                "use_colab": False,
                "colab_hardware": "tpu",
            })

        self.f0_translater = util.generate_f0_translater("./voice_profile.npy")
        # initializing hidden paramaters

        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        # shapes setting
        self.input_size_model = [self.args["batch_size"], 52, 513, 1]
        self.input_size_test = [1, 52, 513, 1]
        self.output_size_model = [self.args["batch_size"], 65, 513, 1]

        if self.args["wave_otp_dir"] is not "False":
            self.args["wave_otp_dir"] = self.args["wave_otp_dir"] + self.args[
                "name_save"] + "/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])

        #inputs place holder
        self.input_model_A = tf.placeholder(tf.float32, self.input_size_model,
                                            "inputs_g_A")
        self.input_model_B = tf.placeholder(tf.float32, self.input_size_model,
                                            "inputs_g_B")
        self.input_model_test = tf.placeholder(
            tf.float32, self.input_size_test, "inputs_g_test")
        self.time = tf.placeholder(tf.float32, [1], "inputs_g_test")
        #creating generator

        with tf.variable_scope("generator_1"):
            fake_aB_image = generator(
                self.input_model_A, reuse=None, training=True)
            self.fake_aB_image_test = generator(
                self.input_model_test, reuse=True, training=False)
        with tf.variable_scope("generator_2"):
            fake_bA_image = generator(
                self.input_model_B, reuse=None, training=True)
        with tf.variable_scope("generator_2", reuse=True):
            fake_Ba_image = generator(fake_aB_image, reuse=True, training=True)
        with tf.variable_scope("generator_1", reuse=True):
            fake_Ab_image = generator(fake_bA_image, reuse=True, training=True)

        #creating discriminator
        with tf.variable_scope("discriminators"):
            inp = tf.concat([
                fake_aB_image, self.input_model_B, fake_bA_image,
                self.input_model_A
            ],
                            axis=0)
            d_judge = discriminator(inp, None)

        #getting individual variabloes
        self.g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        "generator_1")
        self.g_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         "generator_2")
        self.d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        "discriminators")
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #objective-functions of discriminator
        l0 = tf.reshape(tf.one_hot(0, 3), [1, 1, 3])
        l1 = tf.reshape(tf.one_hot(1, 3), [1, 1, 3])
        l2 = tf.reshape(tf.one_hot(2, 3), [1, 1, 3])
        labelA = tf.tile(
            l0, [self.input_size_model[0], self.input_size_model[1], 1])
        labelB = tf.tile(
            l1, [self.input_size_model[0], self.input_size_model[1], 1])
        labelO = tf.tile(
            l2, [self.input_size_model[0], self.input_size_model[1], 1])
        labels = tf.concat([labelO, labelB, labelO, labelA], axis=0)

        self.d_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=d_judge) * 4

        # objective-functions of generator

        # Cyc lossB
        g_loss_cyc_B = tf.pow(tf.abs(fake_Ab_image - self.input_model_B), 2)

        # Gan lossA
        g_loss_gan_A = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=d_judge[self.args["batch_size"] *
                           2:self.args["batch_size"] * 3],
            labels=labelA)

        # Cycle lossA
        g_loss_cyc_A = tf.pow(tf.abs(fake_Ba_image - self.input_model_A), 2)

        # Gan lossB
        g_loss_gan_B = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=d_judge[:self.args["batch_size"]], labels=labelB)

        # generator loss
        self.g_loss = tf.losses.compute_weighted_loss(
            g_loss_cyc_A + g_loss_cyc_B,
            self.args["weight_Cycle"]) + g_loss_gan_B + g_loss_gan_A

        #tensorboard functions
        g_loss_cyc_A_display = tf.summary.scalar(
            "g_loss_cycle_AtoA", tf.reduce_mean(g_loss_cyc_A), family="g_loss")
        g_loss_gan_A_display = tf.summary.scalar(
            "g_loss_gan_BtoA", tf.reduce_mean(g_loss_gan_A), family="g_loss")
        g_loss_sum_A_display = tf.summary.merge(
            [g_loss_cyc_A_display, g_loss_gan_A_display])

        g_loss_cyc_B_display = tf.summary.scalar(
            "g_loss_cycle_BtoB", tf.reduce_mean(g_loss_cyc_B), family="g_loss")
        g_loss_gan_B_display = tf.summary.scalar(
            "g_loss_gan_AtoB", tf.reduce_mean(g_loss_gan_B), family="g_loss")
        g_loss_sum_B_display = tf.summary.merge(
            [g_loss_cyc_B_display, g_loss_gan_B_display])

        d_loss_sum_A_display = tf.summary.scalar(
            "d_loss", tf.reduce_mean(self.d_loss), family="d_loss")

        self.loss_display = tf.summary.merge(
            [g_loss_sum_A_display, g_loss_sum_B_display, d_loss_sum_A_display])
        self.result_score = tf.placeholder(tf.float32, name="FakeFFTScore")
        fale_B_FFT_score_display = tf.summary.scalar(
            "g_error_AtoB", tf.reduce_mean(self.result_score), family="g_test")
        self.g_test_display = tf.summary.merge([fale_B_FFT_score_display])

        #saver
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def convert(self, in_put):
        """ WAVEファイルを最新の学習結果でA⇒B変換する
        Parameters
        ----------
        in_put : ndarray(dtype=np.int16)
            isreadメソッドによって配列化されたWAVEファイル
        
        Returns
        -------
        otp :　ndarray(dtype=np.int16)
            A -> B 変換結果
        time : float
            変換にかかった時間
        """
        #function of test
        #To convert wave file

        conversion_start_time = time.time()
        input_size_one_term = self.args["input_size"]
        executing_times = (in_put.shape[0] - 1) // (
            self.args["input_size"]) + 1
        otp = np.array([], dtype=np.int16)

        for t in range(executing_times):
            # Preprocess

            # Padiing
            start_pos = self.args["input_size"] * t + (
                in_put.shape[0] % self.args["input_size"])
            resorce = in_put[max(0, start_pos - input_size_one_term):start_pos]
            r = max(0, input_size_one_term - resorce.shape[0])
            if r > 0:
                resorce = np.pad(resorce, (r, 0), 'constant')
            # FFT
            f0, resource, ap = util.encode((resorce / 32767).astype(np.float))
            resource = resource.reshape(self.input_size_test)
            #main process

            # running network
            result = self.sess.run(
                self.fake_aB_image_test,
                feed_dict={self.input_model_test: resource})

            # Postprocess

            # IFFT
            f0 = self.f0_translater(f0)
            result_wave = util.decode(
                f0, result[0].copy().reshape(-1, 513).astype(np.float),
                ap) * 32767

            result_wave_fixed = np.clip(result_wave, -32767.0,
                                        32767.0)[:self.args["input_size"]]
            result_wave_int16 = result_wave_fixed.reshape(-1).astype(np.int16)

            #adding result
            otp = np.append(otp, result_wave_int16)

        h = otp.shape[0] - in_put.shape[0]
        if h > 0:
            otp = otp[h:]

        return otp, time.time() - conversion_start_time

    def train(self):
        # naming output-directory
        with tf.control_dependencies(self.update_ops):
            g_optim = tf.train.AdamOptimizer(4e-6, 0.5, 0.999).minimize(
                self.g_loss, var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(4e-6, 0.5, 0.999).minimize(
            self.d_loss, var_list=self.d_vars)
        # logging
        if self.args["tensorboard"]:
            self.writer = tf.summary.FileWriter(
                "./logs/" + self.args["name_save"], self.sess.graph)

        # loading net
        if self.load():
            print(" [I] Load SUCCESSED.")
        else:
            print(" [I] Load FAILED.")

        # loading training data directory
        # loading test data
        self.test = isread(self.args["test_data_dir"] + '/test.wav')
        if self.args["real_data_compare"]:
            self.label = isread(self.args["test_data_dir"] + '/label.wav')

            im = fft(self.label / 32767)
            self.label_spec = np.mean(im, axis=0)
            self.label_spec_v = np.std(im, axis=0)

        # prepareing training-data
        batch_files = self.args["train_data_dir"] + '/A.npy'
        batch_files2 = self.args["train_data_dir"] + '/B.npy'

        print(" [I] loading dataset...")
        self.sounds_r = np.load(batch_files)
        self.sounds_t = np.load(batch_files2)
        # times of one epoch
        train_data_num = min(self.sounds_r.shape[0], self.sounds_t.shape[0])
        self.batch_idxs = train_data_num // self.args["batch_size"]
        index_list = [h for h in range(train_data_num)]
        index_list2 = [h for h in range(train_data_num)]
        print(" [I] %d data loaded!!" % train_data_num)
        self.sounds_r = self.sounds_r.reshape([
            self.sounds_r.shape[0], self.sounds_r.shape[1],
            self.sounds_r.shape[2], 1
        ])
        self.sounds_t = self.sounds_t.reshape([
            self.sounds_t.shape[0], self.sounds_t.shape[1],
            self.sounds_t.shape[2], 1
        ])

        # initializing training infomation
        start_time_all = time.time()
        self.tt_list = list()
        self.start_time = time.time()
        self.train_epoch = self.args["train_iteration"] // self.batch_idxs + 1
        self.one_itr_num = self.batch_idxs * self.args["batch_size"]
        iterations = 0
        # main-training
        for epoch in range(self.train_epoch):
            # shuffling train_data_index
            np.random.shuffle(index_list)
            np.random.shuffle(index_list2)

            if self.args["test"] and epoch % self.args["batch_size"] == 0:
                self.test_and_save(epoch, iterations)
            for idx in range(0, self.batch_idxs):
                # getting batch
                if iterations == self.args["train_iteration"]:
                    break
                st = self.args["batch_size"] * idx
                batch_sounds_resource = np.asarray([
                    self.sounds_r[ind]
                    for ind in index_list[st:st + self.args["batch_size"]]
                ])
                batch_sounds_target = np.asarray([
                    self.sounds_t[ind]
                    for ind in index_list2[st:st + self.args["batch_size"]]
                ])
                ttt = np.array(
                    [1.0 - iterations / self.args["train_iteration"]])
                # update D network
                self.sess.run(
                    d_optim,
                    feed_dict={
                        self.input_model_A: batch_sounds_resource,
                        self.input_model_B: batch_sounds_target,
                        self.time: ttt
                    })
                # update G network
                self.sess.run(
                    g_optim,
                    feed_dict={
                        self.input_model_A: batch_sounds_resource,
                        self.input_model_B: batch_sounds_target,
                        self.time: ttt
                    })

                iterations += 1
            # calculating ETA
            if iterations == self.args["train_iteration"]:
                break

        self.test_and_save(self.train_epoch, iterations)
        taken_time_all = time.time() - start_time_all
        hour_display = taken_time_all // 3600
        minute_display = taken_time_all // 60 % 60
        second_display = int(taken_time_all % 60)
        print(
            " [I] ALL train process finished successfully!! in %06d : %02d : %02d"
            % (hour_display, minute_display, second_display))

    def _get_sample_size(self):
        if self.args["use_colab"]:
            return 2

        import pyaudio
        p = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        sample_size = p.get_sample_size(FORMAT)
        p.terminate()
        return sample_size

    def test_and_save(self, epoch, itr):
        # last testing
        out_puts, _ = self.convert(self.test)
        # fixing havests types
        out_put = out_puts.copy().astype(np.float32) / 32767.0

        # calcurating power spectrum
        im = fft(out_put)
        otp_im = im.copy().reshape(1, -1, 512)
        # writing epoch-result into tensorboard
        if self.args["tensorboard"]:
            tb_result = self.sess.run(
                self.loss_display,
                feed_dict={
                    self.input_model_A:
                    self.sounds_r[0:self.args["batch_size"]],
                    self.input_model_B:
                    self.sounds_t[0:self.args["batch_size"]],
                    self.time: np.zeros([1])
                })
            self.writer.add_summary(tb_result, itr)
            if self.args["real-data-compare"]:
                spec = np.mean(im, axis=0)
                spec_v = np.std(im, axis=0)
                diff = spec - self.label_spec
                diff2 = spec_v - self.label_spec_v
                score = np.mean(diff * diff + diff2 * diff2)
                rs = self.sess.run(
                    self.g_test_display, feed_dict={self.result_score: score})
                self.writer.add_summary(rs, itr)

        # saving test harvests
        if os.path.exists(self.args["wave_otp_dir"]):
            # saving fake spectrum
            plt.clf()
            plt.subplot(2, 1, 1)
            ins = np.transpose(im, (1, 0))
            plt.imshow(ins, vmin=-15, vmax=5, aspect="auto")
            plt.subplot(2, 1, 2)
            plt.plot(out_put)
            plt.ylim(-1, 1)
            path = "%s%04d.png" % (self.args["wave_otp_dir"],
                                   epoch // self.args["batch_size"])
            plt.savefig(path)
            path = "./latest.png"
            plt.savefig(path)

            #saving fake waves
            path = self.args["wave_otp_dir"] + datetime.now().strftime(
                "%m-%d_%H-%M-%S") + "_" + str(epoch // self.args["batch_size"])
            voiced = out_puts.astype(np.int16)[800:156000]

            ww = wave.open(path + ".wav", 'wb')
            ww.setnchannels(1)
            ww.setsampwidth(self._get_sample_size())
            ww.setframerate(16000)
            ww.writeframes(voiced.reshape(-1).tobytes())
            ww.close()
        self.save(self.args["checkpoint_dir"], epoch, self.saver)
        taken_time = time.time() - self.start_time
        self.start_time = time.time()
        if itr != 0:
            self.tt_list.append(taken_time / self.one_itr_num)
            if len(self.tt_list) > 5:
                self.tt_list = self.tt_list[1:-1]
            eta = np.mean(self.tt_list) * (self.args["train_iteration"] - itr)
            print(
                " [I] Iteration %06d / %06d finished. ETA: %02d:%02d:%02d takes %2.3f secs"
                % (itr, self.args["train_iteration"], eta // 3600,
                   eta // 60 % 60, int(eta % 60), taken_time))
        print(" [I] Epoch %04d tested." % epoch)

    def save(self, checkpoint_dir, step, saver):
        model_name = "wave2wave.model"
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(
            self.sess,
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
            self.saver.restore(self.sess,
                               os.path.join(checkpoint_dir, ckpt_name))
            self.epoch = self.saver
            return True
        else:
            return False

def isread(path):
    """
    WAVEファイルから各フレームをINT16配列にして返す

    Returns
    -------
    長さ160,000(10秒分)のdtype=np.int16のndarray

    Notes
    -----
    WAVEファイルのサンプリングレートは16[kHz]でなければならない。
    WAVEファイルが10秒未満の場合、配列の長さが10秒になるようにパディングする
    WAVEファイルが10秒を超える場合、超えた分を切り詰める
    """

    wf = wave.open(path, "rb")
    ans = np.zeros([1], dtype=np.int16)
    dds = wf.readframes(1024)
    while dds != b'':
        ans = np.append(ans, np.frombuffer(dds, "int16"))
        dds = wf.readframes(1024)
    wf.close()
    ans = ans[1:]
    i = 160000 - ans.shape[0]
    if i > 0:
        ans = np.pad(ans, (0, i), "constant")
    else:
        ans = ans[:160000]
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
        r = 1024 - frame.shape[0]
        if r > 0:
            frame = np.pad(frame, (0, r), "constant")
        wined[fft_index] = frame * window
        pos += 512
    fft_r = np.fft.fft(wined, n=1024, axis=1)
    re = fft_r.real.reshape(time_ruler, -1)
    im = fft_r.imag.reshape(time_ruler, -1)
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(
        time_ruler, -1)[:, 512:]
    return np.clip(c, -15.0, 5.0)


if __name__ == '__main__':
    v2d = V2D(
        os.path.join(".", "dataset", "wave"),
        os.path.join(".", "dataset", "train"))
    plof = v2d.convert("A", "B")
    np.save("./voice_profile.npy", plof)

    path = "./setting.json"
    net = Model(path)
    net.train()
    