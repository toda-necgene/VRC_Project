import tensorflow as tf
import util
import time
from datetime import datetime


class Dummy():
    pass


class CycleGAN():
    def __init__(self, model=None):
        self.session = tf.Session()
        self.saver = tf.train.Saver()

        self.name = model.name + model.version

        self.sounds_r = []
        self.sounds_t = []

        self._create_optimizer = None
        self.callback_every_epoch = {}

        self.input = Dummy()
        self.input.A = []
        self.input.B = []
        self.loss = Dummy()
        self.loss.g = []
        self.loss.d = []
        self.loss.display = []
        self.time = []
        self.vars = Dummy()
        self.vars.g = []
        self.vars.d = []
        self.update_ops = []

    def train(self, batch_size=1, train_iteration=100000):
        assert len(self.sounds_r) > 0
        assert len(self.sounds_r) == len(self.sounds_t)

        # naming output-directory
        with tf.control_dependencies(self.update_ops):
            optimizer = self._create_optimizer()
            g_optim = optimizer.minimize(self.loss.g, var_list=self.vars.g)

        optimizer = self._create_optimizer()
        d_optim = optimizer.minimize(self.loss.d, var_list=self.vars.d)

        # initializing training infomation
        start_time_all = time.time()

        batch_idxs = self.sounds_r // batch_size
        train_epoch = train_iteration // batch_idxs + 1

        index_list_r = [h for h in range(self.sounds_r.shape[0])]
        index_list_t = [h for h in range(self.sounds_r.shape[0])]

        iterations = 0
        # main-training
        epoch_count_time = time.time()
        for epoch in range(train_epoch):
            if epoch % batch_size == 0:
                period = time.time() - epoch_count_time
                for f in self.callback_every_epoch.values():
                    f(epoch, iterations, period)
                epoch_count_time = time.time()

            # shuffling train_data_index
            np.random.shuffle(index_list_r)
            np.random.shuffle(index_list_t)

            for idx in range(0, batch_idxs):
                # getting batch
                if iterations == train_iteration:
                    break
                st = batch_size * idx
                batch_sounds_resource = np.asarray([
                    self.sounds_r[ind]
                    for ind in index_list_r[st:st + batch_size]
                ])
                batch_sounds_target = np.asarray([
                    self.sounds_t[ind]
                    for ind in index_list_t[st:st + batch_size]
                ])
                ttt = np.array([1.0 - iterations / train_iteration])
                # update D network
                self.session.run(
                    d_optim,
                    feed_dict={
                        self.input.A: batch_sounds_resource,
                        self.input.B: batch_sounds_target,
                        self.time: ttt
                    })
                # update G network
                self.session.run(
                    g_optim,
                    feed_dict={
                        self.input.A: batch_sounds_resource,
                        self.input.B: batch_sounds_target,
                        self.time: ttt
                    })

                iterations += 1
            # calculating ETA
            if iterations == train_iteration:
                break

        period = time.time() - epoch_count_time
        for f in self.callback_every_epoch.values():
            f(train_epoch, iterations, period)

        taken_time_all = time.time() - start_time_all
        print(" [I] ALL train process finished successfully!! in %d Hours" %
              taken_time_all / 3600)

    def a_to_b(self, array):
        """
        resource = array.reshape(self.input_size_test)
        result = self.session.run(
                self.fake_aB_image_test,
                feed_dict={self.input_model_test: resource})
        return result[0].reshape(-1, 513).astype(np.float)
        """
        pass

    def b_to_a(self, tensor):
        pass

    def save(self, file, global_step):
        self.saver.save(self.session, file, global_step=global_step)

    def load(self, dir):
        # initialize variables
        initializer = tf.global_variables_initializer()
        self.session.run(initializer)
        print(" [I] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(dir)
        if ckpt is not None and ckpt:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # pylint: disable=E1101
            self.saver.restore(self.session,
                               os.path.join(dir, ckpt_name))
            self.epoch = self.saver
            return True
        else:
            return False


class Converter():
    def __init__(self, gan, f0_transfer):
        self.gan = gan
        self.f0_transfer = f0_transfer

    def convert(self, input, term=4096):
        conversion_start_time = time.time()
        executing_times = (input.shape[0] - 1) // term + 1
        otp = np.array([], dtype=np.int16)

        for t in range(executing_times):
            # Preprocess

            # Padiing
            end_pos = term * t + (input.shape[0] % term)
            resorce = input[max(0, end_pos - term):end_pos]
            r = max(0, term - resorce.shape[0])
            if r > 0:
                resorce = np.pad(resorce, (r, 0), 'constant')
            # FFT
            f0, resource, ap = util.encode((resorce / 32767).astype(np.float))

            # IFFT
            f0 = self.f0_transfer(f0)
            result_wave = util.decode(f0, self.gan.a_to_b(resource),
                                      ap) * 32767

            result_wave_fixed = np.clip(result_wave, -32767.0, 32767.0)[:term]
            result_wave_int16 = result_wave_fixed.reshape(-1).astype(np.int16)

            #adding result
            otp = np.append(otp, result_wave_int16)

        h = otp.shape[0] - input.shape[0]
        if h > 0:
            otp = otp[h:]

        return otp, time.time() - conversion_start_time


import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
import wave


class CycleGANFactory():
    def __init__(self, model, config_json):
        self.args = util.config_reader(
            config_json,
            default={
                #
                "checkpoint_dir": "./trained_models",
                "wave_otp_dir": "./harvests",
                "train_data_dir": "./dataset/train",
                #
                "real_data_compare": False,
                "test": False,
                "summary": "console",  # or "tensorboard", False
                #
                "batch_size": 1,
                "weight_Cycle": 100.0,
                "train_iteration": 100000,
                "start_epoch": 0,
                #
                "use_colab": False,
                "colab_hardware": "tpu",
            })

        self.net = CycleGAN(model)
        f0_transfer = util.generate_f0_transfer("./voice_profile.npy")
        self.converter = Converter(self.net, f0_transfer).convert
        
        self.sample_size = 2

        if self.args["use_colab"]:
            pass
        else:
            self.net._create_optimizer = lambda self: tf.train.AdamOptimizer(4e-6, 0.5, 0.999)

        if self.args["summary"]:
            self.summary(self.args["summary"])

        if self.args["test"]:
            self.test(self.args["test"])

        if self.args["checkpoint_dir"]:
            self.checkpoint(self.args["checkpoint_dir"])


        if self.args["train_data_dir"]:
            self.batch_files(self.args["train_data_dir"] + '/A.npy', self.args["train_data_dir"] + '/B.npy')

    def summary(self, summary):
        writer = None
        if summary == "tensorboard":
            writer = tf.summary.FileWriter(
                os.path.join("logs", self.net.name), self.net.session.graph)
        elif summary == "console":
            writer = util.ConsoleSummary()
            self.args["real_data_compare"] = False

        def update_summary(epoch, iteration, period):
            tb_result = self.net.session.run(
                self.net.loss.display,
                feed_dict={
                    self.net.input.A:
                    self.net.sounds_r[0:self.args["batch_size"]],
                    self.net.input.B:
                    self.net.sounds_t[0:self.args["batch_size"]],
                    self.net.time:
                    np.zeros([1])
                })
            print(" [I] finish epoch %04d : iterations %d in %d seconds" %
                  (epoch, iteration, period))
            writer.add_summary(tb_result, iteration)

        self.net.callback_every_epoch["summary"] = update_summary

    def batch_files(self, A, B):
        print(" [I] loading dataset...")
        sounds_r = np.load(A)
        sounds_t = np.load(B)
        
        sounds_r = sounds_r.reshape([
            sounds_r.shape[0], sounds_r.shape[1],
            sounds_r.shape[2], 1
        ])
        sounds_t = sounds_t.reshape([
            sounds_t.shape[0], sounds_t.shape[1],
            sounds_t.shape[2], 1
        ])

        self.net.sounds_r = sounds_r
        self.net.sounds_t = sounds_t
        return self

    def test(self, test_files):
        def save_converting_test_files(epoch, iteration, period):
            for file in glob(test_files):
                basename = os.path.basename(file)

                testdata = util.isread(file)
                converted, _ = self.converter(testdata)
                im = util.fft(converted.copy().astype(np.float32) / 32767.0)

                # saving fake spectrum
                plt.clf()
                plt.subplot(2, 1, 1)
                ins = np.transpose(im, (1, 0))
                plt.imshow(ins, vmin=-15, vmax=5, aspect="auto")
                plt.subplot(2, 1, 2)
                plt.plot(converted)
                plt.ylim(-1, 1)
                path = os.path.join(
                    self.args["wave_otp_dir"],
                    "%s_%d_%s" % (basename, epoch // self.args["batch_size"],
                                  datetime.now().strftime("%m-%d_%H-%M-%S")))
                plt.savefig(path + ".png")
                plt.savefig("latest.png")

                #saving fake waves
                voiced = converted.astype(np.int16)[800:156000]

                ww = wave.open(path + ".wav", 'wb')
                ww.setnchannels(1)
                ww.setsampwidth(self.sample_size)
                ww.setframerate(16000)
                ww.writeframes(voiced.reshape(-1).tobytes())
                ww.close()
                pass

        self.net.callback_every_epoch["test"] = save_converting_test_files
        return self

    def checkpoint(self, checkpoint_dir):
        model_name = "wave2wave.model"
        dir = os.path.join(checkpoint_dir, self.net.name)
        os.makedirs(dir, exist_ok=True)

        def save_checkpoint(epoch, iteration, period):
            self.net.save(os.path.join(dir, model_name), global_step=epoch)

        self.net.callback_every_epoch["save"] = save_checkpoint

        self.net.load(dir)

        return self