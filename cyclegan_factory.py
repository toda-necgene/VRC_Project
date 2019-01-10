import tensorflow as tf
import util
import time
from datetime import datetime


class Dummy():
    pass


class CycleGAN():
    def __init__(self, model=None, processor='', cycle_weight=1.0, create_optimizer=None):
        self.tpu = True

        self.name = model.name + model.version
        self.batch_size = model.input_size[0]

        self.test_size = model.input_size.copy()
        self.test_size[0] = 1

        self.sounds_r = []
        self.sounds_t = []

        self._create_optimizer = None
        self.callback_every_epoch = {}
        self.callback_every_iteration = {}

        input = Dummy()
        input.A = tf.placeholder(tf.float32, model.input_size, "inputs_g_A")
        input.B = tf.placeholder(tf.float32, model.input_size, "inputs_g_B")
        input.test = tf.placeholder(tf.float32, self.test_size, "inputs_g_test")
        # self.time = tf.placeholder(tf.float32, [1], "inputs_g_test")

        #creating generator
        with tf.variable_scope("generator_1"):
            fake_aB = model.generator(input.A, reuse=None, training=True)
            self.fake_aB_test = model.generator(input.test, reuse=True, training=False)
        with tf.variable_scope("generator_2"):
            fake_bA = model.generator(input.B, reuse=None, training=True)
        with tf.variable_scope("generator_2", reuse=True):
            fake_Ba = model.generator(fake_aB, reuse=True, training=True)
        with tf.variable_scope("generator_1", reuse=True):
            fake_Ab = model.generator(fake_bA, reuse=True, training=True)

        #creating discriminator
        with tf.variable_scope("discriminators"):
            inp = tf.concat([fake_aB, input.B, fake_bA, input.A], axis=0)
            d_judge = model.discriminator(inp, None)

        vars = Dummy()
        #getting individual variabloes
        vars.g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator_1")
        vars.g += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator_2")
        vars.d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "discriminators")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #objective-functions of discriminator
        l0 = tf.reshape(tf.one_hot(0, 3), [1, 1, 3])
        l1 = tf.reshape(tf.one_hot(1, 3), [1, 1, 3])
        l2 = tf.reshape(tf.one_hot(2, 3), [1, 1, 3])
        labelA = tf.tile(l0, [model.input_size[0], model.input_size[1], 1])
        labelB = tf.tile(l1, [model.input_size[0], model.input_size[1], 1])
        labelO = tf.tile(l2, [model.input_size[0], model.input_size[1], 1])
        labels = tf.concat([labelO, labelB, labelO, labelA], axis=0)

        loss = Dummy()
        loss.d = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=d_judge) * 4

        # objective-functions of generator

        # Cyc lossB
        g_loss_cyc_B = tf.pow(tf.abs(fake_Ab - input.B), 2)

        # Gan lossA
        g_loss_gan_A = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=d_judge[self.batch_size * 2:self.batch_size * 3],
            labels=labelA)

        # Cycle lossA
        g_loss_cyc_A = tf.pow(tf.abs(fake_Ba - input.A), 2)

        # Gan lossB
        g_loss_gan_B = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=d_judge[:self.batch_size], labels=labelB)

        # generator loss
        loss.g = tf.losses.compute_weighted_loss(
            g_loss_cyc_A + g_loss_cyc_B,
            cycle_weight) + g_loss_gan_B + g_loss_gan_A

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
            "d_loss", tf.reduce_mean(loss.d), family="d_loss")

        loss.display = tf.summary.merge(
            [g_loss_sum_A_display, g_loss_sum_B_display, d_loss_sum_A_display])
        # result_score = tf.placeholder(tf.float32, name="FakeFFTScore")
        # fake_B_FFT_score_display = tf.summary.scalar(
        #    "g_error_AtoB", tf.reduce_mean(result_score), family="g_test")
        # g_test_display = tf.summary.merge([fake_B_FFT_score_display])

        self.input = input
        self.vars = vars
        self.loss = loss
        
        print(processor)
        self.session = tf.Session(processor)
        print(' [I] Devices list: %s' % self.session.list_devices())
        self.saver = tf.train.Saver()

        # naming output-directory
        with tf.control_dependencies(update_ops):
            optimizer = create_optimizer()
            self.g_optim = optimizer.minimize(self.loss.g, var_list=self.vars.g)

        optimizer = create_optimizer()
        self.d_optim = optimizer.minimize(self.loss.d, var_list=self.vars.d)

        print(' [D] created optimizer')

    def train(self, train_iteration=100000):
        assert len(self.sounds_r) > 0
        assert len(self.sounds_r) == len(self.sounds_t)

        # initializing training infomation
        start_time_all = time.time()

        batch_idxs = self.sounds_r.shape[0] // self.batch_size
        train_epoch = train_iteration // batch_idxs + 1

        index_list_r = [h for h in range(self.sounds_r.shape[0])]
        index_list_t = [h for h in range(self.sounds_r.shape[0])]

        iterations = 0
        # main-training
        epoch_count_time = time.time()
        for epoch in range(train_epoch):
            if epoch % self.batch_size == 0:
                period = time.time() - epoch_count_time
                for f in self.callback_every_epoch.values():
                    f(self, epoch, iterations, period)
                epoch_count_time = time.time()

            # shuffling train_data_index
            np.random.shuffle(index_list_r)
            np.random.shuffle(index_list_t)

            for idx in range(0, batch_idxs):
                # getting batch
                if iterations == train_iteration:
                    break
                st = self.batch_size * idx
                batch_sounds_resource = np.asarray([
                    self.sounds_r[ind]
                    for ind in index_list_r[st:st + self.batch_size]
                ])
                batch_sounds_target = np.asarray([
                    self.sounds_t[ind]
                    for ind in index_list_t[st:st + self.batch_size]
                ])
                ttt = np.array([1.0 - iterations / train_iteration])
                # update D network
                self.session.run(
                    self.d_optim,
                    feed_dict={
                        self.input.A: batch_sounds_resource,
                        self.input.B: batch_sounds_target,
                        # self.time: ttt
                    })
                # update G network
                self.session.run(
                    self.g_optim,
                    feed_dict={
                        self.input.A: batch_sounds_resource,
                        self.input.B: batch_sounds_target,
                        # self.time: ttt
                    })

                print('run')

                iterations += 1
            # calculating ETA
            if iterations == train_iteration:
                break

        period = time.time() - epoch_count_time
        for f in self.callback_every_epoch.values():
            f(self, train_epoch, iterations, period)

        taken_time_all = time.time() - start_time_all
        print(" [I] ALL train process finished successfully!! in %f Hours" %
              (taken_time_all / 3600))

    def a_to_b(self, array):
        resource = array.reshape(self.test_size)
        result = self.session.run(
                self.fake_aB_test,
                feed_dict={self.input.test: resource})
        return result[0].reshape(-1, 513).astype(np.float)

    def b_to_a(self, tensor):
        pass

    def save(self, file, global_step):
        if not self.tpu:
            self.saver.save(self.session, file, global_step=global_step)

    def load(self, dir):
        # initialize variables
        print(' [D] load start')
        if self.tpu:
            initializer = tf.contrib.tpu.initialize_system()
            self.session.run(initializer)
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
                "train_data_dir": "./dataset/train",
                #
                "summary": "console",  # or "tensorboard", False
                #
                "weight_Cycle": 100.0,
                "train_iteration": 100000,
                "start_epoch": 0,
                #
                "use_colab": False,
                "colab_hardware": "tpu",
            })

        adam_optimizer = lambda: tf.train.AdamOptimizer(4e-6, 0.5, 0.999)

        optimizer = adam_optimizer
        processor = ''
        if self.args["use_colab"] and self.args["colab_hardware"] == "tpu":
            optimizer = lambda: tf.contrib.tpu.CrossShardOptimizer(adam_optimizer())
            processor = 'grpc://' + os.environ['COLAB_TPU_ADDR']
            print(' [D] use TPU')


        self.net = CycleGAN(model, cycle_weight=self.args["weight_Cycle"], create_optimizer=optimizer, processor=processor)
        
        if self.args["summary"]:
            self.summary(self.args["summary"])

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
            writer = util.ConsoleSummary('./training_value.jsond')
            self.args["real_data_compare"] = False

        def update_summary(net, epoch, iteration, period):
            tb_result = net.session.run(
                net.loss.display,
                feed_dict={
                    self.net.input.A: self.net.sounds_r[0:self.net.batch_size],
                    self.net.input.B: self.net.sounds_t[0:self.net.batch_size],
                    # self.net.time: np.zeros([1])
                })
            print(" [I] finish epoch %04d : iterations %d in %f seconds" %
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

        size = min(sounds_r.shape[0], sounds_t.shape[0])

        self.net.sounds_r = sounds_r[:size]
        self.net.sounds_t = sounds_t[:size]
        return self

    def test(self, callback, append=False):
        if not append:
            self.net.callback_every_epoch["test"] = callback
        else:
            raise Exception("未実装")
        return self

    def checkpoint(self, checkpoint_dir):
        model_name = "wave2wave.model"
        dir = os.path.join(checkpoint_dir, self.net.name)
        os.makedirs(dir, exist_ok=True)

        def save_checkpoint(net, epoch, iteration, period):
            net.save(os.path.join(dir, model_name), global_step=epoch)

        self.net.callback_every_epoch["save"] = save_checkpoint

        self.net.load(dir)

        return self



from model import Model as w2w
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_files = "dataset\\test\\*.wav"
    test_output_dir = "waves"
    f0_transfer = util.generate_f0_transfer("./voice_profile.npy")
    sample_size = 2
    def save_converting_test_files(net, epoch, iteration, period):
        converter = Converter(net, f0_transfer).convert
        for file in glob(test_files):
            basename = os.path.basename(file)

            testdata = util.isread(file)
            converted, _ = converter(testdata)
            converted_norm = converted.copy().astype(np.float32) / 32767.0
            im = util.fft(converted_norm)

            # saving fake spectrum
            plt.clf()
            plt.subplot(2, 1, 1)
            ins = np.transpose(im, (1, 0))
            plt.imshow(ins, vmin=-15, vmax=5, aspect="auto")
            plt.subplot(2, 1, 2)
            plt.plot(converted_norm)
            plt.ylim(-1, 1)
            path = os.path.join(
                test_output_dir,
                "%s_%d_%s" % (basename, epoch // net.batch_size,
                                datetime.now().strftime("%m-%d_%H-%M-%S")))
            plt.savefig(path + ".png")
            plt.savefig("latest.png")

            #saving fake waves
            voiced = converted.astype(np.int16)[800:156000]

            ww = wave.open(path + ".wav", 'wb')
            ww.setnchannels(1)
            ww.setsampwidth(sample_size)
            ww.setframerate(16000)
            ww.writeframes(voiced.reshape(-1).tobytes())
            ww.close()

    net = CycleGANFactory(w2w(1), 'settings.json') \
          .test(save_converting_test_files) \
          .net

    net.train(100000)