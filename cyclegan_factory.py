import tensorflow as tf
import util
import time
from datetime import datetime

class Dummy():
    pass

class CycleGAN():
    def __init__(self, model, input_a, input_b, processor='', cycle_weight=1.0):
        self.name = model.name + model.version
        self.batch_size = model.input_size[0]

        self.test_size = model.input_size.copy()
        self.test_size[0] = 1

        self.sounds_r = input_a
        self.sounds_t = input_b

        self.callback_every_epoch = {}
        self.callback_every_iteration = {}

        input = Dummy()
        input.A = tf.placeholder(tf.float32, model.input_size, "inputs_g_A")
        input.B = tf.placeholder(tf.float32, model.input_size, "inputs_g_B")
        input.test = tf.placeholder(tf.float32, self.test_size, "inputs_g_test")
        self.time = tf.placeholder(tf.float32, [1], "inputs_g_test")

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
        
        self.session = tf.Session(processor)
        self.saver = tf.train.Saver()

    def initialize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.GradientDescentOptimizer(4e-6)
            if self._tpu:
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
            self.g_optim = optimizer.minimize(self.loss.g, var_list=self.vars.g)

        optimizer = tf.train.GradientDescentOptimizer(4e-6)
        if self._tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        self.d_optim = optimizer.minimize(self.loss.d, var_list=self.vars.d)

        if use_tpu:
            self.session.run(tf.contrib.tpu.initialize_system())
        
        initializer = tf.global_variables_initializer()
        self.session.run(initializer)


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
                        self.time: ttt
                    })
                # update G network
                self.session.run(
                    self.g_optim,
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
        # self.saver.save(self.session, file, global_step=global_step)
        pass

    def load(self, dir):
        print(" [I] Reading checkpoint...")
        return

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
import sys


class CycleGANFactory():
    def __init__(self, model):
        self._model = model
        self._processor = ''
        self._tpu = False
        self._cycle_weight = 100.0

        self._input_a = None
        self._input_b = None
        self._checkpoint = None
        self._optimizer = {
            "kind": "GradientDescent",
            "rate": 4e-6,
            "params": {}
        }
        self._summaries = []
        self._test = []


    def _write_message(self, level, str):
        tag = "VEWDI"
        sys.stderr.write("[CycleGAN FACTORY] " + tag[level] + ": " + str)

    def _v(self, str): self._write_message(0, str)
    def _e(self, str): self._write_message(1, str)
    def _w(self, str): self._write_message(2, str)
    def _d(self, str): self._write_message(3, str)
    def _i(self, str): self._write_message(4, str)

    def summary(self, summary):
        self._summaries.append(summary)
        return self
    
    def hardware(self, hardware):
        params = hardware.split(',')
        self._tpu = False
        if 'tpu' in params:
            if not 'COLAB_TPU_ADDR' in os.environ:
                self._w('Failed to get tpu address, do not use TPU (use for CPU or GPU)')
            else:
                self._processor = 'grpc://' + os.environ['COLAB_TPU_ADDR']
                self._tpu = True
                self._d('use TPU')

        return self

    def input(self, A, B):
        self._input_a = A
        self._input_b = B

        return self

    def test(self, callback, append=False):
        self._test.append(callback)
        
        return self

    def checkpoint(self, checkpoint_dir):
        """
        model_name = "wave2wave.model"
        dir = os.path.join(checkpoint_dir, self.net.name)
        os.makedirs(dir, exist_ok=True)

        def save_checkpoint(net, epoch, iteration, period):
            net.save(os.path.join(dir, model_name), global_step=epoch)

        self.net.callback_every_epoch["save"] = save_checkpoint

        self.net.load(dir)
        """
            
        return self
    
    def cycle_weight(self, weight):
        self._cycle_weight = weight
        return self

    def optimizer(self, kind, rate, params={}):
        optimizer_list = ["GradientDescent", "Adam"]
        if kind in optimizer_list:
            self._optimizer["kind"] = kind
        else:
            raise Exception("Unknown optimizer %s" % kind)

        if rate > 0:
            self._optimizer["rate"] = rate
        else:
            raise Exception("Should larger than 0 training rate")

        if params:
            self._optimizer["params"] = params

        return self
        
    def build(self):
        if self._input_a is None or self._input_b is None:
            raise Exception("No defined input data")
        if not self.checkpoint:
            self._w('checkpoint is undefined, trained model is no save')

        net = CycleGAN(model, self._input_a, self._input_b, cycle_weight=self._cycle_weight, processor=self._processor)

        # register summary
        for summary in self._summaries[-1:]: # ラストひとつだけやる。たくさんやるのは未実装
            writer = None
            if summary == "tensorboard":
                writer = tf.summary.FileWriter(
                    os.path.join("logs", net.name), net.session.graph)
            elif summary == "console":
                writer = util.ConsoleSummary('./training_value.jsond')

            def update_summary(net, epoch, iteration, period):
                tb_result = net.session.run(
                    net.loss.display,
                    feed_dict={
                        net.input.A: self._input_a[0:model.batch_size],
                        net.input.B: self._input_b[0:model.batch_size],
                        net.time: np.zeros([1])
                    })
                self._i("finish epoch %04d : iterations %d in %f seconds" %
                    (epoch, iteration, period))
                writer.add_summary(tb_result, iteration)

            net.callback_every_epoch["summary"] = update_summary

        # add checkpoint
        # save

        net.initialize()
        
        # register test
        for test in self._test[-1:]: # ラストひとつだけやる。たくさんやるのは未実装
            net.callback_every_epoch["test"] = test
            
        # load

        return net



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

    data_dir = os.path.join(".", "dataset", "train")
    dataset = list(map(lambda data: data.reshape(list(data.shape) + [1]),
        map(lambda d: np.load(os.path.join(data_dir, d)), ["A.npy", "B.npy"])))
    data_size = min(dataset[0].shape[0], dataset[1].shape[0])
    dataset = list(map(lambda data: data[:data_size], dataset))

    model = w2w(1)
    name = "_".join([model.name, model.version, "tpu"])
    net = CycleGANFactory(model) \
            .cycle_weight(100.00) \
            .optimizer("GradientDescent", 4e-6) \
            .summary("console") \
            .test(save_converting_test_files) \
            .hardware("colab,tpu") \
            .checkpoint(os.path.join(".", "trained_model", name)) \
            .input(dataset[0], dataset[1]) \
            .build()

    net.train(100000)

