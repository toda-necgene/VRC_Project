import tensorflow as tf
import util
import time
import numpy as np
import os

class Dummy():
    pass

class CycleGAN():
    def __init__(self, model, input_a, input_b, processor='', cycle_weight=1.0, generate_optimizer=None, use_tpu=True):
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
        self.saver = tf.train.Saver(max_to_keep=None)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)        
        with tf.control_dependencies(update_ops):
            optimizer = generate_optimizer()
            self.g_optim = optimizer.minimize(self.loss.g, var_list=self.vars.g)

        optimizer = generate_optimizer()
        self.d_optim = optimizer.minimize(self.loss.d, var_list=self.vars.d)


        self._use_tpu = use_tpu
        if self._use_tpu:
            self.session.run(tf.contrib.tpu.initialize_system())
        
    def __enter__(self):        
        initializer = tf.global_variables_initializer()
        self.session.run(initializer)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._use_tpu:
            self.session.run(tf.contrib.tpu.shutdown_system())
        self.session.close()
        return True

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
                    try:
                        f(self, epoch, iterations, period)
                    except:
                        pass
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
            try:
                f(self, train_epoch, iterations, period)
            except:
                pass

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
        self.saver.save(self.session, file, global_step=global_step)
        return True

    def load(self, dir):
        checkpoint = tf.train.get_checkpoint_state(dir)
        if checkpoint:
            latest_model = checkpoint.model_checkpoint_path  # pylint: disable=E1101
            self.saver.restore(self.session, latest_model)
            return True
        else:
            return False

