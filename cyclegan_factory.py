import numpy as np
import tensorflow as tf
from cyclegan import CycleGAN
import os, sys
import util
from datetime import datetime

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
            "kind": tf.train.GradientDescentOptimizer,
            "rate": 4e-6,
            "params": {}
        }
        self._summaries = []
        self._test = []


    def _write_message(self, level, str):
        tag = "VEWDI"
        sys.stderr.write("[CycleGAN FACTORY] " + tag[level] + ": " + str + '\n')

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
        dir = os.path.join(checkpoint_dir, self._model.name + "_" + self._model.version)
        checkpoint_file = os.path.join(dir, datetime.now().strftime('%Y-%m-%d_%H%M%S') + ".ckpt")
        os.makedirs(dir, exist_ok=True)

        self._checkpoint = checkpoint_file
            
        return self
    
    def cycle_weight(self, weight):
        self._cycle_weight = weight
        return self

    def optimizer(self, kind, rate, params={}):
        optimizer_list = {
            "GradientDescent": tf.train.GradientDescentOptimizer,
            "Adam": tf.train.AdamOptimizer,
        }
        if kind in optimizer_list:
            self._optimizer["kind"] = optimizer_list[kind]
        else:
            raise Exception("Unknown optimizer %s" % kind)

        if rate > 0:
            self._optimizer["rate"] = rate
        else:
            raise Exception("Should larger than 0 training rate")

        if type(params) is dict:
            self._optimizer["params"] = params
        else:
            raise Exception("Additional optional params of optimizer should be dict object")

        return self
        
    def build(self):
        if self._input_a is None or self._input_b is None:
            raise Exception("No defined input data")
        if not self.checkpoint:
            self._w('checkpoint is undefined, trained model is no save')

        def generate_optimizer():
            optimizer = self._optimizer["kind"](self._optimizer["rate"], **self._optimizer["params"])
            # if use_tpu:
            #     optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
            return optimizer

        net = CycleGAN(self._model, self._input_a, self._input_b,
                cycle_weight=self._cycle_weight,
                processor=self._processor,
                generate_optimizer=generate_optimizer)

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
                        net.input.A: self._input_a[0:self._model.batch_size],
                        net.input.B: self._input_b[0:self._model.batch_size],
                        net.time: np.zeros([1])
                    })
                self._i("finish epoch %04d : iterations %d in %f seconds" %
                    (epoch, iteration, period))
                writer.add_summary(tb_result, iteration)

            net.callback_every_epoch["summary"] = update_summary

        # add checkpoint
        # save
        # register test
        for test in self._test[-1:]: # ラストひとつだけやる。たくさんやるのは未実装
            net.callback_every_epoch["test"] = test
            
        # load
        if self._checkpoint:
            def save_checkpoint(net, epoch, iteration, period):
                net.save(self._checkpoint, global_step=iteration)

            net.callback_every_epoch["save"] = save_checkpoint

            net.load(os.path.dirname(self._checkpoint))

        return net

if __name__ == '__main__':
    print("Hallo,")

