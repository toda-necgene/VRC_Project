import tensorflow as tf
import os
import numpy as np
import json
from model import generator
import pyworld.pyworld as pw
class Model:
    def __init__(self,path):
        self.args = dict()

        # default setting
        self.args["model_name"] = "VRC"
        self.args["version"] = "18.12.22"

        self.args["checkpoint_dir"] = "./trained_models"

        self.args["input_size"] = 4096
        self.args["pitch_rate_var"]=1.0
        self.args["pitch_rate_mean_s"]=0.0
        self.args["pitch_rate_mean_t"]=0.0

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
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        # shapes of inputs
        self.input_size_test = [1, 52,513,1]
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))

        self.build_model()
    def build_model(self):

        #inputs place holder
        self.input_model_test = tf.placeholder(tf.float32, self.input_size_test, "inputs_G-net_A")
        #creating generator

        with tf.variable_scope("generator_1"):
            self.test_outputaB = generator(self.input_model_test, reuse=None, training=False)

        #saver
        self.saver = tf.train.Saver()


    def load(self):
        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(" [*] Reading checkpoint...")
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(self.args["checkpoint_dir"], model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt :
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




