import tensorflow as tf
import os
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
import json
import shutil
import numpy as np


class Model:
    def __init__(self, path):
        self.args = dict()
        self.args["checkpoint_dir"] = "./trained_models"
        self.args["wave_otp_dir"] = "False"
        self.args["train_data_num"] = 500
        self.args["batch_size"] = 4
        self.args["depth"] = 4
        self.args["d_depth"] = 4
        self.args["train_epoch"] = 500
        self.args["stop_itr"] = -1
        self.args["start_epoch"] = 0
        self.args["test"] = True
        self.args["log"] = True
        self.args["tensorboard"] = False
        self.args["hyperdash"] = False
        self.args["stop_argument"] = True
        self.args["stop_value"] = 0.5
        self.args["input_size"] = 8192
        self.args["weight_Norm"] = 1.0
        self.args["NFFT"] = 128
        self.args["debug"] = False
        self.args["noise"] = False
        self.args["cupy"] = False
        self.args["D_channels"] = [2]
        self.args["G_channel"] = 32
        self.args["G_channel2"] = 32
        self.args["strides_g"] = [2, 2]
        self.args["strides_g2"] = [1, 1]
        self.args["strides_d"] = [2, 2]
        self.args["filter_g"] = [8, 8]
        self.args["filter_g2"] = [2, 2]
        self.args["filter_d"] = [4, 4]
        self.args["model_name"] = "wave2wave"
        self.args["version"] = "1.0.0"
        self.args["log_eps"] = 1e-8
        self.args["g_lr"] = 2e-4
        self.args["g_b1"] = 0.5
        self.args["g_b2"] = 0.999
        self.args["d_b1"] = 0.5
        self.args["d_b2"] = 0.999
        self.args["train_d_scale"] = 1.0
        self.args["train_interval"] = 10
        self.args["pitch_rate"] = 1.0
        self.args["pitch_res"] = 563.0
        self.args["pitch_tar"] = 563.0
        self.args["test_dir"] = "./test"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    keys = data.keys()
                    for k in keys:
                        if k in self.args:
                            if type(self.args[k]) == type(data[k]):
                                self.args[k] = data[k]
                            else:
                                print(" [!] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(
                                    type(self.args[k])) + "\"")
                        else:
                            print(" [!] Argument \"" + k + "\" is not exsits.")
            except json.JSONDecodeError as e:
                print(' [x] JSONDecodeError: ', e)
        else:
            print(" [!] Setting file is not found")
        if len(self.args["D_channels"]) != (self.args['d_depth'] + 1):
            print(" [!] Channels length and depth+1 must be equal ." + str(len(self.args["D_channels"])) + "vs" + str(
                self.args['d_depth'] + 1))
            self.args["D_channels"] = [min([2 ** (i + 1) - 2, 254]) for i in range(self.args['d_depth'] + 1)]
        if self.args["pitch_rate"] == 1.0:
            self.args["pitch_rate"] = self.args["pitch_tar"] / self.args["pitch_res"]
            print(" [!] pitch_rate is not found . calculated value : " + str(self.args["pitch_rate"]))
        self.args["SHIFT"] = self.args["NFFT"] // 2
        ss = int(self.args["input_size"]) * 2 // int(self.args["NFFT"]) + 1
        self.args["name_save"] = self.args["model_name"] + self.args["version"]

        self.input_size_model = [self.args["batch_size"], self.args["NFFT"], self.args["NFFT"], 2]
        print("model input size:" + str(self.input_size_model))
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions()))
        if bool(self.args["debug"]):
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
        if self.args["wave_otp_dir"] is not "False":
            self.args["wave_otp_dir"] = self.args["wave_otp_dir"] + self.args["name_save"] + "/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])
            shutil.copy(path, self.args["wave_otp_dir"] + "info.json")
            self.args["log_file"] = self.args["wave_otp_dir"] + "log.txt"
        self.checkpoint_dir = self.args["checkpoint_dir"]

    def build_model(self):

        # inputs place holder
        # 入力
        self.input_model = tf.placeholder(tf.float32, self.input_size_model, "input")
        # creating generator
        # G-net（生成側）の作成
        with tf.variable_scope("generator_1"):
            ifs = generator(self.input_model,chs=self.args["G_channel"], depth=self.args["depth"], f=self.args["filter_g"],
                                 s=self.args["strides_g"])

        self.fake_B_image = tf.identity(ifs,"output")
        self.g_vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator_1")
        # 保存の準備
        self.saver = tf.train.Saver()


    def save(self, checkpoint_dir, step):
        self.init_fn(self.sess)
        model_name = "wave2wave.model"
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
        # initialize variables
        # 変数の初期化
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(" [*] Reading checkpoint...")
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        print(checkpoint_dir)
        slim = tf.contrib.slim
        self.init_fn = slim.assign_from_checkpoint_fn(tf.train.latest_checkpoint(checkpoint_dir), self.g_vars_1)
        return True

def generator(current_outputs, depth, chs, f, s):
    current = current_outputs
    output_shape = int(current.shape[3])
    # main process
    for i in range(depth):
        current = block(current, output_shape, chs, f, s, i)

    return current


def block(current, output_shape, chs, f, s, depth):
    currents = tf.scalar_mul(1,current)
    ten=current
    # if depth!=0:

    ten = batch_norm(ten,depth,0)

    ten = conv2d(ten,depth, chs, f, s)

    ten = leaky_relu(ten)
    ten = batch_norm(ten, depth, 0)


    ten = deconv2d(ten,depth, output_shape,f,s)
    cct = batch_norm(tf.reshape(ten[:, :, :, 0], [ten.shape[0], ten.shape[1], ten.shape[2], 1]))
    cct2 = tf.reshape(ten[:, :, :, 1], [ten.shape[0], ten.shape[1], ten.shape[2], 1])
    ten = tf.concat([cct, cct2], axis=3)
    ten= tf.reshape(ten+currents,ten.shape)
    # with tf.variable_scope("add_layer_Layer_"+str(depth)):
    #     sc=ten.shape[1:]
    #     fig=tf.reshape(tf.get_variable("add_filter",sc,tf.float32,tf.zeros_initializer(),trainable=True),[1,sc[0],sc[1],sc[2]])
    #     figs=tf.tile(fig,(ten.shape[0],1,1,1))
    #     ten = ten + figs

    return ten

def conv2d(inp,depth,chs,f,s):
    name="conv2d"
    if depth!=0:
        name+="_"+str(depth)
    with tf.variable_scope(name):
        sc1=[f[0],f[1],2,chs]
        sc2=[chs]
        sss=[1,1,s[0],s[1]]
        kernel=tf.get_variable("kernel",sc1,tf.float32,tf.zeros_initializer(),trainable=True)
        bias=tf.get_variable("bias",sc2,tf.float32,tf.zeros_initializer(),trainable=True)

    ten= tf.nn.conv2d(inp,filter=kernel,strides=sss,padding="VALID",name="Conv2d")
    ten= ten+bias

    return ten

def deconv2d(inp,depth,chs,f,s,otp=[1,128,128,2]):
    name = "conv2d_transpose"
    if depth != 0:
        name += "_" + str(depth)
    with tf.variable_scope(name):
        sc1 = [f[0], f[1], 2,inp.shape[3] ]
        sc2 = [chs]
        sss = [1, 1, s[0], s[1]]
        kernel = tf.get_variable("kernel", sc1, tf.float32, tf.zeros_initializer(), trainable=True)
        bias = tf.get_variable("bias", sc2, tf.float32, tf.zeros_initializer(), trainable=True)

    ten = tf.nn.conv2d_transpose( inp,kernel,otp, strides=sss, padding="VALID",name="Deconv2d")
    ten = ten+ bias
    return ten


def leaky_relu(inp):
    # a=tf.constant(-0.01,tf.float32,shape=inp.shape)
    # a2 = tf.constant(-1, tf.float32,shape=inp.shape)
    # return tf.nn.relu(inp)+(a*(tf.nn.relu(a2*inp)))
    return tf.nn.leaky_relu(inp)
def batch_norm(inp,depth,i):
    # return tf.layers.batch_normalization(inp,axis=3, training=False, trainable=True,
    #                                     gamma_initializer=tf.ones_initializer())
    name="batch_normalization"
    a=depth*2+i
    if a!=0:
        name+="_"+str(a)
    with tf.variable_scope(name):
        sc=inp.shape[3]
        gamma=tf.get_variable("gamma",sc,tf.float32,tf.zeros_initializer(),trainable=True)
        beta=tf.get_variable("beta",sc,tf.float32,tf.zeros_initializer(),trainable=True)
        mean = tf.get_variable("moving_mean",sc,tf.float32,tf.zeros_initializer(),trainable=False)
        var=tf.get_variable("moving_variance",sc,tf.float32,tf.zeros_initializer(),trainable=False)
    return tf.nn.batch_normalization(inp,mean,var,beta,gamma,1e-8)
path="../setting.json"
net=Model(path)
net.build_model()
if not net.load():
    exit(-1)
net.sess=tf.Session()
net.saver=tf.train.Saver(tf.global_variables())
net.save("../Network",0)
summary_writer = tf.summary.FileWriter('./log',net.sess.graph)
isa=np.zeros(net.input_size_model)
# print(net.sess.run(net.fake_B_image,feed_dict={net.input_model:isa}))
print(" [*] finished!!")