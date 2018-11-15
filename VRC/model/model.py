
import tensorflow as tf
import math

def discriminator(inp,reuse):
    current=inp
    # setting paramater
    depth=2
    chs=[128,256]

    # convolution(2*4,stride 1*4)
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[3,7], strides=[1,4], padding="VALID",kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),use_bias=True, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        # ten = tf.layers.batch_normalization(ten, axis=3, training=True, trainable=True, reuse=reuse,
        #                                      name="disc_bn_" + str(i))
        current = tf.nn.leaky_relu(ten)
    # dense
    current=tf.reshape(current,[current.shape[0],current.shape[1],current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(current,1,name="dence",reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(0.002)))

    return ten

def generator(ten,reuse,train):
    # setting paramater
    times=2
    chs_enc=[32,64]
    chs_dec=[64,32]
    ten = tf.layers.conv2d(ten, 16, [3, 5], [2, 2], padding="VALID",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 3/2/8)),
                            use_bias=False,
                            data_format="channels_last", reuse=reuse, name="res_conv_A_pre_")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="enc_bn0_")
    ten = tf.nn.relu(ten)

    for i in range(times):
        tenA=tf.layers.conv2d(ten, chs_enc[i], [1, 4], [1, 4], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/4/chs_enc[i])), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_A_" + str(i))

        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="enc_bn1_" + str(i))
        ten = tf.nn.relu(tenA)

    tenB=tf.transpose(ten,[0,1,3,2])
    rs=int(tenB.shape[3])
    tenB=tf.layers.dense(tenB,rs,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),use_bias=False,reuse=reuse,name="dense"+str(i))
    ten = tf.transpose(tenB, [0, 1, 3, 2])

    tenA = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="res_bn_" + str(i))
    ten = tf.nn.leaky_relu(tenA+ten)
    for i in range(9):
        tenA=tf.layers.conv2d(ten, 64, [3, 3], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/9/32)), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_C_" + str(i))

        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn1_" + str(i))
        rate=1-(i/18.0)
        tenA = ShakeDrop(tenA, rate, train)
        ten = tf.nn.leaky_relu(tenA+ten)

    # decodeing
    for i in range(times):

        tenA=deconve_with_ps(ten,[1,4],chs_dec[i],reuse,"dec_"+str(i),False)
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="dec_bn1_" + str(i))
        ten = tf.nn.relu(tenA)
    ten= tf.layers.conv2d_transpose(ten ,1, kernel_size=[3, 3], strides=[2, 2], padding="VALID",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.002),use_bias=True,
                                data_format="channels_last", reuse=reuse, name="last_conv1")
    return tf.sigmoid(ten)

def deconve_with_ps(inp,r,otp_shape,reuses=None,name="",b=True):
    # pixcel shuffler layer

    # calculating output channels
    ch_r=r[0]*r[1]*otp_shape
    in_h = inp.get_shape()[1]
    in_w = inp.get_shape()[2]

    # convolution
    ten = tf.layers.conv2d(inp, ch_r, kernel_size=[1,1], strides=[1,1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/ch_r)), use_bias=b,
                           data_format="channels_last", reuse=reuses, name=name )
    # reshaping
    ten = tf.reshape(ten, [-1, r[0], r[1], in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [-1, in_h * r[0], in_w * r[1], otp_shape])

    return ten
def ShakeDrop(ten,rate,train):
    # shakedrop layer
    # s=ten.get_shape()
    if train:
        s = [int(ten.get_shape()[0]), int(ten.get_shape()[1]), int(ten.get_shape()[2]), int(ten.get_shape()[3])]
        # random noise
        f_rand = tf.random_uniform(s, -1.0, 1.0)
        b_rand = tf.random_uniform(s, 0.0, 1.0)
        # droping
        prop=tf.random_uniform(s,0.0,1.0)+rate
        prop=tf.floor(prop)
        tenA= ten*prop
        tenB = ten*(1-prop)

        # shaking
        ten=tenA+tenB*b_rand+tf.stop_gradient(tenB*(f_rand-b_rand))

        return ten
    else:

        return ten*rate

