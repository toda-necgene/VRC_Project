
import tensorflow as tf
import math

def discriminator(inp,reuse):
    current=inp
    # setting paramater
    depth=4
    chs=[32,64,128,256]

    # convolution(2*4,stride 1*4)
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[3,7], strides=[1,4], padding="SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2/(3*7*chs[i]))),use_bias=True, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        current = tf.nn.leaky_relu(ten)

    # dense
    current=tf.reshape(current,[current.shape[0],current.shape[1],current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(current,1,name="dence",reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2)))

    return ten
def pha_decoder(inp,reuse,train):
    res=2
    ten=inp
    tenP=inp
    for i in range(res):
        #inception resblock
        tenA =tf.layers.conv2d(ten, 16, [2, 5], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/5/2/32)), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_A_" + str(i))


        tenG = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_" + str(i))
        tenG = tf.nn.relu(tenG)

        ten=tenG

    ten = tf.layers.conv2d(ten, 1, kernel_size=[2, 7], strides=[1, 1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/2.0/7.0)), use_bias=False,
                           data_format="channels_last", reuse=reuse, name="last_conv")
    ten=tf.tanh(ten)*3.141593
    ten=tf.concat([tenP,ten],axis=3)
    return ten

def generator(ten,reuse,train):

    # setting paramater
    res=4
    times=2
    chs_enc=[32,32]
    chs_dec=[16,1]
    for i in range(times):
        tenA=tf.layers.conv2d(ten, chs_enc[i], [1, 4], [1, 4], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/4/chs_enc[i])), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_A_" + str(i))

        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="enc_bn1_" + str(i))
        ten = tf.nn.relu(tenA)
    for i in range(res):
        #inception resblock
        tenA =tf.layers.conv2d(ten, 16, [3, 3], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/3/16)), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_B_" + str(i))

        tenB=tf.transpose(ten[:,:,:,:16],[0,1,3,2])
        rs=int(tenB.shape[3])
        tenB=tf.layers.dense(tenB,rs,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/rs)),use_bias=False,reuse=reuse,name="dense"+str(i))
        tenB = tf.transpose(tenB, [0, 1, 3, 2])
        tenA=tf.concat([tenA,tenB],axis=3)

        # adding noise(shakedrop)
        prop = (1 - i / (res * 2))
        tenA = ShakeDrop(tenA, prop, train)

        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_" + str(i))
        tenA = tf.nn.relu(tenA)

        ten=tenA+ten

    # decodeing
    for i in range(times):
        ten=deconve_with_ps(ten,[1,4],chs_dec[i],reuse,"dec_"+str(i),False)
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="dec_bn1_" + str(i))
        ten = tf.nn.relu(ten)

    ten = tf.layers.conv2d(ten, 1, kernel_size=[1, 1], strides=[1, 1], padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0)),use_bias=True,
                                data_format="channels_last", reuse=reuse, name="last_conv1")

    return ten

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

