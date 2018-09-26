
import tensorflow as tf
import math

def discriminator(inp,reuse):
    current=inp
    # setting paramater
    depth=4
    chs=[16,32,64,128]

    # convolution(2*4,stride 1*4)
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[3,7], strides=[1,4], padding="SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2/(3*7*chs[i]))),use_bias=True, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        # ten=tf.layers.dropout(ten,0.4)
        current = tf.nn.leaky_relu(ten)

    # dense
    current=tf.reshape(current,[current.shape[0],current.shape[1],current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(current,1,name="dence",reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2)))

    return ten
def generator(ten,reuse,train):

    # setting paramater
    times=2
    res=3
    chs_enc=[16,128]
    chs_dec=[16,1]

    for i in range(times):
        # Encoding
        tenA = tf.layers.conv2d(ten, chs_enc[i], kernel_size=[1, 4], strides=[1, 4], padding="VALID",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/(4*chs_enc[i]))),use_bias=False,
                                data_format="channels_last", reuse=reuse, name="enc_conv"+str(i),
                                dilation_rate=(1, 1))
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,name="enc_bn"+str(i))
        ten = tf.nn.relu(tenA)

    for i in range(res):
        #inception resblock
        tenA =tf.layers.conv2d(ten, 64, [3, 3], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/(3*5*64))), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_A_" + str(i))

        tenB = tf.layers.conv2d(ten, 64, [1, 1], [1, 1], padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/64)), use_bias=False,
                                data_format="channels_last", reuse=reuse, name="res_conv_B_" + str(i))

        tenB=tf.transpose(tf.nn.relu(tenB),[0,1,3,2])
        r=tenB.get_shape()[-1]
        tenB=tf.layers.dense(tenB, r,use_bias=False,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/int(r)))
                             , reuse=reuse, name="res_dens_D_" + str(i))
        tenB = tf.transpose(tenB, [0, 1, 3, 2])

        # concatnate and BatchNormalization
        tenG = tf.concat([tenA,tenB],axis=3)
        # tenG=tenB
        tenG = tf.layers.batch_normalization(tenG, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_" + str(i))

        # adding noise(shakedrop)
        prop = (1 - i / (res * 2))
        tenG = ShakeDrop(tenG, prop, train)

        # activation
        tenG = tf.nn.relu(tenG)

        # adding skip-connection
        ten=tenG+ten

    # decodeing

    # power spectrum decoder
    tenPOW=ten
    for i in range(times):
        tenPOW = deconve_with_ps(tenPOW, [1, 4], chs_dec[i],  reuses=reuse, name="dec_pish_POW"+str(i),b=(i == times-1))
        if i!=times-1:
            tenPOW = tf.layers.batch_normalization(tenPOW, axis=3, training=train, trainable=True, reuse=reuse,
                                                 name="dec_bn_POW"+str(times+res+i) )
            tenPOW = tf.nn.relu(tenPOW)

    # phase spectrum decoder
    tenPHA=ten
    for i in range(times):
        tenPHA = deconve_with_ps(tenPHA, [1, 4], chs_dec[i], reuses=reuse, name="dec_pish_PHA"+str(i),b=(i == times-1))
        if i != times-1:
            tenPHA = tf.layers.batch_normalization(tenPHA, axis=3, training=train, trainable=True, reuse=reuse,
                                                 name="dec_bn_PHA"+str(times+res+i) )
            tenPHA = tf.nn.relu(tenPHA)

    # concating and activation
    ten=tf.concat([tenPOW,tenPHA],axis=3)
    ten=tf.tanh(ten)

    return ten

def deconve_with_ps(inp,r,otp_shape,reuses=None,name="",b=True):
    # pixcel shuffler layer

    # calculating output channels
    ch_r=r[0]*r[1]*otp_shape
    in_h = inp.get_shape()[1]
    in_w = inp.get_shape()[2]

    # convolution
    ten = tf.layers.conv2d(inp, ch_r, kernel_size=[1,1], strides=[1,1], padding="VALID",
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

