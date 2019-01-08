import tensorflow as tf

def discriminator(inp,reuse):
    # setting paramater
    chs=[64,128,256,512]
    ten = inp
    for i in range(len(chs)):
        ten = conv2d(ten,chs[i],[4,8],[1,1,4,1],True,padding="SAME",kernel_initializer=tf.initializers.he_normal(),use_bias=True,name="disc_"+str(i),reuse=reuse)
        # ten = batch_norm(ten,reuse=reuse,name="disc_bn"+str(i))
        ten = tf.nn.leaky_relu(ten)

    ten = conv2d(ten, 3, [1, 3], [1, 1,1,1],True, "VALID", kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                           use_bias=True, name="disc_last", reuse=reuse)
    return tf.reshape(ten,[ten.shape[0],ten.shape[1],3])

def generator(ten,reuse,training):
    ten = tf.transpose(ten, [0, 1, 3, 2]) # => batch, time_axis, channel, frequency
    ten = conv2d(ten, 64,[1,10],[1,1,1,1],False,"VALID",kernel_initializer=tf.initializers.he_normal()
                 , use_bias=False, reuse=reuse,name="encode_fc")
    ten = batch_norm(ten, reuse=reuse, name="encode_bn")
    ten = tf.nn.leaky_relu(ten)

    for i in range(4):
        tenA = conv2d(ten, 64, [3, 2], [1, 1,1,1], down_sample=True,padding="SAME",
                                kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=False,
                                reuse=reuse,name="guru_conv_A_" + str(i))
        tenA = batch_norm(tenA, reuse=reuse, name="guru_A_bn"+str(i))

        tenB = conv2d(tenA, 64, [1, 1], [1, 1, 1, 1],down_sample=True, padding="SAME",
                                kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=False, reuse=reuse,
                                name="guru_conv_B_"+str(i))
        tenB = batch_norm(tenB, reuse=reuse, name="guru_B_bn" + str(i))

        ten=tf.nn.leaky_relu(tenA*tf.tanh(tenB))

    ten = conv2d(ten, 513 ,[1,10],[1,1,1,1],True,"VALID",kernel_initializer=tf.initializers.random_normal(stddev=0.002),
                 use_bias=True, reuse=reuse,name="decode_fc")
    ten = tf.transpose(ten, [0, 1, 3, 2])

    return tf.tanh(ten)


def conv2d(ten,out_ch,f,s,down_sample,padding,kernel_initializer,use_bias,reuse,name):
    """
    畳み込み、または逆畳み込み(conv2d_transpose)を提供します。

    入力 [batch, height, width, channael] に対して

    畳み込み（down_sample = True)
        出力 [
            batch,
            (height - {padding ? filter_height : 0}) / stride_height + 1,
            (width - {padding ? filter_width : 0}) / stride_width + 1,
            out_channels
        ]

    逆畳み込み（down_sample = True)
        出力 [
            batch,
            (height - {padding ? filter_height : 0}) / stride_height + 1,
            (width - {padding ? filter_width : 0}) / stride_width + 1,
            out_channels
        ]
    """
    with tf.variable_scope(name,reuse=reuse):
        if down_sample:
            filter_shape = [f[0], f[1], ten.shape[-1], int(out_ch)]
            weight = tf.get_variable("kernel", filter_shape, initializer=kernel_initializer,dtype=tf.float32)
            ten=tf.nn.conv2d(ten,weight,s,padding)
        else:
            filter_shape = [f[0], f[1], out_ch, int(ten.shape[-1])]
            output_shape = [int(ten.shape[0]),int(ten.shape[1]),f[1],out_ch]
            weight = tf.get_variable("kernel", filter_shape, initializer=kernel_initializer,dtype=tf.float32)
            ten = tf.nn.conv2d_transpose(ten, weight,output_shape, s, padding)
        if use_bias:
            bias = tf.get_variable("bias",[out_ch],initializer=tf.zeros_initializer())
            ten=tf.nn.bias_add(ten,bias)
    return  ten

def batch_norm(ten,reuse,name):
    """
    与えられたテンソルの平均を0, 分散を1に正規化します

    テンソルshapeに変化無し

    Notes
    -----
    正規化によって大きな学習係数を設定することが可能になります。
    正規化パラメータγ, βは学習が必要です。
    """
    with tf.variable_scope(name, reuse=reuse):
        gamma = tf.get_variable("gamma",shape=ten.shape[-1],initializer=tf.ones_initializer(),dtype=tf.float32)
        beta = tf.get_variable("beta",shape=ten.shape[-1],initializer=tf.zeros_initializer(),dtype=tf.float32)
        mean,var=tf.nn.moments(ten,[1,2],keep_dims=True)
        ten=tf.nn.batch_normalization(ten,mean,var,beta,gamma,1e-6)
    return ten