
import tensorflow as tf

def discriminator(inp,reuse,train=True):
    """
    Discriminator(識別者)ネットワーク
    構造は4層のネットワークにExponential-neuron[https://arxiv.org/abs/1901.00279]に基づく局所解の対策を施しています。
    ちなみに重みにtanhをかけているのはパラメータ発散防止目的です。
    また、周波数軸はチャンネルとして扱います。
    """
    ten = tf.transpose(inp,[0,1,3,2])

    w=tf.get_variable("kernel_e",[3,1,513,3],initializer=tf.initializers.random_normal(stddev=0.002))
    b=tf.get_variable("bias_e",[3],initializer=tf.zeros_initializer())
    alpha=tf.get_variable("alpha_e",[1,1,1,3],initializer=tf.initializers.random_normal(stddev=0.002))
    w=tf.tanh(w)
    tenE=tf.nn.conv2d(ten,w,[1,1,1,1],"SAME")
    tenE=tf.reshape(tf.exp(tf.nn.bias_add(tenE,b))*alpha,[ten.shape[0],ten.shape[1],3])

    ten = tf.layers.conv2d(ten,256,[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,name="disc_0",reuse=reuse)
    ten = tf.nn.leaky_relu(ten)
    ten = tf.layers.conv2d(ten,3,[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,name="disc_1",reuse=reuse)

    ten=tf.reshape(ten,[ten.shape[0],ten.shape[1],3])
    ten+=tenE
    return ten

def generator(ten,reuse):
    """
    Generator(生成)ネットワーク
    構造はAttention-Netowrkです。
    BNにinstance_normを用います。
    """

    ten=tf.transpose(ten,[0,1,3,2])
    ten = tf.layers.conv2d_transpose(ten, 64, [1, 1],kernel_initializer=tf.initializers.random_normal(stddev=0.002), use_bias=False,
                            reuse=reuse,name="encode_conv_")
    ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="bn_00")
    ten = tf.nn.leaky_relu(ten)
    for i in range(2):
        tenA = tf.layers.conv2d(ten,64,[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=False,reuse=reuse, name="mid_GATE_" + str(i))
        tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="bn0" + str(i))
        ten = ten*tf.nn.sigmoid(tenA)
        ten = tf.layers.conv2d(ten,64,[8,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=False,reuse=reuse, name="mid_CONV_" + str(i))
        ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="bn1" + str(i))
    ten = tf.layers.conv2d(ten,513,[1,1],kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse,name="decode_conv")
    ten = tf.transpose(ten, [0, 1, 3, 2])
    return tf.tanh(ten)