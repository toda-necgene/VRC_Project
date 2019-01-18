
import tensorflow as tf

def discriminator(inp:tf.Tensor,reuse:bool):
    """
    #Discriminator(識別者)ネットワーク
    This is Discriminator-net.
    #周波数軸はチャンネルとして扱います。
    It is regard frequency as channels.
    #構造はBN抜きのResnetのようなもの
    Architecture looks like Resnet without BN
    #より良いモデルがあることも否定できない
    There's room for improvement.

    Parameters
    ----------
    inp  : Tensor
        #入力テンソル
        input tensor
        Shape(N,52,1,513)
    reuse: bool (None == False)
        #パラメータ共有のフラグ
        Flag of parameter-sharing

    Returns
    -------
    ten : Tensor
        #出力テンソル
        output tensor
        Shape(N,13,3)
        #soft-maxはしていない
        It not do soft-max activation for outputs.
    """
    ten = tf.layers.conv2d(inp,256,[4,1],[2,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,name="disc_1",reuse=reuse)
    ten = tf.nn.leaky_relu(ten)
    tenA = tf.layers.conv2d(ten,64,[4,1],[2,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,name="disc_2",reuse=reuse)
    ten = tf.nn.leaky_relu(tenA)
    tenA = tf.layers.conv2d(ten,64,[1,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,name="disc_3",reuse=reuse)
    ten = tf.nn.leaky_relu(tenA+ten)
    tenA = tf.layers.conv2d(tenA,64,[1,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,name="disc_4",reuse=reuse)
    ten = tf.nn.leaky_relu(tenA+ten)
    ten = tf.layers.conv2d(ten,3,[1,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,name="disc_last",reuse=reuse)
    ten=tf.reshape(ten,[ten.shape[0],ten.shape[1],3])
    return ten

def generator(ten:tf.Tensor,reuse:bool):
    """
    #Generator(生成)ネットワーク
    This is Generator-net
    #BNにinstance_normを用います。
    using instance_norm as BN

    Parameters
    ----------
    inp  : tensor
        #入力テンソル
        input tensor
        Shape(N,52,1,513)
        ValueRange[-1.0,1.0]

    reuse: bool (None == False)
        #パラメータ共有のフラグ
        Flag of parameter-sharing
    Returns
    -------
    ten : Tensor
        #出力テンソル
        output tensor
        Shape(N,52,1,513)
    """
    ten = tf.layers.conv2d_transpose(ten, 64, [1, 9],kernel_initializer=tf.initializers.random_normal(stddev=0.002), use_bias=False,
                            reuse=reuse,name="encode_conv_")
    ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="bn_00")
    ten = tf.nn.leaky_relu(ten)
    for i in range(6):
        tenA = tf.layers.conv2d(ten,32,[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=False,reuse=reuse, name="mid_CONV_" + str(i))
        tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="bn0_" + str(i))
        tenA = tf.nn.leaky_relu(tenA)
        tenA = tf.layers.conv2d(tenA,64,[1,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=False,reuse=reuse, name="mid_CONV2_" + str(i))
        tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="bn1_" + str(i))
        ten  = ten+tenA
    ten = tf.layers.conv2d(ten,513,[1,9],kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse,name="decode_conv")
    return tf.tanh(ten)