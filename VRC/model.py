"""
製作者：TODA

モデルの定義
"""
import tensorflow as tf

def discriminator(inp: tf.Tensor, reuse: bool):
    """
    Discriminator(識別者)ネットワーク
    周波数軸はチャンネルとして扱います。
    構造はBN抜きのResnetのようなもの
    より良いモデルがあることも否定できない
    Parameters
    ----------
    inp  : Tensor
        入力テンソル
        Shape(N,52,1,513)
    reuse: bool (None == False)
        パラメータ共有のフラグ
    Returns
    -------
    ten : Tensor
        出力テンソル
        Shape(N,13,3)
        soft-maxはしていない
    """
    ten = tf.layers.conv2d(inp, 256, [1, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=True, name="disc_1", reuse=reuse)
    ten = tf.nn.leaky_relu(ten)
    ten_a = tf.layers.conv2d(ten, 64, [1, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=True, name="disc_2", reuse=reuse)
    ten = tf.nn.leaky_relu(ten_a)
    ten_a = tf.layers.conv2d(ten, 32, [4, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=True, name="disc_3", reuse=reuse)
    ten_a = tf.nn.leaky_relu(ten_a)
    ten_a = tf.layers.conv2d(ten_a, 64, [1, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=True, name="disc_4", reuse=reuse)
    ten = tf.clip_by_value(ten_a, 0.0, 1.0)*ten
    ten_a = tf.layers.conv2d(ten, 32, [4, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=True, name="disc_5", reuse=reuse)
    ten_a = tf.nn.leaky_relu(ten_a)
    ten_a = tf.layers.conv2d(ten_a, 64, [1, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=True, name="disc_6", reuse=reuse)
    ten = tf.clip_by_value(ten_a, 0.0, 1.0)*ten
    ten = tf.layers.conv2d(ten, 3, [1, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=True, name="disc_last", reuse=reuse)
    ten = tf.reshape(ten, [ten.shape[0], ten.shape[1], 3])
    return ten

def generator(ten: tf.Tensor, reuse: bool):
    """
    Generator(生成)ネットワーク
    BNにinstance_normを用います。
    Parameters
    ----------
    inp  : tensor
        入力テンソル
        Shape(N,52,1,513)
        ValueRange[-1.0,1.0]

    reuse: bool (None == False)
        パラメータ共有のフラグ
    Returns
    -------
    ten : Tensor
        出力テンソル
        Shape(N,52,1,513)
    """
    ten = tf.layers.conv2d_transpose(ten, 64, [1, 9], kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=True, reuse=reuse, name="encode_conv_")
    ten = tf.nn.leaky_relu(ten)
    for i in range(8):
        ten_a = tf.layers.conv2d(ten, 16, [4, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.002), use_bias=False, reuse=reuse, name="mid_CONV_" + str(i))
        ten_a = tf.contrib.layers.instance_norm(ten_a, reuse=reuse, scope="bn0_" + str(i))
        ten_a = tf.nn.leaky_relu(ten_a)
        ten_a = tf.layers.conv2d(ten_a, 64, [1, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.002), use_bias=False, reuse=reuse, name="mid_CONV2_" + str(i))
        ten_a = tf.contrib.layers.instance_norm(ten_a, reuse=reuse, scope="bn1_" + str(i))
        ten = ten_a+ten
    ten = tf.layers.conv2d(ten, 513, [1, 9], kernel_initializer=tf.initializers.random_normal(stddev=0.002), use_bias=True, reuse=reuse, name="decode_conv")
    return tf.tanh(ten)
