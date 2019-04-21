
"""
製作者:TODA
モデルの定義
"""
import chainer
import chainer.links as L
import chainer.functions as F

class Discriminator(chainer.Chain):
    """
        識別側ネットワーク
        周波数軸はチャンネルとして扱います。
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みはHeの初期化
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            self.c_0 = L.Convolution2D(513, 256, (3, 2), stride=(2, 1), initialW=w_init)
            self.c_1 = L.Convolution2D(256, 128, (3, 1), stride=(2, 1), initialW=w_init)
            self.c_2 = L.Convolution2D(128, 64, (7, 1), stride=(2, 1), initialW=w_init)
            self.d_l = L.Convolution1D(64, 1, 1, initialW=chainer.initializers.Normal(0.02))
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        _y = self.c_0(_x)
        _y = F.leaky_relu(_y)
        _h = self.c_1(_y)
        _y = F.leaky_relu(_h)
        _h = self.c_2(_y)
        _y = F.leaky_relu(_h)
        _y = self.d_l(_y[:, :, :, 0])
        return _y
class Generator(chainer.Chain):
    """
        生成側ネットワーク
        新たな軸を追加して特徴次元数を保持するようにしている
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みの初期化はHeNormal
        最終層のみ分散0.02
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            self.c_0 = L.Convolution2D(2, 64, (1, 9), stride=(1, 8), initialW=chainer.initializers.Normal(0.02))
            self.d_1 = L.Convolution2D(64, 128, (7, 2), stride=(3, 2), initialW=w_init)
            self.d_2 = L.Convolution2D(128, 256, (5, 4), stride=(2, 2), initialW=w_init)
            self.r_1 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_2 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_3 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_4 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_5 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_6 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_7 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_8 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_9 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_10 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_11 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_12 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.u_2 = L.Deconvolution2D(256, 128, (5, 4), stride=(2, 2), initialW=w_init, nobias=True)
            self.b_9 = L.BatchNormalization(128)
            self.u_1 = L.Deconvolution2D(128, 64, (7, 2), stride=(3, 2), initialW=w_init, nobias=True)
            self.b_10 = L.BatchNormalization(64)
            self.d_0_1 = L.Deconvolution2D(64, 2, (1, 9), stride=(1, 8), initialW=chainer.initializers.Normal(0.02))
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                変換前特徴量
                shape: [N,64,104,64]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,64,104,64]
        """
        _y = F.transpose(_x, (0, 3, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_2(_y)
        _y = F.leaky_relu(_y)
        _h = self.r_1(_y)
        _h = F.leaky_relu(_h)
        _h = self.r_2(_h)
        _y = _h + _y
        _h = self.r_3(_y)
        _h = F.leaky_relu(_h)
        _h = self.r_4(_h)
        _y = _h + _y
        _h = self.r_5(_y)
        _h = F.leaky_relu(_h)
        _h = self.r_6(_h)
        _y = _h + _y
        _h = self.r_7(_y)
        _h = F.leaky_relu(_h)
        _h = self.r_8(_h)
        _y = _h + _y
        _h = self.r_9(_y)
        _h = F.leaky_relu(_h)
        _h = self.r_10(_h)
        _y = _h + _y
        _h = self.r_11(_y)
        _h = F.leaky_relu(_h)
        _h = self.r_12(_h)
        _y = _h + _y
        _y = self.u_2(_y)
        _y = self.b_9(_y)
        _y = F.leaky_relu(_y)
        _y = self.u_1(_y)
        _y = self.b_10(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_0_1(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = F.tanh(_y)
        return _y
