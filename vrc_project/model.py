
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
            self.c_0 = L.Convolution2D(513, 128, (7, 2), stride=(3, 1), initialW=w_init)
            self.c_1 = L.Convolution2D(128, 128, (7, 1), pad=(3, 0), initialW=w_init)
            self.c_2 = L.Convolution2D(128, 128, (7, 1), pad=(3, 0), initialW=w_init)
            self.c_3 = L.Convolution2D(128, 64, (7, 1), pad=(3, 0), initialW=w_init)
            self.c_4 = L.Convolution2D(64, 64, (7, 1), pad=(3, 0), initialW=w_init)
            self.c_5 = L.Convolution2D(64, 64, (7, 1), pad=(3, 0), initialW=w_init)
            self.c_6 = L.Convolution2D(64, 32, (7, 1), pad=(3, 0), initialW=w_init)
            self.c_7 = L.Convolution2D(32, 32, (7, 1), pad=(3, 0), initialW=w_init)
            self.c_8 = L.Convolution2D(32, 32, (7, 1), pad=(3, 0), initialW=w_init)
            self.d_l = L.Convolution1D(32, 1, 1, initialW=chainer.initializers.Normal(0.02))
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        _y = self.c_0(_x)
        _y = F.leaky_relu(_y)
        _h = self.c_1(_y)
        _h = F.leaky_relu(_h)
        _h = self.c_2(_h)
        _y = F.leaky_relu(_h + _y)
        _y = self.c_3(_y)
        _y = F.leaky_relu(_y)
        _h = self.c_4(_y)
        _h = F.leaky_relu(_h)
        _h = self.c_5(_h)
        _y = F.leaky_relu(_h + _y)
        _y = self.c_6(_y)
        _y = F.leaky_relu(_y)
        _h = self.c_7(_y)
        _h = F.leaky_relu(_h)
        _h = self.c_8(_h)
        _y = F.leaky_relu(_h + _y)
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
            self.c_1_1 = L.Convolution2D(64, 64, (7, 2), stride=(3, 2), initialW=w_init)
            self.c_2_1 = L.Convolution2D(64, 64, (7, 7), pad=(3, 3), initialW=w_init)
            self.c_1_2 = L.Convolution2D(64, 128, (5, 2), stride=(2, 2), initialW=w_init)
            self.c_2_2 = L.Convolution2D(128, 128, (7, 7), pad=(3, 3), initialW=w_init)
            self.c_1_3 = L.Convolution2D(128, 256, (5, 2), stride=(2, 2), initialW=w_init)
            self.c_2_3 = L.Convolution2D(256, 256, (7, 7), pad=(3, 3), initialW=w_init)
            self.c_1_4 = L.Convolution2D(256, 256, (2, 2), stride=(1, 2), initialW=w_init)
            self.c_1_5 = L.Convolution2D(256, 256, (7, 7), pad=(3, 3), initialW=w_init)
            self.d_1_4 = L.Deconvolution2D(256, 256, (2, 2), stride=(1, 2), initialW=w_init, nobias=True)
            self.b_3_4 = L.BatchNormalization(8)
            self.d_1_3 = L.Deconvolution2D(512, 128, (5, 2), stride=(2, 2), initialW=w_init, nobias=True)
            self.b_3_3 = L.BatchNormalization(16)
            self.d_1_2 = L.Deconvolution2D(256, 64, (5, 2), stride=(2, 2), initialW=w_init, nobias=True)
            self.b_3_2 = L.BatchNormalization(32)
            self.d_1_1 = L.Deconvolution2D(128, 64, (7, 2), stride=(3, 2), initialW=w_init, nobias=True)
            self.b_3_1 = L.BatchNormalization(64)
            self.d_0_1 = L.Convolution2D(64, 2, (1, 1), initialW=chainer.initializers.Normal(0.02))
            self.d_0_2 = L.Convolution2D(64, 513, (1, 1), initialW=chainer.initializers.Normal(0.02))
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
        _y = F.relu(_y)
        _y = self.c_1_1(_y)
        _y = F.relu(_y)
        _y = self.c_2_1(_y)
        _y1 = F.relu(_y)
        _y = self.c_1_2(_y1)
        _y = F.relu(_y)
        _y = self.c_2_2(_y)
        _y2 = F.relu(_y)
        _y = self.c_1_3(_y2)
        _y = F.relu(_y)
        _y = self.c_2_3(_y)
        _y3 = F.relu(_y)
        _y = self.c_1_4(_y3)
        _y4 = F.relu(_y)
        _y = self.c_1_5(_y4)
        _y = F.relu(_y)
        _y = self.d_1_4(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = self.b_3_4(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = F.relu(_y)
        _y = F.concat([_y, _y3])
        _y = self.d_1_3(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = self.b_3_3(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = F.relu(_y)
        _y = F.concat([_y, _y2])
        _y = self.d_1_2(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = self.b_3_2(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = F.relu(_y)
        _y = F.concat([_y, _y1])
        _y = self.d_1_1(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = self.b_3_1(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = F.relu(_y)
        _y = self.d_0_1(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = F.relu(_y)
        _y = self.d_0_2(_y)
        _y = F.tanh(_y)
        return _y
