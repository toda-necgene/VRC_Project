
"""
製作者:TODA
モデルの定義
"""
import random
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
        重みの初期化はHe正規乱数
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            self.c_0 = L.Convolution1D(513, 256, 9, initialW=w_init, pad=4).to_gpu()
            self.a_1 = L.Convolution1D(256, 128, 1, initialW=w_init).to_gpu()
            self.t_1 = L.Convolution1D(256, 128, 1, initialW=w_init).to_gpu()
            self.c_1 = L.Convolution1D(256, 128, 7, initialW=w_init, pad=3).to_gpu()
            self.b_1 = L.GroupNormalization(1).to_gpu()
            self.a_2 = L.Convolution1D(128, 64, 1, initialW=w_init).to_gpu()
            self.t_2 = L.Convolution1D(128, 64, 1, initialW=w_init).to_gpu()
            self.c_2 = L.Convolution1D(128, 64, 7, initialW=w_init, pad=3).to_gpu()
            self.b_2 = L.GroupNormalization(1).to_gpu()
            self.a_3 = L.Convolution1D(64, 32, 1, initialW=w_init).to_gpu()
            self.t_3 = L.Convolution1D(64, 32, 1, initialW=w_init).to_gpu()
            self.c_3 = L.Convolution1D(64, 32, 7, initialW=w_init, pad=3).to_gpu()
            self.b_3 = L.GroupNormalization(1).to_gpu()
            self.a_4 = L.Convolution1D(32, 16, 1, initialW=w_init).to_gpu()
            self.t_4 = L.Convolution1D(32, 16, 1, initialW=w_init).to_gpu()
            self.c_4 = L.Convolution1D(32, 16, 7, initialW=w_init, pad=3).to_gpu()
            self.b_4 = L.GroupNormalization(1).to_gpu()
            w_init = chainer.initializers.Normal(0.002)
            self.c_l = L.Convolution1D(16, 3, 1, initialW=w_init).to_gpu()
    def weight_resampler(self):
        """
        再初期化関数
        重みの平均と分散は維持する。
        """
        n = random.randint(0, 4)
        if n == 0:
            m = self.c_1.W.array.mean()
            v = self.c_1.W.array.var()
            chainer.initializers.Normal(v)(self.c_1.W.array)
            self.c_1.W.array += m
        elif n == 1:
            m = self.c_2.W.array.mean()
            v = self.c_2.W.array.var()
            chainer.initializers.Normal(v)(self.c_2.W.array)
            self.c_2.W.array += m
        elif n == 2:
            m = self.c_3.W.array.mean()
            v = self.c_3.W.array.var()
            chainer.initializers.Normal(v)(self.c_3.W.array)
            self.c_3.W.array += m
        elif n == 3:
            m = self.c_4.W.array.mean()
            v = self.c_4.W.array.var()
            chainer.initializers.Normal(v)(self.c_4.W.array)
            self.c_4.W.array += m
    # @static_graph
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        # 次元削減
        _y = self.c_0(_x)
        _y = F.leaky_relu(_y)
        _y = self.b_1(_y)
        _f = F.softmax(self.a_1(_y)*F.transpose(self.t_1(_y), (0, 2, 1)).reshape(-1, 128, 52))
        _h = self.c_1(_y)
        _y = _h * _f + _y[:, :128, :]
        _y = self.b_2(_y)
        _f = F.softmax(self.a_2(_y)*F.transpose(self.t_2(_y), (0, 2, 1)).reshape(-1, 64, 52))
        _h = self.c_2(_y)
        _y = _h * _f + _y[:, :64, :]
        _y = self.b_3(_y)
        _f = F.softmax(self.a_3(_y)*F.transpose(self.t_3(_y), (0, 2, 1)).reshape(-1, 32, 52))
        _h = self.c_3(_y)
        _y = _h * _f + _y[:, :32, :]
        _y = self.b_4(_y)
        _f = F.softmax(self.a_4(_y)*F.transpose(self.t_4(_y), (0, 2, 1)).reshape(-1, 16, 52))
        _h = self.c_4(_y)
        _y = _h * _f + _y[:, :16, :]
        # 出力変換
        _y = self.c_l(_y)
        return F.clip(_y, 0.0, 1.0)
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
        最終層のみ分散0.004
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            self.c_0 = L.Deconvolution2D(513, 64, ksize=(1, 9), initialW=w_init).to_gpu()
            self.b_1_1 = L.BatchNormalization(64).to_gpu()
            self.c_1_1 = L.Convolution2D(64, 32, (9, 1), initialW=w_init, pad=(4, 0)).to_gpu()
            self.b_1_2 = L.BatchNormalization(32).to_gpu()
            self.c_1_2 = L.Convolution2D(32, 64, (9, 1), initialW=w_init, pad=(4, 0)).to_gpu()
            self.b_2_1 = L.BatchNormalization(64).to_gpu()
            self.c_2_1 = L.Convolution2D(64, 32, (9, 1), initialW=w_init, pad=(4, 0)).to_gpu()
            self.b_2_2 = L.BatchNormalization(32).to_gpu()
            self.c_2_2 = L.Convolution2D(32, 64, (9, 1), initialW=w_init, pad=(4, 0)).to_gpu()
            self.b_3_1 = L.BatchNormalization(64).to_gpu()
            self.c_3_1 = L.Convolution2D(64, 32, (9, 1), initialW=w_init, pad=(4, 0)).to_gpu()
            self.b_3_2 = L.BatchNormalization(32).to_gpu()
            self.c_3_2 = L.Convolution2D(32, 64, (9, 1), initialW=w_init, pad=(4, 0)).to_gpu()
            self.b_n = L.BatchNormalization(64).to_gpu()
            w_init = chainer.initializers.Normal(0.004)
            self.c_n = L.Convolution2D(64, 513, (1, 9), initialW=w_init).to_gpu()
    # @static_graph
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                変換前スペクトラム包絡
                shape: [N,513,52]
                rnage: [-1.0,1.0]
            Returns
            -------
            _y: ndarray
                変換後スペクトラム包絡
                shape: [N,513,52]
                rnage: [-1.0,1.0]
        """
        _y = F.expand_dims(_x, axis=3)
        # Expand second-dimention
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        # ResModule
        _h = self.b_1_1(_y)
        _h = self.c_1_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.b_1_2(_h)
        _h = self.c_1_2(_h)
        _y = F.leaky_relu(_y + _h)
        _h = self.b_2_1(_y)
        _h = self.c_2_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.b_2_2(_h)
        _h = self.c_2_2(_h)
        _y = F.leaky_relu(_y + _h)
        _h = self.b_3_1(_y)
        _h = self.c_3_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.b_3_2(_h)
        _h = self.c_3_2(_h)
        _y = F.leaky_relu(_y + _h)
        # Squeeze second-dimention
        _y = self.b_n(_y)
        _y = self.c_n(_y)
        _y = _y[:, :, :, 0]
        _y = F.tanh(_y)
        return _y
