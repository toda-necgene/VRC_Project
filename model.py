
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
            self.c_0 = L.Convolution1D(513, 256, 3, initialW=w_init, pad=1).to_gpu()
            self.a_1 = L.Convolution1D(256, 128, 11, initialW=w_init, pad=5).to_gpu()
            self.c_1 = L.Convolution1D(256, 128, 11, initialW=w_init, pad=5).to_gpu()
            self.b_1 = L.GroupNormalization(4).to_gpu()
            self.a_2 = L.Convolution1D(128, 64, 11, initialW=w_init, pad=5).to_gpu()
            self.c_2 = L.Convolution1D(128, 64, 11, initialW=w_init, pad=5).to_gpu()
            self.b_2 = L.GroupNormalization(4).to_gpu()
            self.a_3 = L.Convolution1D(64, 32, 11, initialW=w_init, pad=5).to_gpu()
            self.c_3 = L.Convolution1D(64, 32, 11, initialW=w_init, pad=5).to_gpu()
            self.b_3 = L.GroupNormalization(4).to_gpu()
            self.a_4 = L.Convolution1D(32, 16, 11, initialW=w_init, pad=5).to_gpu()
            self.c_4 = L.Convolution1D(32, 16, 11, initialW=w_init, pad=5).to_gpu()
            self.b_4 = L.GroupNormalization(4).to_gpu()
            self.c_l = L.Convolution1D(16, 3, 1, initialW=chainer.initializers.Normal(0.002)).to_gpu()
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
        _f = F.sigmoid(self.a_1(_y))
        _h = self.c_1(_y)
        _y = _h * _f
        _y = F.average_pooling_1d(_y, 7, 3)
        _y = self.b_2(_y)
        _f = F.sigmoid(self.a_2(_y))
        _h = self.c_2(_y)
        _y = _h * _f
        _y = self.b_3(_y)
        _f = F.sigmoid(self.a_3(_y))
        _h = self.c_3(_y)
        _y = _h * _f
        _y = self.b_4(_y)
        _f = F.sigmoid(self.a_4(_y))
        _h = self.c_4(_y)
        _y = _h * _f
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
        最終層のみ分散0.002
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            self.c_0 = L.Convolution1D(513, 512, 1, initialW=w_init).to_gpu()
            self.b_1_1 = L.BatchNormalization(32).to_gpu()
            self.c_1_1 = L.Convolution2D(32, 64, (8, 8), stride=(4, 4), initialW=w_init).to_gpu()
            self.b_1_2 = L.BatchNormalization(64).to_gpu()
            self.c_1_2 = L.Deconvolution2D(64, 32, (8, 8), stride=(4, 4), initialW=w_init).to_gpu()
            self.b_2_1 = L.BatchNormalization(32).to_gpu()
            self.c_2_1 = L.Convolution2D(32, 64, (8, 8), stride=(4, 4), initialW=w_init).to_gpu()
            self.b_2_2 = L.BatchNormalization(64).to_gpu()
            self.c_2_2 = L.Deconvolution2D(64, 32, (8, 8), stride=(4, 4), initialW=w_init).to_gpu()
            self.b_3_1 = L.BatchNormalization(32).to_gpu()
            self.c_3_1 = L.Convolution2D(32, 64, (7, 4), stride=(3, 4), initialW=w_init).to_gpu()
            self.b_3_2 = L.BatchNormalization(64).to_gpu()
            self.c_3_2 = L.Deconvolution2D(64, 32, (7, 4), stride=(3, 4), initialW=w_init).to_gpu()
            self.b_4_1 = L.BatchNormalization(32).to_gpu()
            self.c_4_1 = L.Convolution2D(32, 64, (4, 4), stride=(3, 2), initialW=w_init).to_gpu()
            self.b_4_2 = L.BatchNormalization(64).to_gpu()
            self.c_4_2 = L.Deconvolution2D(64, 32, (4, 4), stride=(3, 2), initialW=w_init).to_gpu()
            self.b_5_1 = L.BatchNormalization(32).to_gpu()
            self.c_5_1 = L.Convolution2D(32, 64, (4, 1), stride=(2, 1), initialW=w_init).to_gpu()
            self.b_5_2 = L.BatchNormalization(64).to_gpu()
            self.c_5_2 = L.Deconvolution2D(64, 32, (4, 1), stride=(2, 1), initialW=w_init).to_gpu()
            self.b_6_1 = L.BatchNormalization(32).to_gpu()
            self.c_6_1 = L.Convolution2D(32, 64, (4, 1), stride=(2, 1), initialW=w_init).to_gpu()
            self.b_6_2 = L.BatchNormalization(64).to_gpu()
            self.c_6_2 = L.Deconvolution2D(64, 32, (4, 1), stride=(2, 1), initialW=w_init).to_gpu()
            self.b_7_1 = L.BatchNormalization(32).to_gpu()
            self.c_7_1 = L.Convolution2D(32, 64, (4, 1), stride=(2, 1), initialW=w_init).to_gpu()
            self.b_7_2 = L.BatchNormalization(64).to_gpu()
            self.c_7_2 = L.Deconvolution2D(64, 32, (4, 1), stride=(2, 1), initialW=w_init).to_gpu()
            self.b_8_1 = L.BatchNormalization(32).to_gpu()
            self.c_8_1 = L.Convolution2D(32, 64, (4, 1), stride=(2, 1), initialW=w_init).to_gpu()
            self.b_8_2 = L.BatchNormalization(64).to_gpu()
            self.c_8_2 = L.Deconvolution2D(64, 32, (4, 1), stride=(2, 1), initialW=w_init).to_gpu()
            self.c_n = L.Convolution1D(512, 513, 1, initialW=chainer.initializers.Normal(0.002)).to_gpu()
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
                range: [-1.0,1.0]
            Returns
            -------
            _y: ndarray
                変換後スペクトラム包絡
                shape: [N,513,52]
                range: [-1.0,1.0]
        """
        # Expand second-dimention
        _y = self.c_0(_x)
        _y = F.reshape(_y, (-1, 32, _y.shape[2], 16))
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
        _h = self.b_4_1(_y)
        _h = self.c_4_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.b_4_2(_h)
        _h = self.c_4_2(_h)
        _y = F.leaky_relu(_y + _h)
        _h = self.b_5_1(_y)
        _h = self.c_5_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.b_5_2(_h)
        _h = self.c_5_2(_h)
        _y = F.leaky_relu(_y + _h)
        _h = self.b_6_1(_y)
        _h = self.c_6_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.b_6_2(_h)
        _h = self.c_6_2(_h)
        _y = F.leaky_relu(_y + _h)
        _h = self.b_7_1(_y)
        _h = self.c_7_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.b_7_2(_h)
        _h = self.c_7_2(_h)
        _y = F.leaky_relu(_y + _h)
        _h = self.b_8_1(_y)
        _h = self.c_8_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.b_8_2(_h)
        _h = self.c_8_2(_h)
        _y = F.leaky_relu(_y + _h)
        _y = F.reshape(_y, (-1, 512, _y.shape[2]))
        # Squeeze second-dimention
        _y = self.c_n(_y)
        _y = F.tanh(_y)
        return _y
