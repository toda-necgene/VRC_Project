
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
            self.c_0 = L.Convolution2D(513, 256, (5, 2), stride=(2, 1), initialW=chainer.initializers.Normal(0.02))
            self.c_1 = L.Convolution1D(256, 64, 5, pad=2, initialW=chainer.initializers.Normal(0.002))
            self.c_2 = L.Convolution1D(64, 64, 5, pad=2, initialW=chainer.initializers.Normal(0.002))
            self.c_3 = L.Convolution1D(64, 64, 5, pad=2, initialW=chainer.initializers.Normal(0.002))
            self.c_4 = L.Convolution1D(64, 64, 5, pad=2, initialW=chainer.initializers.Normal(0.002))
            self.c_5 = L.Convolution1D(64, 64, 5, pad=2, initialW=chainer.initializers.Normal(0.002))
            self.c_6 = L.Convolution1D(64, 64, 5, pad=2, initialW=chainer.initializers.Normal(0.002))
            self.c_l = L.Convolution1D(64, 1, 1, initialW=chainer.initializers.Normal(0.002))
    # @static_graph
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        _y = self.c_0(_x)
        _y = F.leaky_relu(_y)[:, :, :, 0]
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        _h = self.c_2(_y)
        _y = F.leaky_relu(_y + _h)
        _h = self.c_3(_y)
        _y = F.leaky_relu(_y)
        _h = self.c_4(_y)
        _y = F.leaky_relu(_y + _h)
        _h = self.c_5(_y)
        _y = F.leaky_relu(_y + _h)
        _h = self.c_6(_y)
        _y = F.leaky_relu(_y + _h)
        _y = self.c_l(_y)
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
        最終層のみ分散0.002
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.Normal(0.02)
            self.c_0 = L.Convolution2D(513, 32, (1, 2), initialW=w_init, nobias=True)
            self.b_0 = L.BatchNormalization(32)
            self.c_1 = L.Convolution1D(32, 128, 7, stride=3, initialW=w_init, nobias=True)
            self.b_1 = L.BatchNormalization(128)
            self.c_2 = L.Deconvolution1D(128, 32, 7, stride=3, initialW=w_init, nobias=True)
            self.b_2 = L.BatchNormalization(32)
            self.c_3 = L.Convolution1D(32, 128, 7, stride=3, initialW=w_init, nobias=True)
            self.b_3 = L.BatchNormalization(128)
            self.c_4 = L.Deconvolution1D(128, 32, 7, stride=3, initialW=w_init, nobias=True)
            self.b_4 = L.BatchNormalization(32)
            self.c_5 = L.Convolution1D(32, 128, 7, stride=3, initialW=w_init, nobias=True)
            self.b_5 = L.BatchNormalization(128)
            self.c_6 = L.Deconvolution1D(128, 32, 7, stride=3, initialW=w_init, nobias=True)
            self.b_6 = L.BatchNormalization(32)
            self.c_7 = L.Convolution1D(32, 128, 7, stride=3, initialW=w_init, nobias=True)
            self.b_7 = L.BatchNormalization(128)
            self.c_8 = L.Deconvolution1D(128, 32, 7, stride=3, initialW=w_init, nobias=True)
            self.b_8 = L.BatchNormalization(32)
            self.c_n = L.Deconvolution2D(32, 513, (1, 2), initialW=chainer.initializers.Normal(0.002))
    # @static_graph
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                変換前スペクトラム包絡
                shape: [N,513,104]
                range: [-1.0,1.0]
            Returns
            -------
            _y: ndarray
                変換後スペクトラム包絡
                shape: [N,513,104]
                range: [-1.0,1.0]
        """
        # Expand second-dimention
        _y = self.c_0(_x)[:, :, :, 0]
        _y = self.b_0(_y)
        _y = F.leaky_relu(_y)
        # ResModule
        _h = self.c_1(_y)
        _h = self.b_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_2(_h)
        _h = self.b_2(_h)
        _y = F.leaky_relu(_h + _y)
        _h = self.c_3(_y)
        _h = self.b_3(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_4(_h)
        _h = self.b_4(_h)
        _y = F.leaky_relu(_h + _y)
        _h = self.c_5(_y)
        _h = self.b_5(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_6(_h)
        _h = self.b_6(_h)
        _y = F.leaky_relu(_h + _y)
        _h = self.c_7(_y)
        _h = self.b_7(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_8(_h)
        _h = self.b_8(_h)
        _y = F.leaky_relu(_h + _y)
        # Squeeze second-dimention
        _y = F.expand_dims(_y, 3)
        _y = self.c_n(_y)
        _y = F.tanh(_y)
        return _y
