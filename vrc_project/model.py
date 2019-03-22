
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
            self.c_0 = L.Convolution2D(513, 64, (7, 1), stride=(3, 1), initialW=w_init)
            self.d_0 = L.Convolution2D(1, 64, (1, 1), initialW=w_init)
            self.c_1 = L.Convolution2D(64, 64, (6, 8), stride=(3, 4), initialW=w_init)
            self.m_1 = L.Convolution2D(64, 64, (7, 5), pad=(3, 2), initialW=w_init)
            self.d_1 = L.Deconvolution2D(64, 64, (6, 8), stride=(3, 4), initialW=w_init)
            self.c_2 = L.Convolution2D(64, 64, (6, 8), stride=(3, 4), initialW=w_init)
            self.m_2 = L.Convolution2D(64, 64, (7, 5), pad=(3, 2), initialW=w_init)
            self.d_2 = L.Deconvolution2D(64, 64, (6, 8), stride=(3, 4), initialW=w_init)
            self.c_l = L.Convolution2D(64, 1, (1, 1), initialW=chainer.initializers.Normal(0.02))
            self.d_l = L.Convolution1D(64, 1, 1, initialW=chainer.initializers.Normal(0.02))
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        _y = self.c_0(_x[:, :, :, :1])
        _y = F.leaky_relu(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = self.d_0(_y)
        _y = F.leaky_relu(_y)
        _h = self.c_1(_y)
        _h = F.leaky_relu(_h)
        _h = self.m_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.d_1(_h)
        _y = F.leaky_relu(_y + _h)
        _h = self.c_2(_y)
        _h = F.leaky_relu(_h)
        _h = self.m_2(_h)
        _h = F.leaky_relu(_h)
        _h = self.d_2(_h)
        _y = F.leaky_relu(_y + _h)
        _y = self.c_l(_y)
        _y = F.leaky_relu(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))[:, :, :, 0]
        _y = self.d_l(_y)
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
            self.c_1_1 = L.Convolution2D(32, 64, (7, 8), stride=(6, 8), initialW=w_init, nobias=True)
            self.b_1_1 = L.BatchNormalization(64)
            self.c_2_1 = L.Convolution2D(64, 64, (7, 7), pad=(3, 3), initialW=w_init)
            self.b_2_1 = L.BatchNormalization(64)
            self.d_1_1 = L.Deconvolution2D(64, 32, (7, 8), stride=(6, 8), initialW=w_init, nobias=True)
            self.b_3_1 = L.BatchNormalization(32)
            self.c_1_2 = L.Convolution2D(32, 64, (7, 8), stride=(6, 8), initialW=w_init, nobias=True)
            self.b_1_2 = L.BatchNormalization(64)
            self.c_2_2 = L.Convolution2D(64, 64, (7, 7), pad=(3, 3), initialW=w_init)
            self.b_2_2 = L.BatchNormalization(64)
            self.d_1_2 = L.Deconvolution2D(64, 32, (7, 8), stride=(6, 8), initialW=w_init, nobias=True)
            self.b_3_2 = L.BatchNormalization(32)
            self.c_1_3 = L.Convolution2D(32, 64, (7, 8), stride=(6, 8), initialW=w_init, nobias=True)
            self.b_1_3 = L.BatchNormalization(64)
            self.c_2_3 = L.Convolution2D(64, 64, (7, 7), pad=(3, 3), initialW=w_init)
            self.b_2_3 = L.BatchNormalization(64)
            self.d_1_3 = L.Deconvolution2D(64, 32, (7, 8), stride=(6, 8), initialW=w_init, nobias=True)
            self.b_3_3 = L.BatchNormalization(32)
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                変換前特徴量
                shape: [N,16,104,15]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,16,104,15]
        """
        _h = self.c_1_1(_x)
        _h = self.b_1_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_2_1(_h)
        _h = self.b_2_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.d_1_1(_h)
        _h = self.b_3_1(_h)
        _y = F.leaky_relu(_h + _x)
        _h = self.c_1_2(_y)
        _h = self.b_1_2(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_2_2(_h)
        _h = self.b_2_2(_h)
        _h = F.leaky_relu(_h)
        _h = self.d_1_2(_h)
        _h = self.b_3_2(_h)
        _y = F.leaky_relu(_h + _y)
        _h = self.c_1_3(_y)
        _h = self.b_1_3(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_2_3(_h)
        _h = self.b_2_3(_h)
        _h = F.leaky_relu(_h)
        _h = self.d_1_3(_h)
        _h = self.b_3_3(_h)
        _y = F.leaky_relu(_h + _y)
        return _y
class Encoder(chainer.Chain):
    """
        特徴抽出ネットワーク
        スペクトラム包絡/非周期性指標から特徴量
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みの初期化はHeNormal
        最終層のみ分散0.002
        """
        super(Encoder, self).__init__()
        with self.init_scope():
            self.c_0 = L.Convolution2D(2, 32, (1, 9), stride=(1, 8), initialW=chainer.initializers.Normal(0.02))
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                スペクトラム包絡/非周期性指標
                shape: [N,513,104,2]
                range: [-1.0,1.0]
            Returns
            -------
            _y: ndarray
                特徴量
                shape: [N,16,104,64]
        """
        _y = F.transpose(_x, (0, 3, 2, 1))
        _y = self.c_0(_y)
        return _y

class Decoder(chainer.Chain):
    """
        復元ネットワーク
        特徴量からスペクトラム包絡/非周期性指標
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みの初期化はHeNormal
        最終層のみ分散0.002
        """
        super(Decoder, self).__init__()
        with self.init_scope():
            self.c_0 = L.Convolution2D(32, 2, (1, 1), initialW=chainer.initializers.Normal(0.02))
            self.c_n = L.Convolution2D(64, 513, (1, 1), initialW=chainer.initializers.Normal(0.02))
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                特徴量
                shape: [N,64,104,15]
            Returns
            -------
            _y: ndarray
                変換後スペクトラム包絡/非周期性指標
                shape: [N,513,104,2]
                range: [-1.0,1.0]
        """
        _y = self.c_0(_x)
        _y = F.relu(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = self.c_n(_y)
        _y = F.tanh(_y)
        return _y
