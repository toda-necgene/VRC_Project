
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
            self.c_0 = L.Convolution2D(513, 64, (7, 2), stride=(3, 1), initialW=w_init, nobias=True)
            self.b_0 = L.GroupNormalization(4)
            self.c_1 = L.Convolution1D(64, 64, 11, pad=5, initialW=w_init, nobias=True)
            self.b_1 = L.GroupNormalization(4)
            self.c_2 = L.Convolution1D(64, 64, 11, pad=5, initialW=w_init, nobias=True)
            self.b_2 = L.GroupNormalization(4)
            self.c_3 = L.Convolution1D(64, 64, 11, pad=5, initialW=w_init, nobias=True)
            self.b_3 = L.GroupNormalization(4)
            self.c_4 = L.Convolution1D(64, 64, 11, pad=5, initialW=w_init, nobias=True)
            self.b_4 = L.GroupNormalization(4)
            self.c_l = L.Convolution1D(64, 1, 1, initialW=chainer.initializers.Normal(0.02))
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        _y = self.c_0(_x)
        _y = self.b_0(_y)
        _y = F.leaky_relu(_y)[:, :, :, 0]
        _h = self.c_1(_y)
        _h = self.b_1(_h)
        _y = F.leaky_relu(_y+ _h)
        _h = self.c_2(_y)
        _h = self.b_2(_h)
        _y = F.leaky_relu(_y+ _h)
        _h = self.c_3(_y)
        _h = self.b_3(_h)
        _y = F.leaky_relu(_y+ _h)
        _h = self.c_4(_y)
        _h = self.b_4(_h)
        _y = F.leaky_relu(_y+ _h)
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
            w_init = chainer.initializers.Normal(0.002)
            self.c_1 = L.Convolution2D(32, 64, (7, 8), stride=(3, 4), initialW=w_init, nobias=True)
            self.b_1 = L.BatchNormalization(64)
            self.c_2 = L.Deconvolution2D(64, 32, (7, 8), stride=(3, 4), initialW=w_init, nobias=True)
            self.b_2 = L.BatchNormalization(32)
            self.c_3 = L.Convolution2D(32, 64, (7, 8), stride=(3, 4), initialW=w_init, nobias=True)
            self.b_3 = L.BatchNormalization(64)
            self.c_4 = L.Deconvolution2D(64, 32, (7, 8), stride=(3, 4), initialW=w_init, nobias=True)
            self.b_4 = L.BatchNormalization(32)
    # @static_graph
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                変換前特徴量
                shape: [N,64,104,64]
                range: [-1.0,1.0]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,64,104,64]
                range: [-1.0,1.0]
        """
        _h = self.c_1(_x)
        _h = self.b_1(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_2(_h)
        _h = self.b_2(_h)
        _y = F.leaky_relu(_h + _x)
        _h = self.c_3(_y)
        _h = self.b_3(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_4(_h)
        _h = self.b_4(_h)
        _y = F.sigmoid(_h)
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
    # @static_graph
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
                shape: [N,64,104,64]
                range: [-1.0,1.0]
        """
        _y = F.transpose(_x, (0, 3, 2, 1))
        _y = self.c_0(_y)
        _y = F.sigmoid(_y)
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
            self.c_0 = L.Deconvolution2D(32, 2, (1, 1), initialW=chainer.initializers.Normal(0.004))
            self.c_n = L.Deconvolution2D(64, 513, (1, 1), initialW=chainer.initializers.Normal(0.02))
    # @static_graph
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                特徴量
                shape: [N,64,104,64]
                range: [-1.0,1.0]
            Returns
            -------
            _y: ndarray
                変換後スペクトラム包絡/非周期性指標
                shape: [N,513,104,2]
                range: [-1.0,1.0]
        """
        _y = self.c_0(_x)
        _y = F.leaky_relu(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        _y = self.c_n(_y)
        _y = F.tanh(_y)
        return _y
