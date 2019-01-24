
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
        Attention構造にSigmoidを消してみた形
        より良いモデルがあることも否定できない
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みの初期化は0.02分散
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            self.c_d_0 = L.Convolution1D(513, 256, 1, nobias=True, initialW=w_init).to_gpu()
            self.c_d_1 = L.Convolution1D(256, 64, 1, nobias=True, initialW=w_init).to_gpu()
            self.b_d_0 = L.GroupNormalization(256, 256)
            self.b_d_1 = L.GroupNormalization(64, 64)
            self.a_0 = AttentionBlock()
            self.a_1 = AttentionBlock()
            self.a_2 = AttentionBlock()
            self.a_3 = AttentionBlock()
            self.c_l = L.Convolution1D(64, 3, 1, pad=0, initialW=w_init).to_gpu()
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        # 次元削減
        _y = self.c_d_0(_x)
        _y = self.b_d_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_d_1(_y)
        _y = self.b_d_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.a_0(_y)
        _y = self.a_1(_y)
        _y = self.a_2(_y)
        _y = self.a_3(_y)
        # 出力変換
        _y = self.c_l(_y)
        return F.softmax(_y)
class Generator(chainer.Chain):
    """
        生成側ネットワーク
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みの初期化は標準分布で分散0.02
        """
        super(Generator, self).__init__()
        with self.init_scope():
            self.res_block = 4
            w_init = chainer.initializers.Normal(0.002)
            self.c_i = list()
            for _ in range(self.res_block):
                self.c_i.append(ResBlock().to_gpu())
            self.c_n = L.Convolution2D(513, 513, (1, 1), initialW=w_init).to_gpu()
    def forward(self, _x):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray
                変換前スペクトラム包絡
                dtype: float32
                shape: [N,513,52]
                rnage: [-1.0,1.0]
            Returns
            -------
            _y: ndarray
                変換後スペクトラム包絡
                dtype: float32
                shape: [N,513,52]
                rnage: [-1.0,1.0]
        """
        _y = F.reshape(_x, (-1, 513, 52, 1))
        for _cbrcba in self.c_i:
            _y = _cbrcba(_y)
        _y = self.c_n(_y)
        _y = F.tanh(_y)
        _y = F.reshape(_y, (-1, 513, 52))
        return _y

class ResBlock(chainer.Chain):
    """
        生成側ネットワークにおける残差ブロックの中身
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みの初期化は標準分布で分散0.002
        バイアスは使わない
        """
        super(ResBlock, self).__init__()
        w_init = chainer.initializers.Normal(0.002)
        with self.init_scope():
            self.c_0 = L.Convolution2D(513, 128, (3, 1), initialW=w_init, pad=(1, 0), nobias=True)
            self.b_0 = L.GroupNormalization(128, 128)
            self.c_1 = L.Convolution2D(128, 513, (1, 1), initialW=w_init, nobias=True)
            self.b_1 = L.GroupNormalization(513, 513)
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        _h = self.c_0(_x)
        _h = self.b_0(_h)
        _h = F.leaky_relu(_h)
        _h = self.c_1(_h)
        _h = self.b_1(_h)
        _y = _h + _x
        return _y
class AttentionBlock(chainer.Chain):
    """
        識別側ネットワークにおける注意ブロックの中身
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みの初期化は標準分布で分散0.002
        バイアスは使わない
        """
        super(AttentionBlock, self).__init__()
        w_init = chainer.initializers.Normal(0.002)
        with self.init_scope():
            self.c_0 = L.Convolution1D(64, 64, 3, initialW=w_init, pad=1, nobias=True)
            self.b_0 = L.GroupNormalization(64, 64)
            self.c_1 = L.Convolution1D(64, 64, 5, initialW=w_init, pad=2, nobias=True)
            self.b_1 = L.GroupNormalization(64, 64)
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        _a = self.c_0(_x)
        _a = self.b_0(_a)
        _a = F.clip(_a, 0.0, 1.0)
        _h = _x * _a
        _h = self.c_1(_h)
        _h = self.b_1(_h)
        _y = _h + _x
        return _y
