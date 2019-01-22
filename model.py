
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
        重みの初期化は標準分布で分散0.02
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.Normal(0.02)
            self.c_0 = L.Convolution1D(513, 256, 1, initialW=w_init).to_gpu()
            self.c_1 = L.Convolution1D(256, 64, 1, initialW=w_init).to_gpu()
            self.c_2_0 = L.Convolution1D(64, 16, 5, pad=(2), initialW=w_init).to_gpu()
            self.c_2_1 = L.Convolution1D(16, 64, 1, initialW=w_init).to_gpu()
            self.c_3_0 = L.Convolution1D(64, 16, 5, pad=(2), initialW=w_init).to_gpu()
            self.c_3_1 = L.Convolution1D(16, 64, 1, initialW=w_init).to_gpu()
            self.c_4_0 = L.Convolution1D(64, 16, 5, pad=(2), initialW=w_init).to_gpu()
            self.c_4_1 = L.Convolution1D(16, 64, 1, initialW=w_init).to_gpu()
            self.c_5 = L.Convolution1D(64, 3, 1, initialW=w_init).to_gpu()
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        """
        # 次元削減
        _y = self.c_0(_x)
        _y = F.leaky_relu(_y)
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        # クリッピング自己注意
        _h = self.c_2_0(_y)
        _h = F.leaky_relu(_h)
        _h = self.c_2_1(_h)
        _h = F.clip(_h, -1.0, 1.0)
        _y = _y * _h
        _h = self.c_3_0(_y)
        _h = F.leaky_relu(_h)
        _h = self.c_3_1(_h)
        _h = F.clip(_h, -1.0, 1.0)
        _y = _y * _h
        _h = self.c_4_0(_y)
        _h = F.leaky_relu(_h)
        _h = self.c_4_1(_h)
        _h = F.clip(_h, -1.0, 1.0)
        _y = _y * _h
        # 出力
        _y = self.c_5(_y)
        return _y
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
            self.res_block = 6
            w_init = chainer.initializers.Normal(0.02)
            self.c_0 = L.Deconvolution2D(513, 64, (1, 9), initialW=w_init).to_gpu()
            self.c_i = list()
            for _ in range(self.res_block):
                self.c_i.append(ResBlock().to_gpu())
            self.c_n = L.Convolution2D(64, 513, (1, 9), initialW=w_init).to_gpu()
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
        _y = self.c_0(_y)
        _y = F.relu(_y)
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
        重みの初期化は標準分布で分散0.02
        バイアスは使わない
        """
        super(ResBlock, self).__init__()
        w_init = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c_0 = L.Convolution2D(64, 32, (3, 1), initialW=w_init, pad=(1, 0), nobias=True)
            self.b_0 = L.BatchNormalization(32)
            self.c_1 = L.Convolution2D(32, 64, (1, 1), initialW=w_init)
            self.b_1 = L.BatchNormalization(64)
    def forward(self, _x):
        """
        呼び出し関数
        実際の計算を担う
        活性化関数はReLUを使用
        """
        _h = self.c_0(_x)
        _h = self.b_0(_h)
        _h = F.relu(_h)
        _h = self.c_1(_h)
        _h = self.b_1(_h)
        _y = _x + _h
        return _y
