
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
        周波数軸はチャンネル軸として扱います。（全結合される）
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みはHeの初期化
        最終層は0.01の分散
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            self.c_0 = L.Convolution2D(1, 64, (6, 9), stride=(2, 9), initialW=w_init)
            self.c_1 = L.Convolution2D(64, 128, (6, 6), stride=(2, 3), initialW=w_init)
            self.c_2 = L.Convolution2D(128, 256, (9, 4), stride=(2, 2), initialW=w_init)
            self.c_3 = L.Convolution2D(256, 512, (7, 8), stride=(2, 1), initialW=w_init)
            self.d_l = L.Convolution2D(512, 1, (2, 1), stride=(2, 1), initialW=chainer.initializers.Normal(0.005))
    def __call__(self, *_x, **kwargs):
        """
        呼び出し関数
        実際の計算を担う
        Parameter
            ---------
            x: ndarray(tuple)
                特徴量
                range: (-1.0, 1.0)
                shape: [N, 513, 100, 2]
            Returns
            -------
            _y: ndarray
                評価
                range: (-inf, inf)
                shape: [N, 1, 1]
        """
        _y = F.transpose(_x[0], (0, 3, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_2(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_3(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_l(_y)[:, 0, :, :]
        return _y
class Generator(chainer.Chain):
    """
        生成側ネットワーク
        構造は
        (N, 256, 10, 19)形式にエンコード
        single-reluからBNを削除したresnet6ブロック
        入力と同じ(N, 2, 100, 513)にデコード
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
            # (N, 2, 100, 513)
            self.c_0 = L.Convolution2D(1, 64, (2, 3), stride=(2, 3), initialW=w_init)
            # (N, 64, 100, 171)
            self.d_1 = L.Convolution2D(64, 128, (2, 6), stride=(2, 3), initialW=w_init)
            # (N, 128, 50, 56)
            self.d_2 = L.Convolution2D(128, 256, (10, 4), stride=(5, 4), initialW=w_init)
            # (N, 256, 9, 14)
            self.r_1 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            # self.r_2 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_2 = L.Convolution2D(14, 14, (3, 1), pad=(1, 0), initialW=chainer.initializers.Normal(0.02))
            self.r_3 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            # self.r_4 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_4 = L.Convolution2D(14, 14, (3, 1), pad=(1, 0), initialW=chainer.initializers.Normal(0.02))
            self.r_5 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            # self.r_6 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_6 = L.Convolution2D(14, 14, (3, 1), pad=(1, 0), initialW=chainer.initializers.Normal(0.02))
            self.r_7 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            # self.r_8 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_8 = L.Convolution2D(14, 14, (3, 1), pad=(1, 0), initialW=chainer.initializers.Normal(0.02))
            self.r_9 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            # self.r_10 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_10 = L.Convolution2D(14, 14, (3, 1), pad=(1, 0), initialW=chainer.initializers.Normal(0.02))
            self.r_11 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            # self.r_12 = L.Convolution2D(256, 256, (3, 3), pad=(1, 1), initialW=w_init)
            self.r_12 = L.Convolution2D(14, 14, (3, 1), pad=(1, 0), initialW=chainer.initializers.Normal(0.02))
            self.u_2 = L.Deconvolution2D(256, 128, (10, 4), stride=(5, 4), initialW=chainer.initializers.Normal(0.02), nobias=True)
            self.b_a1 = L.BatchNormalization(128)
            self.u_1 = L.Deconvolution2D(128, 64, (2, 6), stride=(2, 3), initialW=chainer.initializers.Normal(0.02), nobias=True)
            self.b_a2 = L.BatchNormalization(64)
            self.d_0_1 = L.Deconvolution2D(64, 1, (2, 3), stride=(2, 3), initialW=chainer.initializers.Normal(0.02))
    def __call__(self, *_x, **kwargs):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray(tuple)
                変換前特徴量
                shape: [N,513,100,2]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,513,100,2]
        """
        _y = F.transpose(_x[0], (0, 3, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_2(_y)
        _y = F.leaky_relu(_y)
        _h = self.r_1(_y)
        _h = F.relu(_h)
        _a = F.transpose(_h, (0, 3, 2, 1))
        _a = self.r_2(_a)
        _a = F.transpose(_a, (0, 3, 2, 1))
        _y = _h * F.tanh(_a) + _y
        _h = self.r_3(_y)
        _h = F.relu(_h)
        _a = F.transpose(_h, (0, 3, 2, 1))
        _a = self.r_4(_a)
        _a = F.transpose(_a, (0, 3, 2, 1))
        _y = _h * F.tanh(_a) + _y
        _h = self.r_5(_y)
        _h = F.relu(_h)
        _a = F.transpose(_h, (0, 3, 2, 1))
        _a = self.r_6(_a)
        _a = F.transpose(_a, (0, 3, 2, 1))
        _y = _h * F.tanh(_a) + _y
        _h = self.r_7(_y)
        _h = F.relu(_h)
        _a = F.transpose(_h, (0, 3, 2, 1))
        _a = self.r_8(_a)
        _a = F.transpose(_a, (0, 3, 2, 1))
        _y = _h * F.tanh(_a) + _y
        _h = self.r_9(_y)
        _h = F.relu(_h)
        _a = F.transpose(_h, (0, 3, 2, 1))
        _a = self.r_10(_a)
        _a = F.transpose(_a, (0, 3, 2, 1))
        _y = _h * F.tanh(_a) + _y
        _h = self.r_11(_y)
        _h = F.relu(_h)
        _a = F.transpose(_h, (0, 3, 2, 1))
        _a = self.r_12(_a)
        _a = F.transpose(_a, (0, 3, 2, 1))
        _y = _h * F.tanh(_a) + _y
        _y = self.u_2(_y)
        _y = self.b_a1(_y)
        _y = F.leaky_relu(_y)
        _y = self.u_1(_y)
        _y = self.b_a2(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_0_1(_y)
        _y = F.tanh(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        return _y
