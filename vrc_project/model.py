
"""
製作者:TODA
モデルの定義
"""
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.link_hooks.spectral_normalization import SpectralNormalization as spn



class Discriminator(chainer.Chain):
    """
        識別側ネットワーク
        CNN構造
        4層構造
        活性化関数はleaky_relu（最後のみ無し）
    """
    def __init__(self):
        """
        モデル定義
        重みはHeの初期化
        最終層は0.0002の分散
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 1, 200, 1025)
            self.c_0 = L.Convolution2D(1, 128, (2, 5), stride=(2, 4), initialW=w_init).add_hook(spn())
            # (N, 128, 100, 256)
            self.c_1 = L.Convolution2D(128, 256, (2, 4), stride=(2, 4), initialW=w_init).add_hook(spn())
            # (N, 256, 50, 64)
            self.c_2 = L.Convolution2D(256, 512, (10, 8), stride=(5, 8), initialW=w_init).add_hook(spn())
            # (N, 512, 9, 8)
            self.c_3 = L.Convolution2D(512, 4, (9, 8), pad=(3, 0), initialW=chainer.initializers.Normal(1e-3)).add_hook(spn())
            # (N, 4, 7)
    def __call__(self, *_x, **kwargs):
        """
        モデルのグラフ実装
        Parameter
            ---------
            x: ndarray(tuple)
                特徴量
                shape: [N, 1024, 200, 1]
            Returns
            -------
            _y: ndarray
                評価
                shape: [N, 4, 3]
        """
        _y = F.transpose(_x[0], (0, 3, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_2(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_3(_y)[:, :, :, 0]
        return _y
class MidBlock(chainer.Chain):
    """
    中間ブロック
    """
    def __init__(self, depth):
        super(MidBlock, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            self.l = L.Convolution2D(256, 128, 1, initialW=w_init).add_hook(spn())
            self.a = L.Convolution2D(256, 64, (11, 1), pad=(5, 0), initialW=w_init).add_hook(spn())
            self.o = L.Convolution2D(128, 256, (3, 1), pad=(1, 0), initialW=w_init).add_hook(spn())
            n = 2 ** (depth+1)
            self.c = L.DilatedConvolution2D(128, 256, (2, 1), dilate=(n, 1), pad=(n//2, 0), initialW=w_init).add_hook(spn())
    def __call__(self, _x):
        _a = F.repeat(self.a(_x), 4, axis=1)
        _y = F.sigmoid(_a) * _x
        _y = self.l(_y)
        _y = F.leaky_relu(_y)
        _otp_for_sum = F.tanh(self.o(_y))
        _otp_to_next = self.c(_y)
        _otp_to_next = F.leaky_relu(_otp_to_next) + _x
        return _otp_to_next, _otp_for_sum
class Generator(chainer.Chain):
    """
        生成側ネットワーク
        DilatedConvを用いた6ブロックResnet（内4ブロックがDilated）
        各ブロック
        Conv(use_bias)-Leaky_relu-add
    """
    def __init__(self):
        """
        モデル定義
        重みの初期化はHeの初期化,最終層のみ0.0001分散で固定
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 1025, 200)
            self.e_0 = L.Convolution2D(1025, 256, (2, 1), stride=(2, 1), initialW=w_init).add_hook(spn())
            # (N, 256, 50)
            self.r_0 = MidBlock(0)
            self.r_1 = MidBlock(1)
            self.r_2 = MidBlock(2)
            self.r_3 = MidBlock(3)
            self.r_4 = MidBlock(4)
            self.r_5 = MidBlock(5)
            # (N, 256, 50)
            # NOTE: 出力を256chにすると学習が崩壊する。
            self.d_1 = L.Deconvolution2D(256, 128, (2, 1), stride=(2, 1), initialW=w_init).add_hook(spn())
            # (N, 128, 200)
            self.d_0 = L.Convolution2D(128, 1025, (7, 1), pad=(3, 0), initialW=w_init).add_hook(spn())
            # (N, 1025, 200)
    def __call__(self, *_x, **kwargs):
        """
            モデルのグラフ実装
            Parameter
            ---------
            x: ndarray(tuple)
                変換前特徴量
                shape: [N,1025,200,1]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,1025,200,1]
        """
        _y = _x[0]
        _y = self.e_0(_y)
        _y = F.leaky_relu(_y)
        _h, _n = self.r_0(_y)
        _y = _n
        _h, _n = self.r_1(_h)
        _y += _n
        _h, _n = self.r_2(_h)
        _y += _n
        _h, _n = self.r_3(_h)
        _y += _n
        _h, _n = self.r_4(_h)
        _y += _n
        _m = self.r_5(_h)
        _y += _m[0] + _m[1]
        _y = self.d_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_0(_y)
        return _y
