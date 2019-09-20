
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
            self.c_0 = L.Convolution2D(1, 128, (6, 5), stride=(2, 4), pad=(2, 0), initialW=w_init).add_hook(spn())
            # (N, 128, 100, 256)
            self.c_1 = L.Convolution2D(128, 256, (2, 4), stride=(2, 4), initialW=w_init).add_hook(spn())
            # (N, 256, 50, 64)
            self.c_2 = L.Convolution2D(256, 1024, (10, 8), stride=(5, 8), initialW=w_init).add_hook(spn())
            # (N, 1024, 9, 8)
            self.c_3 = L.Convolution2D(1024, 4, (3, 8), initialW=chainer.initializers.Normal(1e-3)).add_hook(spn())
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
            self.e_0 = L.Convolution2D(1025, 256, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn())
            # (N, 256, 50)
            self.r_01 = L.Convolution2D(256, 256, (9, 1), pad=(4, 0), initialW=w_init).add_hook(spn())
            self.r_02 = L.Convolution2D(256, 256, (3, 1), pad=(1, 0), initialW=w_init).add_hook(spn())
            self.r_03 = L.Convolution2D(256, 256, (9, 1), pad=(4, 0), initialW=w_init).add_hook(spn())
            self.r_04 = L.Convolution2D(256, 256, (3, 1), pad=(1, 0), initialW=w_init).add_hook(spn())
            self.r_05 = L.Convolution2D(256, 256, (9, 1), pad=(4, 0), initialW=w_init).add_hook(spn())
            self.r_06 = L.Convolution2D(256, 256, (3, 1), pad=(1, 0), initialW=w_init).add_hook(spn())
            self.r_07 = L.Convolution2D(256, 256, (9, 1), pad=(4, 0), initialW=w_init).add_hook(spn())
            self.r_08 = L.Convolution2D(256, 256, (3, 1), pad=(1, 0), initialW=w_init).add_hook(spn())
            self.r_09 = L.Convolution2D(256, 256, (9, 1), pad=(4, 0), initialW=w_init).add_hook(spn())
            self.r_10 = L.Convolution2D(256, 256, (3, 1), pad=(1, 0), initialW=w_init).add_hook(spn())
            self.r_11 = L.Convolution2D(256, 256, (9, 1), pad=(4, 0), initialW=w_init).add_hook(spn())
            self.r_12 = L.Convolution2D(256, 256, (3, 1), pad=(1, 0), initialW=w_init).add_hook(spn())
            # (N, 256, 50)
            # NOTE: 出力を256chにすると学習が崩壊する。
            self.d_1 = L.Deconvolution2D(256, 128, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn())
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
        _h = self.r_01(_y)
        _h = F.sigmoid(_h) * _y
        _h = self.r_02(_h)
        _y = F.leaky_relu(_h) + _y
        _h = self.r_03(_y)
        _h = F.sigmoid(_h) * _y
        _h = self.r_04(_h)
        _y = F.leaky_relu(_h) + _y
        _h = self.r_05(_y)
        _h = F.sigmoid(_h) * _y
        _h = self.r_06(_h)
        _y = F.leaky_relu(_h) + _y
        _h = self.r_07(_y)
        _h = F.sigmoid(_h) * _y
        _h = self.r_08(_h)
        _y = F.leaky_relu(_h) + _y
        _h = self.r_09(_y)
        _h = F.sigmoid(_h) * _y
        _h = self.r_10(_h)
        _y = F.leaky_relu(_h) + _y
        _h = self.r_11(_y)
        _h = F.sigmoid(_h) * _y
        _h = self.r_12(_h)
        _y = F.leaky_relu(_h) + _y
        _y = self.d_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_0(_y)
        return _y
