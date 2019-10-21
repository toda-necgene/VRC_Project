
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
    def __init__(self, chs=None):
        """
        モデル定義
        重みはHeの初期化
        最終層は0.0002の分散
        """
        super(Discriminator, self).__init__()
        if chs is None:
            chs = [16, 32, 64, 128, 256, 512]
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 1, 200, 1025)
            self.c_0 = L.Convolution2D(1, chs[0], (3, 3), stride=(1, 2), pad=(1, 0), initialW=w_init).add_hook(spn())
            # (N, 16, 200, 512)
            self.c_1 = L.Convolution2D(chs[0], chs[1], (3, 4), stride=(1, 2), pad=(1, 1), initialW=w_init).add_hook(spn())
            # (N, 32, 200, 256)
            self.c_2 = L.Convolution2D(chs[1], chs[2], (3, 4), stride=(1, 2), pad=(1, 1), initialW=w_init).add_hook(spn())
            # (N, 64, 200, 128)
            self.c_3 = L.Convolution2D(chs[2], chs[3], (3, 4), stride=(1, 2), pad=(1, 1), initialW=w_init).add_hook(spn())
            # (N, 128, 200, 64)
            # ここから共通処理
            self.c_4 = L.Convolution2D(chs[3], chs[4], (5, 4), stride=(5, 2), pad=(0, 1), initialW=w_init).add_hook(spn())
            # (N, 256, 40, 32)
            self.c_5 = L.Convolution2D(chs[4], chs[5], (10, 4), stride=(5, 4), pad=(0, 0), initialW=w_init).add_hook(spn())
            # (N, 512, 7, 8)
            self.c_6 = L.Convolution2D(chs[5], 4, (4, 8), initialW=w_init).add_hook(spn())
            # (N, 4, 1)
        self.grow = 0
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
        _y = F.transpose(_x[0], (0, 3, 1, 2))
        if self.grow > 3:
            _y = self.c_0(_y)
            _y = F.leaky_relu(_y)
        else:
            _y = F.repeat(_y, 16, axis=1)
        if self.grow > 2:
            _y = self.c_1(_y)
            _y = F.leaky_relu(_y)
        else:
            _y = F.repeat(_y, 2, axis=1)
        if self.grow > 1:
            _y = self.c_2(_y)
            _y = F.leaky_relu(_y)
        else:
            _y = F.repeat(_y, 2, axis=1)
        if self.grow > 0:
            _y = self.c_3(_y)
            _y = F.leaky_relu(_y)
        else:
            _y = F.repeat(_y, 2, axis=1)
        _y = self.c_4(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_5(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_6(_y)[:, :, :, 0]
        return _y
class Generator(chainer.Chain):
    """
        生成側ネットワーク
        nブロックResnet
        各ブロック
        Conv(use_bias)-Leaky_relu-add
    """
    def __init__(self, chs=128, layers=9):
        """
        モデル定義
        重みの初期化はHeの初期化,最終層のみ0.0001分散で固定
        """
        super(Generator, self).__init__()
        self.l_num = layers
        w_init = chainer.initializers.HeNormal()
        # 0,64
        self.add_link("00e_0", L.Convolution2D(64, chs, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn()))
        # 64,128
        self.add_link("00e_1", L.Convolution2D(64, chs, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn()))
        # 128,256
        self.add_link("00e_2", L.Convolution2D(128, chs, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn()))
        # 256,512
        self.add_link("00e_3", L.Convolution2D(256, chs, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn()))
        # 512,1025
        self.add_link("00e_4", L.Convolution2D(513, chs, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn()))
        for i in range(layers):
            self.add_link("0"+str(i+1)+"r_0", L.Convolution2D(chs, chs, (11, 1), pad=(5, 0), initialW=w_init).add_hook(spn()))
        self.add_link(str(layers+1)+"d_0", L.Deconvolution2D(chs, 1025, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn()))
        self.grow = 0
        self.watch = [[0, 64], [64, 128], [128, 256], [256, 512], [512, 1025]]
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
        links = self.children()
        _y = F.transpose(_x[0], (0, 2, 1, 3))
        s = _y.shape[1]
        _s = next(links)(_y[:, self.watch[0][0]:self.watch[0][1]])
        for i in range(0, 4):
            f = next(links)
            if self.grow > i:
                _s += f(_y[:, self.watch[i+1][0]:self.watch[i+1][1]])
        _y = F.leaky_relu(_s)
        for i in range(self.l_num):
            f = next(links)
            _h = f(_y)
            _y = F.relu(_h) + _y
        _y = next(links)(_y)
        _y = F.transpose(_y[:, :s, :, :], (0, 2, 1, 3))
        return _y
