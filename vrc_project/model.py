
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
        4層CNN(Conv1d-leaky_relu)
        最終層のみチャンネルを32づつ4グループにして各グループで最大値をとる
    """
    def __init__(self, chs=None):
        """
        モデル定義
        Parameter
        ---------
        chs : list of int
        各レイヤーごとの出力チャンネル数
        """
        super(Discriminator, self).__init__()
        if chs is None:
            chs = [512, 256, 128]
        with self.init_scope():
            he_init = chainer.initializers.HeNormal()
            # (N, 1025, 200)
            self.c_0 = L.Convolution1D(1025, chs[0], 6, stride=2, pad=2, initialW=he_init).add_hook(spn())
            # (N, 512, 100)
            self.c_1 = L.Convolution1D(chs[0], chs[1], 6, stride=2, pad=2, initialW=he_init).add_hook(spn())
            # (N, 256, 50)
            self.c_2 = L.Convolution1D(chs[1], chs[2], 10, stride=5, pad=0, initialW=he_init).add_hook(spn())
            # (N, 128, 9)
            self.c_3 = L.Convolution1D(chs[2], 128, 9, pad=4, initialW=he_init).add_hook(spn())
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
        _y = F.transpose(_x[0][:, :, :, 0], (0, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_2(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_3(_y)
        _y = F.max(F.reshape(_y, (-1, 4, 32, 9)), axis=2)
        return _y
class Generator(chainer.Chain):
    """
        学習用生成側ネットワーク
    """
    def __init__(self, chs=256, layers=9):
        """
        レイヤー定義
        Parameter
        ---------
        chs : int
        residualblockのチャンネル数
        layres : int
        residualblockの数
        """
        super(Generator, self).__init__()
        self.l_num = layers
        he_init = chainer.initializers.HeNormal()
        self.add_link("00_e", L.Convolution1D(1025, chs, 4, stride=4, initialW=he_init).add_hook(spn()))
        for i in range(self.l_num):
            self.add_link("0"+str(i+1)+"_c", L.Convolution1D(chs, chs, 11, pad=5, initialW=he_init).add_hook(spn()))
        self.add_link("98_d", L.Deconvolution1D(chs, 1, 4, stride=4, initialW=he_init).add_hook(spn()))
        self.add_link("99_d", L.Deconvolution1D(chs, 1025, 4, stride=4, initialW=he_init).add_hook(spn()))

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
        _y = F.transpose(_x[0][:, :, :, 0], (0, 2, 1))
        _y = next(links)(_y)
        _y = F.leaky_relu(_y)
        for _ in range(self.l_num):
            _h = next(links)(_y)
            _y = F.leaky_relu(_h)+ _y
        _a = next(links)(_y)
        _y = next(links)(_y) * F.sigmoid(_a)
        _y = F.transpose(_y, (0, 2, 1))
        _y = F.expand_dims(_y, 3)
        return _y

class GeneratorSimple(chainer.Chain):
    """
        実行用生成側ネットワーク
    """
    def __init__(self, chs=256, layers=6):
        """
        レイヤー定義
        """
        super(GeneratorSimple, self).__init__()
        self.l_num = layers
        self.add_link("00_e_0", L.Convolution1D(1025, chs, 4, stride=4))
        for i in range(self.l_num):
            self.add_link("0"+str(i+1)+"_h_0", L.Convolution1D(chs, chs, 11, pad=5))
        self.add_link("98_d", L.Deconvolution1D(chs, 1, 4, stride=4, initialW=he_init))
        self.add_link("99_d", L.Deconvolution1D(chs, 1025, 4, stride=4, initialW=he_init))
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
        _y = F.transpose(_x[0][:, :, :, 0], (0, 2, 1))
        _y = next(links)(_y)
        _y = F.leaky_relu(_y)
        for _ in range(self.l_num):
            _h = next(links)(_y)
            _y = F.leaky_relu(_h) + _y
        _a = next(links)(_y)
        _y = next(links)(_y) * F.sigmoid(_a)
        _y = F.transpose(_y, (0, 2, 1))
        _y = F.expand_dims(_y, 3)
        return _y