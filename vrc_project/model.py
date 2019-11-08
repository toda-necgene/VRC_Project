
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
# class Generator(chainer.Chain):
#     """
#         学習用生成側ネットワーク
#     """
#     def __init__(self, chs=256, layers=6):
#         """
#         レイヤー定義
#         Parameter
#         ---------
#         chs : int
#         residualblockのチャンネル数
#         layres : int
#         residualblockの数
#         """
#         super(Generator, self).__init__()
#         self.l_num = layers
#         he_init = chainer.initializers.HeNormal()
#         self.add_link("00_e_0", L.Convolution1D(1025, chs, 4, stride=4, initialW=he_init).add_hook(spn()))
#         for i in range(self.l_num):
#             self.add_link("0"+str(i+1)+"_c_0", L.Linear(chs, chs ,initialW=he_init).add_hook(spn()))
#             self.add_link("0"+str(i+1)+"_t_0", L.Linear(50, 50 ,initialW=he_init).add_hook(spn()))
#         self.add_link("97_d_0", L.Deconvolution1D(chs, 1025, 4, stride=4, initialW=he_init).add_hook(spn()))

#     def __call__(self, *_x, **kwargs):
#         """
#             モデルのグラフ実装
#             Parameter
#             ---------
#             x: ndarray(tuple)
#                 変換前特徴量
#                 shape: [N,1025,200,1]
#             Returns
#             -------
#             _y: ndarray
#                 変換後特徴量
#                 shape: [N,1025,200,1]
#         """
#         links = self.children()
#         _y = F.transpose(_x[0][:, :, :, 0], (0, 2, 1))
#         _y = next(links)(_y)
#         _y = F.leaky_relu(_y)
#         for _ in range(self.l_num):
#             c_size, t_size = _y.shape[1:]
#             _h = F.reshape(F.transpose(_y, (0, 2, 1)), (-1, c_size))
#             _h = next(links)(_h)
#             _h = F.reshape(F.transpose(F.reshape(_h, (-1, t_size, c_size)), (0, 2, 1)), (-1, t_size))
#             _y = F.reshape(_y, (-1, t_size))
#             _i = next(links)(_y)
#             _y = F.reshape(F.leaky_relu(_h + _i)+ _y, (-1, c_size, t_size))
#         _y = next(links)(_y)
#         _y = F.transpose(_y, (0, 2, 1))
#         _y = F.expand_dims(_y, 3)
#         return _y
class Generator(chainer.Chain):
    """
        学習用生成側ネットワーク
    """
    def __init__(self, chs=256, layers=6):
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
        with self.init_scope():
            xp = chainer.backend.get_array_module()
            he_init = chainer.initializers.HeNormal()
            self.e0 = L.Convolution1D(1025, chs, 4, stride=4, initialW=he_init).add_hook(spn())
            self.r1c = L.Linear(chs, chs ,initialW=he_init).add_hook(spn())
            self.r1t = L.Linear(50, 50 ,initialW=he_init).add_hook(spn())
            self.a1 = L.Parameter(xp.array([0.0], dtype="float32"))
            self.r2c = L.Linear(chs, chs ,initialW=he_init).add_hook(spn())
            self.r2t = L.Linear(50, 50 ,initialW=he_init).add_hook(spn())
            self.a2 = L.Parameter(xp.array([0.0], dtype="float32"))
            self.r3c = L.Linear(chs, chs ,initialW=he_init).add_hook(spn())
            self.r3t = L.Linear(50, 50 ,initialW=he_init).add_hook(spn())
            self.a3 = L.Parameter(xp.array([0.0], dtype="float32"))
            self.r4c = L.Linear(chs, chs ,initialW=he_init).add_hook(spn())
            self.r4t = L.Linear(50, 50 ,initialW=he_init).add_hook(spn())
            self.a4 = L.Parameter(xp.array([0.0], dtype="float32"))
            self.r5c = L.Linear(chs, chs ,initialW=he_init).add_hook(spn())
            self.r5t = L.Linear(50, 50 ,initialW=he_init).add_hook(spn())
            self.a5 = L.Parameter(xp.array([0.0], dtype="float32"))
            self.r6c = L.Linear(chs, chs ,initialW=he_init).add_hook(spn())
            self.r6t = L.Linear(50, 50 ,initialW=he_init).add_hook(spn())
            self.a6 = L.Parameter(xp.array([0.0], dtype="float32"))
            self.d0 = L.Deconvolution1D(chs, 1025, 4, stride=4, initialW=he_init).add_hook(spn())
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
        # encode
        _y = F.transpose(_x[0][:, :, :, 0], (0, 2, 1))
        _y = self.e0(_y)
        _y = F.leaky_relu(_y)
        c_size, t_size = _y.shape[1:]
        # residual
        #1
        _h = F.reshape(F.transpose(_y, (0, 2, 1)), (-1, c_size))
        _h = self.r1c(_h)
        _h = F.reshape(F.transpose(F.reshape(_h, (-1, t_size, c_size)), (0, 2, 1)), (-1, t_size))
        _y = F.reshape(_y, (-1, t_size))
        _i = self.r1t(_y)
        _act = F.sigmoid(self.a1()*10)
        _y = F.reshape(F.leaky_relu(_h * _act + _i * (1-_act))+ _y, (-1, c_size, t_size))
        #2
        _h = F.reshape(F.transpose(_y, (0, 2, 1)), (-1, c_size))
        _h = self.r2c(_h)
        _h = F.reshape(F.transpose(F.reshape(_h, (-1, t_size, c_size)), (0, 2, 1)), (-1, t_size))
        _y = F.reshape(_y, (-1, t_size))
        _i = self.r2t(_y)
        _act = F.sigmoid(self.a2()*10)
        _y = F.reshape(F.leaky_relu(_h * _act + _i * (1-_act))+ _y, (-1, c_size, t_size))
        #3
        _h = F.reshape(F.transpose(_y, (0, 2, 1)), (-1, c_size))
        _h = self.r3c(_h)
        _h = F.reshape(F.transpose(F.reshape(_h, (-1, t_size, c_size)), (0, 2, 1)), (-1, t_size))
        _y = F.reshape(_y, (-1, t_size))
        _i = self.r3t(_y)
        _act = F.sigmoid(self.a3()*10)
        _y = F.reshape(F.leaky_relu(_h * _act + _i * (1-_act))+ _y, (-1, c_size, t_size))
        #4
        _h = F.reshape(F.transpose(_y, (0, 2, 1)), (-1, c_size))
        _h = self.r4c(_h)
        _h = F.reshape(F.transpose(F.reshape(_h, (-1, t_size, c_size)), (0, 2, 1)), (-1, t_size))
        _y = F.reshape(_y, (-1, t_size))
        _i = self.r4t(_y)
        _act = F.sigmoid(self.a4()*10)
        _y = F.reshape(F.leaky_relu(_h * _act + _i * (1-_act))+ _y, (-1, c_size, t_size))
        #5
        _h = F.reshape(F.transpose(_y, (0, 2, 1)), (-1, c_size))
        _h = self.r5c(_h)
        _h = F.reshape(F.transpose(F.reshape(_h, (-1, t_size, c_size)), (0, 2, 1)), (-1, t_size))
        _y = F.reshape(_y, (-1, t_size))
        _i = self.r5t(_y)
        _act = F.sigmoid(self.a5()*10)
        _y = F.reshape(F.leaky_relu(_h * _act + _i * (1-_act))+ _y, (-1, c_size, t_size))
        #6
        _h = F.reshape(F.transpose(_y, (0, 2, 1)), (-1, c_size))
        _h = self.r6c(_h)
        _h = F.reshape(F.transpose(F.reshape(_h, (-1, t_size, c_size)), (0, 2, 1)), (-1, t_size))
        _y = F.reshape(_y, (-1, t_size))
        _i = self.r6t(_y)
        _act = F.sigmoid(self.a6()*10)
        _y = F.reshape(F.leaky_relu(_h * _act + _i * (1-_act))+ _y, (-1, c_size, t_size))
        # decode
        _y = self.d0(_y)
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
            self.add_link("0"+str(i+1)+"_h_0", L.Convolution1D(chs, chs, 21, pad=10))
        self.add_link("97_d_0", L.Deconvolution1D(chs, 1025, 4, stride=4))
        self.add_link("98_d_0", L.Deconvolution2D(1, 64, (9, 11), pad=(4, 5)))
        self.add_link("99_d_0", L.Deconvolution2D(64, 1, (9, 11), pad=(4, 5)))

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
        _y = next(links)(_y)
        _y = F.transpose(_y, (0, 2, 1))
        _y = F.expand_dims(_y, 1)
        _y = next(links)(_y)
        _y = F.leaky_relu(_y)
        _y = next(links)(_y)
        _y = F.transpose(_y, (0, 2, 3, 1))
        return _y