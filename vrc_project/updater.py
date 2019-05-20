"""
    製作者:TODA
    学習の際に変数を更新するために必要な関数群
"""
import chainer
import chainer.functions as F

class CycleGANUpdater(chainer.training.updaters.StandardUpdater):
    """
        CycleGANのためのUpdater
        以下のものを含む
        - 目的函数
        - 更新のコア関数
    """
    def __init__(self, model, max_itr, *args, **kwargs):
        """
        初期化関数
        Parameters
        ----------
        model: dict
        生成モデルA,生成モデルB,識別モデルA,識別モデルB
        maxitr: int
        最大イテレーション回数
        """
        self.gen_ab1 = model["main"]
        self.gen_ba1 = model["inverse"]
        self.disa = model["disa"]
        self.disb = model["disb"]
        self.max_iteration = max_itr
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
    def update_core(self):
        gen_ab_optimizer1 = self.get_optimizer("gen_ab1")
        gen_ba_optimizer1 = self.get_optimizer("gen_ba1")
        disa_optimizer = self.get_optimizer("disa")
        disb_optimizer = self.get_optimizer("disb")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        batch_size = len(batch_a)
        _xp = chainer.backend.get_array_module(batch_a.data)
        # _cos = _xp.cos(self.iteration / self.max_iteration / 2 * _xp.pi )
        batch_a_n = noise_put(_xp, batch_a, 0.01)
        batch_b_n = noise_put(_xp, batch_b, 0.01)
        # batch_a_n = batch_a
        # batch_b_n = batch_b
        fake_ab_1 = self.gen_ab1(batch_a_n)
        fake_ba_1 = self.gen_ba1(batch_b_n)
        # D update
        for _ in range(1):
            self.disa.cleargrads()
            self.disb.cleargrads()
            y_af = self.disa(fake_ba_1)
            y_bf = self.disb(fake_ab_1)
            y_at = self.disa(batch_a_n)
            y_bt = self.disb(batch_b_n)
            wave_length = y_af.shape[2]
            y_label_o = _xp.ones([batch_size, 1, wave_length], dtype="float32")
            y_label_z = _xp.zeros([batch_size, 1, wave_length], dtype="float32")
            loss_d_a = F.mean_squared_error(y_af, y_label_z) *0.5+F.mean_squared_error(y_at, y_label_o) *0.5
            loss_d_b = F.mean_squared_error(y_bf, y_label_z) *0.5+F.mean_squared_error(y_bt, y_label_o) *0.5
            chainer.report({"D_A": loss_d_a})
            chainer.report({"D_B": loss_d_b})
            loss_d_a.backward()
            loss_d_b.backward()
            disa_optimizer.update()
            disb_optimizer.update()
        # G update
        _lamda = 50.0
        self.gen_ba1.cleargrads()
        self.gen_ab1.cleargrads()
        y_fake_ba = self.disa(fake_ba_1)
        y_fake_ab = self.disb(fake_ab_1)
        fake_aba1 = self.gen_ba1(fake_ab_1)
        fake_bab1 = self.gen_ab1(fake_ba_1)
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_o)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_o)
        loss_cycb = F.mean_absolute_error(fake_bab1, batch_b_n)
        loss_cyca = F.mean_absolute_error(fake_aba1, batch_a_n)
        chainer.report({"G_G_A": loss_ganba})
        chainer.report({"G_G_B": loss_ganab})
        chainer.report({"G_C_A": loss_cyca})
        chainer.report({"G_C_B": loss_cycb})
        glossa = loss_ganba + loss_cyca * _lamda
        glossa.backward()
        glossb = loss_ganab + loss_cycb * _lamda
        glossb.backward()
        gen_ab_optimizer1.update()
        gen_ba_optimizer1.update()

def noise_put(_xp, x, stddev):
    """
    正規分布に沿ったノイズを載せる
    Parameter
    ---------
    _xp: 計算ライブラリ
    x: 入力テンソル（４階）
    stddev: 標準偏差
    """
    noise_shape = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]]
    v = _xp.var(x.array)
    x_s = x +_xp.random.randn(noise_shape[0], noise_shape[1], noise_shape[2], noise_shape[3]) * stddev
    x_s = x_s * (v / _xp.var(x_s.array))
    x_s = F.clip(x_s, -1.0, 1.0)
    return x_s
        