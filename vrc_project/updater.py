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
        self.gen_ab = model["main"]
        self.gen_ba = model["inverse"]
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
        _xp = chainer.backend.get_array_module(batch_a.data)
        batch_a_n = noise_put(_xp, batch_a, 0.02)
        batch_b_n = noise_put(_xp, batch_b, 0.02)
        # batch_a_n = batch_a
        # batch_b_n = batch_b
        # melfilter = _xp.tanh(_xp.linspace(3.14, 0.4, 513)).reshape(1, 513, 1, 1)
        fake_ab = self.gen_ab(batch_a_n)
        fake_ba = self.gen_ba(batch_b_n)
        # D update
        self.disa.cleargrads()
        self.disb.cleargrads()
        y_af = self.disa(fake_ba)
        y_bf = self.disb(fake_ab)
        y_at = self.disa(batch_a_n)
        y_bt = self.disb(batch_b_n)
        y_label_o = _xp.ones(y_af.shape, dtype="float32")
        y_label_z = _xp.zeros(y_af.shape, dtype="float32")
        loss_d_af = F.mean_squared_error(y_af, y_label_z) * 0.5
        loss_d_bf = F.mean_squared_error(y_bf, y_label_z) * 0.5
        loss_d_ar = F.mean_squared_error(y_at, y_label_o) * 0.5
        loss_d_br = F.mean_squared_error(y_bt, y_label_o) * 0.5
        chainer.report({"D_AR": loss_d_ar})
        chainer.report({"D_AF": loss_d_af})
        chainer.report({"D_BR": loss_d_br})
        chainer.report({"D_BF": loss_d_bf})
        loss_d_af.backward()
        loss_d_ar.backward()
        loss_d_bf.backward()
        loss_d_br.backward()
        disa_optimizer.update()
        disb_optimizer.update()
        # G update
        _lamda = 10.0
        self.gen_ba.cleargrads()
        self.gen_ab.cleargrads()
        fake_ab = self.gen_ab(batch_a_n)
        fake_ba = self.gen_ba(batch_b_n)
        y_fake_ba = self.disa(fake_ba)
        y_fake_ab = self.disb(fake_ab)
        fake_aba = self.gen_ba(fake_ab)
        fake_bab = self.gen_ab(fake_ba)
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_o) * 0.5
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_o) * 0.5
        loss_cycb = F.mean_absolute_error(fake_bab, batch_b_n)
        loss_cyca = F.mean_absolute_error(fake_aba, batch_a_n)
        chainer.report({"G_AB": loss_ganba})
        chainer.report({"G_BA": loss_ganab})
        chainer.report({"G_ABA": loss_cyca})
        chainer.report({"G_BAB": loss_cycb})
        gloss = loss_ganba + loss_ganab + (loss_cycb + loss_cyca) * _lamda
        gloss.backward()
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
    x_s = x +_xp.random.randn(noise_shape[0], noise_shape[1], noise_shape[2], noise_shape[3]) * stddev
    return x_s
        