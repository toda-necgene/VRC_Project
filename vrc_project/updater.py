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
        self.gen_am = model["main"]
        self.gen_bm = model["inverse"]
        self.gen_mb = model["main2"]
        self.gen_ma = model["inverse2"]
        self.disa = model["disa"]
        # self.disb = model["disb"]
        self.max_iteration = max_itr
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
    def update_core(self):
        gen_ab_optimizer = self.get_optimizer("gen_ab1")
        gen_ba_optimizer = self.get_optimizer("gen_ba1")
        gen_ab_optimizer2 = self.get_optimizer("gen_ab2")
        gen_ba_optimizer2 = self.get_optimizer("gen_ba2")
        disa_optimizer = self.get_optimizer("disa")
        # disb_optimizer = self.get_optimizer("disb")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        _xp = chainer.backend.get_array_module(batch_a.data)
        # D update
        self.disa.cleargrads()
        # self.disb.cleargrads()
        batch_a_n = batch_a + _xp.random.randn(*(batch_a.shape)) * (0.002 * self.iteration / self.max_iteration +0.0002)
        batch_b_n = batch_b + _xp.random.randn(*(batch_b.shape)) * (0.002 * self.iteration / self.max_iteration +0.0002)
        fake_ab = self.gen_mb(self.gen_am(batch_a_n))
        fake_ba = self.gen_ma(self.gen_bm(batch_b_n))
        y_af = self.disa(fake_ba)
        y_bf = self.disa(fake_ab)
        y_at = self.disa(batch_a_n)
        y_bt = self.disa(batch_b_n)
        y_label_TA = _xp.zeros(y_af.shape, dtype="float32")
        y_label_TA[:, 0] = 1.0
        y_label_TB = _xp.zeros(y_af.shape, dtype="float32")
        y_label_TB[:, 1] = 1.0
        y_label_FA = _xp.zeros(y_af.shape, dtype="float32")
        y_label_FA[:, 2] = 1.0
        y_label_FB = _xp.zeros(y_af.shape, dtype="float32")
        y_label_FB[:, 2] = 1.0
        loss_d_af = F.mean_squared_error(y_af, y_label_FA)
        loss_d_bf = F.mean_squared_error(y_bf, y_label_FB)
        loss_d_ar = F.mean_squared_error(y_at, y_label_TA)
        loss_d_br = F.mean_squared_error(y_bt, y_label_TB)
        chainer.report({"D_A_REAL": loss_d_ar,
                        "D_A_FAKE": loss_d_af,
                        "D_B_REAL": loss_d_br,
                        "D_B_FAKE": loss_d_bf})
        (loss_d_af + loss_d_ar + loss_d_bf + loss_d_br).backward()
        disa_optimizer.update()
        # G update
        self.gen_am.cleargrads()
        self.gen_bm.cleargrads()
        self.gen_mb.cleargrads()
        self.gen_ma.cleargrads()
        fake_bm = self.gen_bm(batch_b_n)
        fake_am = self.gen_am(batch_a_n)
        fake_aa = self.gen_ma(fake_am)
        fake_bb = self.gen_mb(fake_bm)
        fake_ab = self.gen_mb(fake_am)
        fake_ba = self.gen_ma(fake_bm)
        fake_aba = self.gen_bm(fake_ab)
        fake_bab = self.gen_am(fake_ba)
        y_fake_ba = self.disa(fake_ba)
        y_fake_ab = self.disa(fake_ab)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_TB)
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_TA)
        loss_cycbb = F.mean_squared_error(fake_bb, batch_b)
        loss_cycaa = F.mean_squared_error(fake_aa, batch_a)
        loss_cycbam = F.mean_squared_error(fake_bab, fake_bm)
        loss_cycabm = F.mean_squared_error(fake_aba, fake_am)
        gloss = loss_ganba + loss_ganab + (loss_cycaa + loss_cycbb + loss_cycbam+ loss_cycabm) * 10
        gloss.backward()
        chainer.report({"G_AB__GAN": loss_ganab,
                        "G_BA__GAN": loss_ganba,
                        "G_AA__L1N": loss_cycaa,
                        "G_BB__L1N": loss_cycbb,
                        "G_ABA_L1N": loss_cycabm,
                        "G_BAB_L1N": loss_cycbam})
        gen_ba_optimizer.update()
        gen_ab_optimizer.update()
        gen_ba_optimizer2.update()
        gen_ab_optimizer2.update()
        