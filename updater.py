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
        gen_ab_optimizer = self.get_optimizer("gen_ab")
        gen_ba_optimizer = self.get_optimizer("gen_ba")
        disa_optimizer = self.get_optimizer("disa")
        disb_optimizer = self.get_optimizer("disb")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        batch_size = len(batch_a)
        _xp = chainer.backend.get_array_module(batch_a.data)
        fake_ab_ = self.gen_ab(batch_a)
        fake_aba = self.gen_ba(fake_ab_)
        fake_ba_ = self.gen_ba(batch_b)
        fake_bab = self.gen_ab(fake_ba_)
        # D_true update
        self.disa.cleargrads()
        self.disb.cleargrads()
        y_ta = self.disa(batch_a)
        y_tb = self.disb(batch_b)
        wave_length = y_ta.shape[2]
        y_label_o = _xp.ones([batch_size, 1, wave_length], dtype="float32")
        loss_d_t_a = F.mean_squared_error(y_ta, y_label_o)
        loss_d_t_b = F.mean_squared_error(y_tb, y_label_o)
        y_label_z = _xp.zeros([batch_size, 1, wave_length], dtype="float32")
        y_fa = self.disa(fake_ba_)
        y_fb = self.disb(fake_ab_)
        loss_d_f_a = F.mean_squared_error(y_fa, y_label_z)
        loss_d_f_b = F.mean_squared_error(y_fb, y_label_z)
        (loss_d_t_a + loss_d_t_b + loss_d_f_a + loss_d_f_b).backward()
        chainer.report({"loss": loss_d_f_a+loss_d_t_a}, self.disa)
        chainer.report({"loss": loss_d_f_b+loss_d_t_b}, self.disb)
        disa_optimizer.update()
        disb_optimizer.update()
        # G update
        self.gen_ba.cleargrads()
        self.gen_ab.cleargrads()
        y_fake_ba = self.disa(fake_ba_)
        y_fake_ab = self.disb(fake_ab_)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_o)
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_o)
        loss_cyca = F.mean_absolute_error(fake_aba, batch_a)
        loss_cycb = F.mean_absolute_error(fake_bab, batch_b)
        chainer.report({"loss_GAN": loss_ganba, "loss_cyc": loss_cyca}, self.gen_ba)
        chainer.report({"loss_GAN": loss_ganab, "loss_cyc": loss_cycb}, self.gen_ab)
        gloss = (loss_cyca + loss_cycb) + (loss_ganab + loss_ganba)*0.05
        gloss.backward()
        gen_ba_optimizer.update()
        gen_ab_optimizer.update()
