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
        self.gen_en = model["encoder"]
        self.gen_de = model["decoder"]
        self.gen_ab1 = model["main"]
        self.gen_ba1 = model["inverse"]
        self.gen_ab2 = model["main2"]
        self.gen_ba2 = model["inverse2"]
        self.disa = model["disa"]
        self.disb = model["disb"]
        self.max_iteration = max_itr
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
    def update_core(self):
        gen_en_optimizer = self.get_optimizer("gen_en")
        gen_de_optimizer = self.get_optimizer("gen_de")
        gen_ab_optimizer1 = self.get_optimizer("gen_ab1")
        gen_ba_optimizer1 = self.get_optimizer("gen_ba1")
        gen_ab_optimizer2 = self.get_optimizer("gen_ab2")
        gen_ba_optimizer2 = self.get_optimizer("gen_ba2")
        disa_optimizer = self.get_optimizer("disa")
        disb_optimizer = self.get_optimizer("disb")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        batch_size = len(batch_a)
        _xp = chainer.backend.get_array_module(batch_a.data)
        fake_ab_1_p = self.gen_ab1(self.gen_en(batch_a))
        fake_ba_1_p = self.gen_ba1(self.gen_en(batch_b))
        fake_ab_2_p = self.gen_ab2(fake_ab_1_p)
        fake_ba_2_p = self.gen_ba2(fake_ba_1_p)
        fake_ab_12 = self.gen_de(F.concat([fake_ab_1_p, fake_ab_2_p], axis=0))
        fake_ba_12 = self.gen_de(F.concat([fake_ba_1_p, fake_ba_2_p], axis=0))
        fake_aba1 = self.gen_de(self.gen_ba1(fake_ab_1_p))
        fake_bab1 = self.gen_de(self.gen_ab1(fake_ba_1_p))
        fake_aba2 = self.gen_de(self.gen_ba2(self.gen_ba1(fake_ab_2_p)))
        fake_bab2 = self.gen_de(self.gen_ab2(self.gen_ab1(fake_ba_2_p)))
        # D update
        self.disa.cleargrads()
        self.disb.cleargrads()
        batch_mixed_a = F.concat([batch_a, fake_ba_12], axis=0)
        batch_mixed_b = F.concat([batch_b, fake_ab_12], axis=0)
        y_a = self.disa(batch_mixed_a)
        y_b = self.disb(batch_mixed_b)
        wave_length = y_a.shape[2]
        y_label_o = _xp.ones([batch_size, 1, wave_length], dtype="float32")
        y_label_z = _xp.zeros([batch_size, 1, wave_length], dtype="float32")
        y_label = F.concat([y_label_o, y_label_z, y_label_z], axis=0)
        loss_d_a = F.mean_squared_error(y_a, y_label)
        loss_d_b = F.mean_squared_error(y_b, y_label)
        loss_d_a.backward()
        loss_d_b.backward()
        chainer.report({"loss": loss_d_a}, self.disa)
        chainer.report({"loss": loss_d_b}, self.disb)
        disa_optimizer.update()
        disb_optimizer.update()
        # G update
        self.gen_ba1.cleargrads()
        self.gen_ab1.cleargrads()
        self.gen_ba2.cleargrads()
        self.gen_ab2.cleargrads()
        self.gen_en.cleargrads()
        self.gen_de.cleargrads()
        fake_target_label = F.concat([y_label_o, y_label_o], axis=0)
        y_fake_ba = self.disa(fake_ba_12)
        y_fake_ab = self.disb(fake_ab_12)
        loss_ganab = F.mean_squared_error(y_fake_ab, fake_target_label)
        loss_ganba = F.mean_squared_error(y_fake_ba, fake_target_label)
        loss_cyca = F.mean_absolute_error(fake_aba1, batch_a) + F.mean_absolute_error(fake_aba2, batch_a)
        loss_cycb = F.mean_absolute_error(fake_bab1, batch_b) + F.mean_absolute_error(fake_bab2, batch_b)
        chainer.report({"GAN": loss_ganba}, self.gen_ba2)
        chainer.report({"GAN": loss_ganab, "CYC": loss_cyca+loss_cycb}, self.gen_ab2)
        gloss = (loss_cyca + loss_cycb) * 10 + (loss_ganab+loss_ganba) * 0.5
        gloss.backward()
        gen_ba_optimizer2.update()
        gen_ab_optimizer2.update()
        gen_ba_optimizer1.update()
        gen_ab_optimizer1.update()
        gen_en_optimizer.update()
        gen_de_optimizer.update()
