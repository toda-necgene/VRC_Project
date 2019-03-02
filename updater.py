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
    def d_loss(self, dis, y_batch, y_label):
        """
        識別側の目的関数
        Parameters
        ----------
        dis:識別モデル
        y_batch:識別結果
        y_label:目標値
        Returns
        -------
        loss:損失
        """
        loss = F.mean_squared_error(y_batch, y_label)
        chainer.report({"loss": loss}, dis)
        return loss
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
        x_batcha = F.concat((fake_ba_, batch_a), axis=0)
        x_batchb = F.concat((fake_ab_, batch_b), axis=0)
        y_batcha = self.disa(x_batcha)
        y_batchb = self.disb(x_batchb)
        wave_length = y_batcha.shape[2]
        y_label = F.concat((_xp.zeros([batch_size, 1, wave_length], dtype="float32"), _xp.ones([batch_size, 1, wave_length], dtype="float32")), axis=0)
        y_fake_ab = y_batchb[:batch_size]
        y_fake_ba = y_batcha[:batch_size]
        # D update
        self.disa.cleargrads()
        self.d_loss(self.disa, y_batcha, y_label).backward()
        disa_optimizer.update()
        self.disb.cleargrads()
        self.d_loss(self.disb, y_batchb, y_label).backward()
        disb_optimizer.update()
        # G update
        self.gen_ab.cleargrads()
        self.gen_ba.cleargrads()
        loss_gana = F.mean_squared_error(y_fake_ab, y_label[batch_size:])
        loss_ganb = F.mean_squared_error(y_fake_ba, y_label[batch_size:])
        cyca = F.absolute_error(fake_aba, batch_a)
        loss_cyca = F.sum(cyca)/_xp.count_nonzero(cyca.data)
        cycb = F.absolute_error(fake_bab, batch_b)
        loss_cycb = F.sum(cycb)/_xp.count_nonzero(cycb.data)
        chainer.report({"loss_GAN": loss_gana, "loss_cyc": loss_cyca}, self.gen_ab)
        chainer.report({"loss_GAN": loss_ganb, "loss_cyc": loss_cycb}, self.gen_ba)
        gloss = loss_gana + loss_ganb + (loss_cyca + loss_cycb) * 5
        gloss.backward()
        gen_ab_optimizer.update()
        gen_ba_optimizer.update()
