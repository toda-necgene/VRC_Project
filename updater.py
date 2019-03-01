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
        _models: list
        生成モデルA,生成モデルB,識別モデル
            len: 3
        """
        self.gen_ab = model["main"]
        self.gen_ba = model["inverse"]
        self.dis = model["dis"]
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
        dis_optimizer = self.get_optimizer("dis")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        batch_size = len(batch_a)
        _xp = chainer.backend.get_array_module(batch_a.data)
        fake_ab_ = self.gen_ab(batch_a)
        fake_aba = self.gen_ba(fake_ab_)
        fake_ba_ = self.gen_ba(batch_b)
        fake_bab = self.gen_ab(fake_ba_)
        x_batch = F.concat((fake_ab_, fake_ba_, batch_a, batch_b), axis=0)
        y_batch = self.dis(x_batch)
        wave_length = y_batch.shape[2]
        label_a = _xp.zeros([batch_size, 3, wave_length])
        label_a[:, 0] = 1.0
        label_a = label_a.astype(_xp.float32)
        label_b = _xp.zeros([batch_size, 3, wave_length])
        label_b[:, 1] = 1.0
        label_b = label_b.astype(_xp.float32)
        label_f = _xp.zeros([batch_size, 3, wave_length])
        label_f[:, 2] = 1.0
        label_f = label_f.astype(_xp.float32)
        y_label = F.concat((label_f, label_f, label_a, label_b), axis=0)
        y_fake_ab = y_batch[:batch_size]
        y_fake_ba = y_batch[batch_size:batch_size*2]
        # D update
        self.dis.cleargrads()
        self.d_loss(self.dis, y_batch, y_label).backward()
        dis_optimizer.update()
        # G update
        self.gen_ab.cleargrads()
        self.gen_ba.cleargrads()
        loss_gana = F.mean_squared_error(y_fake_ab, label_b)
        loss_ganb = F.mean_squared_error(y_fake_ba, label_a)
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
