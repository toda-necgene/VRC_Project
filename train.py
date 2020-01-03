"""
製作者:TODA
"""
import os
import shutil
import torch
import torch.nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from setting import get_setting
from core.model import Discriminator, Generator
from core.seq_dataset import SeqData
from core.voice_to_dataset_cycle import create_dataset
from core.eval import TestModel
from core.notify  import send_msg_img
from core.world_and_wave import load_wave_file, wave2world_hifi, wave2world_lofi

def w_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

def load_model(_checkpoint_dir: str,
                g_ab: Generator,
                g_ba: Generator,
                dis: Discriminator) -> bool:
    print(" [*] Reading checkpoint...")
    if not os.path.exists(_checkpoint_dir):
            os.makedirs(_checkpoint_dir)
            print(" [I] created dir.")
            g_ab.apply(w_init)
            g_ba.apply(w_init)
            dis.apply(w_init)
    else:
        print(" [I] checkpoint is found. loading file name : %s " % (_checkpoint_dir))
        if os.path.exists(_checkpoint_dir) and os.path.exists(os.path.join(_checkpoint_dir,"gen_ab.pth")):
            g_ab.load_state_dict(torch.load(os.path.join(_checkpoint_dir,"gen_ab.pth")))
            print(" [o] 'gen_ab_pth' loaded checkpoint successfully.")
        else:
            g_ab.apply(w_init)
            print(" [x] 'gen_ab_pth' is not found.")
        if os.path.exists(_checkpoint_dir) and os.path.exists(os.path.join(_checkpoint_dir,"gen_ba.pth")):
            print(" [o] 'gen_ba_pth' loaded checkpoint successfully.")
            g_ba.load_state_dict(torch.load(os.path.join(_checkpoint_dir,"gen_ba.pth")))
        else:
            g_ba.apply(w_init)
            print(" [x] 'gen_ba_pth' is not found.")
        if os.path.exists(_checkpoint_dir) and os.path.exists(os.path.join(_checkpoint_dir,"dis.pth")):
            dis.load_state_dict(torch.load(os.path.join(_checkpoint_dir,"dis.pth")))
            print(" [o] 'dis_pth' loaded checkpoint successfully.")
        else:
            dis.apply(w_init)
            print(" [x] 'dis_pth' is not found.")
        return True
    return False

def dataset_prepare_controler(args):
    """
    Parameters
    ----------
    _args: dict
    Returns
    -------
    _train_iter_a: numpy.ndarray
    _train_iter_b: numpy.ndarray
    _voice_profile: dict (float64)
        f0 parameters.
        keys: (pre_sub, pitch_rate, postad)
    _length_sp: int
    """
    _sounds_a = None
    _sounds_b = None
    if not (args["use_old_dataset"] and os.path.exists("./dataset/patch/A.npy") and os.path.exists("./dataset/patch/B.npy")):
        wave2world_function = wave2world_lofi
        if args["f0_estimation_plan"] is "harvest":
            wave2world_function = wave2world_hifi
        _sounds_a, _sounds_b = create_dataset(wave2world_function)
    else:
        # preparing training-data
        _sounds_a = np.load("./dataset/patch/A.npy")
        _sounds_b = np.load("./dataset/patch/B.npy")
    _length_sp = 200
    # f0 parameters(./vrc_project/voice_to_dataset_cycle.py L65)
    _voice_profile = np.load("./voice_profile.npz")
    if not os.path.exists(args["name_save"]):
        os.mkdir(args["name_save"])
    shutil.copy("./voice_profile.npz", args["name_save"]+"/voice_profile.npz")
    return _sounds_a, _sounds_b, _voice_profile, _length_sp
if __name__ == '__main__':
    _args = get_setting()
    test = load_wave_file("./dataset/test/test.wav") / 32767.0
    _label_sample = load_wave_file("./dataset/test/label.wav") / 32767.0    
    if  _args["wave_otp_dir"] is not "False":
        _args["wave_otp_dir"] = _args["wave_otp_dir"] + _args["model_name"] +  _args["version"]+"/"
        writer = SummaryWriter(log_dir="log/"+ _args["model_name"] +  _args["version"])
        if not os.path.exists(_args["wave_otp_dir"]):
            os.makedirs(_args["wave_otp_dir"])
    sounds_a, sounds_b, voice_profile, length_sp = dataset_prepare_controler(_args)
    g_a_to_b = Generator()
    g_b_to_a = Generator()
    dis = Discriminator()
    loss_mse = torch.nn.MSELoss()
    loss_abs = torch.nn.L1Loss()
    # to_gpu
    if _args["gpu"] >= 0:
        torch.device("cuda:0")
        loss_mse.cuda()
        loss_abs.cuda()
        g_a_to_b.cuda()
        g_b_to_a.cuda()
        dis.cuda()
    else:
        torch.device("cpu")
    Tensor = torch.cuda.FloatTensor if _args["gpu"] >= 0 else torch.Tensor
    load_model(_args["name_save"], g_a_to_b, g_b_to_a, dis)
    if os.path.isfile(_args["line_notify"]):
        with open(_args["line_notify"], "rb") as s:
                key = s.readline().decode("utf8")
    all_iter = 0
    d_optimizer = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))
    g_optimizer_ab = torch.optim.Adam(g_a_to_b.parameters(), lr=1e-4, betas=(0.5, 0.999))
    g_optimizer_ba = torch.optim.Adam(g_b_to_a.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # iterator reset
    _train_iter_a = torch.utils.data.DataLoader(SeqData(sounds_a, 200), batch_size= _args["batch_size"][0], shuffle=True)
    _train_iter_b = torch.utils.data.DataLoader(SeqData(sounds_b, 200), batch_size= _args["batch_size"][0], shuffle=True)
    # main training
    test_obj = None
    if _args["test"]:
        _args["length_sp"] = length_sp
        test_obj = TestModel(_args, [test, _label_sample, voice_profile])
    print("iters\tloss_d\tloss_g\tg_cyc\ttest\tenv")
    batch_index = 0
    for j in trange(_args["train_iteration"], leave=False):
        batch_a = torch.autograd.Variable(next(iter(_train_iter_a)).type(Tensor))
        batch_b = torch.autograd.Variable(next(iter(_train_iter_b)).type(Tensor))
        batch_a = batch_a.permute([0, 2, 1, 3])
        batch_b = batch_b.permute([0, 2, 1, 3])
        process_rate = (1 - all_iter/ _args["train_iteration"]) ** 2
        batch_al = batch_a * (torch.rand(_args["batch_size"][batch_index], 1, 1, 1) + 0.5).type(Tensor)
        batch_bl = batch_b * (torch.rand(_args["batch_size"][batch_index], 1, 1, 1) + 0.5).type(Tensor)
        batch_an = batch_al * (torch.randn(_args["batch_size"][batch_index], batch_a.shape[1], 1, 1) * process_rate * 0.002 + 1).type(Tensor)
        batch_bn = batch_bl * (torch.randn(_args["batch_size"][batch_index], batch_a.shape[1], 1, 1) * process_rate * 0.002 + 1).type(Tensor)
        eye = np.eye(4)
        y_label_TA = np.repeat(np.repeat(eye[0].reshape(1, 4, 1), _args["batch_size"][batch_index], axis=0), 9, axis=2)
        y_label_TA = Tensor(y_label_TA)
        y_label_TB = np.repeat(np.repeat(eye[1].reshape(1, 4, 1), _args["batch_size"][batch_index], axis=0), 9, axis=2)
        y_label_TB = Tensor(y_label_TB)
        y_label_FA = np.repeat(np.repeat(eye[2].reshape(1, 4, 1), _args["batch_size"][batch_index], axis=0), 9, axis=2)
        y_label_FA = Tensor(y_label_FA)
        y_label_FB = np.repeat(np.repeat(eye[3].reshape(1, 4, 1), _args["batch_size"][batch_index], axis=0), 9, axis=2)
        y_label_FB = Tensor(y_label_FB)
        # D update
        d_optimizer.zero_grad()
        fake_ab = g_a_to_b(batch_an)
        fake_ba = g_b_to_a(batch_bn)
        y_af = dis(fake_ba)
        y_bf = dis(fake_ab)
        y_at = dis(batch_an)
        y_bt = dis(batch_bn)
        loss_d_af = loss_mse(y_af, y_label_FA)
        loss_d_bf = loss_mse(y_bf, y_label_FB)
        loss_d_ar = loss_mse(y_at, y_label_TA)
        loss_d_br = loss_mse(y_bt, y_label_TB)
        d_lossa = loss_d_af + loss_d_ar
        d_lossb = loss_d_bf + loss_d_br
        d_loss = d_lossa + d_lossb
        d_lossa.backward()
        d_lossb.backward()
        d_optimizer.step()
        # G update
        g_optimizer_ab.zero_grad()
        g_optimizer_ba.zero_grad()
        fake_ba = g_b_to_a(batch_bn)
        fake_ab = g_a_to_b(batch_an)
        fake_aa = g_b_to_a(batch_an)
        fake_bb = g_a_to_b(batch_bn)
        y_fake_ba = dis(fake_ba)
        y_fake_ab = dis(fake_ab)
        fake_aba = g_b_to_a(fake_ab)
        fake_bab = g_a_to_b(fake_ba)
        loss_ganab = loss_mse(y_fake_ab, y_label_TB)
        loss_ganba = loss_mse(y_fake_ba, y_label_TA)
        loss_cycb = loss_abs(fake_bab, batch_b)
        loss_cyca = loss_abs(fake_aba, batch_a)
        loss_ideb = loss_abs(fake_bb, batch_b)
        loss_idea = loss_abs(fake_aa, batch_a)
        gloss_gan = loss_ganba + loss_ganab
        gloss_cyc = loss_cyca + loss_cycb
        gloss_ide = loss_idea + loss_ideb
        gloss = gloss_gan + gloss_cyc * 5 + gloss_ide
        gloss.backward()
        g_optimizer_ab.step()
        g_optimizer_ba.step()
        all_iter += 1
        if all_iter % 100 == 0:
            torch.save(g_a_to_b.state_dict(), _args["name_save"]+"/gen_ab.pth")
            torch.save(g_b_to_a.state_dict(), _args["name_save"]+"/gen_ba.pth")
            torch.save(dis.state_dict(), _args["name_save"]+"/dis.pth")
            print("\r", end="")
            if test_obj is not None:
                test_loss, env_loss = test_obj(all_iter, writer, g_a_to_b)
                writer.add_scalar("disc/loss", d_loss, all_iter)
                writer.add_scalar("disc/loss_a", d_lossa, all_iter)
                writer.add_scalar("disc/loss_b", d_lossb, all_iter)
                writer.add_scalar("disc/loss_a_fake", loss_d_af, all_iter)
                writer.add_scalar("disc/loss_b_fake", loss_d_bf, all_iter)
                writer.add_scalar("disc/loss_a_real", loss_d_ar, all_iter)
                writer.add_scalar("disc/loss_b_real", loss_d_br, all_iter)
                writer.add_scalar("gene/loss", gloss, all_iter)
                writer.add_scalar("gene/cyc", gloss_cyc, all_iter)
                writer.add_scalar("gene/gan", gloss_gan, all_iter)
                writer.add_scalar("gene/cyc_aba", loss_cyca, all_iter)
                writer.add_scalar("gene/gan_ab", loss_ganab, all_iter)
                writer.add_scalar("gene/cyc_bab", loss_cycb, all_iter)
                writer.add_scalar("gene/gan_ba", loss_ganba, all_iter)
                writer.add_scalar("gene/test_spectrumL2", test_loss, all_iter)
                writer.add_scalar("gene/test_envelopeL2", env_loss, all_iter)
                print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(all_iter, d_loss, gloss, gloss_cyc, test_loss, env_loss))
                if all_iter % 2000 == 0 :
                    # reset optimizer and batchsize
                    send_msg_img(key, "{} Finished. score:{}".format(all_iter, test_loss), "latest.png")
                    if batch_index < len(_args["batch_size"])-1:
                        batch_index += 1
                        _train_iter_a = torch.utils.data.DataLoader(SeqData(sounds_a, 200), batch_size=_args["batch_size"][batch_index], shuffle=True)
                        _train_iter_b = torch.utils.data.DataLoader(SeqData(sounds_b, 200), batch_size=_args["batch_size"][batch_index], shuffle=True)
                    d_optimizer = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))
                    g_optimizer_ab = torch.optim.Adam(g_a_to_b.parameters(), lr=1e-4, betas=(0.5, 0.999))
                    g_optimizer_ba = torch.optim.Adam(g_b_to_a.parameters(), lr=1e-4, betas=(0.5, 0.999))
            else:
                print("\r{}\t{:.4f}\t{:.4f}".format(all_iter, d_loss, gloss))
            
    print("[*] all_finish")
    