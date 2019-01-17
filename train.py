import os
from glob import glob
from datetime import datetime

import wave
import numpy as np

from VRC.cyclegan_factory import CycleGANFactory
from VRC.waver import Waver
from VRC.model import Model as w2w

import VRC.log as log



def _get_dataset(a, b):
    data_a = np.load(a)
    data_a = data_a.reshape(list(data_a.shape) + [1])

    data_b = np.load(b)
    data_b = data_b.reshape(list(data_b.shape) + [1])

    data_size = min(data_a.shape[0], data_b.shape[0])

    return data_a[:data_size], data_b[:data_size]

    # dataset = list(map(lambda data: data.reshape(list(data.shape) + [1]),
    #     map(lambda d: np.load(os.path.join(data_dir, d)), ["A.npy", "B.npy"])))
    # data_size = min(dataset[0].shape[0], dataset[1].shape[0])
    # dataset = list(map(lambda data: data[:data_size], dataset))

import original.util as util
from VRC.converter import Converter
from VRC.waveplot import WavePlot
from VRC.waver import Waver
def _generate_test_callback(files, output_dir, f0_transfer):
    os.makedirs(output_dir, exist_ok=True)

    tester = []
    waver = Waver()
    for file in glob(files):
        f0, _, ap, psp = waver.encode(file)
        tester.append({
            'f0': f0_transfer(f0),
            'ap': ap,
            'psp': psp,
            'format': os.path.basename(file) + '_%d_%s',
        })
    savefig = WavePlot().savefig

    def save_converting_test_files(net, epoch, iteration, period):
        for d in tester:
            path = d['format'] % (epoch // net.batch_size, datetime.now().strftime("%m-%d_%H-%M-%S"))
            psp = net.a_to_b(d['psp'])
            data = waver.decode(d['f0'], d['ap'], psp=psp, file=path + '.wav')

            savefig(path + ".png", [psp, data])

    return save_converting_test_files


from argparse import ArgumentParser

if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("-a", "--input-a",
                            type=str, default=os.path.join(".", "dataset", "train", "A.npy"),
                            help="A of dataset")
    argparser.add_argument("-b", "--input-b",
                            type=str, default=os.path.join(".", "dataset", "train", "B.npy"),
                            help="B of dataset")
    argparser.add_argument("-c", "--checkpoint",
                            type=str, default="gs://colab_bucket",
                            help="checkpoint address, default=gs://colab_bucket")
    argparser.add_argument("-s", "--batchsize",
                            type=int, default=8,
                            help="train batch size")
    argparser.add_argument("-w", "--weight",
                            type=float, default=100.0,
                            help="gan cycle weight")
    argparser.add_argument("-i", "--iteration",
                            type=int, default=100000,
                            help="train iterations")
    argparser.add_argument("-v", "--voice-profile",
                            type=str, default="voice_profile.npy",
                            help="data of voice profile (f0a mean, f0b mean, rate of stdev)")
    argparser.add_argument("-t", "--test-data",
                            type=str, default=os.path.join("dataset", "test", "*.wav"),
                            help="test data, be able wildcar (compiant of glob)")
    argparser.add_argument("-d", "--test-dir",
                            type=str, default="waves",
                            help="output converted test data")
    argparser.add_argument("-p", "--processor",
                            type=str, default="colab,tpu",
                            help="specification of hardware")
                            
    args = argparser.parse_args()

    waver = Waver(block=4096, fs=16000, fft_size=1024, bit_rate=16)
    # バッチファイル作成
    if not args.input_a.endswith(".npy"):
        f0_a, sp, ap, psp = waver.encode(args.input_a, filter_silent=True)
        args.input_a = os.path.dirname(args.input_a) + ".npy"
        np.save(args.input_a, psp)
        log.i("save A")
        
    if not args.input_b.endswith(".npy"):
        f0_b, sp, ap, psp = waver.encode(args.input_b, filter_silent=True)
        args.input_b = os.path.dirname(args.input_b) + ".npy"
        np.save(args.input_b, psp)
        log.i("save B")

    if  'f0_a' in locals() and 'f0_b' in locals():
        profile = waver.get_f0_transfer_params(f0_a, f0_b)
        np.save(args.voice_profile, profile)
        log.i("save profile")

    # 中間状況確認
    f0_mean_diff, f0_std_rate = np.load(args.voice_profile)
    tester = _generate_test_callback(
        args.test_data,
        args.test_dir,
        waver.generate_f0_transfer(f0_mean_diff, f0_std_rate))

    # データセット準備
    data_a, data_b = _get_dataset(args.input_a, args.input_b)

    model = w2w(args.batchsize)
    with CycleGANFactory(model) \
            .cycle_weight(args.weight) \
            .optimizer("Adam", 4e-6, {"beta1": 0.5, "beta2": 0.999}) \
            .summary("console") \
            .test(tester) \
            .hardware(args.processor) \
            .checkpoint(args.checkpoint) \
            .input(data_a, data_b) \
            .build() as net:
        net.train(args.iteration)

