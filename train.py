from model import Model as w2w
import util
import os
from glob import glob
import wave
import numpy as np
from datetime import datetime
from cyclegan_factory import CycleGANFactory


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

from converter import Converter
from waveplot import WavePlot
def _generate_test_callback(files, output_dir, f0_transfer):
    os.makedirs(output_dir, exist_ok=True)
    sample_size = 2
    def save_converting_test_files(net, epoch, iteration, period):
        converter = Converter(net, f0_transfer).convert
        savefig = WavePlot().savefig
        for file in glob(files):
            basename = os.path.basename(file)

            testdata = util.isread(file)
            converted, _ = converter(testdata)

            converted_norm = converted.copy().astype(np.float32) / 32767.0
            im = util.fft(converted_norm)
            ins = np.transpose(im, (1, 0))

            path = os.path.join(
                output_dir,
                "%s_%d_%s" % (basename, epoch // net.batch_size,
                                datetime.now().strftime("%m-%d_%H-%M-%S")))

            savefig(path + ".png", [ins, converted_norm])

            #saving fake waves
            voiced = converted.astype(np.int16)[800:156000]

            ww = wave.open(path + ".wav", 'wb')
            ww.setnchannels(1)
            ww.setsampwidth(sample_size)
            ww.setframerate(16000)
            ww.writeframes(voiced.reshape(-1).tobytes())
            ww.close()

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
    
    # 中間状況確認
    tester = _generate_test_callback(
        args.test_data,
        args.test_dir,
        util.generate_f0_transfer(args.voice_profile))

    # データセット準備
    data_a, data_b = _get_dataset(args.input_a, args.input_b)

    model = w2w(args.batchsize)
    with CycleGANFactory(model) \
            .cycle_weight(args.weight) \
            .optimizer("Adam", 4e-6, {"beta1": 0.5, "beta2": 0.999}) \
            .summary("console") \
            .test(tester) \
            .hardware(args.hardware) \
            .checkpoint(args.checkpoint) \
            .input(data_a, data_b) \
            .build() as net:
        net.train(args.iteration)

