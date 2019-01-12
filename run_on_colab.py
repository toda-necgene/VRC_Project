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
    # 中間状況確認
    tester = _generate_test_callback(
        os.path.join("dataset", "test", "*.wav"),
        "waves",
        util.generate_f0_transfer("./voice_profile.npy"))

    argparser = ArgumentParser()
    argparser.add_argument("-c", "--checkpoint",
                            type=str, default="gs://colab_bucket",
                            help="checkpoint address, default=gs://colab_bucket")
    argparser.add_argument("-b", "--batchsize",
                            type=int, default=128,
                            help="train batch size")
    argparser.add_argument("-w", "--weight",
                            type=float, default=100.0,
                            help="gan cycle weight")
    argparser.add_argument("-i", "--iteration",
                            type=int, default=100000,
                            help="train iterations")

    args = argparser.parse_args()

    # データセット準備
    input_a = os.path.join(".", "dataset", "train", "A.npy")
    input_b = os.path.join(".", "dataset", "train", "B.npy")
    data_a, data_b = _get_dataset(input_a, input_b)

    model = w2w(args.batchsize)
    with CycleGANFactory(model) \
            .cycle_weight(args.weight) \
            .optimizer("Adam", 4e-6, {"beta1": 0.5, "beta2": 0.999}) \
            .summary("console") \
            .test(tester) \
            .hardware("colab,tpu") \
            .checkpoint(args.checkpoint) \
            .input(data_a, data_b) \
            .build() as net:
        net.train(args.iteration)

