from model import Model as w2w
from converter import Converter
from waveplot import WavePlot
import util
import os
from glob import glob
import wave
import numpy as np
from datetime import datetime
from cyclegan_factory import CycleGANFactory

if __name__ == '__main__':
    test_files = os.path.join("dataset", "test", "*.wav")
    test_output_dir = "waves"
    os.makedirs(test_output_dir)
    f0_transfer = util.generate_f0_transfer("./voice_profile.npy")
    sample_size = 2
    def save_converting_test_files(net, epoch, iteration, period):
        converter = Converter(net, f0_transfer).convert
        savefig = WavePlot().savefig
        for file in glob(test_files):
            basename = os.path.basename(file)

            testdata = util.isread(file)
            converted, _ = converter(testdata)

            converted_norm = converted.copy().astype(np.float32) / 32767.0
            im = util.fft(converted_norm)
            ins = np.transpose(im, (1, 0))

            path = os.path.join(
                test_output_dir,
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

    data_dir = os.path.join(".", "dataset", "train")
    data_a = np.load(os.path.join(data_dir, "A.npy"))
    data_a = data_a.reshape(list(data_a.shape) + [1])
    data_b = np.load(os.path.join(data_dir, "B.npy"))
    data_b = data_b.reshape(list(data_b.shape) + [1])
    data_size = min(data_a.shape[0], data_b.shape[0])
    dataset = [data_a[:data_size], data_b[:data_size]]
    # dataset = list(map(lambda data: data.reshape(list(data.shape) + [1]),
    #     map(lambda d: np.load(os.path.join(data_dir, d)), ["A.npy", "B.npy"])))
    # data_size = min(dataset[0].shape[0], dataset[1].shape[0])
    # dataset = list(map(lambda data: data[:data_size], dataset))

    model = w2w(1)
    name = "_".join([model.name, model.version, "tpu"])
    net = CycleGANFactory(model) \
            .cycle_weight(100.00) \
            .optimizer("Adam", 4e-6, {"beta1": 0.5, "beta2": 0.999}) \
            .summary("console") \
            .test(save_converting_test_files) \
            .hardware("colab,tpu") \
            .checkpoint(os.path.join(".", "trained_model", name)) \
            .input(dataset[0], dataset[1]) \
            .build()

    net.train(100000)

