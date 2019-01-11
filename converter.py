import util
import numpy as np
import time


class Converter():
    def __init__(self, gan, f0_transfer):
        self.gan = gan
        self.f0_transfer = f0_transfer

    def convert(self, input, term=4096):
        conversion_start_time = time.time()
        executing_times = (input.shape[0] - 1) // term + 1
        otp = np.array([], dtype=np.int16)

        for t in range(executing_times):
            # Preprocess

            # Padiing
            end_pos = term * t + (input.shape[0] % term)
            resorce = input[max(0, end_pos - term):end_pos]
            r = max(0, term - resorce.shape[0])
            if r > 0:
                resorce = np.pad(resorce, (r, 0), 'constant')
            # FFT
            f0, resource, ap = util.encode((resorce / 32767).astype(np.float))

            # IFFT
            f0 = self.f0_transfer(f0)
            result_wave = util.decode(f0, self.gan.a_to_b(resource),
                                      ap) * 32767

            result_wave_fixed = np.clip(result_wave, -32767.0, 32767.0)[:term]
            result_wave_int16 = result_wave_fixed.reshape(-1).astype(np.int16)

            #adding result
            otp = np.append(otp, result_wave_int16)

        h = otp.shape[0] - input.shape[0]
        if h > 0:
            otp = otp[h:]

        return otp, time.time() - conversion_start_time

