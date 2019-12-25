import torch
import numpy as np
class SeqData(torch.utils.data.Dataset):
    def __init__(self, data, size):
        self.data = data
        self.data_num = size
    def __len__(self):
        return self.data.shape[0] // (self.data_num//2)
    def __getitem__(self, idx):
        p = self.data_num // 2
        st = int(idx * p + np.random.randint(-p//2, p//2, 1))
        if st < 0:
            _r = self.data[0:st+self.data_num]
            pa = int(self.data_num-_r.shape[0])
            _r = np.pad(_r, ((pa, 0), (0, 0), (0, 0)), "constant", constant_values=-1)
        else:
            _r = self.data[st:st+self.data_num]
            if _r.shape[0] < self.data_num:
                pa = int(self.data_num-_r.shape[0])
                _r = np.pad(_r, ((0, pa), (0, 0), (0, 0)), "constant", constant_values=-1)
        return _r
