
"""
製作者:TODA
モデルの定義
"""
import torch
import torch.nn as nn



class Discriminator(nn.Module):
    """
        識別側ネットワーク
        4層CNN(Conv1d-leaky_relu)
        最終層のみチャンネルを32づつ4グループにして各グループで最大値をとる
        出力の多様化を促し、Gとの収束スピードのバランスを保つ
    """
    def __init__(self):
        """
        モデル定義
        """
        super(Discriminator, self).__init__()
        self.model =nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(1025, 512, 6, stride=2, padding=2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv1d(512, 256, 6, stride=2, padding=2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv1d(256, 256, 10, stride=5, padding=0)),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, 9, padding=4))
        self.last_pooing = nn.MaxPool1d(128, stride=128) 
    def forward(self, x):
        """
        [n, 1025, 200, 1] -> [n, 4, 9]
        """
        return self.last_pooing(self.model(x[:,:,:,0]).permute([0, 2, 1])).permute([0, 2, 1])
class Generator(nn.Module):
    """
        学習用生成側ネットワーク
    """
    def __init__(self):
        """
        レイヤー定義
        """
        super(Generator, self).__init__()
        self.model_encode = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1025, 256, (4, 1), stride=(4, 1))),
            nn.LeakyReLU()
        )
        self.model_twist=[
            nn.Sequential(
                nn.ZeroPad2d((0, 0, 20, 0)),
                nn.utils.spectral_norm(nn.Conv2d(256, 256, (21, 1), stride=(1, 1))),
                nn.LeakyReLU(),
                nn.ZeroPad2d((0, 0, 20, 0)),
                nn.utils.spectral_norm(nn.Conv2d(256, 256, (21, 1), stride=(1, 1))),
                nn.LeakyReLU()),
            nn.Sequential(
                nn.ZeroPad2d((0, 0, 20, 0)),
                nn.utils.spectral_norm(nn.Conv2d(256, 256, (21, 1), stride=(1, 1))),
                nn.LeakyReLU(),
                nn.ZeroPad2d((0, 0, 20, 0)),
                nn.utils.spectral_norm(nn.Conv2d(256, 256, (21, 1), stride=(1, 1))),
                nn.LeakyReLU()),
            nn.Sequential(
                nn.ZeroPad2d((0, 0, 20, 0)),
                nn.utils.spectral_norm(nn.Conv2d(256, 256, (21, 1), stride=(1, 1))),
                nn.LeakyReLU(),
                nn.ZeroPad2d((0, 0, 20, 0)),
                nn.utils.spectral_norm(nn.Conv2d(256, 256, (21, 1), stride=(1, 1))),
                nn.LeakyReLU()),
            nn.Sequential(
                nn.ZeroPad2d((0, 0, 20, 0)),
                nn.utils.spectral_norm(nn.Conv2d(256, 256, (21, 1), stride=(1, 1))),
                nn.LeakyReLU(),
                nn.ZeroPad2d((0, 0, 20, 0)),
                nn.utils.spectral_norm(nn.Conv2d(256, 256, (21, 1), stride=(1, 1))),
                nn.LeakyReLU())]
        self.model_decode =nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(256, 1025, (4, 1), stride=(4, 1))),
        )
        self.shapes= [
            [-1, 256, 5, 10],
            [-1, 256, 10, 5],
            [-1, 256, 25, 2],
            [-1, 256, 50, 1]
        ]
    def cuda(self):
        super(Generator, self).cuda()
        for b in self.model_twist:
            b.cuda()
    def forward(self, x):
        """
            [n, 1025, 200, 1] -> [n, 1025, 200, 1]
        """
        _y = self.model_encode(x)
        # 1-3 layers
        for i, block in enumerate(self.model_twist):
            _y = _y.view(*self.shapes[i])
            _h = block(_y)
            _y = _h + _y
        y = self.model_decode(_y)
        return y