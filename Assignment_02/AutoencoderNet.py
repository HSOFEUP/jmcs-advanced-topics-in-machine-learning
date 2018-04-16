import torch
import torch.nn as nn


class AutoencoderNet(nn.Module):
    def __init__(self):
        super(AutoencoderNet, self).__init__()

        # Input size: 1 x 28 x 28 = 784 dimensions

        # Calculation of output size: O = (W - K + 2P) / S + 1  ; O: output, W: input, K: kernel, P: padding, S: stride

        self.channel_num = 8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=2),  # 16 x 12 x 12
            nn.MaxPool2d(kernel_size=3, stride=3),  # 16 x 4 x 4
            nn.Conv2d(16, 8, kernel_size=2, stride=2)  # 8 x 2 x 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 12, kernel_size=3, stride=2),  # 12 x 5 x 5
            nn.ConvTranspose2d(12, 12, kernel_size=5, stride=3),  # 12 x 15 x 15
            nn.ConvTranspose2d(12, 1, kernel_size=2, stride=2),  # 1 x 30 x 30
        )

    def forward(self, x):
        en = self.encoder(x)
        out = self.decoder(en)

        # crop the centeral box with the same size as input.
        x_off = int((out.shape[2] - x.shape[2]) / 2)
        y_off = int((out.shape[3] - x.shape[3]) / 2)
        out = out[:, :, x_off:x_off + x.shape[2], y_off:y_off + x.shape[3]]
        return out

    def get_features(self, x):
        en = self.encoder(x)
        en = en.view(-1, 32)
        return en
