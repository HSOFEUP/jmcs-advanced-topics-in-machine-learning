import torch
import torch.nn as nn


class AutoencoderNet(nn.Module):
    def __init__(self):
        super(AutoencoderNet, self).__init__()

        # Input size: 1 x 28 x 28 = 784 dimensions

        self.channel_num = 8
        self.encoder = nn.Sequential(
            # Calculation of conv output size: O = (W - K + 2P) / S + 1  ; O: output, W: input, K: kernel, P: padding, S: stride
            #nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1),  # 16 x 10 x 10
            #nn.MaxPool2d(kernel_size=3, stride=3, padding=1),  # 16 x 4 x 4
            #nn.ReLU(True),
            #nn.Conv2d(16, 8, kernel_size=2, stride=2)  # 8 x 2 x 2

            nn.Conv2d(1, 16, kernel_size=2, stride=2, padding=0),  # 16 x 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # 16 x 8 x 8
            nn.ReLU(True),
            nn.Conv2d(16, 12, kernel_size=3, stride=1, padding=0),  # 12 x 6 x 6
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 12 x 3 x 3
            nn.ReLU(True),
            nn.Conv2d(12, 8, kernel_size=3, stride=2, padding=1)  # 8 x 2 x 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 12, kernel_size=3, stride=2, padding=0),  # 12 x 5 x 5
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 12, kernel_size=3, stride=3),  # 12 x 15 x 15 (k=3) OR 12 x 17 x 17 (k=5)
            nn.ReLU(True),
            nn.ConvTranspose2d(12, 1, kernel_size=2, stride=2),  # 1 x 30 x 30  OR  1 x 34 x 34
            nn.ReLU(True)
            #nn.ConvTranspose2d(8, 10, kernel_size=3, stride=2, padding=0),  # 12 x 5 x 5
            #nn.ReLU(True),
            #nn.ConvTranspose2d(10, 12, kernel_size=2, stride=2, padding=0),  # 12 x 10 x 10
            #nn.ReLU(True),
            #nn.ConvTranspose2d(12, 12, kernel_size=3, stride=2, padding=2),  # 12 x 17 x 17
            #nn.ReLU(True),
            #nn.ConvTranspose2d(12, 1, kernel_size=3, stride=2, padding=2),  # 1 x 31 x 31
            #nn.ReLU(True)
        )

    def forward(self, x):
        en = self.encoder(x)

        #print('en: ')
        #print(type(en))
        #print(en.size())
        #print('-------------en')

        out = self.decoder(en)

        #print('out: ')
        #print(out.size())
        #print('-------------out')

        # crop the centeral box with the same size as input.
        x_off = int((out.shape[2] - x.shape[2]) / 2)
        y_off = int((out.shape[3] - x.shape[3]) / 2)
        out = out[:, :, x_off:x_off + x.shape[2], y_off:y_off + x.shape[3]]
        #print('out: ')
        #print(out.size())
        #print('-------------out2')
        #print (2/0)
        return out

    def get_features(self, x):
        en = self.encoder(x)
        en = en.view(-1, 32)
        return en
