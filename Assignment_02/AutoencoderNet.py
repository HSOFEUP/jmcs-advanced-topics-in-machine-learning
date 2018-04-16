

import torch
import torch.nn as nn

class AutoencoderNet(nn.Module):

    def __init__(self):
        super(AutoencoderNet, self).__init__()

        self.channel_num = 8
        self.encoder = nn.Sequential( #TODO : encoder architecture
            )

        self.decoder = nn.Sequential( #TODO : decoder architecture
            )

    def forward(self, x):
        en = self.encoder(x)
        out  = self.decoder(en)

        #crop the centeral box with the same size as input.
        x_off = int((out.shape[2]-x.shape[2])/2)
        y_off = int((out.shape[3]-x.shape[3])/2)
        out = out[:,:,x_off:x_off+x.shape[2], y_off:y_off+x.shape[3]]
        return out

    def get_features(self, x):
        en = self.encoder(x)
        en  = en.view(-1, 32)
        return en
