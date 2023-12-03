# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# ESTCNN
class ESTCNN(nn.Module):
    def __init__(self, channel, length=3000):
        super(ESTCNN, self).__init__()
        self.lenlist = {250: 7616, 500: 17408, 1000: 36992, 1500: 56576,
                        2000: 75072, 2500: 94656, 3000: 114240}
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channel, (1, 3)),
            nn.Conv2d(channel, channel, (1, 3)),
            nn.Conv2d(channel, channel, (1, 3)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, 2 * channel, (1, 3)),
            nn.Conv2d(2 * channel, 2 * channel, (1, 3)),
            nn.Conv2d(2 * channel, 2 * channel, (1, 3)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * channel, 4 * channel, (1, 3)),
            nn.Conv2d(4 * channel, 4 * channel, (1, 3)),
            nn.Conv2d(4 * channel, 4 * channel, (1, 3)),
            nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 7))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Flatten(),
            nn.Linear(self.lenlist[length], 512),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.classifier(x)
        return out


from torchsummary import summary
import hiddenlayer as hl
import netron
import tensorwatch as tw

"""print network parameters and usage"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input eeg signal
    n_channel = 17
    n_sample = 3000
    model = ESTCNN(16).to(device)
    print(model)
    summary(model, (1, n_channel, n_sample))  # Success!

    input = torch.randn(1, 1, n_channel, n_sample).to(device)
    out = model(input)
    print(out.shape)  # [1, 4]

    print("Successfully.")
