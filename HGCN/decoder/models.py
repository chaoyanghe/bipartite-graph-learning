from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.optim as optim
from torch.nn import init

from utils import (LEARNING_RATE, WEIGHT_DECAY, HIDDEN_DIMENSIONS)


class Decoder(nn.Module):
    def __init__(self, infeat, outfeat):
        super(Decoder, self).__init__()

        hidfeat = HIDDEN_DIMENSIONS  # define the hidden layer dimension
        self.main = nn.Sequential(
            nn.Linear(infeat, hidfeat),
            nn.ReLU(),
            nn.Linear(hidfeat, outfeat)
        )

    def forward(self, input):
        output = self.main(input)
        return output


def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0)


class HGCNDecoder(object):
    def __init__(self, netG, infeat, outfeat, device):
        self.netD = Decoder(infeat, outfeat).to(device)
        self.netD.apply(_weights_init)
        self.netG = netG
        self.optimizer = optim.Adam(list(self.netG.parameters()) + list(self.netD.parameters()),
                                    lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
        self.decoder_output = 0

    def _loss(self, input, target):
        criterion = nn.MSELoss()
        loss = criterion(input, target)
        return loss

    def forward_backward(self, target, input, step, epoch, iter):
        self.optimizer.zero_grad()
        output = self.netD(input)  # input == output from GCN
        self.decoder_output = output
        loss = self._loss(output, target)
        loss.backward()
        self.optimizer.step()

        print('Step: {:01d}'.format(step),
              'Epoch: {:04d}'.format(epoch),
              'Iterations: {:04d}'.format(iter),
              'Loss: {:.4f}'.format(loss.item())
              )

    # validation
    def forward(self, target, input, iter):
        output = self.netD(input)
        self.decoder_output = output
        loss = self._loss(output, target)

        print('Iterations: {:.04d}'.format(iter),
              'Validation Loss: {:.4f}'.format(loss.item()))
