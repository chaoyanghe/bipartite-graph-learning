from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.optim as optim
import logging

from torch.nn import init


class Decoder(nn.Module):
    def __init__(self, infeat, hidfeat, outfeat, dropout):
        super(Decoder, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(infeat, hidfeat),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=True),
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
    def __init__(self, netG, infeat, outfeat, hidfeat, learning_rate, weight_decay, dropout, device):
        self.netD = Decoder(infeat, hidfeat, outfeat, dropout).to(device)
        self.netD.apply(_weights_init)
        self.netG = netG
        self.optimizer = optim.Adam(list(self.netG.parameters()) + list(self.netD.parameters()),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
        self.decoder_output = 0

    def _loss(self, input, target):
        criterion = nn.MSELoss()
        loss = criterion(input, target)
        return loss

    def forward_backward(self, target, input, step, epoch, iter):
        self.optimizer.zero_grad()
        output = self.netD(input)
        self.decoder_output = output
        loss = self._loss(output, target)
        loss.backward()
        self.optimizer.step()

        logging.info('Step: {:01d}'.format(step),
                     'Epoch: {:04d}'.format(epoch),
                     'Iterations: {:04d}'.format(iter),
                     'Loss: {:.4f}'.format(loss.item())
                     )

    # # validation
    # def forward(self, target, input, iter):
    #     output = self.netD(input)
    #     self.decoder_output = output
    #     loss = self._loss(output, target)
    #
    #     logging.info('Iterations: {:.04d}'.format(iter),
    #                  'Validation Loss: {:.4f}'.format(loss.item()))
