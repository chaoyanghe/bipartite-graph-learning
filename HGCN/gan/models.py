from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

from data.utils import (REAL_LABEL, FAKE_LABEL)


class Discriminator(nn.Module):
    def __init__(self, infeat, hidfeat, outfeat, dropout):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(infeat, hidfeat),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidfeat, outfeat),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output


def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0)


# generator is GCN
class GAN(object):
    def __init__(self, netG, infeat, hidfeat, learning_rate, weight_decay,
                 dropout, device, outfeat):  # binary classification
        """ Data from set U and V are used for generator alternatively
        """

        self.netD = Discriminator(infeat, hidfeat, outfeat, dropout).to(device)
        self.netD.apply(_weights_init)
        self.netG = netG
        self.optimizerG = optim.Adam(self.netG.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        self.optimizerD = optim.Adam(self.netD.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        self.device = device

    def _loss(self, logits, labels):
        criterion = nn.BCELoss()
        loss = criterion(logits, labels)
        # binary_cross_entropy = nn.BCELoss(logits, labels)
        return loss

    def forward_backward(self, real_data, netG_output, step, epoch, iter):
        # output from generator, and real data
        real = real_data
        fake = netG_output
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        label = torch.full((real.size()[0], 1), REAL_LABEL, device=self.device)
        self.optimizerD.zero_grad()
        output = self.netD(real)
        label.fill_(REAL_LABEL)
        lossD_real = self._loss(output, label)
        lossD_real.backward()

        # train with fake
        output = self.netD(fake.detach())  # calculate the gradient for discriminator
        label.fill_(FAKE_LABEL)
        lossD_fake = self._loss(output, label)
        lossD_fake.backward()
        lossD = lossD_real + lossD_fake
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.optimizerG.zero_grad()
        label.fill_(REAL_LABEL)
        output = self.netD(fake)
        lossG = self._loss(output, label)
        lossG.backward()
        self.optimizerG.step()

        logging.info("Step: %s, Epoch: %s, Iterations: %s, dis loss: %s, gen loss: %s" % (
            step, epoch, iter, lossD.item(), lossG.item()))

    # # validation
    # def forward(self, real_data, netG_output, iter):
    #     # output from generator, and real data
    #     real = real_data
    #     fake = netG_output
    #     ############################
    #     # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    #     ###########################
    #     # train with real
    #     label = torch.full((real.size()[0], 1), REAL_LABEL, device=self.device)
    #     output = self.netD(real)
    #     label.fill_(REAL_LABEL)
    #     lossD_real = self._loss(output, label)
    #
    #     # train with fake
    #     output = self.netD(fake.detach())  # calculate the gradient for discriminator
    #     label.fill_(FAKE_LABEL)
    #     lossD_fake = self._loss(output, label)
    #     lossD = lossD_real + lossD_fake
    #
    #     ############################
    #     # (2) Update G network: maximize log(D(G(z)))
    #     ###########################
    #     label.fill_(REAL_LABEL)
    #     output = self.netD(fake)
    #     lossG = self._loss(output, label)
    #
    #     logging.info("terations: %s, dis loss: %s, gen loss: %s" % (
    #         iter, lossD.item(), lossG.item()))
