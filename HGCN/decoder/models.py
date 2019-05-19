from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
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


class Encoder(nn.Module):
    def __init__(self, infeat, hidfeat, outfeat, dropout):
        super(Encoder, self).__init__()
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


class HGCNDecoder(nn.Module):
    def __init__(self, netG, infeat, outfeat, encoder_hidfeat, decoder_hidfeat, learning_rate, weight_decay, dropout,
                 device):
        super(HGCNDecoder, self).__init__()
        self.encoder = Encoder(infeat, encoder_hidfeat, outfeat, dropout).to(device)
        # These two layers are to generate the mean and standard from the hidden output from encoder
        self.dense_mean = nn.Linear(outfeat, outfeat).to(device)
        self.dense_stddev = nn.Linear(outfeat, outfeat).to(device)
        self.decoder = Decoder(outfeat, decoder_hidfeat, outfeat, dropout).to(device)
        self.encoder.apply(_weights_init)
        self.decoder.apply(_weights_init)
        self.netG = netG
        self.optimizer = optim.Adam(
            list(self.netG.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters())
            + list(self.dense_mean.parameters()) + list(self.dense_stddev.parameters()),
            lr=learning_rate, weight_decay=weight_decay)
        self.device = device
        self.mean = torch.FloatTensor()
        self.exp_stddev = torch.FloatTensor()

    def _latent_loss(self):
        mean_sq = self.mean * self.mean
        exp_stddev_sq = self.exp_stddev * self.exp_stddev
        return 0.5 * torch.mean(mean_sq + exp_stddev_sq - torch.log(exp_stddev_sq) - 1)

    def _construct_loss(self, construct, target):
        criterion = nn.MSELoss()
        loss = criterion(construct, target)
        return loss

    def _sample_latent(self, hidden_vec):
        """
        function to generate latent vector
        :param hidden_vec:
        :return:
        """
        self.mean = self.dense_mean(hidden_vec)
        stddev = self.dense_stddev(hidden_vec)
        # whether to add exponential to the output
        # will make the stddev all positive
        self.exp_stddev = torch.exp(stddev)
        std = torch.randn(self.exp_stddev.size()).to(self.device)
        return self.mean + self.exp_stddev * torch.autograd.Variable(std, requires_grad=False)

    def forward_backward(self, target, input, step, epoch, iter):
        self.optimizer.zero_grad()
        hidden_vec = self.encoder(input)
        latent_vec = self._sample_latent(hidden_vec)
        construct_vec = self.decoder(latent_vec)
        construct_loss = self._construct_loss(construct_vec, target)
        latent_loss = self._latent_loss()
        loss = construct_loss + latent_loss
        loss.backward()
        self.optimizer.step()

        logging.info("Step: %s, Epoch: %s, Iterations: %s, loss: %s" % (
            step, epoch, iter, loss.item()))

    # extract the latent vector for input
    def forward(self, input):
        hidden_vec = self.encoder(input)
        latent_vec = self._sample_latent(hidden_vec)
        return latent_vec
# class HGCNDecoder(object):
#     def __init__(self, netG, infeat, outfeat, hidfeat, learning_rate, weight_decay, dropout, device):
#         self.netD = Decoder(infeat, hidfeat, outfeat, dropout).to(device)
#         self.netD.apply(_weights_init)
#         self.netG = netG
#         self.optimizer = optim.Adam(list(self.netG.parameters()) + list(self.netD.parameters()),
#                                     lr=learning_rate,
#                                     weight_decay=weight_decay)
#
#     # here is simple MSE loss, probably has different loss
#     def _loss(self, input, target):
#         criterion = nn.MSELoss()
#         loss = criterion(input, target)
#         return loss
#
#     def forward_backward(self, target, input, step, epoch, iter):
#         self.optimizer.zero_grad()
#         output = self.netD(input)
#         loss = self._loss(output, target)
#         loss.backward()
#         self.optimizer.step()
#
#         logging.info("Step: %s, Epoch: %s, Iterations: %s, loss: %s" % (
#             step, epoch, iter, loss.item()))
#
#     def forward(self, input):
#         output = self.netD(input)
#         return output
#     # # validation
#     # def forward(self, target, input, iter):
#     #     output = self.netD(input)
#     #     self.decoder_output = output
#     #     loss = self._loss(output, target)
#     #
#     #     logging.info('Iterations: {:.04d}'.format(iter),
#     #                  'Validation Loss: {:.4f}'.format(loss.item()))
