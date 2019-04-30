from __future__ import division
from __future__ import print_function

import torch

from data.utils import load_data
from decoder.models import HGCNDecoder
from gan.models import GAN

from pygcn.models import GCN
from utils import (EPOCHS, VALIDATE_ITER)

"""Single layer for adversarial loss"""


class AdversarialHGCNLayer(object):
    def __init__(self, featuresU_dimensions, featuresV_dimensions, device):
        self.gcn_explicit = GCN(featuresV_dimensions, featuresU_dimensions).to(device)
        self.gcn_implicit = GCN(featuresU_dimensions, featuresV_dimensions).to(device)
        self.gcn_merge = GCN(featuresV_dimensions, featuresU_dimensions).to(device)
        self.gcn_opposite = GCN(featuresU_dimensions, featuresV_dimensions).to(device)
        self.gan_explicit = GAN(self.gcn_explicit, featuresU_dimensions, device, outfeat=1)
        self.gan_implicit = GAN(self.gcn_implicit, featuresV_dimensions, device, outfeat=1)
        self.gan_merge = GAN(self.gcn_merge, featuresU_dimensions, device, outfeat=1)
        self.gan_opposite = GAN(self.gcn_opposite, featuresV_dimensions, device, outfeat=1)

    def relation_learning(self, Uinput, Vinput, adjU, adjV):
        # explicit
        print('Step 1: Explicit relation learning')
        for i in range(EPOCHS):
            gcn_explicit_output = self.gcn_explicit(Vinput, adjU)
            self.gan_explicit.forward_backward(Uinput, gcn_explicit_output, step=1, epoch=i)

        # implicit
        print('Step 2: Implicit relation learning')
        for i in range(EPOCHS):
            gcn_implicit_output = self.gcn_implicit(Uinput, adjV)
            self.gan_implicit.forward_backward(Vinput, gcn_implicit_output, step=2, epoch=i)

        # merge
        print('Step 3: Merge relation learning')
        for i in range(EPOCHS):
            gcn_merge_output = self.gcn_merge(gcn_implicit_output.detach(), adjU)
            self.gan_merge.forward_backward(gcn_explicit_output.detach(), gcn_merge_output, step=3, epoch=i)

        # opposite
        print('Step 4: Opposite relation learning')
        for i in range(EPOCHS):
            gcn_opposite_output = self.gcn_opposite(gcn_merge_output.detach(), adjV)
            self.gan_opposite.forward_backward(gcn_implicit_output.detach(), gcn_opposite_output, step=4, epoch=i)


"""Single layer for decoder layer"""


class DecoderGCNLayer(object):
    def __init__(self, featuresU_dimensions, featuresV_dimensions, device):
        """For decoder layer, we can define any output dimension from GCN layer"""
        dimenson_output_from_gcn = 2
        self.gcn_explicit = GCN(featuresV_dimensions, dimenson_output_from_gcn).to(device)
        self.gcn_implicit = GCN(featuresU_dimensions, dimenson_output_from_gcn).to(device)
        self.gcn_merge = GCN(featuresV_dimensions, dimenson_output_from_gcn).to(device)
        self.gcn_opposite = GCN(featuresU_dimensions, dimenson_output_from_gcn).to(device)
        self.decoder_explicit = HGCNDecoder(self.gcn_explicit, dimenson_output_from_gcn, featuresU_dimensions, device)
        self.decoder_implicit = HGCNDecoder(self.gcn_explicit, dimenson_output_from_gcn, featuresV_dimensions, device)
        self.decoder_merge = HGCNDecoder(self.gcn_explicit, dimenson_output_from_gcn, featuresU_dimensions, device)
        self.decoder_opposite = HGCNDecoder(self.gcn_explicit, dimenson_output_from_gcn, featuresV_dimensions, device)


    def relation_learning(self, Uinput, Vinput, adjU, adjV):
        # explicit
        print('Step 1: Explicit relation learning')
        for i in range(EPOCHS):  # separate the data into batches
            gcn_explicit_output = self.gcn_explicit(Vinput, adjU)
            self.decoder_explicit.forward_backward(Uinput, gcn_explicit_output, step=1, epoch=i)
            if i % VALIDATE_ITER == 0:
                gcn_explicit_output = self.gcn_explicit(Vinput, adjU)
                self.decoder_explicit.forward(Uinput, gcn_explicit_output)

        # implicit
        print('Step 2: Implicit relation learning')
        for i in range(EPOCHS):
            gcn_implicit_output = self.gcn_implicit(Uinput, adjV)
            self.decoder_implicit.forward_backward(Vinput, gcn_implicit_output, step=2, epoch=i)
            if i % VALIDATE_ITER == 0:
                gcn_implicit_output = self.gcn_implicit(Uinput, adjV)
                self.decoder_implicit.forward(Vinput, gcn_implicit_output)

        # merge
        print('Step 3: Merge relation learning')

        # the output is the output from decoder, since we can set different dimensions of the output from GCN
        gcn_implicit_output = self.decoder_implicit.decoder_output
        gcn_explicit_output = self.decoder_explicit.decoder_output
        for i in range(EPOCHS):
            gcn_merge_output = self.gcn_merge(gcn_implicit_output.detach(), adjU)
            self.decoder_merge.forward_backward(gcn_explicit_output.detach(), gcn_merge_output, step=3, epoch=i)
            if i % VALIDATE_ITER == 0:
                gcn_merge_output = self.gcn_merge(gcn_implicit_output.detach(), adjU)
                self.decoder_merge.forward(gcn_explicit_output.detach(), gcn_merge_output)

        # opposite
        print('Step 4: Opposite relation learning')
        gcn_merge_output = self.decoder_merge.decoder_output
        for i in range(EPOCHS):
            gcn_opposite_output = self.gcn_opposite(gcn_merge_output.detach(), adjV)
            self.decoder_opposite.forward_backward(gcn_implicit_output.detach(), gcn_opposite_output, step=4, epoch=i)
            if i % VALIDATE_ITER == 0:
                gcn_opposite_output = self.gcn_opposite(gcn_merge_output.detach(), adjV)
                self.decoder_opposite.forward(gcn_implicit_output.detach(), gcn_opposite_output)

        gcn_opposite_output = self.decoder_opposite.decoder_output



"""Model selection / start training"""


class HeterogeneousGCN(object):
    def __init__(self, adjU, adjV, featuresU, featuresV, device):
        self.adjU = torch.FloatTensor(adjU).to(device)
        self.adjV = torch.FloatTensor(adjV).to(device)
        self.featuresU = torch.FloatTensor(featuresU).to(device)
        self.featuresV = torch.FloatTensor(featuresV).to(device)

        self.featuresU_dimensions = featuresU.shape[1]
        self.featuresV_dimensions = featuresV.shape[1]
        # self.hidden_dimensions = hidden_dimensions
        # self.dropout = dropout

        self.device = device

    def decoder_train(self):
        decoder_hgcn = DecoderGCNLayer(self.featuresU_dimensions, self.featuresV_dimensions, self.device)
        Uinput, Vinput = self.featuresU, self.featuresV
        # training_dict = {'Uinput': Uinput, 'Vinput': Vinput, 'adjU': adjU, 'adjV': adjV}
        # val_dict = {'Uinput': Uinput, 'Vinput': Vinput, 'adjU': adjU, 'adjV': adjV}
        for i in range(5):
            decoder_hgcn.relation_learning(Uinput, Vinput, self.adjU, self.adjV)

    def adversarial_train(self):
        adversarial_hgcn = AdversarialHGCNLayer(self.featuresU_dimensions, self.featuresV_dimensions, self.device)
        Uinput, Vinput = self.featuresU, self.featuresV
        for i in range(5):
            adversarial_hgcn.relation_learning(Uinput, Vinput, self.adjU, self.adjV)



if __name__ == '__main__':
    adjU, adjV, featuresU, featuresV = load_data()
    h_gcn = HeterogeneousGCN(adjU, adjV, featuresU, featuresV)
    # h_gcn.adversarial_train()
    h_gcn.decoder_train()
