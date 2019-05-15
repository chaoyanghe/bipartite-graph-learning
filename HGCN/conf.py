#!/usr/bin/env python

MODEL = ['gan_gcn', 'decoder_gcn']
RANDOM_SEED = 42
BATCH_SIZE = 100
EPOCHS = 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
HIDDEN_DIMENSIONS = 10  # hidden layer units for discriminator in GAN model
VALIDATE_ITER = 5  # validate the model every # iterations
DROPOUT = 0.5
GCN_OUTPUT_DIM = 10
ENCODER_HIDDEN_DIMENSIONS = 10
LATENT_DIMENSIONS = 8  # must be the same as the opposite set
DECODER_HIDDEN_DIMENSIONS = 10

"""Parameters in writing and loading data to / from files."""
TRAINING_LOSS_PATH = 'metrics/experiments_results/decoder_training_loss.csv'
STEP1 = 'explicit_relation'
STEP2 = 'implicit_relation'
STEP3 = 'merge_relation'
STEP4 = 'opposite_relation'

# classification
it = 3000
method = "hgcn"
input_folder = "./data/Tencent-QQ"
output_folder = "./out"
