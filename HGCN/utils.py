MODEL = ['gan_gcn', 'decoder_gcn']
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
DROPOUT = 0.5
HIDDEN_DIMENSIONS = 2  # hidden layer in Decoder
VALIDATE_ITER = 5  # validate the model every # iterations




"""Parameters in writing and loading data to / from files."""
TRAINING_LOSS_PATH = 'metrics/experiments_results/decoder_training_loss.csv'
STEP1 = 'explicit_relation'
STEP2 = 'implicit_relation'
STEP3 = 'merge_relation'
STEP4 = 'opposite_relation'

