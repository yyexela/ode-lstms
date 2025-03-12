# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import torch
import helpers
import argparse
from torch_node_cell import IrregularSequenceLearner

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="person")
parser.add_argument("--seq_length", default=100, type=int) # Length of sequence for ODE and PDE datasets
parser.add_argument("--matrix_id", default='X1', type=str) # Matrix X1 through X10
args = parser.parse_args()

# training shape: 64         x 100     x 3
#                 batch size x seq_len x ODE dim

seq_len = args.seq_length
train_mat, test_mat = helpers.load_dataset_raw(args)
train_mat_in, _, timespans_in = helpers.load_dataset_lstm_input(args, seq_len = seq_len)

model = IrregularSequenceLearner.load_from_checkpoint("lightning_logs/version_8/checkpoints/epoch=1-step=30.ckpt")

output_timesteps = test_mat.shape[0]
output = helpers.forward_model(model, train_mat_in, timespans_in, output_timesteps, model.device)

print("Done!")
