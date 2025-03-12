# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import torch
import helpers
import argparse
import numpy as np
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

version = 0
ckpt_dir = f"lightning_logs/version_{version}/checkpoints/"
ckpt_name = helpers.get_single_file_name(ckpt_dir)
ckpt_path = os.path.join(ckpt_dir, ckpt_name)
model = IrregularSequenceLearner.load_from_checkpoint(ckpt_path)


output_timesteps = test_mat.shape[0]
output = helpers.forward_model(model, train_mat_in, timespans_in, output_timesteps, model.device)

np.save(f'{model.hp_dict["dataset_dict"]["matrix_id"]}_{version}.npy', output.detach().cpu())

print("Done!")
