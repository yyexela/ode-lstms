# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import torch
import helpers
import argparse
import numpy as np
from pathlib import Path
from torch_node_cell import IrregularSequenceLearner

file_dir = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="person")
parser.add_argument("--seq_length", default=100, type=int) # Length of sequence for ODE and PDE datasets
parser.add_argument("--seed", default=0, type=int) # Seed
parser.add_argument("--matrix_id", default='1', type=int) # Matrix X1 through X10, enter only the integer
parser.add_argument("--gpu", default=0, nargs="+", type=int) # List of GPUs to train on 
args = parser.parse_args()

helpers.seed_everything(args.seed)

# training shape: 64         x 100     x 3
#                 batch size x seq_len x ODE dim

seq_len = args.seq_length
train_mat, test_mat = helpers.load_dataset_raw(args)
train_mat_in, _, timespans_in = helpers.load_dataset_lstm_input(args, seq_len = seq_len)

version = 0
ckpt_dir = file_dir / "lightning_logs" / f"version_{version}" / "checkpoints" 
ckpt_name = helpers.get_single_file_name(ckpt_dir)
ckpt_path = os.path.join(ckpt_dir, ckpt_name)
model = IrregularSequenceLearner.load_from_checkpoint(ckpt_path)


output_timesteps = test_mat.shape[0]
output = helpers.forward_model(model, train_mat_in, timespans_in, output_timesteps, model.device)

output = np.asarray(output.detach().cpu()).T

# Make tmp output dir
(file_dir / 'tmp_pred').mkdir(exist_ok=True)

# Save file
np.save(file_dir / 'tmp_pred' / 'pred_mat.npy', output)

print("Done!")
