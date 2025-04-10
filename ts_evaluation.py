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
parser.add_argument("--reconstruct_ids", nargs="+", default=None, type=int) # Matrix X1 through X10, enter list of integers
parser.add_argument("--forecast_ids", nargs="+", default=None, type=int) # Matrix X1 through X10, enter list of integers
parser.add_argument("--forecast_lengths", nargs="+", default=None, type=int) # Length of forecasts, argument must be same length as forecast_ids
parser.add_argument("--gpu", default=0, nargs="+", type=int) # List of GPUs to train on 
args = parser.parse_args()

helpers.seed_everything(args.seed)

# training shape: 64         x 100     x 3
#                 batch size x seq_len x ODE dim

# Load model
version = 0
ckpt_dir = file_dir / "lightning_logs" / f"version_{version}" / "checkpoints" 
ckpt_name = helpers.get_single_file_name(ckpt_dir)
ckpt_path = os.path.join(ckpt_dir, ckpt_name)
model = IrregularSequenceLearner.load_from_checkpoint(ckpt_path)

outputs = dict()
outputs['reconstructions'] = dict()
outputs['forecasts'] = dict()

# Generate reconstructions
for matrix_id in args.reconstruct_ids:
    # Get input data to initialize LSTM
    test_mat_input, timespans_in, test_mat_shape = helpers.load_dataset_lstm_input(args.dataset, 'train', 'reconstruction', matrix_id, args.seq_length)
    # Generate the rest of the output
    # alexey: replace 10 with test_mat_shape[0] - args.seq_length
    output_mat = helpers.forward_model(model, test_mat_input, timespans_in, 10, model.device)
    # Save output
    output_mat = np.asarray(output_mat.detach().cpu()).T
    test_mat = np.asarray(test_mat_input.squeeze().T)
    output_mat = np.concatenate([test_mat, output_mat], axis=1)
    outputs['reconstructions'][f'{matrix_id}'] = output_mat

# Generate forecasts
for matrix_id, output_timesteps in zip(args.forecast_ids, args.forecast_lengths):
    # Get input data to initialize LSTM
    test_mat_input, timespans_in, test_mat_shape = helpers.load_dataset_lstm_input(args.dataset, 'train', 'forecast', matrix_id, args.seq_length)
    # Generate the rest of the output
    # alexey: replace 10 with output_timesteps
    output_mat = helpers.forward_model(model, test_mat_input, timespans_in, 10, model.device)
    # Save output
    output_mat = np.asarray(output_mat.detach().cpu()).T
    outputs['forecasts'][f'{matrix_id}'] = output_mat

# Make tmp output dir
(file_dir / 'tmp_pred').mkdir(exist_ok=True)

# Save file
torch.save(outputs, file_dir / 'tmp_pred' / 'results.torch')

print("Done!")
