# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import torch
import helpers
import argparse
import numpy as np
from pathlib import Path
from torch_node_cell import IrregularSequenceLearner
from ctf4science.data_module import load_dataset

file_dir = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="person")
parser.add_argument("--seq_length", default=100, type=int) # Length of sequence for ODE and PDE datasets
parser.add_argument("--seed", default=0, type=int) # Seed
parser.add_argument("--reconstruct_id", default=None, type=int) # Matrix X1 through X10, enter integer
parser.add_argument("--forecast_id", default=None, type=int) # Matrix X1 through X10, enter integer
parser.add_argument("--forecast_length", default=None, type=int) # Length of forecast,
parser.add_argument("--train_ids", default=[1], nargs="+", type=int) # Matrices X1 through X10, enter a list of integers
parser.add_argument("--pair_id", default=None, type=int)
parser.add_argument("--burn_in", action='store_true') # Use provided forecast_id matrix as part of output
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

# Generate reconstructions
if args.reconstruct_id is not None:
    # Get input data to initialize LSTM
    train_mats, _, _ = load_dataset(args.dataset, args.pair_id)
    train_mat = train_mats[args.train_ids.index(args.reconstruct_id)]
    output_timesteps = train_mat.shape[1] - args.seq_length
    train_mat = np.swapaxes(train_mat, 0, 1)
    train_mat = torch.Tensor(train_mat.astype(np.float32))
    timespans = np.ones((train_mat.shape[0], 1))/args.seq_length
    timespans = torch.Tensor(timespans.astype(np.float32))
    train_mat = torch.unsqueeze(train_mat[0:args.seq_length,:],0)
    timespans = torch.unsqueeze(timespans[0:args.seq_length,:],0)
    # Generate the rest of the output
    output_mat = helpers.forward_model(model, train_mat, timespans, output_timesteps, model.device)
    # Save output
    output_mat = np.asarray(output_mat.detach().cpu()).T
    train_mat = np.asarray(train_mat.squeeze().T)
    output_mat = np.concatenate([train_mat, output_mat], axis=1)

# Generate forecasts
if args.forecast_id is not None:
    # Get input data to initialize LSTM
    train_mats, _, init_mat = load_dataset(args.dataset, args.pair_id)
    if args.burn_in:
        train_mat = init_mat
    else:
        train_mat = train_mats[args.train_ids.index(args.forecast_id)]
    output_timesteps = args.forecast_length - train_mat.shape[1]
    train_mat = np.swapaxes(train_mat, 0, 1)
    train_mat = torch.Tensor(train_mat.astype(np.float32))
    timespans = np.ones((train_mat.shape[0], 1))/args.seq_length
    timespans = torch.Tensor(timespans.astype(np.float32))
    train_mat = torch.unsqueeze(train_mat[0:args.seq_length,:],0)
    timespans = torch.unsqueeze(timespans[0:args.seq_length,:],0)
    # Generate the rest of the output
    output_mat = helpers.forward_model(model, train_mat, timespans, output_timesteps, model.device)
    # Prepend burn-in
    output_mat = torch.cat([train_mat[0],output_mat.detach().cpu()])
    # Save output
    output_mat = np.asarray(output_mat).T

# Make tmp output dir
(file_dir / 'tmp_pred').mkdir(exist_ok=True)

# Save file
torch.save(output_mat, file_dir / 'tmp_pred' / 'output_mat.torch')
