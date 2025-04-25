# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import torch
import helpers
import argparse
import numpy as np
from pathlib import Path
from torch_node_cell import IrregularSequenceLearner
from ctf4science.data_module import load_dataset, load_validation_dataset, get_validation_prediction_timesteps, get_prediction_timesteps

file_dir = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="person")
parser.add_argument("--seq_length", default=100, type=int) # Length of sequence for ODE and PDE datasets
parser.add_argument("--seed", default=0, type=int) # Seed
parser.add_argument("--pair_id", default=None, type=int)
parser.add_argument("--burn_in", action='store_true') # Use provided forecast_id matrix as part of output
parser.add_argument("--validation", action='store_true')
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
if args.pair_id in [2, 4]:
    # Get input data to initialize spacetime
    if args.validation:
        train_mats, _, init_data = load_validation_dataset(args.dataset, args.pair_id)
        output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0] - args.seq_length
    else:
        train_mats, init_data = load_dataset(args.dataset, args.pair_id)
        output_timesteps = get_prediction_timesteps(args.dataset, args.pair_id).shape[0] - args.seq_length
    train_mat = train_mats[0]
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
else:
    # Get input data to initialize spacetime
    if args.validation:
        train_mats, _, init_mat = load_validation_dataset(args.dataset, args.pair_id)
        if args.pair_id in [8,9]:
            train_mat = init_mat
            output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0] - args.seq_length
        else:
            train_mat = train_mats[0]
            output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0]
    else:
        train_mats, init_mat = load_dataset(args.dataset, args.pair_id)
        if args.pair_id in [8,9]:
            train_mat = init_mat
            output_timesteps = get_prediction_timesteps(args.dataset, args.pair_id).shape[0] - args.seq_length
        else:
            train_mat = train_mats[0]
            output_timesteps = get_prediction_timesteps(args.dataset, args.pair_id).shape[0]
    train_mat = np.swapaxes(train_mat, 0, 1)
    train_mat = torch.Tensor(train_mat.astype(np.float32))
    timespans = np.ones((train_mat.shape[0], 1))/args.seq_length
    timespans = torch.Tensor(timespans.astype(np.float32))
    train_mat = torch.unsqueeze(train_mat[-args.seq_length:,:],0)
    timespans = torch.unsqueeze(timespans[-args.seq_length:,:],0)
    # Generate the rest of the output
    # alexey: replace 10 with output_timesteps
    output_mat = helpers.forward_model(model, train_mat, timespans, output_timesteps, model.device)
    # Save output
    output_mat = output_mat.detach().cpu()
    output_mat = np.asarray(output_mat).T
    if args.pair_id in [8,9]:
        train_mat = np.asarray(train_mat.squeeze().T)
        output_mat = np.concatenate([train_mat, output_mat], axis=1)

# Make tmp output dir
(file_dir / 'tmp_pred').mkdir(exist_ok=True)

# Save file
torch.save(output_mat, file_dir / 'tmp_pred' / 'output_mat.torch')
