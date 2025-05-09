# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import numpy as np
import torch
import argparse
import helpers
import shutil
from tqdm import tqdm
from torch_node_cell import ODELSTM, NonPLLearner
from ctf4science.data_module import load_validation_dataset, load_dataset, get_prediction_timesteps, get_validation_prediction_timesteps, get_validation_training_timesteps
from pathlib import Path

# file dir
file_dir = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="person")
parser.add_argument("--seed", default=0, type=int) # Seed
parser.add_argument("--model", default="ode-lstms")
parser.add_argument("--solver", default="dopri5")
parser.add_argument("--hidden_state_size", default=64, type=int) # Hidden state size
parser.add_argument("--seq_length", default=100, type=int) # Length of sequence for ODE and PDE datasets
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--pair_id", default=None, type=int)
parser.add_argument("--gradient_clip_val", default=1.00, type=float)
parser.add_argument("--device", default="cuda", type=str) # PyTorch device
parser.add_argument("--validation", action='store_true')
parser.add_argument("--batch_id", default='0', type=str)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

print("Command line arguments:")
print(args)

helpers.seed_everything(args.seed)

classification_task = args.dataset in ["et_mnist", "xor", "person"]

trainloader, testloader, in_features, out_features, return_sequences, batch_size = helpers.load_dataset_trainer(args)
if batch_size is None:
    batch_size = args.batch_size

# Remove old model saves
try:
    shutil.rmtree(file_dir / "lightning_logs")
except:
    pass

ode_lstm = ODELSTM(
    in_features,
    args.hidden_state_size, # hidden state size
    out_features,
    return_sequences=return_sequences,
    solver_type=args.solver,
    model=args.model
)

ode_lstm.train()
ode_lstm.to(args.device)

learner = NonPLLearner(
    model=ode_lstm,
    args=args,
    classification_task=classification_task)

losses = learner.training_loop(trainloader)

ode_lstm.eval()

# Generate reconstructions
if args.pair_id in [2, 4]:
    # Get input data to initialize spacetime
    if args.validation:
        train_mats, _, init_data = load_validation_dataset(args.dataset, args.pair_id, transpose=True)
        output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0]
    else:
        train_mats, init_data = load_dataset(args.dataset, args.pair_id, transpose=True)
        output_timesteps = get_prediction_timesteps(args.dataset, args.pair_id).shape[0]
    train_mat = train_mats[0]
    train_mat = np.swapaxes(train_mat, 0, 1)
    train_mat = torch.Tensor(train_mat.astype(np.float32))
    timespans = np.ones((train_mat.shape[0], 1))/args.seq_length
    timespans = torch.Tensor(timespans.astype(np.float32))
    train_mat = torch.unsqueeze(train_mat,0)
    timespans = torch.unsqueeze(timespans,0)
    # Generate the rest of the output
    if args.debug:
        output_mat = np.zeros((train_mat.shape[2], output_timesteps+args.seq_length))
    else:
        output_mat_full = None
        #while output_mat_full is None or output_mat_full.shape[1] < output_timesteps:
        for start_idx in tqdm(range(output_timesteps - args.seq_length), "Unrolling Model"):
            train_mat_tmp = train_mat[:,start_idx:start_idx + args.seq_length,:]
            timespans_tmp = timespans[:,start_idx:start_idx + args.seq_length,:]
            output_mat = helpers.forward_model(ode_lstm, train_mat_tmp, timespans_tmp, 1, args.device)
            # Save output
            output_mat = np.asarray(output_mat.detach().cpu()).T
            train_mat_tmp = np.asarray(train_mat_tmp.squeeze().T)
            if start_idx == 0:
                output_mat_full = np.concatenate([train_mat_tmp, output_mat], axis=1)
            else:
                output_mat_full = np.concatenate([output_mat_full, output_mat], axis=1)
        output_mat_full = output_mat_full[:,0:output_timesteps]
        output_mat = output_mat_full

# Generate forecasts
else:
    # Get input data to initialize spacetime
    if args.validation:
        train_mats, _, init_mat = load_validation_dataset(args.dataset, args.pair_id, transpose=True)
        if args.pair_id in [8,9]:
            train_mat = init_mat
            output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0] - args.seq_length
        else:
            train_mat = train_mats[0]
            output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0]
    else:
        train_mats, init_mat = load_dataset(args.dataset, args.pair_id, transpose=True)
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
    if args.debug:
        output_mat = np.zeros((train_mat.shape[2], output_timesteps + args.seq_length))
    else:
        output_mat = helpers.forward_model(ode_lstm, train_mat, timespans, output_timesteps, args.device)
        # Save output
        output_mat = output_mat.detach().cpu()
        output_mat = np.asarray(output_mat).T
        if args.pair_id in [8,9]:
            train_mat = np.asarray(train_mat.squeeze().T)
            output_mat = np.concatenate([train_mat, output_mat], axis=1)

# Make tmp output dir
(file_dir / 'tmp_pred').mkdir(exist_ok=True)

# Save file
torch.save(output_mat, file_dir / 'tmp_pred' / f'output_mat_{args.batch_id}.torch')
