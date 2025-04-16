# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import argparse
import helpers
import shutil
import pytorch_lightning as pl
from torch_node_cell import ODELSTM, IrregularSequenceLearner
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
parser.add_argument("--train_ids", default=[1], nargs="+", type=int) # Matrices X1 through X10, enter a list of integers
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--pair_id", default=None, type=int)
parser.add_argument("--gradient_clip_val", default=1.00, type=float)
parser.add_argument("--gpu", default=0, nargs="+", type=int) # List of GPUs to train on 
parser.add_argument("--accelerator", default="gpu", type=str)
parser.add_argument("--log_every_n_steps", default=1, type=int)
args = parser.parse_args()

helpers.seed_everything(args.seed)

classification_task = args.dataset in ["et_mnist", "xor", "person"]

trainloader, testloader, in_features, out_features, return_sequences, batch_size = helpers.load_dataset_trainer(args)

# Hyperparameters dictionary
hp_dict = {
    "dataset_dict": {
        "dataset": args.dataset,
        "batch_size": batch_size,
        "train_ids": args.train_ids,
        "pair_id": args.pair_id,
        "seq_length": args.seq_length
    },
    "model_dict": {
        "in_features": in_features,
        "out_features": out_features,
        "hidden_state_size": args.hidden_state_size,
        "return_sequences": return_sequences,
        "model": args.model,
        "solver": args.solver
    },
    "learner_dict": {
        "lr": args.lr,
        "classification_task": classification_task
    },
    "trainer_dict": {
        "max_epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "devices": args.gpu,
        "accelerator": args.accelerator,
        "log_every_n_steps": args.log_every_n_steps
    }
}

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

learn = IrregularSequenceLearner(
    model=ode_lstm,
    lr=args.lr,
    classification_task=classification_task,
    hp_dict=hp_dict)

trainer = pl.Trainer(
    max_epochs=args.epochs,
    #progress_bar_refresh_rate=1, # Deprecated
    gradient_clip_val=args.gradient_clip_val,
    devices=args.gpu,
    accelerator=args.accelerator,
    log_every_n_steps=args.log_every_n_steps,
    default_root_dir=str(file_dir)
)

trainer.fit(learn, trainloader)

results = trainer.test(learn, testloader)
