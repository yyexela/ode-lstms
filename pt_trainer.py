# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import argparse
import helpers
import pytorch_lightning as pl
from torch_node_cell import ODELSTM, IrregularSequenceLearner

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="person")
parser.add_argument("--solver", default="dopri5")
parser.add_argument("--hidden_state_size", default=64, type=int) # Hidden state size
parser.add_argument("--seq_length", default=100, type=int) # Length of sequence for ODE and PDE datasets
parser.add_argument("--matrix_id", default='X1', type=str) # Matrix X1 through X10
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--gpus", default=0, nargs="+", type=int)
args = parser.parse_args()

trainloader, testloader, in_features, num_classes, return_sequences = helpers.load_dataset_trainer(args)

ode_lstm = ODELSTM(
    in_features,
    args.hidden_state_size, # hidden state size
    num_classes,
    return_sequences=return_sequences,
    solver_type=args.solver,
)

classification_task = args.dataset in ["et_mnist", "xor", "person"]
learn = IrregularSequenceLearner(
    model=ode_lstm,
    lr=args.lr,
    classification_task=classification_task)

trainer = pl.Trainer(
    max_epochs=args.epochs,
    #progress_bar_refresh_rate=1, # Deprecated
    gradient_clip_val=1,
    devices=args.gpus,
    accelerator="gpu",
    log_every_n_steps=1
)

trainer.fit(learn, trainloader)

results = trainer.test(learn, testloader)
