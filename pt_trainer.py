# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import torch
import argparse
from irregular_sampled_datasets import PersonData, ETSMnistData, XORData, ODELorenzData
import torch.utils.data as data
from torch_node_cell import ODELSTM, IrregularSequenceLearner
import pytorch_lightning as pl

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

# ODE LSTMS
#train_x.shape
#torch.Size([1900, 100, 3])
#train_ts.shape
#torch.Size([1900, 100, 1])
#train_y.shape
#torch.Size([1900, 3])
#train_mask.shape
#torch.Size([1900, 100, 3])

# XOR
#train_x.shape
#torch.Size([100000, 32, 1])
#train_ts.shape
#torch.Size([100000, 32, 1])
#train_y.shape
#torch.Size([100000])
#train_mask.shape
#torch.Size([100000, 32])

def load_dataset(args):
    if args.dataset == "person":
        dataset = PersonData()
        train_x = torch.Tensor(dataset.train_x)
        train_y = torch.LongTensor(dataset.train_y)
        train_ts = torch.Tensor(dataset.train_t)
        test_x = torch.Tensor(dataset.test_x)
        test_y = torch.LongTensor(dataset.test_y)
        test_ts = torch.Tensor(dataset.test_t)
        train = data.TensorDataset(train_x, train_ts, train_y)
        test = data.TensorDataset(test_x, test_ts, test_y)
        return_sequences = True
    elif args.dataset == "ODELorenz":
        return_sequences = False
        dataset = ODELorenzData(seq_length=args.seq_length, matrix_id=args.matrix_id)
        train_x = torch.Tensor(dataset.train_events)
        train_y = torch.Tensor(dataset.train_y)
        train_ts = torch.Tensor(dataset.train_elapsed)
        test_x = torch.Tensor(dataset.test_events)
        test_y = torch.Tensor(dataset.test_y)
        test_ts = torch.Tensor(dataset.test_elapsed)
        train = data.TensorDataset(train_x, train_ts, train_y)
        test = data.TensorDataset(test_x, test_ts, test_y)
    else:
        if args.dataset == "et_mnist":
            dataset = ETSMnistData(time_major=False)
        elif args.dataset == "xor":
            dataset = XORData(time_major=False, event_based=False, pad_size=32)
        elif args.dataset == "ODELorenz":
            dataset = ODELorenzData(seq_length=args.seq_length, matrix_id=args.matrix_id)
        else:
            raise ValueError("Unknown dataset '{}'".format(args.dataset))
        return_sequences = False
        train_x = torch.Tensor(dataset.train_events)
        train_y = torch.LongTensor(dataset.train_y)
        train_ts = torch.Tensor(dataset.train_elapsed)
        train_mask = torch.Tensor(dataset.train_mask)
        test_x = torch.Tensor(dataset.test_events)
        test_y = torch.LongTensor(dataset.test_y)
        test_ts = torch.Tensor(dataset.test_elapsed)
        test_mask = torch.Tensor(dataset.test_mask)
        train = data.TensorDataset(train_x, train_ts, train_y, train_mask)
        test = data.TensorDataset(test_x, test_ts, test_y, test_mask)
    trainloader = data.DataLoader(train, batch_size=64, shuffle=True, num_workers=4)
    testloader = data.DataLoader(test, batch_size=64, shuffle=False, num_workers=4)
    in_features = train_x.size(-1)
    num_classes = 3 if args.dataset == "ODELorenz" else int(torch.max(train_y).item() + 1)
    return trainloader, testloader, in_features, num_classes, return_sequences

trainloader, testloader, in_features, num_classes, return_sequences = load_dataset(args)

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
