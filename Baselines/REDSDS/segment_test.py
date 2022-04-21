# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib
from tensorboardX import SummaryWriter
import src.utils as utils
import src.datasets as datasets
import src.tensorboard_utils as tensorboard_utils
from src.model_utils import build_model
from src.evaluation import evaluate_segmentation
from src.torch_utils import torch2numpy

available_datasets = {"bouncing_ball", "3modesystem", "bee"}


def train_step(batch, model, optimizer, step, config):
    model.train()

    def _set_lr(lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    switch_temp = utils.get_temperature(step, config, "switch_")
    extra_args = dict()
    dur_temp = 1.0
    if config["model"] == "REDSDS":
        dur_temp = utils.get_temperature(step, config, "dur_")
        extra_args = {"dur_temperature": dur_temp}
    lr = utils.get_learning_rate(step, config)
    xent_coeff = utils.get_cross_entropy_coef(step, config)
    cont_ent_anneal = config["cont_ent_anneal"]
    optimizer.zero_grad()
    result = model(
        batch,
        switch_temperature=switch_temp,
        num_samples=config["num_samples"],
        cont_ent_anneal=cont_ent_anneal,
        **extra_args,
    )
    objective = -1 * (
        result[config["objective"]] + xent_coeff * result["crossent_regularizer"]
    )
    print(
        step,
        f"obj: {objective.item():.4f}",
        f"lr: {lr:.6f}",
        f"s-temp: {switch_temp:.2f}",
        f"cross-ent: {xent_coeff}",
        f"cont ent: {cont_ent_anneal}",
    )
    objective.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
    _set_lr(lr)
    optimizer.step()
    result["objective"] = objective
    result["lr"] = lr
    result["switch_temperature"] = switch_temp
    result["dur_temperature"] = dur_temp
    result["xent_coeff"] = xent_coeff
    return result

# class BouncingBallDataset(torch.utils.data.Dataset):
#     def __init__(self, path="./data/bouncing_ball.npz"):
#         npz = np.load(path)
#         self.data_y = npz["y"].astype(np.float32)
#         self.data_z = npz["z"].astype(np.int32)

#     def __getitem__(self, i):
#         return self.data_y[i], self.data_z[i]

#     def __len__(self):
#         return self.data_y.shape[0]

if __name__ == "__main__":
    script_path = os.path.dirname(__file__)
    data_path = script_path+'/data/'
    config_path = script_path+'/configs/bouncing_ball_duration.yaml'
    device = 'cuda'
    gpu = 0
    config = utils.get_config_and_setup_dirs(config_path)
    device = torch.device(device)
    print(config)
    with open(os.path.join(config["log_dir"], "config.json"), "w") as fp:
        json.dump(config, fp)

    train_dataset = datasets.BouncingBallDataset(path=data_path+"bouncing_ball.npz")
    test_dataset = datasets.BouncingBallDataset(path=data_path+"bouncing_ball_test.npz")
    
    num_workers = 4
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=80, pin_memory=True
    )

    train_gen = iter(train_loader)
    test_gen = iter(test_loader)

    print(f'Running {config["model"]} on {config["dataset"]}.')
    print(f"Train size: {len(train_dataset)}. Test size: {len(test_dataset)}.")

    # MODEL
    model = build_model(config=config)
    start_step = 1
    model = model.to(device)

    # TRAIN AND EVALUATE
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=config["weight_decay"]
    )

    for step in range(start_step, config["num_steps"] + 1):
        try:
            train_batch, train_label = next(train_gen)
            train_batch = train_batch.to(device)
        except StopIteration:
            train_gen = iter(train_loader)
        train_result = train_step(train_batch, model, optimizer, step, config)

        if True:
            # Evaluate Segmentation
            if step == config["num_steps"]:
                extra_args = dict()
                if config["model"] == "REDSDS":
                    extra_args = {"dur_temperature": train_result["dur_temperature"]}
                true_segs = []
                pred_segs = []
                true_tss = []
                recons_tss = []
                for test_batch, test_label in test_loader:
                    test_batch = test_batch.to(device)
                    test_result = model(
                        test_batch,
                        switch_temperature=train_result["switch_temperature"],
                        num_samples=1,
                        deterministic_inference=True,
                        **extra_args,
                    )
                    pred_seg = torch2numpy(torch.argmax(test_result["log_gamma"][0], dim=-1))
                    true_seg = torch2numpy(test_label[:, : config["context_length"]])
                    print(pred_seg)