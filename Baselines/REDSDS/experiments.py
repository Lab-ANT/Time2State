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
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import src.utils as utils
import pandas as pd
from src.model_utils import build_model
from src.torch_utils import torch2numpy
from TSpy.dataset import load_USC_HAD
from TSpy.eval import *
from TSpy.utils import *
from TSpy.label import *

dataset_info = {'amc_86_01.4d':{'n_segs':4, 'label':{588:0,1200:1,2006:0,2530:2,3282:0,4048:3,4579:2}},
        'amc_86_02.4d':{'n_segs':8, 'label':{1009:0,1882:1,2677:2,3158:3,4688:4,5963:0,7327:5,8887:6,9632:7,10617:0}},
        'amc_86_03.4d':{'n_segs':7, 'label':{872:0, 1938:1, 2448:2, 3470:0, 4632:3, 5372:4, 6182:5, 7089:6, 8401:0}},
        'amc_86_07.4d':{'n_segs':6, 'label':{1060:0,1897:1,2564:2,3665:1,4405:2,5169:3,5804:4,6962:0,7806:5,8702:0}},
        'amc_86_08.4d':{'n_segs':9, 'label':{1062:0,1904:1,2661:2,3282:3,3963:4,4754:5,5673:6,6362:4,7144:7,8139:8,9206:0}},
        'amc_86_09.4d':{'n_segs':5, 'label':{921:0,1275:1,2139:2,2887:3,3667:4,4794:0}},
        'amc_86_10.4d':{'n_segs':4, 'label':{2003:0,3720:1,4981:0,5646:2,6641:3,7583:0}},
        'amc_86_11.4d':{'n_segs':4, 'label':{1231:0,1693:1,2332:2,2762:1,3386:3,4015:2,4665:1,5674:0}},
        'amc_86_14.4d':{'n_segs':3, 'label':{671:0,1913:1,2931:0,4134:2,5051:0,5628:1,6055:2}},
}

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

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


if __name__ == "__main__":
    script_path = os.path.dirname(__file__)
    data_path = script_path+'/../../data/'
    config_path = script_path+'/configs/test_duration.yaml'
    device = 'cuda'
    gpu = 0
    config = utils.get_config_and_setup_dirs(config_path)
    device = torch.device(device)

    dataset_path = data_path+'/MoCap/4d/amc_86_01.4d'
    df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
    data = df.to_numpy()[::2,:]
    data = normalize(data)
    n_state=dataset_info['amc_86_01.4d']['n_segs']
    groundtruth = seg_to_label(dataset_info['amc_86_01.4d']['label'])[::2]
    groundtruth = groundtruth[:-1]
    print(data.shape)
    length, dim = data.shape
    data = torch.tensor(data.astype(np.float32))
    data = data.unsqueeze(0).to(device)
    print(data.shape)
    config['context_length'] = length
    config['obs_dim'] = dim
    config['num_categories'] = n_state

    # MODEL
    model = build_model(config=config)
    start_step = 1
    model = model.to(device)

    # TRAIN
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=config["weight_decay"]
    )

    for step in range(start_step, config["num_steps"] + 1):
        train_result = train_step(data, model, optimizer, step, config)

    extra_args = dict()
    if config["model"] == "REDSDS":
        extra_args = {"dur_temperature": train_result["dur_temperature"]}

    df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
    test = df.to_numpy()
    test = normalize(test)[::2,:]
    test = torch.tensor(test.astype(np.float32))
    test = test.unsqueeze(0).to(device)

    test_result = model(test,
        switch_temperature=train_result["switch_temperature"],
        num_samples=1,
        deterministic_inference=True,
        **extra_args,)
    
    pred_seg = torch2numpy(torch.argmax(test_result["log_gamma"][0], dim=-1))
    print(pred_seg.shape)
    print(evaluate_clustering(groundtruth, pred_seg.flatten()))
    df = pd.read_csv(dataset_path, sep=' ',usecols=range(0,4))
    test = df.to_numpy()
    plt.subplot(311)
    for i in range(4):
        plt.plot(test[:,i])
    plt.subplot(312)
    plt.imshow(pred_seg.reshape(1, -1), aspect='auto', cmap='tab10',
          interpolation='nearest')
    plt.subplot(313)
    plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab10',
          interpolation='nearest')
    plt.savefig('result_redsds.png')