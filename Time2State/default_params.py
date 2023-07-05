import torch

cuda = False
gpu = 0
if torch.cuda.is_available():
    print('Cuda is available.')
    cuda = True
else:
    print('Cuda is not available.')
# cuda = False

params_CausalConvNetwork = {
    "batch_size": 1,
    "channels": 30,
    "compared_length": 256,
    "depth": 10,
    "nb_steps": 10,
    "in_channels": 1,
    "kernel_size": 3,
    "penalty": None,
    "early_stopping": None,
    "lr": 0.003,
    "nb_random_samples": 10,
    "negative_penalty": 1,
    "out_channels": 4,
    "reduced_size": 80,
    "cuda": cuda,
    "gpu": gpu
}

params_TNC = {
    'win_size' : 512,
    'in_channels' : 4,
    'nb_steps' : 5,
    'out_channels' : 4,
}

params_CPC = {
    'win_size' : 512,
    'in_channels' : 4,
    'nb_steps' : 5,
    'out_channels' : 4,
}

params_Triplet = {
    "batch_size": 1,
    "channels": 30,
    "compared_length": 256,
    "depth": 10,
    "nb_steps": 20,
    "in_channels": 1,
    "kernel_size": 3,
    "penalty": None,
    "early_stopping": None,
    "lr": 0.003,
    "nb_random_samples": 10,
    "negative_penalty": 1,
    "out_channels": 4,
    "reduced_size": 80,
    "cuda": cuda,
    "gpu": gpu
}

params_LSE = {
    "batch_size": 1,
    "channels": 30,
    "win_size": 256,
    "win_type": 'rect', # {rect, hanning}
    "depth": 10,
    "nb_steps": 20,
    "in_channels": 1,
    "kernel_size": 3,
    "lr": 0.003,
    "out_channels": 4,
    "reduced_size": 80,
    "cuda": cuda,
    "gpu": gpu,
    "M" : 10,
    "N" : 4
}