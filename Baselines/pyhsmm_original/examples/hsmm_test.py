
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy, os

import pyhsmm
from pyhsmm.util.text import progprint_xrange

SAVE_FIGURES = False

data = np.loadtxt(os.path.join(os.path.dirname(__file__),'amc_86_01.4d'))[:,:4]
T = data.shape[0]
# data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))[:T]

# Set the weak limit truncation level
Nmax = 60

# and some hyperparameters
obs_dim = data.shape[1]
obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.25,
                'nu_0':obs_dim+2}
dur_hypparams = {'alpha_0':10,
                 'beta_0':2}

obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
posteriormodel.add_data(data,trunc=600) # duration truncation speeds things up when it's possible
# posteriormodel.add_data(data,trunc=600) # duration truncation speeds things up when it's possible
# posteriormodel.add_data(data,trunc=600) # duration truncation speeds things up when it's possible

for idx in progprint_xrange(10):
    posteriormodel.resample_model()

posteriormodel.plot()
posteriormodel.stateseqs[0]

plt.show()