'''
Created by Chengyu on 2022/2/26.
'''

from sklearn import mixture
import numpy as np
# import pyhsmm
# from pyhsmm.util.text import progprint_xrange
# np.seterr(divide='ignore') # these warnings are usually harmless for this code
from sklearn import cluster
# from hmmlearn.hmm import GaussianHMM, GMMHMM
from Time2State.abstractions import *

# class GMM(BasicClusteringClass):
#     def __init__(self, n_states):
#         self.n_states = n_states

#     def fit(self, X):
#         gmm = mixture.GaussianMixture(n_components=self.n_states, covariance_type="full").fit(X)
#         return gmm.predict(X)

# class GHMM(BasicClusteringClass):
#     def __init__(self, n_component):
#         self.n_component = n_component

#     def fit(self, X):
#         model = GaussianHMM(n_components=self.n_component, covariance_type='diag', n_iter=10000)
#         model.fit(X)
#         prediction = model.decode(X, algorithm='viterbi')[1]
#         return prediction

# class GMM_HMM(BasicClusteringClass):
#     def __init__(self, n_states):
#         self.n_states = n_states
        
#     def fit(self, X):
#         model = GMMHMM(n_components=self.n_states, covariance_type='diag', n_iter=10000)
#         model.fit(X)
#         prediction = model.decode(X, algorithm='viterbi')[1]
#         return prediction

class DPGMM(BasicClusteringClass):
    def __init__(self, n_states, alpha=1e3):
        self.alpha = alpha
        if n_states is not None:
            self.n_states = n_states
        else:
            self.n_states = 20

    def fit(self, X):
        dpgmm = mixture.BayesianGaussianMixture(init_params='kmeans',
                                                n_components=self.n_states,
                                                covariance_type="full",
                                                weight_concentration_prior=self.alpha, # alpha
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(X)
        return dpgmm.predict(X)

class KMeansClustering(BasicClusteringClass):
    def __init__(self, n_component):
        self.n_component = n_component

    def fit(self, X):
        clust = cluster.KMeans(n_clusters=self.n_component).fit(X)
        return clust.labels_

class SpectralClustering_(BasicClusteringClass):
    def __init__(self, n_component):
        self.n_component = n_component

    def fit(self, X):
        clust = cluster.SpectralClustering(n_clusters=self.n_component).fit(X)# KMeans(n_clusters=self.n_component).fit(X)
        return clust.labels_
        
# class HDP_HSMM(BasicClusteringClass):
#     def fit(self, X):
#         data = X
#         # Set the weak limit truncation level
#         Nmax = 25

#         # and some hyperparameters
#         obs_dim = data.shape[1]
#         obs_hypparams = {'mu_0':np.zeros(obs_dim),
#                         'sigma_0':np.eye(obs_dim),
#                         'kappa_0':0.25,
#                         'nu_0':obs_dim+2}
#         dur_hypparams = {'alpha_0':1e3,
#                         'beta_0':20}

#         obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
#         dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

#         posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
#                 alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
#                 init_state_concentration=6., # pretty inconsequential
#                 obs_distns=obs_distns,
#                 dur_distns=dur_distns)
#         posteriormodel.add_data(data,trunc=600) # duration truncation speeds things up when it's possible

#         for idx in progprint_xrange(20):
#             posteriormodel.resample_model()

#         # posteriormodel.plot()
#         # plt.show()
#         return posteriormodel.stateseqs[0]