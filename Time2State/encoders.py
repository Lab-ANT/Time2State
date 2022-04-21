import torch
import numpy
import sklearn
import math
import sys
import os
sys.path.append(os.path.dirname(__file__))
import utils
import networks
import losses

class BasicEncoder():
    def encode(self, X):
        pass

    def save(self, X):
        pass

    def load(self, X):
        pass

class LSTM_LSE(BasicEncoder):
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, M, N):
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu)

        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.LSE_loss.LSELoss(
            compared_length, nb_random_samples, negative_penalty, M, N
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adagrad(self.network.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=lr)
        self.loss_list = []
    
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        network = networks.rnn.RnnEncoder(256, in_channels, out_channels, num_layers=2, cell_type='GRU', device='cuda', dropout=0.1)
        # network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """

        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                # print(batch.size(2))
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.network, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.network, train, save_memory=save_memory
                    )
                self.loss_list.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (
                ratio >= 5 and train_size >= 50
            ):
                # Computes the best regularization parameters
                features = self.encode(X)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.network)(**self.params)
                    best_encoder.double()
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.network.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.network

    def encode(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.network(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.network(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.network = self.network.train()
        return features

    def encode_window(self, X, win_size=128, batch_size=50, window_batch_size=1000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param step size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        @param step Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length-win_size)/step)+1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window/window_batch_size)):
                masking = numpy.array([X[b,:,j:j+win_size] for j in range(step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window),step)])
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b,:,i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T
    
    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self


class CausalConv_LSE(BasicEncoder):
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, M, N):
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu)

        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.LSE_loss.LSELoss(
            compared_length, nb_random_samples, negative_penalty, M, N
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.loss_list = []
    
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        network = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """

        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                # print(batch.size(2))
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.network, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.network, train, save_memory=save_memory
                    )
                self.loss_list.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (
                ratio >= 5 and train_size >= 50
            ):
                # Computes the best regularization parameters
                features = self.encode(X)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.network)(**self.params)
                    best_encoder.double()
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.network.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.network

    def encode(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.network(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.network(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.network = self.network.train()
        return features

    def encode_window(self, X, win_size=128, batch_size=50, window_batch_size=1000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param step size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        @param step Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length-win_size)/step)+1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window/window_batch_size)):
                masking = numpy.array([X[b,:,j:j+win_size] for j in range(step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window),step)])
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b,:,i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T
    
    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self

class CausalConv_Triplet(BasicEncoder):
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.network = self.__create_network(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu)

        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty
        )
        self.loss_varying = losses.triplet_loss.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.loss_list = []
    
    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        network = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """

        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                # print(batch.size(2))
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.network, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.network, train, save_memory=save_memory
                    )
                self.loss_list.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (
                ratio >= 5 and train_size >= 50
            ):
                # Computes the best regularization parameters
                features = self.encode(X)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.network)(**self.params)
                    best_encoder.double()
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.network.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.network

    def encode(self, X, batch_size=500):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.network = self.network.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.network(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.network(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.network = self.network.train()
        return features

    def encode_window(self, X, win_size=128, batch_size=50, window_batch_size=1000, step=10):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param step size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        @param step Step length of the sliding window.
        """
        # _, dim = X.shape
        # X = numpy.transpose(numpy.array(X, dtype=float)).reshape(1, dim, -1)

        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length-win_size)/step)+1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window/window_batch_size)):
                masking = numpy.array([X[b,:,j:j+win_size] for j in range(step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window),step)])
                # print(masking.shape,step*i*window_batch_size, step*min((i+1)* window_batch_size, num_window))
                embeddings[b,:,i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T
    
    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self

