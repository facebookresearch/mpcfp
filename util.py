#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import torch
import torch.distributed as dist
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive


# Record the number of deaths from horsekicks in 14 corps of the Prussian army
# each year from 1875 to 1894, as reported by Ladislaus Bortkiewicz in his book
# "The Law of Small Numbers"
horsekicks = [
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
    [2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0],
    [1, 2, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 2, 2, 0, 1, 0, 0, 2, 1, 0],
    [0, 3, 2, 1, 1, 1, 0, 0, 0, 2, 1, 4, 3, 0],
    [1, 0, 0, 2, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 2, 0, 0, 0, 0, 1, 0, 1, 1, 2, 1, 4, 1],
    [0, 0, 1, 2, 0, 1, 2, 1, 0, 1, 0, 3, 0, 0],
    [3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 1],
    [2, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 3, 0],
    [1, 1, 2, 1, 0, 0, 3, 2, 1, 1, 0, 1, 2, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 2, 2, 0, 2],
    [1, 2, 0, 2, 0, 1, 1, 2, 0, 2, 1, 1, 2, 2],
    [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 3, 3, 1, 0],
    [1, 3, 2, 0, 1, 1, 3, 0, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]]


def inverse_probit(x):
    return 0.5 * (1 + (x / math.sqrt(2)).erf())


def load_horses(degree=2, onehotenc=False):
    """
    Loads the classical data on deaths from horsekicks of Bortkiewicz.
    """
    # Construct the design matrix
    m = len(horsekicks)
    n = len(horsekicks[0])
    if onehotenc:
        flat = np.zeros((m * n, degree + 1 + n))
    else:
        flat = np.zeros((m * n, degree + 1))
    targets = np.zeros(m * n)
    for j in range(m):
        for k in range(n):
            monomial = np.array([j**i / (m - 1)**i for i in range(degree + 1)])
            index = j + m * k
            if onehotenc:
                onehot = np.zeros((n))
                onehot[k] = 1
                # Concatenate the monomial features and one-hot encodings
                flat[index, :(degree + 1)] = monomial
                flat[index, (degree + 1):] = onehot
            else:
                flat[index, :] = monomial
            targets[index] = horsekicks[j][k]
    # Return data
    return torch.FloatTensor(flat), torch.FloatTensor(targets)


def load_mnist(path="/tmp", n_classes=2):
    '''
    Loads the MNIST train and test datasets.
    '''
    assert n_classes <= 10

    # download the MNIST dataset:
    from torchvision.datasets.mnist import MNIST
    datasets = []
    for train in [True, False]:
        mnist = MNIST(path, download=True, train=train)

        # preprocess the MNIST dataset:
        images = mnist.data.float().div_(255.)
        images = images.view(images.size(0), -1)
        targets = mnist.targets.float()

        # Keep only 0, 1, ..., n_classes-1
        images = images[targets <= n_classes - 1, :]
        targets = targets[targets <= n_classes - 1]
        datasets.extend([images, targets])

    # return data:
    return datasets


def load_covtype(path="/tmp", n_classes=2):
    '''
    Loads data for forest cover type.
    '''
    assert n_classes == 2 or n_classes == 7
    train_split = 500000

    # Set paths and filenames
    if n_classes == 2:
        filename = 'covtype.libsvm.binary'
    elif n_classes == 7:
        filename = 'covtype'
    url = 'http://tygert.com/'
    url += filename + '.gz'
    # Read data from disk
    targets = []
    samples = []
    if not os.path.exists(os.path.join(path, filename)):
        download_and_extract_archive(url, path)
    with open(os.path.join(path, filename), 'r') as f:
        for line in f:
            parsed = line.split()

            # Ensure that the targets start from 0, not from 1
            targets.append(float(parsed[0]) - 1)
            features = np.zeros((54))
            for term in parsed[1:]:
                both = term.split(':')
                features[int(both[0]) - 1] = float(both[1])
            samples.append(features)

    targets = np.stack(targets)
    samples = np.stack(samples)

    # Shuffle prior to split
    perm = np.random.permutation(samples.shape[0])
    samples = samples[perm, :]
    targets = targets[perm]

    train_targets, test_targets = np.split(targets, [train_split])
    train_samples, test_samples = np.split(samples, [train_split], axis=0)

    # Normalize samples (but not the one-hot part)
    means = train_samples[:, :10].mean(axis=0, keepdims=True)
    train_samples[:, :10] = (train_samples[:, :10] - means)
    maxes = np.abs(train_samples[:, :10]).max(axis=0, keepdims=True)
    train_samples[:, :10] /= maxes

    test_samples[:, :10] = (test_samples[:, :10] - means) / maxes

    # Return data
    return (torch.Tensor(train_samples), torch.Tensor(train_targets),
            torch.Tensor(test_samples), torch.Tensor(test_targets))


def load_synth(len_targets, num_features, link=None):
    if link is None or link == 'identity':
        assert len_targets > num_features
        # Construct a design matrix whose columns are orthonormal
        # together with a vector orthogonal to its column space
        designmat = torch.randn(
            size=(len_targets, num_features + 1), dtype=torch.float64)
        designmat, _ = torch.qr(designmat)
        ortho = designmat[:, -1]
        lossmin = 10
        ortho = ortho * math.sqrt(lossmin)
        designmat = designmat[:, :-1]
        # Construct the optimal solution
        weights = torch.randn(size=(num_features,), dtype=torch.float64)
        # Define targets to be the result of the optimal solution plus ortho
        bias = 0
        targets = designmat @ weights + ortho
    elif link == 'logit' or link == 'probit' or link == 'log':
        assert len_targets > num_features
        # Construct a design matrix whose columns are orthonormal
        designmat = torch.randn(
            size=(len_targets, num_features), dtype=torch.float64)
        designmat, _ = torch.qr(designmat)
        # Construct the optimal solution
        weights = torch.randn(size=(num_features,), dtype=torch.float64)
        weights = weights / torch.norm(weights)
        if link == 'log':
            weights *= 10
        elif link == 'logit' or link == 'probit':
            # Place 10 pairs of points straddling the decision hyperplane.
            assert len_targets >= 40
            for j in range(10):
                v = torch.randn(size=(num_features,), dtype=torch.float64)
                v = v - weights * torch.dot(v, weights)
                for k in range(designmat.shape[1]):
                    designmat[2 * j, k] = v[k] + weights[k] * .02
                    designmat[2 * j + 1, k] = v[k] - weights[k] * .02
        # Define targets to be the result of the optimal solution
        targets = designmat @ weights
        if link == 'logit':
            bias = 0
            targets = targets.sigmoid().round()
        elif link == 'probit':
            bias = 0
            targets = inverse_probit(targets).round()
        elif link == 'log':
            bias = 3
            targets += bias
            targets = targets.exp().round()
    # Convert the data to be processed from double- to single-precision
    return designmat.float(), targets.float(), weights, float(bias)


def load_data(dataset='mnist', *args, **kwargs):
    if dataset == 'horses':
        return load_horses(*args, **kwargs)
    elif dataset == 'mnist':
        return load_mnist(*args, **kwargs)
    elif dataset == 'synth':
        return load_synth(*args, **kwargs)
    elif dataset == 'covtype':
        return load_covtype(*args, **kwargs)


def load_data_sampler(A, y, batchsize=8):
    perm = torch.randperm(A.size(0))

    if dist.is_initialized():
        dist.broadcast(perm, 0)

    # define simple dataset sampler:
    def sampler():
        while True:
            for idx in range(0, A.size(0), batchsize):
                yield (
                    A[perm[idx:(idx + batchsize)], :],
                    y[perm[idx:(idx + batchsize)]],
                )

    # return sampler:
    return sampler
