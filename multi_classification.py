#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time
import torch
import numpy as np

from shared import SharedTensor
import util


def parse_args():
    '''
    Parse input arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Train private or public multinomial logistic regression.')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help="dataset to use.",
                        choices=['mnist', 'synth', 'covtype'])
    parser.add_argument('--plaintext', action="store_true",
                        help="use a non-private algorithm")
    parser.add_argument('--width', default=100, type=float,
                        help="width of uniform distribution for secret shares")
    parser.add_argument('--seed', default=2019, type=float,
                        help="Seed the torch RNG.")
    parser.add_argument('--iter', default=10000, type=int,
                        help="Iterations of SGD.")
    parser.add_argument('--batchsize', default=8, type=int,
                        help="Batch size for SGD.")
    parser.add_argument('--cuda', action="store_true",
                        help="Run on CUDA device.")
    parser.add_argument('--data_path', default="/tmp", type=str,
                        help="Path to cache downloaded MNIST data.")
    parser.add_argument('--n_classes', default=2, type=int,
                        help="Number of classes.")
    return parser.parse_args()


def test_classifier(A, y, weights, bias):
    out = A.matmul(weights) + bias
    out = out.softmax(1)

    # Loss
    # Ignore values outside [0, 1] from round-off
    out.clamp_(min=1e-7, max=1 - 1e-7)
    loss = -((y * out.log()).sum(1)).mean()
    # Accuracy
    accuracy = torch.mean((np.argmax(out, 1) == np.argmax(y, 1)).float())
    logging.info(
        "Test Set: Loss {:.3f}, Accuracy {:.3f}".format(loss, accuracy))


def train_classifier(
        sampler, weights, bias, maxiter=10000, learning_rate=1e-2,
        weight_decay=0, report_iter=100, iweights=None, ibias=None):

    it = 0
    losses = []
    accuracies = []

    start = time.time()
    for A, y in sampler():
        n = A.shape[0]
        out = A.matmul(weights) + bias

        out = out.softmax(1)
        diffs = out - y
        gradw = (1 / n) * A.T.matmul(diffs)
        gradb = diffs.mean(axis=0)
        weights -= learning_rate * (gradw + weight_decay * weights)
        bias -= learning_rate * gradb

        # Open up for computing loss, accuracy
        if isinstance(A, SharedTensor):
            out = out.get_plain_text().float()
            y = y.get_plain_text().float()

        # Loss
        # Ignore values outside [0, 1] from round-off
        out.clamp_(min=1e-7, max=1 - 1e-7)
        loss = -((y * out.log()).sum(1)).mean()

        it += 1
        accuracy = torch.mean((np.argmax(out, 1) == np.argmax(y, 1)).float())
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        if it % report_iter == 0:
            if isinstance(weights, SharedTensor):
                plainweights = weights.get_plain_text()
                plainbias = bias.get_plain_text()
            else:
                plainweights = weights
                plainbias = bias
            logging.info(
                'norm of weights = {:.3f}'.format(torch.norm(plainweights)))
            if iweights is not None:
                plainweights = plainweights[:, 1] - plainweights[:, 0]
                plainnormed = plainweights / torch.norm(plainweights)
                iweightsnormed = iweights / torch.norm(iweights)
                logging.info(
                    'norm of difference of weights from iweights = {:.3f}'
                    .format(torch.norm(plainweights - iweights)))
                logging.info(
                    'same, normalizing both weights and iweights = {:.3f}'
                    .format(torch.norm(plainnormed - iweightsnormed)))
            if ibias is not None:
                logging.info(
                    'bias-ibias = ({:.3f}, {:.3f})'
                    .format((plainbias - ibias)[0], (plainbias - ibias)[1]))
                logging.info(
                    'bias-ibias, divided by norm of weights = ({:.3f}, {:.3f})'
                    .format((plainbias - ibias)[0] / torch.norm(plainweights),
                            (plainbias - ibias)[1] / torch.norm(plainweights)))
            sec_per_it = (time.time() - start) / report_iter
            avg_loss = sum(losses) / len(losses)
            avg_acc = sum(accuracies) / len(accuracies)
            logging.info(
                "Iter {}: Loss {:.3f}, Accuracy {:.3f}, msec/it {:.3f}".format(
                    it, avg_loss, avg_acc, sec_per_it * 1000))
            losses = []
            accuracies = []
            start = time.time()
        if it >= maxiter:
            break


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # set up loggers:
    logger = logging.getLogger()
    level = logging.INFO
    if 'RANK' in os.environ and os.environ['RANK'] != '0':
        level = logging.CRITICAL
    logger.setLevel(level)

    kwargs = {}
    if args.dataset == 'mnist':
        kwargs['path'] = args.data_path
        kwargs['n_classes'] = args.n_classes
        learning_rate = 1e-2
        weight_decay = 1e-3
    elif args.dataset == 'synth':
        assert args.n_classes == 2, 'dataset synth supports n_classes=2 only'
        kwargs['len_targets'] = 64
        kwargs['num_features'] = 8
        kwargs['link'] = 'logit'
        learning_rate = 3
        weight_decay = 0
    elif args.dataset == 'covtype':
        kwargs['path'] = args.data_path
        kwargs['n_classes'] = args.n_classes
        learning_rate = 1
        weight_decay = 1e-3

    returns = util.load_data(dataset=args.dataset, **kwargs)
    train_samples, train_targets = returns[0], returns[1]
    train_targets = torch.nn.functional.one_hot(
        train_targets.long(), num_classes=args.n_classes)

    test_samples = test_targets = None
    iweights = ibias = None

    if args.dataset == "synth":
        iweights, ibias = returns[2:]
    elif args.dataset in ["mnist", "covtype"]:
        test_samples, test_targets = returns[2:]
        test_targets = torch.nn.functional.one_hot(
            test_targets.long(), num_classes=args.n_classes)

    weights = 1e-2 * torch.randn((train_samples.shape[1], args.n_classes))
    bias = -5 * torch.ones(args.n_classes)

    if not args.plaintext:
        SharedTensor.config.width = args.width
        train_samples = SharedTensor(train_samples)
        train_targets = SharedTensor(train_targets)
        weights = SharedTensor(weights)
        bias = SharedTensor(bias)

    if args.cuda:
        train_samples = train_samples.cuda()
        train_targets = train_targets.cuda()
        weights = weights.cuda()
        bias = bias.cuda()

    sampler = util.load_data_sampler(
        train_samples, train_targets, batchsize=args.batchsize)

    # Default to reporting the loss exactly once per epoch
    report_iter = train_samples.shape[0] // args.batchsize
    if train_samples.shape[0] % args.batchsize != 0:
        logging.info('The number of targets is NOT a multiple of batchsize.')
    train_classifier(
        sampler, weights, bias, maxiter=args.iter, learning_rate=learning_rate,
        weight_decay=weight_decay, report_iter=report_iter, iweights=iweights,
        ibias=ibias)

    if test_samples is not None:
        if isinstance(weights, SharedTensor):
            weights = weights.get_plain_text().type(torch.float32)
            bias = bias.get_plain_text().type(torch.float32)
        if args.cuda:
            test_samples = test_samples.cuda()
            test_targets = test_targets.cuda()
        test_classifier(
            test_samples, test_targets, weights, bias)


if __name__ == "__main__":
    main()
