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

from shared import SharedTensor
import util


def parse_args():
    '''
    Parse input arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Train private or public linear logistic regression.')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help="dataset to use.",
                        choices=['horses', 'mnist', 'synth', 'covtype'])
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
                        help="Path to cache downloaded data set.")
    parser.add_argument('--link', default="identity", type=str,
                        help="Link function",
                        choices=['identity', 'logit', 'probit', 'log'])
    parser.add_argument('--degree', default=2, type=int,
                        help="Degree of polynomial over years for horses.")
    parser.add_argument('--onehotenc', action="store_true",
                        help="Use one-hot encodings for Prussian corps.")
    parser.add_argument('--tanhterms', default=120, type=int,
                        help="Order of Chebyshev approximation to tanh.")
    parser.add_argument('--erfterms', default=100, type=int,
                        help="Order of Chebyshev approximation to erf.")
    return parser.parse_args()


def test_classifier(A, y, weights, bias, link):
    out = A.matmul(weights) + bias
    if link == "logit":
        out = out.sigmoid()
    elif link == "probit":
        out = util.inverse_probit(out)

    # Loss
    if link == 'logit' or link == 'probit':
        # Ignore values outside [0, 1] from round-off
        out.clamp_(min=1e-7, max=1 - 1e-7)
        loss = -(y * out.log() + (1 - y) * (1 - out).log())
        loss = loss.mean()
    elif link == 'log':
        # Ignore small values from round-off
        out.clamp_(min=1e-7)
        loss = -(y * out.log() - out - (y + 1).lgamma()).mean()
    else:
        loss = ((out - y) ** 2).mean()

    # Accuracy
    if link == 'logit' or link == 'probit':
        accuracy = torch.mean(((out > 0.5).float() == y).float())
        logging.info(
            "Test Set: Loss {:.3f}, Accuracy {:.3f}".format(loss, accuracy))
    else:
        logging.info(
            "Test Set: Loss {:.3f}".format(loss))


def train_classifier(
        sampler, weights, bias, maxiter=10000, learning_rate=1e-2,
        weight_decay=0, report_iter=100, link="identity", iweights=None,
        ibias=None):

    it = 0
    losses = []
    ilosses = []
    accuracies = []

    start = time.time()
    for A, y in sampler():
        n = A.shape[0]
        out = A.matmul(weights) + bias

        if link == "logit":
            out = out.sigmoid()
        elif link == "probit":
            out = util.inverse_probit(out)
        elif link == "log":
            out = out.exp()

        diffs = out - y
        gradw = (1 / n) * A.T.matmul(diffs)
        gradb = diffs.mean()
        weights -= learning_rate * (gradw + weight_decay * weights)
        bias -= learning_rate * gradb

        # Open up for computing loss, accuracy
        if isinstance(A, SharedTensor):
            out = out.get_plain_text().float()
            y = y.get_plain_text().float()

        # Loss
        if link == 'logit' or link == 'probit':
            # Ignore values outside [0, 1] from round-off
            out.clamp_(min=1e-7, max=1 - 1e-7)
            loss = -(y * out.log() + (1 - y) * (1 - out).log()).mean()
        elif link == 'log':
            # Ignore small values from round-off
            out.clamp_(min=1e-7)
            loss = -(y * out.log() - out - (y + 1).lgamma()).mean()
            if iweights is not None:
                # Calculate ideal loss
                if isinstance(A, SharedTensor):
                    iout = (A.get_plain_text().matmul(iweights) + ibias).exp()
                else:
                    iout = (A.matmul(iweights.float()) + ibias).exp()
                iout.clamp_(min=1e-7)
                iloss = -(y * iout.log() - iout - (y + 1).lgamma()).mean()
        else:
            loss = ((out - y) ** 2).sum() / n

        it += 1
        if link == 'logit' or link == 'probit':
            accuracy = torch.mean(((out > 0.5).float() == y).float())
            accuracies.append(accuracy.item())
        losses.append(loss.item())
        if link == 'log' and iweights is not None:
            ilosses.append(iloss.item())
        if it % report_iter == 0:
            if isinstance(weights, SharedTensor):
                plainweights = weights.get_plain_text()
                plainbias = bias.get_plain_text()
            else:
                plainweights = weights
                plainbias = bias
            plainnormed = plainweights / torch.norm(plainweights)
            logging.info(
                'norm of weights = {:.3f}'.format(torch.norm(plainweights)))
            if iweights is not None:
                iweightsnormed = iweights / torch.norm(iweights)
                logging.info(
                    'norm of difference of weights from iweights = {:.3f}'
                    .format(torch.norm(plainweights - iweights)))
                logging.info(
                    'same, normalizing both weights and iweights = {:.3f}'
                    .format(torch.norm(plainnormed - iweightsnormed)))
            if ibias is not None:
                logging.info(
                    'bias-ibias = {:.3f}, divided by norm of weights = {:.3f}'
                    .format((plainbias - ibias)[0],
                            (plainbias - ibias)[0] / torch.norm(plainweights)))
            sec_per_it = (time.time() - start) / report_iter
            avg_loss = sum(losses) / len(losses)
            if link == 'logit' or link == 'probit':
                avg_acc = sum(accuracies) / len(accuracies)
                logging.info(
                    "Iter {}: Loss {:.3f}, Accuracy {:.3f}, msec/it {:.3f}"
                    .format(it, avg_loss, avg_acc, sec_per_it * 1000))
            elif link == 'log' and iweights is not None:
                avg_iloss = sum(ilosses) / len(ilosses)
                logging.info(
                    "Iter {}: Loss {:.3f}, Ideal Loss {:.3f}, msec/it {:.3f}"
                    .format(it, avg_loss, avg_iloss, sec_per_it * 1000))
            else:
                logging.info(
                    "Iter {}: Loss {:.3f}, msec/it {:.3f}"
                    .format(it, avg_loss, sec_per_it * 1000))
            losses = []
            ilosses = []
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
    if args.dataset == 'horses':
        assert args.link == 'log', "dataset horses requires link='log'"
        kwargs['degree'] = args.degree
        kwargs['onehotenc'] = args.onehotenc
        learning_rate = 2e-2
    elif args.dataset == 'mnist':
        kwargs['path'] = args.data_path
        learning_rate = 1e-3
    elif args.dataset == 'synth':
        kwargs['len_targets'] = 64
        kwargs['num_features'] = 8
        kwargs['link'] = args.link
        if args.link == 'identity':
            learning_rate = 3e-2
        elif args.link == 'logit' or args.link == 'probit':
            learning_rate = 3
        elif args.link == 'log':
            learning_rate = 3e-3
    elif args.dataset == 'covtype':
        kwargs['path'] = args.data_path
        if args.link == "logit" or args.link == "probit":
            learning_rate = 3
        else:
            learning_rate = 1e-1

    returns = util.load_data(dataset=args.dataset, **kwargs)
    train_samples, train_targets = returns[0], returns[1]

    test_samples = test_targets = None
    iweights = ibias = None

    if args.dataset == "synth":
        iweights, ibias = returns[2:]
    elif args.dataset in ["mnist", "covtype"]:
        test_samples, test_targets = returns[2:]

    weights = 1e-1 * torch.randn(train_samples.shape[1])
    bias = torch.zeros(1)

    if not args.plaintext:
        SharedTensor.config.width = args.width
        SharedTensor.config.set_tanh(terms=args.tanhterms)
        SharedTensor.config.set_erf(terms=args.erfterms)
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
        logging.info('The number of samples is NOT a multiple of batchsize.')

    train_classifier(
        sampler, weights, bias, maxiter=args.iter, link=args.link,
        learning_rate=learning_rate, report_iter=report_iter,
        iweights=iweights, ibias=ibias)

    # Evaluate classifier on the test if available
    if test_samples is not None:
        if isinstance(weights, SharedTensor):
            weights = weights.get_plain_text().type(torch.float32)
            bias = bias.get_plain_text().type(torch.float32)
        if args.cuda:
            test_samples = test_samples.cuda()
            test_targets = test_targets.cuda()
        test_classifier(
            test_samples, test_targets, weights, bias, link=args.link)


if __name__ == "__main__":
    main()
