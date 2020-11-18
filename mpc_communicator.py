#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import logging
import os


state = {
    'backend': 'gloo',
    'rendezvous': None,
    'world_size': 1,
    'rank': 0,
    'comm_rounds': 0,
    'comm_bytes': 0,
}

defaults = {
    'BACKEND': 'gloo',
    'RENDEZVOUS': 'file:///tmp/sharedfile',
}

BYTES_PER_ELEMENT = 8


def multiprocess(func):
    def single_process_wrapper(*args, **kwargs):
        if state['world_size'] < 2:
            if func.__name__ in ['gather', 'all_gather']:
                return [args[0]]
            else:
                return args[0]
        else:
            return func(*args, **kwargs)

    return single_process_wrapper


def initialize(**kwargs):
    if dist.is_initialized():
        return

    for var in state.keys():
        config_arg = None
        var = var.upper()
        if var in kwargs:
            config_arg = kwargs[var]
        elif var in os.environ:
            config_arg = os.environ[var]
        elif var in defaults:
            config_arg = defaults[var]

        var = var.lower()
        if config_arg is not None:
            state[var] = config_arg

    state['world_size'] = int(state['world_size'])
    state['rank'] = int(state['rank'])

    level = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    logging.info('==================')
    logging.info('Initializing communicator with rank %d' % state['rank'])
    logging.info('==================')
    logging.getLogger().setLevel(level)

    dist.init_process_group(
        backend=state['backend'],
        init_method=state['rendezvous'],
        world_size=state['world_size'],
        rank=state['rank'],
    )


@multiprocess
def send(tensor, dst):
    dist.send(tensor, dst)


@multiprocess
def recv(tensor, src=None):
    dist.recv(tensor, src=src)
    return tensor


@multiprocess
def isend(tensor, dst):
    dist.isend(tensor, dst)


@multiprocess
def irecv(tensor, src=None):
    dist.irecv(tensor, src=src)
    return tensor


@multiprocess
def scatter(scatter_list, src, size=None, async_op=False, dtype=torch.long,
            device=None):
    if src != state['rank']:
        if size is None:
            size = scatter_list[state['rank']].size()
        tensor = torch.empty(size=size, dtype=dtype, device=device)
        dist.scatter(tensor, [], src, async_op=async_op)
    else:
        tensor = scatter_list[state['rank']]
        assert all(t.is_contiguous() for t in scatter_list), \
            "Tensors must be contiguous for scatter"
        dist.scatter(tensor, scatter_list, src, async_op=async_op)
    return tensor


@multiprocess
def all_reduce(tensor, async_op=False):
    dist.all_reduce(tensor, async_op=async_op)
    return tensor


@multiprocess
def gather(tensor, dst, async_op=False, dtype=torch.long):
    if state['rank'] == dst:
        result = []
        for _ in range(state['world_size']):
            result.append(torch.empty(size=tensor.size(), dtype=dtype))
        dist.gather(tensor, result, dst, async_op=async_op)
        return result
    dist.gather(tensor, [], dst, async_op=async_op)


@multiprocess
def all_gather(tensor, async_op=False, dtype=torch.long):
    result = []
    for _ in range(state['world_size']):
        result.append(torch.empty(size=tensor.size(), dtype=dtype))
    dist.all_gather(result, tensor, async_op=async_op)
    return result


@multiprocess
def broadcast(tensor, src, async_op=False):
    dist.broadcast(tensor, src, async_op=async_op)
    return tensor


def get_backend():
    return state['backend']


def get_rank():
    return state['rank']


def get_world_size():
    return state['world_size']


def _log_communication(nelement):
    state['comm_rounds'] += 1
    state['comm_bytes'] += (nelement * BYTES_PER_ELEMENT)


def reset_communication_stats():
    state['comm_rounds'] = 0
    state['comm_bytes'] = 0


def print_communication_stats():
    logging.info('====Communication Stats====')
    logging.info('Rounds: %d' % state['comm_rounds'])
    logging.info('Bytes : %d' % state['comm_bytes'])
