#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
#
# This file contains a simple PyTorch based implementation of multi-party
# computation (MPC) using additive sharing with uniform noise.
#
# =============================================================================

# dependencies:
import collections
import math
import torch

import approximations
import mpc_communicator as comm

SENTINEL = -1


def sample_uniform(shape, device):
    return torch.empty(
        size=shape, dtype=torch.float64, device=device).uniform_(
        -SharedTensor.config.width / 2, SharedTensor.config.width / 2)


def share(tensor, num_parties):
    if num_parties == 1:
        return tensor
    shares = [sample_uniform(tensor.size(), tensor.device)
              for _ in range(num_parties - 1)]
    # TODO: Use Ruiyu Zhu's trick when num_parties > 2
    shares.append(tensor - sum(shares))
    return shares


def beaver_triple(size1, size2, binary_op, device):
    a = sample_uniform(size1, device)
    b = sample_uniform(size2, device)
    c = binary_op(a, b)
    a, b = SharedTensor(a), SharedTensor(b)
    # Use w^2 noise for c since c = a*b
    w = SharedTensor.config.width
    SharedTensor.config.width = w**2
    c = SharedTensor(c)
    SharedTensor.config.width = w
    return a, b, c


def beaver_protocol(x, y, binary_op):
    a, b, c = beaver_triple(x.shape, y.shape, binary_op, x.device)
    epsilon = x.sub(a).get_plain_text()
    delta = y.sub(b).get_plain_text()
    result = c._tensor
    result += binary_op(epsilon, b._tensor) + binary_op(a._tensor, delta)
    # Aggregate in all-reduce only at one of the parties
    # (the party with rank == 0), instead of downweighting
    # by the total number of parties
    if x._rank == 0:
        result += binary_op(epsilon, delta)
    return result


def beaver_square(x):
    a = sample_uniform(x.shape, x.device)
    c = a * a
    a = SharedTensor(a)
    # Use w^2 noise for c since c = a^2
    w = SharedTensor.config.width
    SharedTensor.config.width = w**2
    c = SharedTensor(c)
    SharedTensor.config.width = w
    a = a.to(device=x.device)
    c = c.to(device=x.device)
    epsilon = x.sub(a).get_plain_text()
    result = c._tensor + 2 * epsilon * a._tensor
    # Aggregate in all-reduce only at one of the parties
    # (the party with rank == 0), instead of downweighting
    # by the total number of parties
    if x._rank == 0:
        result += epsilon * epsilon
    return result


def _broadcast(a, b):
    if torch.is_tensor(b):
        return torch.broadcast_tensors(a, b)[0].clone()
    return a.clone()


class SharedConfig:
    """
    A configuration object for the SharedTensor class.

    The object stores key properties of the sharing scheme,
    including the width of the uniform noise being added
    and the parameters used by approximating functions
    (sign, exp, tanh, erf, etc.)
    """

    ChebyshevConfig = collections.namedtuple(
        "ChebyshevConfig", "maxval terms coeffs")

    def __init__(self):
        self.width = 1000

        self.set_tanh()
        self.set_erf()

        self.exp_scale = 20
        self.sign_iters = 60
        self.invsqrt_iters = 26
        self.inv8root_iters = 26
        self.reciprocal_iters = 30

    def set_tanh(self, maxval=20, terms=120):
        self.tanh = SharedConfig.ChebyshevConfig(
            maxval=maxval,
            terms=terms,
            coeffs=approximations.chebseries(math.tanh, maxval, terms))

    def set_erf(self, maxval=20, terms=100):
        self.erf = SharedConfig.ChebyshevConfig(
            maxval=maxval,
            terms=terms,
            coeffs=approximations.chebseries(math.erf, maxval, terms))


# Arithmetically Shared Real Tensor
class SharedTensor:
    """
    Encrypted tensor type that is private.
    """
    config = SharedConfig()

    # constructors:
    def __init__(
        self,
        tensor=None,
        shares=None,
        size=None,
        src=0,
    ):
        if src == SENTINEL:
            return

        comm.initialize()

        # _rank is the rank of the current processes
        # _src is the rank of the source process that will provide data shares
        self._rank = comm.get_rank()
        self._src = src

        if self._rank == self._src:
            # encrypt tensor into private pair:
            if tensor is not None:
                assert torch.is_tensor(tensor), 'input must be a tensor'
                shares = share(tensor.type(torch.float64).contiguous(),
                               num_parties=comm.get_world_size())

            assert shares is not None, 'inputting some tensor is necessary'
            self._tensor = comm.scatter(shares, src, dtype=torch.float64)
        else:
            # TODO: Remove this line & adapt tests to use size arg
            if tensor is not None:
                size = tensor.size()
                device = tensor.device
            else:
                size = shares[0].size()
                device = shares[0].device
            self._tensor = comm.scatter(
                None, src, size=size, dtype=torch.float64, device=device)

    @staticmethod
    def init_comm(**kwargs):
        comm.initialize(**kwargs)

    @property
    def shape(self):
        return self._tensor.shape

    def size(self, *args):
        """Return tensor's size (shape)"""
        return self._tensor.size(*args)

    def dim(self):
        """Return number of dimensions in the tensor"""
        return len(self.size())

    def nelement(self):
        """Return number of elements in the tensor"""
        return self._tensor.nelement()

    def __len__(self):
        """Return length of the tensor"""
        return self.size(0)

    def view(self, *args):
        """Resize the tensor"""
        result = self.shallow_copy()
        result._tensor = self._tensor.view(*args)
        return result

    def unsqueeze(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = self._tensor.unsqueeze(*args, **kwargs)
        return result

    def squeeze(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = self._tensor.squeeze(*args, **kwargs)
        return result

    def __getitem__(self, index):
        """Index into tensor"""
        result = self.shallow_copy()
        result._tensor = self._tensor[index]
        return result

    def __setitem__(self, index, value):
        """Set tensor values by index"""
        if torch.is_tensor(value):
            value = SharedTensor(value)
        assert isinstance(value, SharedTensor), \
            'Unsupported input type %s for __setitem__' % type(value)
        self._tensor.__setitem__(index, value._tensor)

    def get_plain_text(self):
        """Decrypt the tensor"""
        return comm.all_reduce(self._tensor.clone())

    def get_plain_text_async(self):
        """Decrypt the tensor asynchronously"""
        tensor = self._tensor.clone()
        return tensor, comm.dist.all_reduce(tensor, async_op=True)

    def clone(self):
        """Create a deepcopy"""
        result = SharedTensor(src=SENTINEL)
        result._rank = self._rank
        result._src = self._src
        result._tensor = self._tensor.clone()
        return result

    def add_(self, y):
        """Perform element-wise addition"""
        return self.add(y, self)

    def add(self, y, out=None):
        """Perform element-wise addition"""
        if out is None:
            out = self.shallow_copy()
        other = y._tensor if isinstance(y, SharedTensor) else y
        if self._rank == 0 or isinstance(y, SharedTensor):
            torch.add(self._tensor, other=other, out=out._tensor)
        elif out._tensor.numel() == 0:
            out._tensor = _broadcast(self._tensor, other)
        return out

    def __iadd__(self, y):
        return self.add_(y)

    def __add__(self, y):
        return self.add(y)

    def __radd__(self, y):
        return self.add(y)

    def sub_(self, y):
        """Perform element-wise subtraction"""
        return self.sub(y, self)

    def sub(self, y, out=None):
        """Perform element-wise subtraction"""
        if out is None:
            out = self.shallow_copy()
        other = y._tensor if isinstance(y, SharedTensor) else y
        if self._rank == 0 or isinstance(y, SharedTensor):
            torch.sub(self._tensor, other=other, out=out._tensor)
        elif out._tensor.numel() == 0:
            out._tensor = _broadcast(self._tensor, other)
        return out

    def __isub__(self, y):
        return self.sub_(y)

    def __sub__(self, y):
        return self.sub(y)

    def __rsub__(self, y):
        return -self.sub(y)

    def shallow_copy(self):
        result = SharedTensor(src=SENTINEL)
        result._rank = self._rank
        result._src = self._src
        if self._tensor.is_cuda:
            result._tensor = torch.cuda.DoubleTensor()
        else:
            result._tensor = torch.DoubleTensor()
        return result

    def mul(self, y, out=None):
        if out is None:
            out = self.shallow_copy()
        if isinstance(y, SharedTensor):
            result = beaver_protocol(self, y, torch.mul)
            out._tensor.resize_(result.shape)
            out._tensor.copy_(result)
        else:
            if isinstance(y, (int, float)):
                y = torch.tensor(data=y)
            torch.mul(self._tensor, other=y, out=out._tensor)
        return out

    def mul_(self, y):
        return self.mul(y, out=self)

    def __imul__(self, y):
        return self.mul_(y)

    def __mul__(self, y):
        return self.mul(y)

    def __rmul__(self, y):
        return self.mul(y)

    def div(self, y, out=None, scale=2):
        """Perform division by private nos. in [1, scale] or any public nos."""
        if out is None:
            out = self.shallow_copy()
        if isinstance(y, SharedTensor):
            self.mul(y.reciprocal(scale=scale), out=out)
        else:
            if isinstance(y, (int, float)):
                y = torch.tensor(data=y)
            torch.div(self._tensor, other=y, out=out._tensor)
        return out

    def div_(self, y, scale=2):
        return self.div(y, out=self, scale=scale)

    def __truediv__(self, y, scale=2):
        return self.div(y, scale=scale)

    def matmul(self, y):
        out = self.shallow_copy()
        if isinstance(y, SharedTensor):
            result = beaver_protocol(self, y, torch.matmul)
            out._tensor = result
        else:
            out._tensor = self._tensor.matmul(y)
        return out

    def conv2d(self, kernel, **kwargs):
        """Perform a 2D convolution using the given kernel"""
        def _conv2d(input, weight):
            return torch.nn.functional.conv2d(input, weight, **kwargs)
        out = self.shallow_copy()
        if isinstance(kernel, SharedTensor):
            result = beaver_protocol(self, kernel, _conv2d)
            out._tensor = result
        else:
            out._tensor = _conv2d(self._tensor, kernel)
        return out

    def avg_pool2d(self, kernel_size, **kwargs):
        """Applied 2D average-pooling."""
        out = self.shallow_copy()
        out._tensor = torch.nn.functional.avg_pool2d(
            self._tensor, kernel_size, **kwargs)
        return out

    def square(self):
        return self.clone().square_()

    def square_(self):
        result = beaver_square(self)
        self._tensor.copy_(result)
        return self

    def abs(self):
        return self.clone().abs_()

    def abs_(self):
        return self.mul_(self.sign())

    def __abs__(self):
        return self.abs()

    def exp(self):
        return self.clone().exp_()

    def exp_(self):
        w = SharedTensor.config.width
        n = SharedTensor.config.exp_scale
        SharedTensor.config.width = w / 2**n
        self.div_(2**n).add_(1)
        for _ in range(n):
            self.square_()
            SharedTensor.config.width = min(w, SharedTensor.config.width * 2)
        return self

    def softmax(self, axis):
        """Perform the softmax transformation"""
        return self.clone().softmax_(axis)

    def softmax_(self, axis):
        """Perform the softmax transformation"""
        # Subtract the sum of the numbers that are positive to ensure that
        # all numbers are non-positive. Subtracting the max is more common,
        # but hard to calculate with polynomials. Approximating the max
        # with a p-norm, where p is sufficiently large, is also possible,
        # yet risks leaking information if any intermediate results are large.
        x = self.clone()
        x.relu_()
        x = x.sum(axis, keepdim=True)
        self.sub_(x)
        # Exponentiate each entry
        self.exp_()
        # Normalize
        return self.mul_(
            self.sum(axis, keepdim=True).reciprocal_(
                scale=2 * self.shape[axis]))

    def sigmoid(self):
        return self.clone().sigmoid_()

    def sigmoid_(self):
        return self.div_(2).tanh_().div_(2).add_(0.5)

    def tanh_(self):
        coeffs = SharedTensor.config.tanh.coeffs
        t = approximations.chebyshev(
            SharedTensor.config.tanh.terms,
            self / SharedTensor.config.tanh.maxval)
        self._tensor.fill_(0)
        for c, x in zip(coeffs[1::2], t):
            self.add_(x.mul_(c))
        return self

    def tanh(self):
        return self.clone().tanh_()

    def erf_(self):
        coeffs = SharedTensor.config.erf.coeffs
        t = approximations.chebyshev(
            SharedTensor.config.erf.terms,
            self / SharedTensor.config.erf.maxval)
        self._tensor.fill_(0)
        for c, x in zip(coeffs[1::2], t):
            self.add_(x.mul_(c))
        return self

    def erf(self):
        return self.clone().erf_()

    def sign(self):
        return self.clone().sign_()

    def sign_(self):
        w = SharedTensor.config.width
        # Divide by 1e5, assuming that the absval. of the input is at most 1e4
        self.div_(1e5)
        # Reduce the width of the added noise in accordance with 1e5/1e4 = 10
        SharedTensor.config.width /= 10
        n_iter = SharedTensor.config.sign_iters
        for _ in range(n_iter):
            self.mul_(self.square().neg_().add_(3).div_(2))
        SharedTensor.config.width = w
        return self

    def reciprocal(self, scale=2):
        """
        Reciprocal on the range [1, scale].
        """
        return self.clone().reciprocal_(scale=scale)

    def reciprocal_(self, scale=2):
        """
        Reciprocal on the range [1, scale].
        """
        self.mul_(1 / scale)
        x = self.clone()
        self.neg_().add_(2)
        n_iter = SharedTensor.config.reciprocal_iters
        for _ in range(n_iter):
            self.mul_(self.mul(x).neg_().add_(2))
        self.mul_(1 / scale)
        return self

    def invsqrt(self):
        return self.clone().invsqrt_()

    def invsqrt_(self):
        x = self.clone()
        self.neg_().add_(3).div_(2)
        n_iter = SharedTensor.config.invsqrt_iters
        for _ in range(n_iter):
            self.mul_(self.square().mul(x).neg_().add_(3)).div_(2)
        return self

    def inv8root(self):
        return self.clone().inv8root_()

    def inv8root_(self):
        x = self.clone()
        self.neg_().add_(9).div_(8)
        n_iter = SharedTensor.config.inv8root_iters
        for _ in range(n_iter):
            y = self.clone()
            y.square_()
            y.square_()
            y.square_()
            self.mul_(y.mul(x).neg_().add_(9)).div_(8)
        return self

    def relu(self):
        return self.clone().relu_()

    def relu_(self):
        return self.add_(self.abs()).div_(2)

    def neg_(self):
        self._tensor.neg_()
        return self

    def neg(self):
        return self.clone().neg_()

    def __neg__(self):
        return self.neg()

    def sum(self, *args, **kwargs):
        """Sum the entries of a tensor, along a given dimension if given"""
        out = self.shallow_copy()
        out._tensor = self._tensor.sum(*args, **kwargs)
        return out

    def mean(self, *args, **kwargs):
        """Average the entries of a tensor, along a given dimension if given"""
        out = self.shallow_copy()
        out._tensor = self._tensor.mean(*args, **kwargs)
        return out

    def t(self):
        return self.clone().t_()

    def t_(self):
        return self.transpose_(0, 1)

    def transpose(self, dim0, dim1):
        return self.clone().transpose_(dim0, dim1)

    def transpose_(self, dim0, dim1):
        self._tensor.transpose_(dim0, dim1)
        return self

    @property
    def T(self):
        return self.clone().transpose_(0, 1)

    @property
    def device(self):
        return self._tensor.device

    def to(self, device=None, copy=False, non_blocking=False):
        result = self.shallow_copy()
        result._tensor = self._tensor.to(
            device=device, copy=copy, non_blocking=non_blocking)
        return result

    def cpu(self):
        result = self.shallow_copy()
        result._tensor = self._tensor.cpu()
        return result

    def cuda(self, *args, **kwargs):
        result = self.shallow_copy()
        result._tensor = self._tensor.cuda(*args, **kwargs)
        return result
