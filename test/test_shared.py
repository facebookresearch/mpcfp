#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# dependencies:
import itertools
import sys
import torch
import unittest
import torch.distributed as dist

import sys
sys.path.append("..")
from shared import SharedTensor

from multiprocess_test_case import MultiProcessTestCase


def get_random_test_tensor(max_value=6, size=(5, 5)):
    tensor = (2 * torch.rand(*size, dtype=torch.float64) - 1) * max_value
    if dist.is_initialized():
        dist.broadcast(tensor, 0)
    return tensor.type(torch.float64)


class TestShared(MultiProcessTestCase):
    """
        This class tests all functions of the shared tensors.
    """
    benchmarks_enabled = False

    def setUp(self):
        super().setUp()

    def _check(self, encrypted_tensor, reference, msg, tol=1e-4):
        tensor = encrypted_tensor.get_plain_text()

        if self.rank != 0:   # Do not check for non-0 rank
            return

        # Check sizes match
        self.assertTrue(tensor.size() == reference.size(), msg)
        test_passed = torch.allclose(
            tensor, reference, rtol=tol, atol=tol)
        self.assertTrue(test_passed, msg=msg)

    def test_encrypt_decrypt(self):
        """
            Tests tensor encryption and decryption for both positive
            and negative values.
        """
        reference = get_random_test_tensor()
        encrypted_tensor = SharedTensor(reference)
        self._check(encrypted_tensor, reference, 'en/decryption failed')

    def test_clone(self):
        reference = get_random_test_tensor()
        encrypted_tensor = SharedTensor(reference)
        cloned = encrypted_tensor.clone()
        self._check(cloned, reference, 'cloning failed')

    def test_arithmetic(self):
        """Tests arithmetic functions on encrypted tensor."""
        arithmetic_functions = ['add', 'add_', 'sub', 'sub_', 'mul', 'mul_']
        for func in arithmetic_functions:
            for tensor_type in [lambda x: x, SharedTensor]:
                tensor1 = get_random_test_tensor()
                tensor2 = get_random_test_tensor()
                encrypted = SharedTensor(tensor1)
                encrypted2 = tensor_type(tensor2)
                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted, func)(encrypted2)
                msg = '%s %s failed' % (
                    'private' if tensor_type is SharedTensor else 'public',
                    func)
                self._check(encrypted_out, reference, msg)
                if '_' in func:
                    # Check in-place op worked
                    self._check(encrypted, reference, msg)
                else:
                    # Check original is not modified
                    self._check(encrypted, tensor1, msg)

            # Check encrypted vector with encrypted scalar works.
            tensor1 = get_random_test_tensor()
            tensor2 = get_random_test_tensor(size=(1, 1))
            encrypted1 = SharedTensor(tensor1)
            encrypted2 = SharedTensor(tensor2)
            reference = getattr(tensor1, func)(tensor2)
            encrypted_out = getattr(encrypted1, func)(encrypted2)
            self._check(encrypted_out, reference, msg)

        # Test radd, rsub, and rmul
        tensor = get_random_test_tensor()
        reference = 2 + tensor
        encrypted = SharedTensor(tensor)
        encrypted_out = 2 + encrypted
        self._check(encrypted_out, reference, 'right add failed')

        reference = 2 - tensor
        encrypted_out = 2 - encrypted
        self._check(encrypted_out, reference, 'right sub failed')

        reference = 2 * tensor
        encrypted_out = 2 * encrypted
        self._check(encrypted_out, reference, 'right mul failed')

    def test_broadcast(self):
        """Test broadcast functionality."""
        arithmetic_functions = ['add', 'sub', 'mul']
        sizes = [(2, 2), (2, 1), (1, 2), (1, 1)]
        for func in arithmetic_functions:
            for tensor_type in [lambda x: x, SharedTensor]:
                for size1, size2 in itertools.combinations(sizes, 2):
                    tensor1 = get_random_test_tensor(size=size1)
                    tensor2 = get_random_test_tensor(size=size2)
                    encrypted = SharedTensor(tensor1)
                    encrypted2 = tensor_type(tensor2)
                    reference = getattr(tensor1, func)(tensor2)
                    encrypted_out = getattr(encrypted, func)(encrypted2)
                    self._check(
                        encrypted_out, reference,
                        '%s %s failed' % ('private' if tensor_type
                                          is SharedTensor else 'public', func)
                    )

    def test_transpose(self):
        """Tests transpose on encrypted tensor."""
        funcs = ['transpose', 'transpose_']
        for func in funcs:
            tensor = get_random_test_tensor()
            encrypted = SharedTensor(tensor)
            reference = getattr(tensor, func)(0, 1)
            encrypted_out = getattr(encrypted, func)(0, 1)
            msg = 'private %s failed' % func
            self._check(encrypted_out, reference, msg)
            if '_' in func:
                # Check in-place op worked
                self._check(encrypted, reference, msg)
            else:
                # Check original is not modified
                self._check(encrypted, tensor, msg)

        # Check property
        tensor = get_random_test_tensor()
        encrypted = SharedTensor(tensor)
        self._check(encrypted.T, tensor.T, msg)

    def test_matmul(self):
        """Test matrix multiplication."""
        for tensor_type in [lambda x: x, SharedTensor]:
            tensor = get_random_test_tensor()
            for width in range(2, tensor.shape[1]):
                matrix_size = (tensor.shape[1], width)
                matrix = get_random_test_tensor(size=matrix_size)
                reference = tensor.matmul(matrix)
                encrypted_tensor = SharedTensor(tensor)
                matrix = tensor_type(matrix)
                encrypted_tensor = encrypted_tensor.matmul(matrix)
                self._check(
                    encrypted_tensor, reference,
                    'Private-%s matrix multiplication failed' %
                    ('private' if tensor_type is SharedTensor else 'public')
                )

    def test_reductions(self):
        """Test reduction operations."""
        funcs = ['sum']
        for func in funcs:
            tensor = get_random_test_tensor()
            encrypted = SharedTensor(tensor)
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, 'private %s failed' % func)

            for dim in [0, 1]:
                reference = getattr(tensor, func)(dim)
                encrypted_out = getattr(encrypted, func)(dim)
                self._check(
                    encrypted_out, reference, 'private %s failed' % func)

    def test_get_set(self):
        for size in range(1, 5):
            # Test __getitem__
            tensor = get_random_test_tensor(size=(size, size))
            reference = tensor[:, 0]

            encrypted_tensor = SharedTensor(tensor)
            encrypted_out = encrypted_tensor[:, 0]
            self._check(encrypted_out, reference, 'getitem failed')

            reference = tensor[0, :]
            encrypted_out = encrypted_tensor[0, :]
            self._check(encrypted_out, reference, 'getitem failed')

            for encrypted_type in [lambda x: x, SharedTensor]:
                # Test __setitem__
                tensor2 = get_random_test_tensor(size=(size,))
                reference = tensor.clone()
                reference[:, 0] = tensor2

                encrypted_out = SharedTensor(tensor)
                encrypted2 = encrypted_type(tensor2)
                encrypted_out[:, 0] = encrypted2

                self._check(
                    encrypted_out, reference,
                    '%s setitem failed' % type(encrypted2))

                reference = tensor.clone()
                reference[0, :] = tensor2

                encrypted_out = SharedTensor(tensor)
                encrypted2 = encrypted_type(tensor2)
                encrypted_out[0, :] = encrypted2

                self._check(
                    encrypted_out, reference,
                    '%s setitem failed' % type(encrypted2))

    def test_cuda(self):
        if not torch.cuda.is_available():
            return
        tensor1 = SharedTensor(get_random_test_tensor())
        tensor2 = SharedTensor(get_random_test_tensor())
        reference = (tensor1 * tensor2).get_plain_text()
        tensor1 = tensor1.cuda()
        tensor2 = tensor2.cuda()
        out = tensor1 * tensor2
        self._check(out.cpu(), reference, "CUDA op failed")

    def test_conv(self):
        """Test convolution of encrypted tensor with public/private tensors."""
        for kernel_type in [lambda x: x, SharedTensor]:
            for matrix_width in range(2, 5):
                for kernel_width in range(1, matrix_width):
                    for padding in range(kernel_width // 2 + 1):
                        matrix_size = (5, matrix_width)
                        matrix = get_random_test_tensor(size=matrix_size)

                        kernel_size = (kernel_width, kernel_width)
                        kernel = get_random_test_tensor(size=kernel_size)

                        matrix = matrix.unsqueeze(0).unsqueeze(0)
                        kernel = kernel.unsqueeze(0).unsqueeze(0)

                        reference = torch.nn.functional.conv2d(
                            matrix, kernel, padding=padding)
                        encrypted_matrix = SharedTensor(matrix)
                        encrypted_kernel = kernel_type(kernel)
                        encrypted_conv = encrypted_matrix.conv2d(
                            encrypted_kernel, padding=padding
                        )

                        self._check(encrypted_conv, reference, 'conv2d failed')

    def test_pooling(self):
        """Test average pooling on encrypted tensor."""
        for width in range(2, 5):
            for width2 in range(1, width):
                matrix_size = (4, 5, width)
                matrix = get_random_test_tensor(size=matrix_size)
                pool_size = width2
                for stride in range(1, width2):
                    for padding in range(2):
                        reference = torch.nn.functional.avg_pool2d(
                            matrix.unsqueeze(0), pool_size,
                            stride=stride, padding=padding
                        )

                        encrypted_matrix = SharedTensor(matrix)
                        encrypted_pool = encrypted_matrix.avg_pool2d(
                            pool_size, stride=stride, padding=padding)
                        self._check(
                            encrypted_pool, reference[0], 'avg_pool2d failed')

    def test_square(self):
        """Test square."""
        funcs = ['square', 'square_']
        for func in funcs:
            tensor = get_random_test_tensor()
            encrypted = SharedTensor(tensor)
            reference = tensor * tensor
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, 'private %s failed' % func)

    def test_div(self):
        """Tests division by numbers in [1, 2] on encrypted tensor."""
        funcs = ['div', 'div_']
        for func in funcs:
            for tensor_type in [lambda x: x, SharedTensor]:
                tensor1 = get_random_test_tensor()
                tensor2 = get_random_test_tensor(max_value=0.5) + 1.5
                encrypted = SharedTensor(tensor1)
                encrypted2 = tensor_type(tensor2)
                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted, func)(encrypted2)
                msg = '%s %s failed' % (
                    'private' if tensor_type is SharedTensor else 'public',
                    func)
                self._check(encrypted_out, reference, msg)
                if '_' in func:
                    # Check in-place op worked
                    self._check(encrypted, reference, msg)
                else:
                    # Check original is not modified
                    self._check(encrypted, tensor1, msg)

    def test_sign(self):
        """Tests sign on encrypted tensor."""
        funcs = ['sign', 'sign_', 'abs', 'abs_', 'relu', 'relu_']
        for func in funcs:
            tensor = get_random_test_tensor(max_value=1e4)
            if func != 'sign' and func != 'sign_':
                # Make sure we test with some entry, say entry (0, 0), being 0
                tensor[0, 0] = 0
            encrypted = SharedTensor(tensor)
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_exp(self):
        """Tests exp on encrypted tensor."""
        funcs = ['exp', 'exp_']
        for func in funcs:
            tensor = get_random_test_tensor(max_value=2)
            encrypted = SharedTensor(tensor)
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_tanh(self):
        funcs = ['tanh', 'tanh_']
        for func in funcs:
            tensor = get_random_test_tensor(max_value=2)
            encrypted = SharedTensor(tensor)
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_sigmoid(self):
        funcs = ['sigmoid', 'sigmoid_']
        for func in funcs:
            tensor = get_random_test_tensor(max_value=2)
            encrypted = SharedTensor(tensor)
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_softmax(self):
        axes = [0, 1, 2]
        funcs = ['softmax', 'softmax_']
        for axis in axes:
            for func in funcs:
                tensor = get_random_test_tensor(max_value=5, size=(5, 5, 5))
                encrypted = SharedTensor(tensor)
                encrypted_out = getattr(encrypted, func)(axis)
                # Calculate the plaintext reference.
                x = tensor.clone()
                x.relu_()
                x = x.sum(axis, keepdim=True)
                tensor.sub_(x)
                tensor.exp_()
                tensor.mul_(tensor.sum(axis, keepdim=True).reciprocal_())
                reference = tensor
                # Reduce the tolerance requested for softmax.
                self._check(encrypted_out, reference, "%s failed" % func)

    def test_erf(self):
        funcs = ['erf', 'erf_']
        for func in funcs:
            tensor = get_random_test_tensor(max_value=2)
            encrypted = SharedTensor(tensor)
            reference = getattr(tensor, func)()
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, "%s failed" % func)

    def test_reciprocal(self):
        """Test reciprocal."""
        funcs = ['reciprocal', 'reciprocal_']
        for func in funcs:
            for scale in [2, 10]:
                tensor = get_random_test_tensor(max_value=(scale - 1) / 2)
                tensor += 1 + (scale - 1) / 2
                encrypted = SharedTensor(tensor)
                reference = getattr(tensor, func)()
                encrypted_out = getattr(encrypted, func)(scale=scale)
                self._check(
                    encrypted_out, reference, 'private %s failed' % func)

    def test_invsqrt(self):
        """Test invsqrt."""
        funcs = ['invsqrt', 'invsqrt_']
        for func in funcs:
            tensor = (get_random_test_tensor(max_value=1000) + 1001) / 2001
            encrypted = SharedTensor(tensor)
            reference = tensor.sqrt().reciprocal()
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, 'private %s failed' % func)

    def test_inv8root(self):
        """Test inv8root."""
        funcs = ['inv8root', 'inv8root_']
        for func in funcs:
            tensor = (get_random_test_tensor(max_value=1000) + 1001) / 2001
            encrypted = SharedTensor(tensor)
            reference = tensor.sqrt()
            reference = reference.sqrt()
            reference = reference.sqrt()
            reference = reference.reciprocal()
            encrypted_out = getattr(encrypted, func)()
            self._check(encrypted_out, reference, 'private %s failed' % func)


# run all the tests:
if __name__ == '__main__':
    unittest.main()
