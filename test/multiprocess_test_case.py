#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import os
import sys
import tempfile
import unittest
from functools import wraps
import torch
import torch.distributed as dist


def configure_logging():
    """Configures a logging template for debugging multiple processes."""

    level = logging.INFO
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format=(
            "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]"
            + "[%(processName)s] %(message)s"
        ),
    )


class MultiProcessTestCase(unittest.TestCase):
    MAIN_PROCESS_RANK = -1

    @property
    def world_size(self):
        return 2

    @staticmethod
    def join_or_run(fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn(self)

        return wrapper

    # The main process spawns N subprocesses that run the test.
    # This function patches overwrites every test function to either
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    @classmethod
    def setUpClass(cls):
        for attr in dir(cls):
            if attr.startswith("test"):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))

    def __init__(self, methodName):
        super().__init__(methodName)

        self.rank = self.MAIN_PROCESS_RANK
        self.mp_context = multiprocessing.get_context("spawn")

    def setUp(self):
        super(MultiProcessTestCase, self).setUp()

        configure_logging()

        # This gets called in the children process as well to give subclasses a
        # chance to initialize themselves in the new process
        if self.rank == self.MAIN_PROCESS_RANK:
            self.file = tempfile.NamedTemporaryFile(delete=True).name
            self.processes = [
                self._spawn_process(rank)
                for rank in range(int(self.world_size))
            ]

    def tearDown(self):
        super(MultiProcessTestCase, self).tearDown()
        for p in self.processes:
            p.terminate()

    def _current_test_name(self):
        # self.id() == e.g.
        # '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def _spawn_process(self, rank):
        name = "process " + str(rank)
        test_name = self._current_test_name()
        process = self.mp_context.Process(
            target=self.__class__._run, name=name,
            args=(test_name, rank, self.file)
        )
        process.start()
        return process

    @classmethod
    def _run(cls, test_name, rank, file):
        self = cls(test_name)

        self.file = file
        self.rank = int(rank)

        # set environment variables:
        communicator_args = {
            "WORLD_SIZE": self.world_size,
            "RANK": self.rank,
            "RENDEZVOUS": "file://%s" % self.file,
            "BACKEND": "gloo",
        }
        for key, val in communicator_args.items():
            os.environ[key] = str(val)

#        crypten.init()
        self.setUp()

        # We're retrieving a corresponding test and executing it.
        getattr(self, test_name)()
#        crypten.uninit()
        sys.exit(0)

    def _join_processes(self, fn):
        for p in self.processes:
            p.join()
            self._check_return_codes(p)

    def _check_return_codes(self, process):
        self.assertEqual(process.exitcode, 0)
