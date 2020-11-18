#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

for w in 1000 10000 100000 1000000
do
  echo "Launching width $w"
  ./launch_private_multi.sh --width $w --iter 24000 --n_classes 10 --batchsize 50 &> results/mnist_width${w}_multi.txt
done
