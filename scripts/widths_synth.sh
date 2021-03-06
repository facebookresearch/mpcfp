#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

for link in 'identity' 'logit' 'probit'
do
  for w in 10 100 1000 10000 100000 1000000 5000000
  do
    echo "Launching width $w"
    ./launch_private.sh --width $w --iter 10000 --link $link --dataset synth &> results/synth_width${w}_link_${link}.txt
  done
done
