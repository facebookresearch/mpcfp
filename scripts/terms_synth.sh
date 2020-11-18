#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

for t in 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40
do
  echo "Launching terms $t"
  ./launch_private.sh --iter 10000 --link logit --tanhterms $t --dataset synth &> results/synth_terms${t}_link_logit.txt
  ./launch_private.sh --iter 10000 --link probit --erfterms $t --dataset synth &> results/synth_terms${t}_link_probit.txt
done
