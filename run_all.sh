#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p results
echo "Plotting approximations"
python scripts/plot_approximations.py
echo "Running synthetic widths"
./scripts/widths_synth.sh
echo "Running synthetic terms"
./scripts/terms_synth.sh
echo "Running mnist widths"
./scripts/widths_mnist.sh
echo "Running covtype widths"
./scripts/widths_covtype.sh
echo "Running mnist multi-class widths"
./scripts/widths_mnist_multi.sh
echo "Running covtype multi-class widths"
./scripts/widths_covtype_multi.sh
echo "Making plots"
python scripts/make_plots.py
