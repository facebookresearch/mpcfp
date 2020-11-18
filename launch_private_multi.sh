#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

REPO_ROOT=`pwd`
CMD="$REPO_ROOT/multi_classification.py"
WORLD_SIZE=2

python $REPO_ROOT/distributed_launcher.py  --world_size "$WORLD_SIZE" "$CMD" "$@"
