#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

tbsm_py="python tbsm_pytorch.py "

$tbsm_py  --mode="train"  --debug-mode --dlrm-path=$1 --datatype="synthetic" \
--model-type="tsl" --tsl-inner="def" --tsl-num-heads=1 \
--save-model=./output/model.pt --num-train-pts=100 --num-val-pts=20 --points-per-user=1 --mini-batch-size=32 \
--nepochs=1 --numpy-rand-seed=123 --arch-embedding-size="5-4-3" --print-freq=10 --test-freq=10 --num-batches=0  \
--pro-train-file=./output/train.npz --pro-val-file=./output/val.npz --ts-length=20  --device-num=0   \
--tsl-interaction-op="dot" --tsl-mechanism="mlp" --learning-rate=0.05 --no-select-seed
