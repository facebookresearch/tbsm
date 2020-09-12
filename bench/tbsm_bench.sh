#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

tbsm_py="python tbsm_pytorch.py "

$tbsm_py  --use-gpu  --mode="train-test"  --dlrm-path=$1 --datatype="taobao" \
--model-type="tsl" --tsl-inner="def"  --tsl-num-heads=1 \
--save-model=./output/model.pt --num-train-pts=690000 --num-val-pts=300000 --points-per-user=10 --mini-batch-size=256 \
--nepochs=1 --numpy-rand-seed=123 --arch-embedding-size="987994-4162024-9439" --print-freq=100 --test-freq=2000 --num-batches=0  \
--raw-train-file=./input/taobao_train.txt --raw-test-file=./input/taobao_test.txt --pro-train-file=./output/taobao_train_t20.npz \
--pro-val-file=./output/taobao_val_t20.npz --pro-test-file=./output/taobao_test_t20.npz --ts-length=20  \
--device-num=0 --tsl-interaction-op="dot" --tsl-mechanism="mlp" --learning-rate=0.05  --arch-sparse-feature-size=16 \
--arch-mlp-bot="1-16" --arch-mlp-top="15-15" --tsl-mlp="15-15" --arch-mlp="60-1"
