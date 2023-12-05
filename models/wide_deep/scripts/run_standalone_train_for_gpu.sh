#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# 
echo "Usage: bash run_standalone_train_for_gpu.sh EPOCHS DATASET"
execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)

EPOCHS=$1
DATASET=$2
DEVICE_ID=$3

if [[ "X$DEVICE_TARGET" == "XGPU" ]]; then
  if [[ ! -n "$DEVICE_ID" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    export CUDA_VISIBLE_DEVICES=$DEVICE_ID
  fi
fi

rm -rf ${execute_path}/log/
mkdir ${execute_path}/log/
cp ${self_path}/../op_precision.ini ${execute_path}/log/
cd ${execute_path}/log/ || exit

python -s ${self_path}/../train_and_eval.py             \
    --device_target="GPU"                               \
    --data_path=$DATASET                                \
    --batch_size=16000                                  \
    --epochs=$EPOCHS > log.txt 2>&1 &
