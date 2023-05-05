#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

# Usage: bash run_parameter_server_standalone_train_terabyte_scale_model.sh EPOCHS DEVICE_TARGET
#            DATASET_PATH SERVER_NUM SCHED_HOST SCHED_PORT VOCAB_CACHE_SIZE [DEVICE_ID]
# DEVICE_ID is optional, default value is zero

execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export EPOCHS=$1
export DEVICE_TARGET=$2
export DATASET_PATH=$3
export MS_SCHED_NUM=1
export MS_WORKER_NUM=1
export MS_SERVER_NUM=$4
export MS_SCHED_HOST=$5
export MS_SCHED_PORT=$6
# Device cache size, default value: 4600000
export VOCAB_CACHE_SIZE=$7
DEVICE_ID=$8

# Embeding dimension: 240
EMB_DIM=240
# Total feature number in tb scale model.
VOCAB_SIZE=900000000
# Remote(server) host memory cache size(10GB), the size is adjustable, larger cache size may get better performance.
export MS_EMBEDDING_REMOTE_CACHE_MEMORY_SIZE=10
# Local device cache size(4600000 embeddings)
DEFAULT_CACHE_SIZE=4600000

if [[ ! -n "$7" ]]; then
  export VOCAB_CACHE_SIZE=${DEFAULT_CACHE_SIZE}
fi

# Set device id
if [[ "X$DEVICE_TARGET" == "XGPU" ]]; then
  if [[ ! -n "$DEVICE_ID" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    export CUDA_VISIBLE_DEVICES=$DEVICE_ID
  fi
else
  if [[ ! -n "$DEVICE_ID" ]]; then
    export DEVICE_ID=0
  else
    export DEVICE_ID=$DEVICE_ID
  fi
fi

# Start Scheduler
export MS_ROLE=MS_SCHED
rm -rf ${execute_path}/sched/
mkdir ${execute_path}/sched/
cd ${execute_path}/sched/ || exit
python -s ${self_path}/../train_and_eval_parameter_server_standalone.py --device_target=$DEVICE_TARGET          \
       --epochs=$EPOCHS --data_path=$DATASET_PATH --parameter_server=1  --vocab_size=$VOCAB_SIZE --emb_dim=$EMB_DIM  \
       --vocab_cache_size=$VOCAB_CACHE_SIZE >sched.log 2>&1 &

# Start Servers
export MS_ROLE=MS_PSERVER
for((i=0;i<$MS_SERVER_NUM;i++));
do
  rm -rf ${execute_path}/server_$i/
  mkdir ${execute_path}/server_$i/
  cd ${execute_path}/server_$i/ || exit
  python -s ${self_path}/../train_and_eval_parameter_server_standalone.py --device_target=$DEVICE_TARGET         \
         --epochs=$EPOCHS --data_path=$DATASET_PATH --parameter_server=1 --vocab_size=$VOCAB_SIZE --emb_dim=$EMB_DIM  \
         --vocab_cache_size=$VOCAB_CACHE_SIZE >server_$i.log 2>&1 &
done

# Start Worker
export MS_ROLE=MS_WORKER
rm -rf ${execute_path}/worker/
mkdir ${execute_path}/worker/
cd ${execute_path}/worker/ || exit
python -s ${self_path}/../train_and_eval_parameter_server_standalone.py --device_target=$DEVICE_TARGET         \
       --epochs=$EPOCHS --data_path=$DATASET_PATH --parameter_server=1  --vocab_size=$VOCAB_SIZE --emb_dim=$EMB_DIM \
       --vocab_cache_size=$VOCAB_CACHE_SIZE --dropout_flag=True >worker.log 2>&1 &
