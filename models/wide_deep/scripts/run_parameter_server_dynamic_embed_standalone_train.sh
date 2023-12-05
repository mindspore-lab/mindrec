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

echo "Usage: bash run_parameter_server_dynamic_embed_standalone_train.sh EPOCHS DEVICE_TARGET DATASET SERVER_NUM
                                                                         SCHED_HOST SCHED_PORT VOCAB_CACHE_SIZE [DEVICE_ID]
      DEVICE_ID is optional, default value is zero"

execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export EPOCHS=$1
export DEVICE_TARGET=$2
export DATASET=$3
export MS_SCHED_NUM=1
export MS_WORKER_NUM=1
export MS_SERVER_NUM=$4
export MS_SCHED_HOST=$5
export MS_SCHED_PORT=$6
export VOCAB_CACHE_SIZE=$7
DEVICE_ID=$8

if [[ ! -n "$7" ]]; then
  export VOCAB_CACHE_SIZE=0
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
cp ${self_path}/../op_precision.ini ${execute_path}/sched/
cd ${execute_path}/sched/ || exit
python -s ${self_path}/../train_and_eval_parameter_server_standalone.py --device_target=$DEVICE_TARGET  \
       --epochs=$EPOCHS --data_path=$DATASET --parameter_server=1                                       \
       --dynamic_embedding=True --sparse=True                                                           \
       --vocab_cache_size=$VOCAB_CACHE_SIZE >sched.log 2>&1 &

# Start Servers
export MS_ROLE=MS_PSERVER
for((i=0;i<$MS_SERVER_NUM;i++));
do
  rm -rf ${execute_path}/server_$i/
  mkdir ${execute_path}/server_$i/
  cp ${self_path}/../op_precision.ini ${execute_path}/server_$i/
  cd ${execute_path}/server_$i/ || exit
  python -s ${self_path}/../train_and_eval_parameter_server_standalone.py --device_target=$DEVICE_TARGET  \
         --epochs=$EPOCHS --data_path=$DATASET --parameter_server=1                                       \
         --dynamic_embedding=True --sparse=True                                                           \
         --vocab_cache_size=$VOCAB_CACHE_SIZE >server_$i.log 2>&1 &
done

# Start Worker
export MS_ROLE=MS_WORKER
rm -rf ${execute_path}/worker/
mkdir ${execute_path}/worker/
cp ${self_path}/../op_precision.ini ${execute_path}/worker_$i/
cd ${execute_path}/worker/ || exit
python -s ${self_path}/../train_and_eval_parameter_server_standalone.py --device_target=$DEVICE_TARGET  \
       --epochs=$EPOCHS --data_path=$DATASET --parameter_server=1                                       \
       --dynamic_embedding=True --sparse=True                                                           \
       --vocab_cache_size=$VOCAB_CACHE_SIZE --dropout_flag=True >worker.log 2>&1 &
