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

echo "Usage: bash run_parameter_server_distribute_train.sh RANK_SIZE EPOCHS DEVICE_TARGET DATASET
                                              SERVER_NUM SCHED_HOST SCHED_PORT
                                              VOCAB_CACHE_SIZE [RANK_TABLE_FILE]
      RANK_TABLE_FILE is optional and only need on Ascend platform"

execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export RANK_SIZE=$1
export EPOCHS=$2
export DEVICE_TARGET=$3
export DATASET=$4
export MS_WORKER_NUM=$RANK_SIZE
export MS_SERVER_NUM=$5
export MS_SCHED_HOST=$6
export MS_SCHED_PORT=$7
export VOCAB_CACHE_SIZE=$8
export RANK_TABLE_FILE=$9 #Only need on Ascend platform

if [[ ! -n "$8" ]]; then
  export VOCAB_CACHE_SIZE=0
fi

# Start Scheduler
export MS_ROLE=MS_SCHED
rm -rf ${execute_path}/sched/
mkdir ${execute_path}/sched/
cp ${self_path}/../op_precision.ini ${execute_path}/sched/
cd ${execute_path}/sched/ || exit
python -s ${self_path}/../train_and_eval_parameter_server_distribute.py                             \
        --device_target=$DEVICE_TARGET --data_path=$DATASET --epochs=$EPOCHS --parameter_server=1   \
        --vocab_cache_size=$VOCAB_CACHE_SIZE >sched.log 2>&1 &

# Start Servers
export MS_ROLE=MS_PSERVER
for((i=0;i<$MS_SERVER_NUM;i++));
do
  rm -rf ${execute_path}/server_$i/
  mkdir ${execute_path}/server_$i/
  cp ${self_path}/../op_precision.ini ${execute_path}/server_$i/
  cd ${execute_path}/server_$i/ || exit
  python -s ${self_path}/../train_and_eval_parameter_server_distribute.py                           \
         --device_target=$DEVICE_TARGET --data_path=$DATASET --epochs=$EPOCHS --parameter_server=1  \
         --vocab_cache_size=$VOCAB_CACHE_SIZE >server_$i.log 2>&1 &
done

# Start Workers
export MS_ROLE=MS_WORKER
for((i=0;i<$MS_WORKER_NUM;i++));
do
  rm -rf ${execute_path}/worker_$i/
  mkdir ${execute_path}/worker_$i/
  cp ${self_path}/../op_precision.ini ${execute_path}/worker_$i/
  cd ${execute_path}/worker_$i/ || exit
  export RANK_ID=$i
  export DEVICE_ID=$i
  python -s ${self_path}/../train_and_eval_parameter_server_distribute.py                     \
    --device_target=$DEVICE_TARGET --data_path=$DATASET --epochs=$EPOCHS --parameter_server=1 \
    --vocab_cache_size=$VOCAB_CACHE_SIZE --dropout_flag=False >worker_$i.log 2>&1 &
done
