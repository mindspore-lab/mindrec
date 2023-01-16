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

echo "Usage: bash run_distribute_train_for_gpu.sh RANK_SIZE EPOCHS DATASET SCHED_HOST SCHED_PORT"

execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export RANK_SIZE=$1
export EPOCHS=$2
export DATASET=$3
export MS_WORKER_NUM=$RANK_SIZE
export MS_SCHED_HOST=$4
export MS_SCHED_PORT=$5

# Start Scheduler
export MS_ROLE=MS_SCHED
rm -rf ${execute_path}/sched/
mkdir ${execute_path}/sched/
cd ${execute_path}/sched/ || exit
  python -s ${self_path}/../train_and_eval_distribute.py  \
      --device_target="GPU"                               \
      --data_path=$DATASET                                \
      --batch_size=16000                                  \
      --epochs=$EPOCHS > sched.log 2>&1 &

# Start Workers
export MS_ROLE=MS_WORKER
for((i=0;i<$MS_WORKER_NUM;i++));
do
  rm -rf ${execute_path}/worker_$i/
  mkdir ${execute_path}/worker_$i/
  cd ${execute_path}/worker_$i/ || exit
  python -s ${self_path}/../train_and_eval_distribute.py  \
      --device_target="GPU"                               \
      --data_path=$DATASET                                \
      --batch_size=16000                                  \
      --epochs=$EPOCHS > worker_$i.log 2>&1 &
done

