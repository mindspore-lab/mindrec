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
# ===========================================================================

# bash run_dist_online_train.sh WORKER_NUM SHED_HOST SCHED_PORT LOCAL_HOST_IP
execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export MS_WORKER_NUM=$1
export MS_SCHED_HOST=$2
export MS_SCHED_PORT=$3
LOCAL_HOST_IP=$4

# Start Scheduler
export MS_ROLE=MS_SCHED
rm -rf ${execute_path}/sched/
mkdir ${execute_path}/sched/
cd ${execute_path}/sched/ || exit
python -s ${self_path}/dist_online_train.py --device_target=GPU  \
       --address=$LOCAL_HOST_IP >sched.log 2>&1 &


# Start Worker
export MS_ROLE=MS_WORKER
for((i=0;i<$MS_WORKER_NUM;i++));
do
  rm -rf ${execute_path}/worker_$i/
  mkdir ${execute_path}/worker_$i/
  cd ${execute_path}/worker_$i/ || exit
  python -s ${self_path}/dist_online_train.py --device_target=GPU  \
        --address=$LOCAL_HOST_IP                               \
       --dropout_flag=True >worker.log 2>&1 &
done

