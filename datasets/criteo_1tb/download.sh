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

if [ $# -ne 1 ];
then
  echo "Usage: download.sh [DAY_NUM]"
  exit 2
fi

echo "start to download criteo_1tb data of $1 day" 

mkdir -p ~/mindspore/rec/criteo_1tb_data 

cd ~/mindspore/rec/criteo_1tb_data

end_idx=$(($1-1))
for i in `seq 0 $end_idx`
do
  echo "download https://storage.googleapis.com/criteo-cail-datasets/day_$i.gz"
  curl -O -k https://storage.googleapis.com/criteo-cail-datasets/day_$i.gz
  gzip -d day_$i.gz
done

echo "finish to download criteo_1tb data of $1 day" 

