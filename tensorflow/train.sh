#!/bin/bash
clear
echo "-----------------------------------"
cur_dir=$(cd "$(dirname $0)";pwd -P)
log_dir="${cur_dir}/logs"
log_file="${log_dir}/log.log"
echo "log dir: " ${log_dir}

python ds2_trainer.py \
    --batch_size=32 \
    --use_dummy=True \
    --max_iter=100 \
    --learning_rate=0.001 \
    --data_format="nhwc" \
    --loss_iter_interval=10 \
    --log_dir=$log_dir \
    --profil_iter=30 \
    --checkpoint_iter=1000 \
    --debug \
    2>&1 | tee -a $log_file

#tensorboard --logdir=$log_dir

