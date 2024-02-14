#!/usr/bin/env bash

echo running training of prob-detr, HIERARCHICAL dataset

set -x

# EXP_DIR=exps/MOWODB/PROB
PY_ARGS=${@:1}
WANDB_NAME=Hierarchical_split

## Task 1 
CUDA_LAUNCH_BLOCKING=1  python -u main_open_world.py \
    --output_dir "exps/hypow_hierarchical_t1" --dataset 'HIERARCHICAL'  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set 't1_train_hierarchical' --test_set 'hierarchical_test' --eval_every 5 --batch_size 3 --lr 1e-4 --num_workers 3  --data_root '/workspace/Hyp-OW/data/OWOD/' \
    --model_type 'hypow'  --obj_temp 1.3  --seed 0 \
    --wandb_name "${WANDB_NAME}_t1" --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
    --save_buffer  --relabel  \
    --epochs 50  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
    --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
     --all_background --empty_weight 0.1  --unknown_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
     --pretrain '/workspace/Hyp-OW/exps/hypow_t1_hierarchical_split.pth'  --load_buffer 
   

## Task 2 fine-tuning
# CUDA_LAUNCH_BLOCKING=1  python -u main_open_world.py \
#     --output_dir "exps/hypow_hierarchical_t2" --dataset 'HIERARCHICAL'  --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
#     --train_set 't2_train_hierarchical' --test_set 'hierarchical_test' --eval_every 5 --batch_size 3 --lr 1e-4 --num_workers 3  --data_root '/workspace/Hyp-OW/data/OWOD/' \
#     --model_type 'hypow'  --obj_temp 1.3  --seed 0 \
#     --wandb_name "${WANDB_NAME}_t1" --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
#     --save_buffer  --relabel  \
#     --epochs 70  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
#     --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
#      --all_background --empty_weight 0.1  --unknown_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
#      --pretrain '/workspace/Hyp-OW/exps/hypow_t2_ft_hierarchical_split.pth'  --load_buffer
     


# ## Task 3 fine-tuning
# CUDA_LAUNCH_BLOCKING=1  python -u main_open_world.py \
#     --output_dir "exps/hypow_hierarchical_t3_ft" --dataset 'HIERARCHICAL'  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
#     --train_set 'hierarchical_t3_ft' --test_set 'hierarchical_test' --eval_every 5 --batch_size 3 --lr 1e-4 --num_workers 3  --data_root '/workspace/Hyp-OW/data/OWOD/' \
#     --model_type 'hypow'  --obj_temp 1.3  --seed 0 \
#     --wandb_name "${WANDB_NAME}_t1" --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
#     --save_buffer  --relabel   \
#     --epochs 210  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
#     --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
#      --all_background --empty_weight 0.1  --unknown_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
#      --pretrain '/workspace/Hyp-OW/exps/hypow_t3_ft_hierarchical_split'  --load_buffer 
     
     


     
# ## Task 4 fine-tuning
# CUDA_LAUNCH_BLOCKING=1  python -u main_open_world.py \
#     --output_dir "exps/hypow_hierarchical_t4_ft" --dataset 'HIERARCHICAL'  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
#     --train_set 'hierarchical_t4_ft' --test_set 'hierarchical_test' --eval_every 5 --batch_size 3 --lr 1e-4 --num_workers 3  --data_root '/workspace/Hyp-OW/data/OWOD/' \
#     --model_type 'hypow'  --obj_temp 1.3  --seed 0 \
#     --wandb_name "${WANDB_NAME}_t1" --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
#     --save_buffer  --relabel \
#     --epochs 300  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
#     --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
#      --all_background --empty_weight 0.1  --unknown_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
#      --pretrain '/workspace/Hyp-OW/exps/hypow_t4_ft_hierarchical_split.pth'  --load_buffer 