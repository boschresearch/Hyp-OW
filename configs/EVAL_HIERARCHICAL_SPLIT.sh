#!/usr/bin/env bash

echo running eval of HYP-OW on Hierarchical Split

set -x

EXP_DIR=exps/MOWODB/PROB
PY_ARGS=${@:1}
WANDB_NAME=PROB_V1
 
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "exps/eval_t1_hierarchical_split" --dataset HIERARCHICAL --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set "t1_train_hierarchical" --test_set 'hierarchical_test' --epochs 191 --lr_drop 35\
    --model_type 'hypow' --obj_loss_coef 8e-4 --obj_temp 1.3 --batch_size 1 \
    --pretrain '/workspace/Hyp-OW/exps/hypow_t1_hierarchical_split.pth' --eval --wandb_project ""\
     --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
    --save_buffer  --relabel  --eval \
    --epochs 50  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
    --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
     --all_background --empty_weight 0.1  --unknown_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
    ${PY_ARGS}
   
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "exps/eval_t2_ft_hierarchical_split" --dataset HIERARCHICAL --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
    --train_set "t1_train_hierarchical" --test_set 'hierarchical_test' --epochs 191 --lr_drop 35\
    --model_type 'hypow' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain  '/workspace/Hyp-OW/exps/hypow_t2_ft_hierarchical_split.pth'  --eval --wandb_project ""\
     --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
    --save_buffer  --relabel  --eval \
    --epochs 50  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
    --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
     --all_background --empty_weight 0.1  --unknown_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "exps/eval_t3_ft_hierarchical_split" --dataset HIERARCHICAL --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "t1_train_hierarchical" --test_set 'hierarchical_test' --epochs 191 --lr_drop 35\
    --model_type 'hypow' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain '/workspace/Hyp-OW/exps/hypow_t3_ft_hierarchical_split.pth' --eval --wandb_project ""\
    --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
    --save_buffer  --relabel  --eval \
    --epochs 50  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
    --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
     --all_background --empty_weight 0.1  --unknown_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "exps/eval_t4_ft_hierarchical_split" --dataset HIERARCHICAL --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
    --train_set "t1_train_hierarchical" --test_set 'hierarchical_test' --epochs 191 --lr_drop 35\
    --model_type 'hypow' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain '/workspace/Hyp-OW/exps/ours_t4_ft_hierarchical_split.pth' --eval --wandb_project ""\
      --wandb_name "${WANDB_NAME}_t1" --wandb_project '' --lr_drop 40  --num_queries 100 --logging_freq 40 --use_focal_cls  \
    --save_buffer  --relabel  --eval \
    --epochs 50  --clip_r 1.0 --use_hyperbolic --unknown_weight 0.1  \
    --hyperbolic_c 0.1 --hyperbolic_temp 0.2 --samples_per_category 1 --hyperbolic_coeff 0.05 --checkpoint_period 10 --start_relabelling 0 --emb_per_class 10 \
     --all_background --empty_weight 0.1  --unknown_weight 0.1 --use_max_uperbound  --family_regularizer --family_hyperbolic_coeff 0.02  \
    ${PY_ARGS}
    
    