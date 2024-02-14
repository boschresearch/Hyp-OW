#!/bin/bash

GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/EVAL_HIERARCHICAL_SPLIT.sh

##  for other splits
##  configs/EVAL_OWOD_SPLIT.sh
##  configs/EVAL_OWDETR_SPLIT.sh 
