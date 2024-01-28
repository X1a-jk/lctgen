#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8,9,10
export TORCH_DISTRIBUTED_DEBUG=INFO
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml
