#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,3
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
unset http_proxy
unset https_proxy
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml

