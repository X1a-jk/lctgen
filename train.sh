#!/bin/bash
export CUDA_VISIBLE_DEVICES=10
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TORCH_DISTRIBUTED_DEBUG=INFO
unset http_proxy
unset LD_LIBRARY_PATH
unset https_proxy
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml

