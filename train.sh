#!/bin/bash
export CUDA_VISIBLE_DEVICES=9,10,11
export TORCH_DISTRIBUTED_DEBUG=INFO
unset http_proxy
unset https_proxy
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml

