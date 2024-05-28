#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4
export TORCH_DISTRIBUTED_DEBUG=INFO
unset http_proxy
unset https_proxy
unset no_proxy
python lctgen/main.py  --run-type eval --exp-config cfgs/inference.yaml

