#!/usr/bin/bash
# How to run:
# conda activate plp && bash baseline_pseudo_msp_clip.sh
arch="openclip_ViT-L-14/openai"
arch_name=$(echo $arch | tr '/' '_')

for dataset in CIFAR10 CIFAR100 IN1K; do
    out_dir="experiments/pseudo-msp-clip/arch=$arch_name/dataset=$dataset/"
    python logit_evaluation.py --clip_arch=$arch --dataset=$dataset --out_dir=$out_dir
done