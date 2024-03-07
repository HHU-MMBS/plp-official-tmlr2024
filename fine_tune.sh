#!/usr/bin/bash
# conda activate plp && bash fine_tune.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3" 
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
for seed in 42
do
    for dataset in "CIFAR10"
    do
        port=$(( 47000 + $RANDOM % 1000 ))
        arch=$"timm_vit_large_patch16_224"
        arch_name=$(echo $arch | tr '/' '_')

        torchrun --master_port $port --nproc_per_node=$nproc_per_node finetune.py  --dataset=$dataset \
        --arch $arch   --batch_size=64  --epochs 100  --seed=$seed --warmup_epochs 5
    done
done

