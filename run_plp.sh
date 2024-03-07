# conda activate plp && bash run_plp.sh
export CUDA_VISIBLE_DEVICES=1
arch="openclip_ViT-bigG-14/laion2b" # "openclip_ViT-L-14/openai"
arch_name=$(echo $arch | tr '/' '_')
seed=42

# Pseudo label probing
## CIFAR benchmarks
for dataset in CIFAR100 CIFAR10; do
    batch_size=256
    output_dir="experiments/PLP/dataset=$dataset/arch=$arch_name/seed=$seed"
    python linear_probing.py --arch="$arch" --dataset=$dataset --num_epochs=100 --batch_size=$batch_size --output_dir=$output_dir --seed=$seed --pseudo_labels=True \
        --pseudo_prompt "a photo of a {c}." "a blurry photo of a {c}." "a photo of many {c}." "a photo of the large {c}." "a photo of the small {c}."
    python logit_evaluation.py --probing_path=$output_dir --dataset=$dataset
done

# ImageNet experiments (large scale ood evaluations)
# All considered OOD datasets are used by default
dataset=IN1K
batch_size=8192
output_dir="experiments/PLP/dataset=$dataset/arch=$arch_name/seed=$seed"
python linear_probing.py --arch="$arch" --dataset=$dataset --num_epochs=100 --batch_size=$batch_size --output_dir=$output_dir --seed=$seed --pseudo_labels=True \
        --pseudo_prompt "a photo of a {c}." "a blurry photo of a {c}." "a photo of many {c}." "a photo of the large {c}." "a photo of the small {c}."

python logit_evaluation.py --probing_path=$output_dir --dataset=$dataset
