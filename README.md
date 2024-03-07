# [TMLR2024 Adapting Contrastive Language-Image Pretrained (CLIP) Models for Out-of-Distribution Detection](https://arxiv.org/abs/2303.05828)

[Arxiv link](https://arxiv.org/abs/2303.05828), [Youtube](https://www.youtube.com/watch?v=eVUKVfxpx8A), [TMLR](https://openreview.net/forum?id=YCgX7sJRF1) , [Slides](https://docs.google.com/presentation/d/17cio7YZQIu2JApCp_YjLKXGmqT8a1qEt-0LqdyKI_cY/edit#slide=id.p)

#### Abstract 
We present a comprehensive experimental study on pre-trained feature extractors for visual out-of-distribution (OOD) detection, focusing on leveraging contrastive language-image pre-trained (CLIP) models. Without fine-tuning on the training data, we are able to establish a positive correlation ($R^2\geq0.92$) between in-distribution classification and unsupervised OOD detection for CLIP models in $4$ benchmarks. We further propose a new, simple, and scalable method called pseudo-label probing (PLP) that adapts vision-language models for OOD detection. Given a set of label names of the training set, PLP trains a linear layer using the pseudo-labels derived from the text encoder of CLIP. Intriguingly, we show that without modifying the weights of CLIP or training additional image/text encoders (i) PLP outperforms the previous state-of-the-art on all $5$ large-scale benchmarks based on ImageNet, specifically by an average AUROC gain of 3.4\% using the largest CLIP model (ViT-G), (ii) linear probing outperforms fine-tuning by large margins for CLIP architectures (i.e. CLIP ViT-H achieves a mean gain of 7.3\% AUROC on average on all ImageNet-based benchmarks), and (iii) billion-parameter CLIP models still fail at detecting feature-based adversarially manipulated OOD images. The code and adversarially created datasets will be made publicly available.

## Project setup 
```
conda create -n plp python=3.7
conda activate plp
pip install -r requirements.txt
```
Next, you need to set the global paths that the embeddings will be saved  `_PRECOMPUTED_PATH`, the folder where the default Pytorch ImageFolder deads the OOD datasets such as NINCO in `_DEFAULT_PATH` and the ImageNet path in `_IMAGENET_PATH`

On `loaders/datasets.py`:
```python
_DEFAULT_PATH = 'path_to_ood_datasets'
_PRECOMPUTED_PATH = './data' 
_IMAGENET_PATH = 'Path_to_imagenet/.../ILSVRC/Data/CLS-LOC'
```
On `model_builders/model_builders.py`:
```python
_PRECOMPUTED_PATH = './data'
```

## Supported model names
```python
from model_builders import available_models
print(available_models())
```
Such as
```
mae_vit_base convnext_base msn_vit_base  mae_vit_large mae_vit_huge ibot_vit_large ibot_vit_large_in21k beit_vit_large_in21k    
timm_vit_base_patch16_224 timm_vit_large_patch16_224 timm_vit_large_patch16_224_in21k timm_convnext_base_in22k timm_vit_large_patch16_224_in21k 
```
- Timm models are dependent on the `timm` version and need to be tested!

#### Supported CLIP models
```
openclip_RN50/openai openclip_ViT-B-16/openai openclip_ViT-B-16/laion2b openclip_ViT-L-14/openai openclip_ViT-L-14/laion2b
openclip_ViT-H-14/laion2b openclip_ViT-bigG-14/laion2b openclip_convnext_base/laion400m_s13b_b51k openclip_convnext_base_w
laion2b openclip_convnext_large_d/laion2b
```
## Dataset names 
We mainly use the following dataset names
```
CIFAR10 CIFAR100 IN1K inat SUN Places IN_O texture NINCO
```
Apart from the CIFAR datasets, you need to download the datasets and place them in  `_DEFAULT_PATH`

### Generated pre-computed image embeddings/representations

Here is an example of how to generate embeddings for one model and one dataset
```
python gen_embeds.py --arch openclip_ViT-L-14/openai --dataset CIFAR10 --batch_size 512  --no_eval_knn --overwrite  
```

# PLP: pseudo label probing
You can modify `run_plp.sh` and run it with `conda activate plp && bash run_plp.sh`. Here is an example for ImageNet(IN1K):

**Important:** You need the precomputed embeddings to run this (`gen_embeds.py`).

```bash
dataset=IN1K
batch_size=8192
arch="openclip_ViT-L-14/openai"
output_dir="experiments/PLP/dataset=$dataset/arch=$arch_name"
python linear_probing.py --arch="$arch" --dataset=$dataset --num_epochs=100 \ 
        --batch_size=$batch_size --output_dir=$output_dir   --seed=$seed  --pseudo_labels=True \
        --pseudo_prompt "a photo of a {c}." "a blurry photo of a {c}." "a photo of many {c}." "a photo of the large {c}." "a photo of the small {c}."
python logit_evaluation.py --probing_path=$output_dir --dataset=$dataset
```

# Supervised linear probing using in-distribution labels

With the same scripts, you can run linear probing using all the supported models. Here is an example on ImageNet. 

```bash
dataset=IN1K
batch_size=8192
arch="openclip_ViT-L-14/openai"
output_dir="experiments/PLP/dataset=$dataset/arch=$arch_name"
python linear_probing.py --arch="$arch" --dataset=$dataset --num_epochs=100 \ 
        --batch_size=$batch_size --output_dir=$output_dir   --seed=$seed  
python logit_evaluation.py --probing_path=$output_dir --dataset=$dataset
```

# Instructions for all the considered comparisons and baselines 

## Step 1. Find `timm` names for baselines
To run the baselines in the polished open-source version, we use `timm==0.9.2`. Use the model names from `timm` using:
```python
import timm
print(timm.list_models("*ft1k*", pretrained=True)) # finetuned imagenet1k models
print(timm.list_models("*in22k", pretrained=True)) # pretrained in21k models 
print(timm.list_models("*in21k*", pretrained=True)) # pretrained in21k models (different naming convention)
```

#### Baseline 1: Fine-tuned 1K models from timm
```bash
export CUDA_VISIBLE_DEVICES=1 && conda activate plp && python baseline_probe_in21k_models.py --dataset IN1K --archs convnext_base_in22k
```

#### Baseline 2: supervised linear probing of IN21K models from timm
```bash
export CUDA_VISIBLE_DEVICES=2 && conda activate plp && python baseline_probe_in21k_models.py --dataset IN1K --archs convnext_base_in22k
```

#### Baseline 3: Fine-tune any model on the supported datasets like CIFAR10 and CIFAR100 
Modify `fine_tune.sh` and pass one of the suggested models above with our naming convention or run
```
torchrun --nproc_per_node=4 finetune.py  --dataset=CIFAR10 \
        --arch=timm_vit_large_patch16_224   --batch_size=64  \
        --epochs 100  --seed=$seed --warmup_epochs 5
```

Then use `notebooks/ood_ft_model.ipynb` to get the OOD detection performance metrics.

#### Baseline 4: CLIP zero-shot Pseudo-MSP (from [Ming et al.](https://arxiv.org/abs/2211.13445))

Supported CLIP model names
```
openclip_RN50/openai openclip_ViT-B-16/openai openclip_ViT-B-16/laion2b openclip_ViT-L-14/openai openclip_ViT-L-14/laion2b
openclip_ViT-H-14/laion2b openclip_ViT-bigG-14/laion2b openclip_convnext_base/laion400m_s13b_b51k openclip_convnext_base_w
laion2b openclip_convnext_large_d/laion2b
```

**Important:** You need the precomputed embeddings to run this!

Mofidy and launch the script `baseline_pseudo_msp_clip.sh`:
```bash
conda activate plp && bash baseline_pseudo_msp_clip.sh
```
Or run it directly via:
```bash
python logit_evaluation.py --clip_arch=openclip_ViT-L-14/openai --dataset "CIFAR10" --out_dists "CIFAR100" --out_dir=$"experiments/pseudo-msp-clip/arch=openclip-ViT-L-14_openai/dataset=CIFAR10"   --eval_maha=True
```


# Download OOD datasets

As explained in [MOS](https://github.com/deeplearning-wisc/large_scale_ood), iNaturalist, SUN, and Places can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

### NINCO dataset download  

We use `NINCO/NINCO_OOD_classes` subfolder as an OOD detection benchmark. For more details check the [NINCO github](https://github.com/j-cb/NINCO/tree/main)

Copied from NINCO github to facilitate reproduction:
- [To evaluate models and view the NINCO images, please download and extract the dataset contained in this tar.gz file.](https://zenodo.org/record/8013288/files/NINCO_all.tar.gz?download=1)
- [Google Drive mirror.](https://drive.google.com/file/d/1lGH9aWDZLGpniqs4JHkgM0Yy4_DsQ0rx/view?usp=share_link)

### ImageNet-O
From [Natural Adversarial Examples from Dan Hendrycks et al.](https://github.com/hendrycks/natural-adv-examples) and their corresponding [github](https://github.com/hendrycks/natural-adv-examples):

__[Download the natural adversarial example dataset ImageNet-O for out-of-distribution detectors here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar).__


# Citation
```
@article{adaloglou2023adapting,
  title={Adapting Contrastive Language-Image Pretrained (CLIP) Models for Out-of-Distribution Detection},
  author={Adaloglou, Nikolas and Michels, Felix and Kaiser, Tim and Kollmann, Markus},
  journal={arXiv e-prints},
  pages={arXiv--2303},
  year={2023}
}
```

# Acknowledgments and Licence
The current codebase is a wild mixture of other GitHub repositories and packages listed below:
- [TEMI](https://github.com/HHU-MMBS/TEMI-official-BMVC2023)
- [DEiT](https://github.com/facebookresearch/deit)
- [iBOT](https://github.com/bytedance/ibot/)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [DINO](https://github.com/facebookresearch/dino)

 The codebase follows the licences of the above codebases.

