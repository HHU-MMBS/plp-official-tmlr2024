# conda activate plp && bash generate_representations.sh
export CUDA_VISIBLE_DEVICES=3
for arch in openclip_ViT-L-14/openai
do
    for ds in CIFAR10 CIFAR100
    do
    python gen_embeds.py --arch $arch --dataset $ds --batch_size 512  --no_eval_knn --overwrite  
    done
done

# Example for imageNet
# # ImageNet
# for arch in openclip_ViT-L-14/openai
# do
#     for ds in IN1K inat SUN Places IN_O texture NINCO
#     do
#     python gen_embeds.py --arch $arch --dataset $ds --batch_size 512  --no_eval_knn --overwrite  
#     done
# done
