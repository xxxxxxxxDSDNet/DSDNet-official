CUDA_VISIBLE_DEVICES=3 python train.py -opt ./options/train/train_tmm22.yml 

CUDA_VISIBLE_DEVICES=2 python train.py -opt  ./options/train/train_nips23.yml  


CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 \
train.py -opt ./options/train/train_tmm22.yml --launcher pytorch

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4322 \
train.py -opt ./options/train/train_nips23.yml  --launcher pytorch