## 启动训练
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --model convnextv2_large --batch_size 64 --update_freq 8 --blr 1.5e-4 --epochs 3 --warmup_epochs 1 --data_path /mnt/rj200t/patches/output_sdpc --output_dir /mnt/rj200t/ckpt --txt_file_for_train /mnt/rj200t/img_names/train.txt


nohup python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
    --model convnextv2_large \
    --batch_size 64 \
    --update_freq 8 \
    --blr 1.5e-4 \
    --epochs 3 \
    --warmup_epochs 1 \
    --data_path /mnt/rj200t/patches/output_sdpc \
    --output_dir /mnt/rj200t/ckpt \
    --txt_file_for_train /mnt/rj200t/img_names/train.txt \
    > /mnt/rj200t/ckpt/training.log 2>&1 &