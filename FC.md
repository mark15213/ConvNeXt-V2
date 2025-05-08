## 启动训练
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --model convnextv2_huge --batch_size 16 --update_freq 8 --blr 1.5e-4 --epochs 10 --warmup_epochs 40 --data_path /mnt/rj200t/patches/output_sdpc --output_dir /mnt/rj200t/ckpt


