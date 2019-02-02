dataDir=

python train.py \
    --name default \
    --dataroot ${dataDir}/ContourDrawing/ \
    --checkpoints_dir ${dataDir}/Exp/PhotoSketch/Checkpoints/ \
    --model pix2pix \
    --which_direction AtoB \
    --dataset_mode 1_to_n \
    --no_lsgan \
    --norm batch \
    --pool_size 0 \
    --output_nc 1 \
    --which_model_netG resnet_9blocks \
    --which_model_netD global_np \
    --batchSize 2 \
    --lambda_A 200 \
    --lr 0.0002 \
    --aug_folder width-5 \
    --crop --rotate --color_jitter \
    --niter 400 \
    --niter_decay 400 \
