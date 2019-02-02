@set "dataDir="

python test.py ^
    --name default ^
    --dataroot %dataDir%\ContourDrawing\ ^
    --phase val ^
    --how_many 100 ^
    --checkpoints_dir %dataDir%\Exp\PhotoSketch\Checkpoints\ ^
    --results_dir %dataDir%\Exp\PhotoSketch\Results\ ^
    --model pix2pix ^
    --which_direction AtoB ^
    --dataset_mode 1_to_n ^
    --norm batch ^
    --input_nc 3 ^
    --output_nc 1 ^
    --which_model_netG resnet_9blocks ^
    --which_model_netD global_np ^
    --aug_folder width-5 ^
    --no_dropout ^
