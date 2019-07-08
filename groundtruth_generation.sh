SCANNET_DAT_PATH=/media/root/data/ScanNet_DAT

for scene_path in $(ls -d $SCANNET_DAT_PATH/scene*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name

    # Run total variation on the datacost obtained from all depth
    # to generate the ground truth voxel grid
    rm $scene_path/datacost.ply
    rm $scene_path/groundtruth_datacost.ply
    
    python eval_tv_l1.py \
        --datacost_path $scene_path/groundtruth_datacost.npz \
        --output_path $scene_path/groundtruth_model \
        --label_map_path $scene_path/labels.txt \
        --niter_steps 10 --lam 10 --nclasses 42
done

