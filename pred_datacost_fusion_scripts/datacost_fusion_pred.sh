SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet_v2
SCANNET_TRAIN_FRAMES_SEG_PATH=/cluster/scratch/xiaojwan/ScanNet_v2_TRAIN_FRAMES_SEG


for scene_path in $(ls -d $SCANNET_PATH/scans/scene*)
do

    scene_name=$(basename $scene_path)
    echo $scene_name

    ## delete existed process result to avoid overwritten
    
    if [ -d ${scene_path}/sensor ]; then
        rm -r ${scene_path}/sensor    
    fi
    $SCANNET_PATH/SensReader/c++/sens \
        $scene_path/${scene_name}.sens $scene_path/sensor

    if [ -d ${scene_path}/converted ]; then
        rm -r ${scene_path}/converted
    fi
    

    mkdir -p $scene_path/converted/
    mkdir -p $scene_path/converted/images


    # Convert scannet data into a format read by our code
    python3 $SCANNET_PATH/TSDF/convert_scannet_train_scenes.py \
        --scene_path $scene_path \
        --output_path $scene_path/converted \
        --scene_2d_seg_path $SCANNET_TRAIN_FRAMES_SEG_PATH \
        --resolution 0.05 \
        --frame_rate 50


    rm -r ${scene_path}/sensor

    # Fuse all depth maps and segmentation to create the ground-truth
    python3 $SCANNET_PATH/TSDF/tsdf_fusion.py \
        --input_path $scene_path/converted/ \
        --output_path $scene_path/converted/pred_datacost \
        --resolution 0.05

    rm -r $scene_path/converted/images

done



