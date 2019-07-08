SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet_v2
SCANNET_TEST_FRAMES_SEG_PATH=/cluster/scratch/xiaojwan/ScanNet_v2_TEST_FRAMES_SEG

declare -i count=0
for scene_path in $(ls -d $SCANNET_PATH/scans_test/scene*)
do

    scene_name=$(basename $scene_path)

    if [ $count -lt 30 ]; then
        count=count+1
        continue
    fi
    
    if [ $count -eq 40 ]; then
        echo $scene_name
        break
    fi
    

    ## delete existed process result to avoid overwritten
   
    if [ -d ${scene_path}/converted ]; then
        rm -r ${scene_path}/converted
    fi
    

    mkdir -p $scene_path/converted/
    mkdir -p $scene_path/converted/images


    # Convert scannet data into a format read by our code
    python3 $SCANNET_PATH/TSDF/convert_scannet_test_scenes.py \
        --scene_path $scene_path \
        --output_path $scene_path/converted \
        --scene_2d_seg_path $SCANNET_TEST_FRAMES_SEG_PATH \
        --resolution 0.05


    rm -r ${scene_path}/sensor

    # Fuse all depth maps and segmentation to create the ground-truth
    python3 $SCANNET_PATH/TSDF/tsdf_fusion.py \
        --input_path $scene_path/converted/ \
        --output_path $scene_path/converted/datacost \
        --resolution 0.05

    rm -r $scene_path/converted/images

    count=count+1

done



