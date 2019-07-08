SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet_v2

declare -i count=0
for scene_path in $(ls -d $SCANNET_PATH/scans/scene*)
do
    scene_name=$(basename $scene_path)

    
    if [ $count -lt 900 ]; then
        count=count+1
        continue
    fi
    
    
    if [ $count -eq 1050 ]; then
        echo $scene_name
        break
    fi
    

    if [ -d ${scene_path}/sensor ]; then
        rm -r ${scene_path}/sensor    
    fi

    if [ -d ${scene_path}/converted ]; then
        rm -r ${scene_path}/converted
    fi


    $SCANNET_PATH/SensReader/c++/sens \
        $scene_path/${scene_name}.sens $scene_path/sensor

   
    
    mkdir -p $scene_path/converted/
    mkdir -p $scene_path/converted/images

    # Convert scannet data into a format read by our code
    python3 $SCANNET_PATH/TSDF/convert_raw_scannet_for_color_sdf_fusion.py \
        --scene_path $scene_path \
        --output_path $scene_path/converted \
        --resolution 0.05


    rm -r ${scene_path}/sensor

    # Fuse all depth maps and color to create the input
    python3 $SCANNET_PATH/TSDF/color_sdf_fusion.py \
        --input_path $scene_path/converted/ \
        --output_path $scene_path/converted/color_sdf \
        --resolution 0.05

    rm -r $scene_path/converted/images

    count=count+1

done



