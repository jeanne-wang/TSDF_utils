SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet_v2

declare -i count=0
for scene_path in $(ls -d $SCANNET_PATH/scans/scene*)
do
	scene_name=$(basename $scene_path)

	if [ $count -eq 500 ]; then
		echo $scene_name
		break
    fi


    if [ -d ${scene_path}/sensor ]; then
    	rm -r ${scene_path}/sensor    
    fi


    $SCANNET_PATH/SensReader/c++/sens \
     	$scene_path/${scene_name}.sens $scene_path/sensor


    # Fuse all depth maps and segmentation to create the ground-truth
    python3 $SCANNET_PATH/TSDF/estimate_distance_sign_from_2D.py \
        --input_path $scene_path/converted/ \
        --output_path $scene_path/converted/groundtruth_datacost \
        --resolution 0.05


    rm -r ${scene_path}/sensor  

    count=count+1

done



