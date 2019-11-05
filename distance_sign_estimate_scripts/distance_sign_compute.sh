SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet_v2
SCANNET_POINTS_PATH=/cluster/scratch/xiaojwan/ScanNet_v2_points_along_surf
OUTPUT_PATH=/cluster/scratch/xiaojwan/ScanNet_v2_points_along_surf_tsdf
TOOL_PATH=/cluster/scratch/xiaojwan/TSDF_utils
DISTANCE_TOOL_PATH=$TOOL_PATH/scannet_signed_distance_compute/closest-point-in-mesh-for-a-sampled-point

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


    $TOOL_PATH/SensReader/c++/sens \
     	$scene_path/${scene_name}.sens $scene_path/sensor

    python3 $TOOL_PATH/sample_along_camera_rays/sample_along_camera_rays.py \
        --scene_path $scene_path \
        --output_file $scene_path/"points_along_camera_rays.dat" \
        --num_sample 10000 \
        --observed_threshold 0.01 \
        --gaussian_variance 0.1

    $DISTANCE_TOOL_PATH/compute_distance_transform_scannet \
        -f $scene_path/$scene_name"_vh_clean_2.ply" \
        -p $scene_path/"points_along_camera_rays.dat" \
        -o $scene_path \
        -v 10000

    python3 $DISTANCE_TOOL_PATH/python/read_dat_save_npy_scannet.py


    python3 $TOOL_PATH/scannet_signed_distance_compute/estimate_distance_sign_from_2D_cameras.py \
        --scene_path $scene_path \
        --scene_sampled_point_file $SCANNET_POINTS_PATH/$scene_name"_vh_clean_2.npy" \
        --output_file $OUTPUT_PATH/$scene_name".npz" \
        --truncated_distance 0.1

    rm -r ${scene_path}/sensor  

    count=count+1

done



