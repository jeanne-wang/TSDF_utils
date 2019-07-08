SCANNET_PATH=/media/root/data/ScanNet_v2_eval


for scene_path in $(ls -d $SCANNET_PATH/scans_val/scene*)
do
	scene_name=$(basename $scene_path)

    if [ -d ${scene_path}/sensor ]; then
    	rm -r ${scene_path}/sensor    
    fi

    SensReader/c++/sens \
     	$scene_path/${scene_name}.sens $scene_path/sensor

done



