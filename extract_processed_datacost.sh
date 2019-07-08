SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet
OUTPUT_PATH=/cluster/scratch/xiaojwan/ScanNet_DAT
for scene_path in $(ls -d $SCANNET_PATH/scans/scene*)
do
    scene_name=$(basename $scene_path)


    if [ -d ${scene_path}/converted ]; then
        mkdir -p  $OUTPUT_PATH/$scene_name/
        mv $scene_path/converted/* $OUTPUT_PATH/$scene_name/
    fi
    
done



