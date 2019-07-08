SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet_v2

for scene_path in $(ls -d $SCANNET_PATH/scans_test/scene*)
do

    scene_name=$(basename $scene_path)

    mv $scene_path/converted/* $scene_path
    rm -r $scene_path/converted

   

done



