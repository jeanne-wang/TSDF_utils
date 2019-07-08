SCANNET_PATH=/cluster/scratch/xiaojwan/ScanNet

for scene_path in $(ls -d $SCANNET_PATH/scans/scene*)
do
    scene_name=$(basename $scene_path)


    if [ -d ${scene_path}/converted ]; then
    	rm -r $scene_path/converted/
    fi
    
done



