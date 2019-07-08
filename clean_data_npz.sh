SCANNET_PATH=/media/root/data/ScanNet_DAT

for scene_path in $(ls -d $SCANNET_PATH/scene*)
do
    scene_name=$(basename $scene_path)
    rm -r ${scene_path}/groundtruth_model
    rm ${scene_path}/datacost.npz
    
done



