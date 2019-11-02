REPLICA_PATH=/Users/xiaojwan/CVPR2020/replica_v1
for scene_path in $(ls -d $REPLICA_PATH/*/)
do
	scene_name=$(basename $scene_path)
   
	mkdir -p "scene_"$scene_name
	echo "==========================="
	echo $scene_name
	./compute_distance_transform -f $scene_path/mesh.ply -p $scene_path/sample_along_surf_points.dat -o "scene_"$scene_name -v 10000
	echo "==========================="

done

