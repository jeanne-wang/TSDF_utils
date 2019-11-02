REPLICA_PATH=/Users/xiaojwan/CVPR2020/replica_v1
for scene_path in $(ls -d $REPLICA_PATH/*/)
do
	scene_name=$(basename $scene_path)
    echo $scene_name

    python sample_points_along_surface.py --habitat_mesh_file $scene_path/mesh.ply \
        --density 1000 \
        --variance 0.05 \
        --output_path $scene_path/

done



