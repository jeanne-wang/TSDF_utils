REPLICA_PATH=/Users/xiaojwan/CVPR2020/replica_v1
REPLICA_SAMPLED_POINT_PATH=/Users/xiaojwan/CVPR2020/replica-points
OUTPUT_PATH=/Users/xiaojwan/CVPR2020/replica-points-tsdf
for scene_path in $(ls -d $REPLICA_PATH)
do
	scene_name=$(basename $scene_path)

    python estimate_distance_sign_from_2D_cameras.py --frames_2d_path $scene_path/frames/ \
        --scene_semantic_mesh $scene_path/habitat/mesh_semantic.ply \
        --scene_semantic_json_file $scene_path/habitat/info_semantic.json \
        --scene_sampled_point_file $REPLICA_SAMPLED_POINT_PATH/$scene_name".npy" \
        --output_file $OUTPUT_PATH/$scene_name".npz"

done



