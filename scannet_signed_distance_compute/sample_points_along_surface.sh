MESH_ROOT="/media/root/data/ScanNet_v2_data/*/scene*2.ply"
OUTPUT_PATH="/media/root/data/ScanNet_v2_data/ScanNet_v2_points_along_surf"
for MESH_FILE in $(ls ${MESH_ROOT})
do
	SCENE_NAME=$(basename ${MESH_FILE})
	SCENE_NAME=${SCENE_NAME//.ply/}

    python3.6 sample_points_along_surface.py --mesh_file $MESH_FILE \
        --density 0 \
        --num_sample 10000 \
        --variance 0.05 \
        --output_path $OUTPUT_PATH/$SCENE_NAME"_"

done



