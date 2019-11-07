MESH_ROOT="/media/root/data/ScanNet_v2_data/*/scene*2.ply"
POINT_ROOT="/home/xiaojwan/CVPR2020/Experiment/ScanNet_v2_points_depthmap_camera_rays_0_1"
TOOL_PATH=/home/xiaojwan/CVPR2020/TSDF_utils/scannet_signed_distance_compute/closest-point-in-mesh-for-a-sampled-point
declare -i count=0
for MESH_FILE in $(ls ${MESH_ROOT})
do
    SCENE_NAME=$(basename ${MESH_FILE})
    SCENE_NAME=${SCENE_NAME//_vh_clean_2.ply/}

    if [ $count -eq 500 ]; then
        echo $SCENE_NAME
        break
    fi

    mkdir -p $TOOL_PATH/scannet/${SCENE_NAME}
    echo "==========================="
    echo ${SCENE_NAME}
    ./$TOOL_PATH/compute_distance_transform_scannet \
        -f ${MESH_FILE} \
        -p $POINT_ROOT/$SCENE_NAME"_points_depthmap_camera_rays.dat" \
        -o $TOOL_PATH/scannet/${SCENE_NAME} \
        -v 10000
    echo "==========================="

    count=count+1

done



