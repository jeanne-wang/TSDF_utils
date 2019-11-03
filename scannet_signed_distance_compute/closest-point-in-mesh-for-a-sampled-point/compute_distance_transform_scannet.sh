g++ -std=c++11 compute_distance_transform_scannet.cpp -o compute_distance_transform_scannet -lpthread
MESH_ROOT="/home/xiaojwan/CVPR2020/Experiment/SparseConvNet/examples/ScanNet/*/scene*2.ply"
POINT_ROOT="/home/xiaojwan/CVPR2020/Experiment/ScanNet_v2_points_along_surf"
for MESH_FILE in $(ls ${MESH_ROOT})
do
	SCENE_NAME=$(basename ${MESH_FILE})
	SCENE_NAME=${SCENE_NAME//.ply/}
	mkdir -p "./scannet/"${SCENE_NAME}
	echo "==========================="
	echo ${SCENE_NAME}
	./compute_distance_transform_scannet -f ${MESH_FILE} -p $POINT_ROOT/$SCENE_NAME"_sample_along_surf_points.dat" -o "./scannet/"${SCENE_NAME} -v 10000
	echo "==========================="

done

